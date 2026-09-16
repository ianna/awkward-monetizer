import sqlite3

import awkward as ak
import numpy as np
import pandas as pd
import pytest

from awkward_monetizer import benchmark
from awkward_monetizer.reconstruct import (
    fetch_dimuon_events,
    fetch_tables,
    reconstruct_events,
)
from awkward_monetizer.timing import stage_time


@pytest.mark.parametrize('where', [None, 'mass BETWEEN 60 AND 120', 'events.mass > 80', '1=0'])
@pytest.mark.parametrize('project', [False, True])
@pytest.mark.parametrize('id_offset', [0, 2**53])
def test_join_projection_preserves_selected_events_and_objects(where, project, id_offset):
    events = pd.DataFrame({'event_id': [2, 0, 1], 'mass': [90., 90., 50.],
                           'unused': ['zero muons', 'selected', 'excluded']})
    muons = pd.DataFrame({'event_id': [1, 0, 0], 'muon_index': [0, 1, 0],
                          'e': [25., 45., 45.], 'px': [25., -45., 45.],
                          'py': [0., 0., 0.], 'pz': [0., 0., 0.],
                          'charge': [1, -1, 1], 'mass': [999., 999., 999.]})
    events['event_id'] += id_offset
    muons['event_id'] += id_offset
    with sqlite3.connect(':memory:') as conn:
        events.to_sql('events', conn, index=False)
        muons.to_sql('muons', conn, index=False)
        statements = []
        conn.set_trace_callback(statements.append)
        stages = {}
        kwargs = {'event_columns': ('event_id',),
                  'object_columns': ('event_id', 'muon_index', 'e', 'px', 'py', 'pz', 'charge')}
        ev, obj = fetch_tables(conn, where, timings=stages, **(kwargs if project else {}))
        # Independent reference selection, retaining events without objects.
        expected_ev = pd.read_sql_query('SELECT * FROM events' +
                                       (f' WHERE {where}' if where else ''), conn)
        expected_obj = muons[muons.event_id.isin(expected_ev.event_id)]
        expected = reconstruct_events(expected_ev, expected_obj)
        actual = reconstruct_events(ev, obj, timings=stages)
        assert ak.to_list(actual.event_id) == ak.to_list(expected.event_id)
        np.testing.assert_array_equal(benchmark.zmumu_masses(actual),
                                      benchmark.zmumu_masses(expected))
        assert ak.to_list(ak.num(actual.muons)) == ak.to_list(ak.num(expected.muons))
        if project:
            assert tuple(ev.columns) == kwargs['event_columns']
            assert tuple(obj.columns) == kwargs['object_columns']
        if where:
            assert 'JOIN (SELECT event_id FROM events' in statements[1]
            assert ' IN (' not in statements[1]
        assert set(stages) == {'execute', 'fetch', 'dataframe', 'sort_group', 'array_build'}
        assert all(value >= 0 for value in stages.values())
        direct = fetch_dimuon_events(conn, where)
        assert ak.to_list(direct.event_id) == ak.to_list(expected.event_id)
        for field in ('muon_index', 'e', 'px', 'py', 'pz', 'charge'):
            assert ak.to_list(direct.muons[field]) == ak.to_list(expected.muons[field])
        np.testing.assert_array_equal(benchmark.zmumu_masses(direct),
                                      benchmark.zmumu_masses(expected))


def test_stage_time_adds_repeated_calls(monkeypatch):
    ticks = iter([1., 3., 4., 7.])
    monkeypatch.setattr('awkward_monetizer.timing.time.perf_counter', lambda: next(ticks))
    stages = {}
    with stage_time(stages, 'fetch'):
        pass
    with stage_time(stages, 'fetch'):
        pass
    assert stages == {'fetch': 5.}


def test_root_books_actions_in_one_event_loop(cms_root, monkeypatch):
    root = pytest.importorskip('ROOT')
    factory = root.RDataFrame
    frames = []

    def capture(*args):
        frame = factory(*args)
        frames.append(frame)
        return frame

    monkeypatch.setattr(root, 'RDataFrame', capture)
    result = benchmark.backend_rdataframe(cms_root, 'dimuon', repeats=1)
    reference = benchmark.backend_awkward(cms_root, 'dimuon', repeats=1)
    assert len(frames) == 2  # Warmup and one measured query.
    assert [frame.GetNRuns() for frame in frames] == [1, 1]
    assert result['metric']['n'] == reference['metric']['n']
    assert result['metric']['mean'] == pytest.approx(reference['metric']['mean'])


def test_hybrid_stage_report_excludes_warmup(monkeypatch, capsys):
    from awkward_monetizer import db

    tables = {
        'events': pd.DataFrame({'event_id': [0], 'mass': [90.]}),
        'muons': pd.DataFrame({'event_id': [0, 0], 'muon_index': [0, 1],
                              'e': [45., 45.], 'px': [45., -45.],
                              'py': [0., 0.], 'pz': [0., 0.], 'charge': [1, -1]}),
    }
    conn = sqlite3.connect(':memory:')
    monkeypatch.setattr(db, 'open_embedded', lambda _: conn)
    monkeypatch.setattr(benchmark, 'read_root', lambda *args: (None, 'events'))
    monkeypatch.setattr(benchmark, 'build_tables', lambda *args: tables)

    def load(conn, tables, **kwargs):
        for name, frame in tables.items():
            frame.to_sql(name, conn, if_exists='append', index=False)

    monkeypatch.setattr(benchmark, 'load_tables', load)
    result = benchmark.backend_hybrid('unused.root', 'dimuon', repeats=2)
    assert result['metric'] == {'n': 1, 'mean': 90.}
    assert set(result['stages']) == {
        'execute', 'fetch', 'column_arrays', 'sort_group', 'array_build', 'analysis',
    }
    for timing in result['stages'].values():
        assert timing['n'] == 2
        assert 0 <= timing['min'] <= timing['median']
    benchmark._print_table({'hybrid': result})
    assert 'hybrid query stages' in capsys.readouterr().out
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute('SELECT 1')
