"""Scatter-gather over range-partitioned shards (item 4 of the scaling plan)."""
import importlib.util
import json
import sqlite3

import awkward as ak
import numpy as np
import pytest

from awkward_monetizer import adl, db
from awkward_monetizer.cli import main
from awkward_monetizer.datasets import DATASETS
from awkward_monetizer.ingest import ingest_root_chunked
from awkward_monetizer.keys import build_manifest, file_id_range, split_event_ids
from awkward_monetizer.sampledata import write_nanoaod
from awkward_monetizer.shards import (
    Shard,
    ShardMap,
    combine,
    fetch_events,
    fetch_shard_events,
    ingest_manifest,
    map_events,
    verify,
)

NANO = DATASETS["nanoaod"]
_has_monetdbe = importlib.util.find_spec("monetdbe") is not None


class _Cursor(sqlite3.Cursor):
    def executemany(self, sql, rows):
        return super().executemany(sql.replace("%s", "?"), rows)


class _Connection(sqlite3.Connection):
    def cursor(self):
        return super().cursor(factory=_Cursor)


def _sqlite(path):
    return lambda: sqlite3.connect(path, factory=_Connection)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("shards")
    return [write_nanoaod(str(d / f"f{i}.root"), n_events=n, seed=10 + i)
            for i, n in enumerate([700, 500, 900])]


@pytest.fixture
def sharded(tmp_path, files):
    """3 files over 2 SQLite-backed shards, plus a single-DB reference."""
    man = build_manifest(files)
    smap = ShardMap([
        Shard("s0", (0, 2), connector=_sqlite(str(tmp_path / "s0.db"))),
        Shard("s1", (2, 10), connector=_sqlite(str(tmp_path / "s1.db"))),
    ])
    counts = ingest_manifest(man, smap, NANO, step_size=300, tree="Events",
                             use_copy_into=False, create_schema=True)
    ref = _sqlite(str(tmp_path / "ref.db"))()
    db.create_schema(ref, db.read_schema("nanoaod"))
    for f in man.files:
        ingest_root_chunked(f.path, NANO, ref, step_size=300, file_id=f.file_id,
                            use_copy_into=False)
    yield smap, counts, ref
    ref.close()


# ---------------------------------------------------------------- shard map
def test_shard_map_validation_and_lookup(tmp_path):
    smap = ShardMap([Shard("b", (5, 9), host="h1"), Shard("a", (0, 5), host="h0")])
    assert [s.name for s in smap] == ["a", "b"]           # sorted by range
    assert smap.shard_for(4).name == "a" and smap.shard_for(5).name == "b"
    with pytest.raises(KeyError):
        smap.shard_for(9)
    assert smap.shards[1].event_id_range == (file_id_range(5)[0],
                                             file_id_range(8)[1])
    with pytest.raises(ValueError, match="overlap"):
        ShardMap([Shard("a", (0, 5)), Shard("b", (4, 9))])
    with pytest.raises(ValueError):
        Shard("x", (3, 3))
    with pytest.raises(ValueError):
        Shard("x", (0, 1), backend="embedded")            # no target
    out = tmp_path / "shards.json"
    smap.to_json(str(out))
    again = ShardMap.from_json(str(out))
    assert [(s.name, s.file_ids, s.host) for s in again] == \
           [("a", (0, 5), "h0"), ("b", (5, 9), "h1")]


def test_split_is_even_and_open_ended():
    smap = ShardMap.split(10, [{"name": f"s{i}"} for i in range(3)])
    assert [s.file_ids[0] for s in smap] == [0, 3, 7]
    assert smap.shard_for(10**6).name == "s2"             # appended files fit


# ---------------------------------------------------------------- ingest
def test_ingest_routes_files_to_owning_shard(sharded):
    smap, counts, _ = sharded
    assert counts == {"s0": 700 + 500, "s1": 900}
    reports = verify(smap)
    assert all(not r["problems"] for r in reports), reports
    conn = smap.shards[1].connect()
    fids, _ = split_event_ids([r[0] for r in conn.execute("SELECT event_id FROM jets")])
    conn.close()
    assert set(fids.tolist()) == {2}


def test_ingest_manifest_is_idempotent(sharded, files):
    smap, counts, _ = sharded
    again = ingest_manifest(build_manifest(files), smap, NANO, step_size=500,
                            tree="Events", use_copy_into=False)
    assert again == counts
    assert [r["events"] for r in verify(smap)] == [1200, 900]


def test_verify_detects_misplaced_and_orphan_rows(sharded):
    smap, _, _ = sharded
    conn = smap.shards[0].connect()
    stray = file_id_range(5)[0]                  # belongs to shard s1
    conn.execute(f"INSERT INTO events (event_id) VALUES ({stray})")
    conn.execute("INSERT INTO jets (event_id, jet_index) VALUES (-7, 0)")
    conn.commit()
    conn.close()
    problems = verify(smap)[0]["problems"]
    assert any("outside shard range" in p for p in problems)
    assert any("orphan" in p for p in problems)


# ---------------------------------------------------------------- query
@pytest.mark.parametrize("where", [None, "met_pt > 40", "n_jets >= 3"])
def test_fetch_events_matches_single_database(sharded, where):
    smap, _, ref = sharded
    got = fetch_events(smap, where=where)
    want = fetch_shard_events(ref, where)
    assert ak.to_list(got.event_id) == ak.to_list(want.event_id)
    assert np.all(np.diff(ak.to_numpy(got.event_id)) > 0)      # globally ordered
    for coll in ("jets", "muons"):
        assert ak.to_list(ak.num(got[coll])) == ak.to_list(ak.num(want[coll]))
        assert ak.almost_equal(got[coll].pt, want[coll].pt)


def test_map_events_adl_matches_single_database(sharded):
    smap, _, ref = sharded
    got = map_events(smap, adl.run_all)
    want = adl.run_all(fetch_shard_events(ref))
    assert got.keys() == want.keys()
    for name in want:
        np.testing.assert_allclose(got[name], want[name], err_msg=name)


def test_map_events_aggregates_and_empty_shards(sharded):
    smap, _, _ = sharded
    n = map_events(smap, len)
    assert n == 2100
    # a selection that is empty on shard s1 (file 2) but not on s0
    lo = file_id_range(2)[0]
    where = f"event_id < {lo} AND met_pt > 20"
    per_shard = map_events(smap, len, where=where, reduce=None)
    assert per_shard[1] is None and per_shard[0] > 0
    assert map_events(smap, len, where=where) == per_shard[0]
    assert len(fetch_events(smap, where=where)) == per_shard[0]
    assert len(fetch_events(smap, where="met_pt < 0")) == 0     # empty everywhere


def test_events_only_projection(sharded):
    smap, _, _ = sharded
    ev = fetch_events(smap, "met_pt > 40", collections=(),
                      event_columns=("event_id", "met_pt"))
    assert ak.fields(ev) == ["event_id", "met_pt"] and ak.all(ev.met_pt > 40)


def test_combine_rules():
    assert combine([1, None, 2]) == 3
    assert combine([None, None]) is None
    merged = combine([{"a": 1, "b": np.array([1])}, {"a": 2, "b": np.array([2])}])
    assert merged["a"] == 3 and merged["b"].tolist() == [1, 2]
    assert combine([[1], [2, 3]]) == [1, 2, 3]
    hist = pytest.importorskip("hist")
    h = [hist.Hist.new.Reg(4, 0, 4).Double().fill([i]) for i in range(3)]
    assert combine(h).sum() == 3


def test_embedded_shards_need_processes(tmp_path):
    smap = ShardMap([Shard("a", (0, 1), backend="embedded", target=str(tmp_path / "a")),
                     Shard("b", (1, 2), backend="embedded", target=str(tmp_path / "b"))])
    from awkward_monetizer.shards import scatter
    with pytest.raises(ValueError, match="process"):
        scatter(smap, lambda s, c: None, executor="thread")


# ---------------------------------------------------------------- real MonetDB
@pytest.mark.skipif(not _has_monetdbe, reason="monetdbe not installed")
def test_cli_end_to_end_two_embedded_monetdb_shards(tmp_path, files, capsys):
    man, shards_json = str(tmp_path / "manifest.json"), str(tmp_path / "shards.json")
    main(["manifest", *files, "--out", man])
    main(["shards", "init", "--manifest", man, "--out", shards_json,
          "--shard", f"embedded:{tmp_path / 'db0'}",
          "--shard", f"embedded:{tmp_path / 'db1'}"])
    with open(shards_json) as fh:
        assert [s["file_ids"][0] for s in json.load(fh)["shards"]] == [0, 2]
    main(["shards", "ingest", shards_json, "--manifest", man, "--tree", "Events",
          "--create-schema", "--step-size", "400"])
    main(["shards", "verify", shards_json])
    main(["shards", "adl", shards_json, "--where", "met_pt > 10"])
    out = capsys.readouterr().out
    assert "done: 2100 events on 2 shards." in out
    assert out.count(": OK") == 2
    assert "Q6: trijet pT" in out

    # the scattered ADL result equals running on all files in one process
    smap = ShardMap.from_json(shards_json)
    got = map_events(smap, adl.q4_met_ge2jets40, where="met_pt > 10")
    ev = ak.concatenate([adl.events_from_root(f) for f in files])
    want = adl.q4_met_ge2jets40(ev[ev.met_pt > 10])
    np.testing.assert_allclose(np.sort(got), np.sort(want))


# ---------------------------------------------------------------- errors
def test_bad_file_in_manifest_fails_before_loading(tmp_path, files):
    import uproot
    dimuon = str(tmp_path / "dimuon.root")
    with uproot.recreate(dimuon) as f:
        f["events"] = {"pt1": np.array([1.0, 2.0])}
    smap = ShardMap([Shard("s0", (0, 2), connector=_sqlite(str(tmp_path / "a.db"))),
                     Shard("s1", (2, 9), connector=_sqlite(str(tmp_path / "b.db")))])
    for sh in smap:
        c = sh.connect()
        db.create_schema(c, db.read_schema("nanoaod"))
        c.close()
    with pytest.raises(ValueError, match="dimuon.root") as exc:
        ingest_manifest(build_manifest([dimuon, *files]), smap, NANO,
                        use_copy_into=False)
    assert "nothing was loaded" in str(exc.value)
    assert [r["events"] for r in verify(smap)] == [0, 0]


def test_scatter_reports_every_failing_shard(tmp_path):
    from awkward_monetizer.shards import ShardError, scatter
    smap = ShardMap([Shard(f"s{i}", (i, i + 1),
                           connector=_sqlite(str(tmp_path / f"{i}.db")))
                     for i in range(3)])

    def task(shard, conn):
        if shard.name != "s1":
            raise RuntimeError(f"boom on {shard.name}")
        return 1
    with pytest.raises(ShardError) as exc:
        scatter(smap, task)
    assert set(exc.value.errors) == {"s0", "s2"}
    assert "2 of 3 shard(s) failed" in str(exc.value)


def test_manifest_cli_warns_and_fresh(tmp_path, files, capsys):
    man = str(tmp_path / "m.json")
    main(["manifest", files[0], files[1], "--out", man])
    main(["manifest", files[1], "--out", man])
    assert "also contains files not given here" in capsys.readouterr().out
    main(["manifest", files[1], "--out", man, "--fresh"])
    with open(man) as fh:
        assert [f["path"] for f in json.load(fh)["files"]] == [files[1]]
