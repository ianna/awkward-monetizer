"""Event-level columns that make cuts pushable: stored multiplicities
(``n_jets``, ``n_muons``), optional trigger flags (NULL when a file's menu lacks
the path), and reading NanoAOD from either a TTree or an RNTuple."""
import awkward as ak
import numpy as np
import pandas as pd
import pytest
import uproot

from awkward_monetizer import db
from awkward_monetizer.datasets import DATASETS
from awkward_monetizer.ingest import (
    build_tables,
    find_tree_name,
    ingest_root_chunked,
    is_tree_class,
    open_tree,
    read_root,
)
from awkward_monetizer.reconstruct import fetch_tables, reconstruct_multi
from awkward_monetizer.sampledata import make_nanoaod, write_nanoaod

NANO = DATASETS["nanoaod"]


@pytest.fixture(scope="module", params=["rntuple", "ttree"])
def nano_any(request, tmp_path_factory):
    p = tmp_path_factory.mktemp(request.param) / "nano.root"
    return request.param, write_nanoaod(str(p), n_events=800, seed=5,
                                        fmt=request.param)


def test_is_tree_class():
    assert is_tree_class("TTree")
    assert is_tree_class("ROOT::RNTuple")
    assert is_tree_class("ROOT::Experimental::RNTuple")
    assert not is_tree_class("TH1F")


def test_auto_detect_ttree_and_rntuple(nano_any):
    fmt, path = nano_any
    with uproot.open(path) as f:
        cls = next(iter(f.classnames().values()))
        assert (cls == "TTree") == (fmt == "ttree")
        assert find_tree_name(f) == "Events"
        assert open_tree(f, None).num_entries == 800


def test_ttree_and_rntuple_give_identical_tables(tmp_path):
    tabs = {}
    for fmt in ("rntuple", "ttree"):
        p = write_nanoaod(str(tmp_path / f"{fmt}.root"), 300, seed=8, fmt=fmt)
        events, _ = read_root(p, NANO, tree=None)          # auto-detect
        tabs[fmt] = build_tables(events, NANO)
    for name in ("events", "jets", "muons"):
        pd.testing.assert_frame_equal(tabs["rntuple"][name], tabs["ttree"][name],
                                      check_dtype=False)


def test_kinematics_independent_of_triggers():
    a, b = make_nanoaod(200, seed=3), make_nanoaod(200, seed=3, triggers=False)
    assert "HLT_IsoMu24" in a and "HLT_IsoMu24" not in b
    for k in b:
        assert ak.to_list(a[k]) == ak.to_list(b[k]), k


def test_stored_counts_and_flags(nano_any):
    _, path = nano_any
    events, _ = read_root(path, NANO, None)
    ev = build_tables(events, NANO)["events"]
    np.testing.assert_array_equal(ev["n_muons"], ak.num(events.Muon_pt, axis=1))
    np.testing.assert_array_equal(ev["n_jets"], ak.num(events.Jet_pt, axis=1))
    # present paths are plain bools, copied from the file
    assert ev["hlt_isomu24"].dtype == bool
    np.testing.assert_array_equal(ev["hlt_isomu24"], ak.to_numpy(events.HLT_IsoMu24))
    assert not (ev["hlt_isomu27"] & ~ev["hlt_isomu24"]).any()  # 27 implies 24 here
    # the path the sample lacks is NULL, never False
    assert ev["hlt_mu17_mu8_dz"].isna().all()
    # column order must match the DDL (COPY INTO is positional)
    ddl = db.read_schema("nanoaod").split("CREATE TABLE jets")[0]
    assert [c for c in ev.columns if c not in ddl] == []
    positions = [ddl.index(f" {c} ") for c in ev.columns]
    assert positions == sorted(positions)


def test_file_without_any_trigger_branches(tmp_path):
    p = write_nanoaod(str(tmp_path / "notrig.root"), 100, seed=1, triggers=False)
    events, _ = read_root(p, NANO, None)
    ev = build_tables(events, NANO)["events"]
    for col in NANO.flags:
        assert ev[col].isna().all(), col
    # without a database, NULL flags reconstruct as an option type
    arr = reconstruct_multi(ev, {})
    assert ak.to_list(arr.hlt_isomu24[:3]) == [None, None, None]


def test_pushable_cuts_in_monetdb(nano_any):
    """Embedded MonetDB (the real engine) via INSERT; the CSV/binary upload
    paths' NULL handling is covered by test_upload's fake server, and the
    engine's own BOOLEAN parsing of CSV True/False/'' by test_csv_booleans."""
    monetdbe = pytest.importorskip("monetdbe")
    _, path = nano_any
    events, _ = read_root(path, NANO, None)
    conn = monetdbe.connect(":memory:")
    try:
        db.create_schema(conn, db.read_schema("nanoaod"))
        n = ingest_root_chunked(path, NANO, conn, step_size=300, method="insert")
        assert n == 800

        def count(where):
            # fetchall, not fetchone: with monetdbe 0.11, dropping a cursor read
            # via fetchone() can corrupt event_ids returned by a later fetchall()
            # on another cursor (the fetch_tables call below)
            return conn.execute(
                f"SELECT count(*) FROM events WHERE {where}").fetchall()[0][0]

        n_mu = ak.to_numpy(ak.num(events.Muon_pt, axis=1))
        trig = ak.to_numpy(events.HLT_IsoMu24)
        # object-count cut straight on events, no GROUP BY over muons
        assert count("n_muons >= 2") == int((n_mu >= 2).sum())
        assert count("hlt_isomu24") == int(trig.sum())
        assert count("hlt_isomu24 AND n_muons >= 2") == int((trig & (n_mu >= 2)).sum())
        # missing path is NULL: neither true nor false
        assert count("hlt_mu17_mu8_dz IS NULL") == 800
        assert count("hlt_mu17_mu8_dz = FALSE") == 0

        ev_df, mu_df = fetch_tables(conn, "hlt_isomu24 AND n_muons >= 2")
        got = reconstruct_multi(ev_df, {"muons": mu_df})
        assert len(got) == int((trig & (n_mu >= 2)).sum())
        assert ak.all(ak.num(got.muons, axis=1) == got.n_muons)
        assert ak.all(got.hlt_isomu24)
    finally:
        conn.close()


def test_csv_booleans(tmp_path):
    """What the CSV writers emit for flags (pandas True/False, '' for NA) is
    what MonetDB's COPY parses as BOOLEAN / NULL."""
    monetdbe = pytest.importorskip("monetdbe")
    df = pd.DataFrame({"i": [1, 2, 3],
                       "b": pd.array([True, False, pd.NA], dtype="boolean")})
    p = tmp_path / "b.csv"
    df.to_csv(p, index=False, header=False)
    conn = monetdbe.connect(":memory:")
    try:
        conn.execute("CREATE TABLE t (i INT, b BOOLEAN)")
        conn.execute(f"COPY 3 RECORDS INTO t FROM '{p}' "
                     "USING DELIMITERS ',', '\\n', '\"' NULL AS ''")
        rows = conn.execute("SELECT i, b FROM t ORDER BY i").fetchall()
    finally:
        conn.close()
    assert [(i, None if b is None else bool(b)) for i, b in rows] == \
        [(1, True), (2, False), (3, None)]
