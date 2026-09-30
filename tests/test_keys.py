"""Collision-free event_id: (file_id << 40) | entry."""
import json
import sqlite3

import numpy as np
import pytest

from awkward_monetizer import db
from awkward_monetizer.cli import main
from awkward_monetizer.datasets import DATASETS
from awkward_monetizer.ingest import build_tables, ingest_root_chunked, read_root
from awkward_monetizer.keys import (
    ENTRY_BITS,
    MAX_ENTRY,
    MAX_FILE_ID,
    Manifest,
    build_manifest,
    file_id_range,
    make_event_id,
    make_event_ids,
    split_event_ids,
)
from awkward_monetizer.sampledata import write_nanoaod

NANO = DATASETS["nanoaod"]


class _Cursor(sqlite3.Cursor):
    def executemany(self, sql, rows):
        return super().executemany(sql.replace("%s", "?"), rows)


class _Connection(sqlite3.Connection):
    def cursor(self):
        return super().cursor(factory=_Cursor)


def _nano_db():
    conn = sqlite3.connect(":memory:", factory=_Connection)
    db.create_schema(conn, db.read_schema("nanoaod"))
    return conn


def _ids(conn, table="events"):
    return [r[0] for r in conn.execute(
        f"SELECT event_id FROM {table} ORDER BY event_id")]


# ---------------------------------------------------------------- encoding
def test_pack_unpack_roundtrip():
    entries = np.array([0, 1, 12345, MAX_ENTRY], dtype=np.int64)
    for fid in (0, 1, 7, MAX_FILE_ID):
        eid = make_event_ids(fid, entries)
        assert eid.dtype == np.int64 and (eid >= 0).all()   # fits signed BIGINT
        f, e = split_event_ids(eid)
        assert (f == fid).all() and (e == entries).all()


def test_file_id_zero_is_legacy_numbering():
    assert (make_event_ids(0, np.arange(10)) == np.arange(10)).all()


def test_ranges_are_disjoint_and_ordered():
    lo0, hi0 = file_id_range(0)
    lo1, hi1 = file_id_range(1)
    assert (lo0, hi0) == (0, 1 << ENTRY_BITS) and hi0 == lo1 < hi1
    assert make_event_id(1, 0) == lo1 and make_event_id(0, MAX_ENTRY) == hi0 - 1
    _, top = file_id_range(MAX_FILE_ID)
    assert top == 2**63                        # last id is 2**63 - 1


@pytest.mark.parametrize("fid, entry", [(-1, 0), (MAX_FILE_ID + 1, 0),
                                        (0, -1), (0, MAX_ENTRY + 1)])
def test_out_of_range_rejected(fid, entry):
    with pytest.raises(ValueError):
        make_event_id(fid, entry)


# ---------------------------------------------------------------- tables
def test_build_tables_encodes_file_id_everywhere(nano_file):
    events, _ = read_root(nano_file, NANO, None, entry_stop=50)
    t = build_tables(events, NANO, id_offset=100, file_id=3)
    f, e = split_event_ids(t["events"]["event_id"])
    assert (f == 3).all() and (e == np.arange(100, 150)).all()
    for coll in ("jets", "muons"):
        assert set(t[coll]["event_id"]) <= set(t["events"]["event_id"])


def test_branch_event_id_rejects_file_id(nano_file):
    ds = DATASETS["nanoaod"]
    import dataclasses
    by_branch = dataclasses.replace(ds, event_id="event")
    events, _ = read_root(nano_file, by_branch, None, entry_stop=5)
    with pytest.raises(ValueError, match="file_id"):
        build_tables(events, by_branch, file_id=1)


# ---------------------------------------------------------------- ingest
def test_ids_independent_of_chunking(nano_file):
    a, b = _nano_db(), _nano_db()
    ingest_root_chunked(nano_file, NANO, a, step_size=1000, file_id=5,
                        use_copy_into=False)
    ingest_root_chunked(nano_file, NANO, b, step_size=137, file_id=5,
                        use_copy_into=False)
    assert _ids(a) == _ids(b)
    assert _ids(a, "jets") == _ids(b, "jets")


def test_split_file_equals_single_job(nano_file):
    whole, parts = _nano_db(), _nano_db()
    ingest_root_chunked(nano_file, NANO, whole, step_size=400, file_id=2,
                        use_copy_into=False)
    # two "workers" each take half of the same file
    ingest_root_chunked(nano_file, NANO, parts, step_size=400, file_id=2,
                        entry_start=700, use_copy_into=False)
    ingest_root_chunked(nano_file, NANO, parts, step_size=400, file_id=2,
                        entry_stop=700, use_copy_into=False)
    for table in ("events", "jets", "muons"):
        assert _ids(whole, table) == _ids(parts, table)


def test_two_files_same_tables_no_collisions(tmp_path, nano_file):
    other = write_nanoaod(str(tmp_path / "other.root"), n_events=900, seed=9)
    conn = _nano_db()
    n0 = ingest_root_chunked(nano_file, NANO, conn, step_size=500, file_id=0,
                             use_copy_into=False)
    n1 = ingest_root_chunked(other, NANO, conn, step_size=500, file_id=1,
                             use_copy_into=False)
    n, distinct = conn.execute(
        "SELECT count(*), count(DISTINCT event_id) FROM events").fetchone()
    assert n == distinct == n0 + n1
    orphans = conn.execute("SELECT count(*) FROM jets WHERE event_id NOT IN "
                           "(SELECT event_id FROM events)").fetchone()[0]
    assert orphans == 0
    # the file each event came from is recoverable in SQL
    per_file = dict(conn.execute(
        f"SELECT event_id / {1 << ENTRY_BITS}, count(*) FROM events GROUP BY 1"))
    assert per_file == {0: n0, 1: n1}


def test_replace_only_touches_own_range(tmp_path, nano_file):
    other = write_nanoaod(str(tmp_path / "other.root"), n_events=600, seed=4)
    conn = _nano_db()
    ingest_root_chunked(nano_file, NANO, conn, step_size=500, file_id=0,
                        use_copy_into=False)
    ingest_root_chunked(other, NANO, conn, step_size=500, file_id=1,
                        use_copy_into=False)
    before = {t: _ids(conn, t) for t in ("events", "jets", "muons")}
    # a worker re-runs file 1 (e.g. after a crash): no duplicates, file 0 intact
    ingest_root_chunked(other, NANO, conn, step_size=250, file_id=1,
                        replace=True, use_copy_into=False)
    assert {t: _ids(conn, t) for t in before} == before

    # replace on a sub-range leaves the rest of the same file alone
    ingest_root_chunked(other, NANO, conn, step_size=250, file_id=1,
                        entry_start=100, entry_stop=200, replace=True,
                        use_copy_into=False)
    assert {t: _ids(conn, t) for t in before} == before


# ---------------------------------------------------------------- manifest
def test_manifest_stable_when_extended(tmp_path):
    paths = [str(tmp_path / f"f{i}.root") for i in range(3)]
    m = build_manifest(paths[:2])
    out = tmp_path / "m.json"
    m.to_json(str(out))
    m2 = Manifest.from_json(str(out)).add(paths)       # f0, f1 already present
    assert [m2.file_id(p) for p in paths] == [0, 1, 2]
    with pytest.raises(KeyError):
        m2.file_id(str(tmp_path / "nope.root"))
    with pytest.raises(ValueError):
        Manifest([m2.files[0], m2.files[0]])


def test_cli_manifest_and_ingest(tmp_path, nano_file, capsys):
    from unittest.mock import MagicMock, patch
    other = write_nanoaod(str(tmp_path / "b.root"), n_events=300, seed=3)
    man = str(tmp_path / "manifest.json")
    main(["manifest", nano_file, other, "--out", man, "--count-entries",
          "--tree", "Events"])
    with open(man) as fh:
        data = json.load(fh)
    assert [f["file_id"] for f in data["files"]] == [0, 1]
    assert [f["n_entries"] for f in data["files"]] == [1500, 300]

    conn = _nano_db()
    proxy = MagicMock(wraps=conn)
    proxy.close.side_effect = lambda: None
    with patch("awkward_monetizer.db.open_server", return_value=proxy):
        for f in (nano_file, other):
            main(["ingest", f, "--dataset", "nanoaod", "--step-size", "400",
                  "--manifest", man, "--replace", "--no-copy-into"])
    fids, entries = split_event_ids(_ids(conn))
    assert np.bincount(fids).tolist() == [1500, 300]
    assert entries.max() == 1499


def test_cli_flag_conflicts(nano_file, tmp_path):
    with pytest.raises(SystemExit):
        main(["ingest", nano_file, "--dataset", "nanoaod", "--replace"])
    with pytest.raises(SystemExit):
        main(["ingest", nano_file, "--dataset", "nanoaod", "--step-size", "10",
              "--replace", "--truncate", "--dry-run"])
    with pytest.raises(SystemExit):
        main(["ingest", nano_file, "--file-id", "-1"])
