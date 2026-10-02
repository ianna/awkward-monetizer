"""Configurable key layout (file bits / entry bits) and key migration."""
import importlib.util
import json
import os
import sqlite3

import awkward as ak
import numpy as np
import pytest

from awkward_monetizer import adl, db
from awkward_monetizer.cli import main
from awkward_monetizer.datasets import DATASETS
from awkward_monetizer.ingest import ingest_root_chunked
from awkward_monetizer.keys import (
    DEFAULT_LAYOUT,
    EXABYTE_LAYOUT,
    FileEntry,
    KeyLayout,
    Manifest,
    build_manifest,
    migrate_tables,
)
from awkward_monetizer.sampledata import write_nanoaod
from awkward_monetizer.shards import (
    Shard,
    ShardMap,
    fetch_events,
    fetch_shard_events,
    ingest_manifest,
    map_events,
    verify,
)

NANO = DATASETS["nanoaod"]
TABLES = ["events", "jets", "muons"]
_has_monetdbe = importlib.util.find_spec("monetdbe") is not None


class _Cursor(sqlite3.Cursor):
    def executemany(self, sql, rows):
        return super().executemany(sql.replace("%s", "?"), rows)


class _Connection(sqlite3.Connection):
    def cursor(self):
        return super().cursor(factory=_Cursor)


def _sqlite(path):
    return lambda: sqlite3.connect(path, factory=_Connection)


def _nano_db(path=":memory:"):
    conn = _sqlite(path)()
    db.create_schema(conn, db.read_schema("nanoaod"))
    return conn


def _ids(conn, table="events"):
    return [r[0] for r in conn.execute(
        f"SELECT event_id FROM {table} ORDER BY event_id")]


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("layout")
    return [write_nanoaod(str(d / f"f{i}.root"), n_events=n, seed=20 + i)
            for i, n in enumerate([400, 300, 500])]


# ---------------------------------------------------------------- layout
def test_exabyte_layout_capacity_and_bounds():
    lay = EXABYTE_LAYOUT
    assert (lay.file_bits, lay.entry_bits) == (32, 31)
    assert lay.max_file_id == 2**32 - 1 and lay.max_entry == 2**31 - 1
    top = lay.make_event_ids(lay.max_file_id, [lay.max_entry])[0]
    assert int(top) == 2**63 - 1                      # fits a signed BIGINT
    f, e = lay.split_event_ids([top])
    assert (int(f[0]), int(e[0])) == (lay.max_file_id, lay.max_entry)
    assert lay.file_id_range(3) == (3 << 31, 4 << 31)
    # a file id that the default layout cannot represent
    with pytest.raises(ValueError):
        DEFAULT_LAYOUT.make_event_ids(2**23, [0])
    assert int(lay.make_event_ids(2**23, [5])[0]) == (2**23 << 31) | 5
    with pytest.raises(ValueError):
        lay.make_event_ids(0, [2**31])
    for bad in (0, 63, -1):
        with pytest.raises(ValueError):
            KeyLayout(bad)


def test_convert_event_ids_is_exact_or_refuses():
    old, new = DEFAULT_LAYOUT, EXABYTE_LAYOUT
    eids = np.concatenate([old.make_event_ids(f, np.arange(0, 2000, 7))
                           for f in (0, 1, 9, 8_000_000)])
    conv = old.convert_event_ids(eids, new)
    f0, e0 = old.split_event_ids(eids)
    f1, e1 = new.split_event_ids(conv)
    assert (f0 == f1).all() and (e0 == e1).all()
    assert (new.convert_event_ids(conv, old) == eids).all()       # round trip
    with pytest.raises(ValueError, match="entry numbers"):        # entry too big
        old.convert_event_ids(old.make_event_ids(1, [2**31]), new)
    with pytest.raises(ValueError, match="file ids"):             # file id too big
        new.convert_event_ids(new.make_event_ids(2**23, [0]), old)


# ---------------------------------------------------------------- manifest
def test_manifest_records_layout(tmp_path, files):
    m = build_manifest(files, layout=EXABYTE_LAYOUT)
    out = str(tmp_path / "m.json")
    m.to_json(out)
    with open(out) as fh:
        assert json.load(fh)["entry_bits"] == 31
    again = Manifest.from_json(out)
    assert again.layout == EXABYTE_LAYOUT
    assert [again.file_id(p) for p in files] == [0, 1, 2]
    with pytest.raises(ValueError, match="do not fit"):
        Manifest([FileEntry(0, "big.root", 2**31 + 1)], layout=EXABYTE_LAYOUT)
    with pytest.raises(ValueError, match="do not fit"):
        Manifest([FileEntry(0, "big.root", 2**31 + 1)]).with_layout(EXABYTE_LAYOUT)


def test_cli_manifest_layout_cannot_change_on_append(tmp_path, files, capsys):
    man = str(tmp_path / "m.json")
    main(["manifest", files[0], "--out", man, "--entry-bits", "31"])
    assert "32 file bits, 31 entry bits" in capsys.readouterr().out
    main(["manifest", files[1], "--out", man])                 # append keeps layout
    assert Manifest.from_json(man).layout == EXABYTE_LAYOUT
    with pytest.raises(SystemExit):
        main(["manifest", files[2], "--out", man, "--entry-bits", "40"])


# ---------------------------------------------------------------- ingest / shards
def test_ingest_and_shards_under_exabyte_layout(tmp_path, files):
    man = build_manifest(files, layout=EXABYTE_LAYOUT)
    specs = [{"name": f"s{i}", "connector": _sqlite(str(tmp_path / f"s{i}.db"))}
             for i in range(2)]
    smap = ShardMap.split(len(files), specs, layout=EXABYTE_LAYOUT)
    assert smap.shards[-1].file_ids[1] == 2**32            # open-ended, new range
    counts = ingest_manifest(man, smap, NANO, step_size=250, tree="Events",
                             create_schema=True)
    assert sum(counts.values()) == 1200
    assert all(not r["problems"] for r in verify(smap))

    ref = _nano_db()
    for f in man.files:
        ingest_root_chunked(f.path, NANO, ref, step_size=250, file_id=f.file_id,
                            use_copy_into=False, layout=EXABYTE_LAYOUT)
    got, want = fetch_events(smap, "met_pt > 30"), fetch_shard_events(ref, "met_pt > 30")
    assert ak.to_list(got.event_id) == ak.to_list(want.event_id)
    fids, entries = EXABYTE_LAYOUT.split_event_ids(_ids(ref))
    assert np.bincount(fids).tolist() == [400, 300, 500] and entries.max() == 499

    # the JSON shard map carries the layout
    out = str(tmp_path / "shards.json")
    smap.to_json(out)
    assert ShardMap.from_json(out).layout == EXABYTE_LAYOUT


def test_mixed_layouts_are_rejected(tmp_path, files):
    man40 = build_manifest(files)                                   # default layout
    smap31 = ShardMap([Shard("s0", (0, 9), connector=_sqlite(str(tmp_path / "a.db")))],
                      layout=EXABYTE_LAYOUT)
    with pytest.raises(ValueError, match="key layouts differ"):
        ingest_manifest(man40, smap31, NANO, tree="Events")
    with pytest.raises(ValueError, match="exceed"):                 # range too wide
        ShardMap([Shard("s0", (0, 2**24))])


# ---------------------------------------------------------------- migration
def test_migrate_tables_matches_direct_ingest(files):
    old, new = DEFAULT_LAYOUT, EXABYTE_LAYOUT
    a, b = _nano_db(), _nano_db()
    for fid, f in enumerate(files):
        ingest_root_chunked(f, NANO, a, step_size=250, file_id=fid,
                            use_copy_into=False, layout=old)
        ingest_root_chunked(f, NANO, b, step_size=250, file_id=fid,
                            use_copy_into=False, layout=new)
    before = {t: _ids(a, t) for t in TABLES}
    assert migrate_tables(a, TABLES, old, new, check_only=True)["events"] == 1200
    assert {t: _ids(a, t) for t in TABLES} == before             # nothing changed
    counts = migrate_tables(a, TABLES, old, new)
    assert counts["events"] == 1200
    for t in TABLES:
        assert _ids(a, t) == _ids(b, t)                          # same as direct
    orphans = a.execute("SELECT count(*) FROM jets WHERE event_id NOT IN "
                        "(SELECT event_id FROM events)").fetchone()[0]
    assert orphans == 0
    migrate_tables(a, TABLES, new, old)                          # and back again
    assert {t: _ids(a, t) for t in TABLES} == before


def test_migration_refuses_and_changes_nothing_when_keys_do_not_fit(files):
    conn = _nano_db()
    ingest_root_chunked(files[0], NANO, conn, step_size=250, file_id=2,
                        use_copy_into=False)
    before = {t: _ids(conn, t) for t in TABLES}
    with pytest.raises(ValueError, match="entry numbers"):
        migrate_tables(conn, TABLES, DEFAULT_LAYOUT, KeyLayout(5))   # entries > 31
    with pytest.raises(ValueError, match="file ids"):
        migrate_tables(conn, TABLES, DEFAULT_LAYOUT, KeyLayout(62))  # 1 file bit
    assert {t: _ids(conn, t) for t in TABLES} == before


def test_migration_survives_old_new_key_collisions():
    """A new key may equal another row's old key (here 1<<6 == 4<<4)."""
    old, new = KeyLayout(4), KeyLayout(6)
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE events (event_id BIGINT PRIMARY KEY)")
    eids = np.concatenate([old.make_event_ids(f, np.arange(4)) for f in range(6)])
    conn.executemany("INSERT INTO events VALUES (?)", [(int(e),) for e in eids])
    assert int(new.make_event_ids(1, [0])[0]) in set(eids.tolist())
    migrate_tables(conn, ["events"], old, new)
    assert _ids(conn) == sorted(old.convert_event_ids(eids, new).tolist())


@pytest.mark.skipif(not _has_monetdbe, reason="monetdbe not installed")
def test_cli_migrate_two_embedded_monetdb_shards(tmp_path, files, capsys):
    man, sj = str(tmp_path / "manifest.json"), str(tmp_path / "shards.json")
    main(["manifest", *files, "--out", man, "--count-entries", "--tree", "Events"])
    main(["shards", "init", "--manifest", man, "--out", sj,
          "--shard", f"embedded:{tmp_path / 'db0'}",
          "--shard", f"embedded:{tmp_path / 'db1'}"])
    main(["shards", "ingest", sj, "--manifest", man, "--tree", "Events",
          "--create-schema", "--step-size", "300"])
    before = map_events(ShardMap.from_json(sj), adl.q4_met_ge2jets40)

    main(["shards", "migrate", sj, "--manifest", man, "--entry-bits", "31"])
    main(["shards", "verify", sj])
    out = capsys.readouterr().out
    assert "done: 40 -> 31 entry bits" in out and out.count(": OK") == 2

    smap = ShardMap.from_json(sj)
    assert smap.layout == EXABYTE_LAYOUT == Manifest.from_json(man).layout
    np.testing.assert_allclose(map_events(smap, adl.q4_met_ge2jets40), before)
    # re-ingesting under the migrated layout is idempotent
    main(["shards", "ingest", sj, "--manifest", man, "--tree", "Events",
          "--step-size", "300"])
    assert [r["events"] for r in verify(smap)] == [700, 500]


# ---------------------------------------------------------------- real server
_REAL = os.environ.get("AWKWARD_MONETIZER_TEST_DB")


@pytest.mark.skipif(not _REAL, reason="set AWKWARD_MONETIZER_TEST_DB=host/db "
                                      "(a scratch database; it is overwritten)")
def test_real_server_migration_matches_direct_ingest(files):
    hostport, _, database = _REAL.partition("/")
    host, _, port = hostport.partition(":")
    conn = db.open_server(database=database or "hep_test", host=host or "localhost",
                          port=int(port or 50000))
    try:
        db.create_schema(conn, db.read_schema("nanoaod"))
        for fid, f in enumerate(files):
            ingest_root_chunked(f, NANO, conn, step_size=250, file_id=fid,
                                method="binary")
        counts = migrate_tables(conn, TABLES, DEFAULT_LAYOUT, EXABYTE_LAYOUT)
        cur = conn.cursor()
        cur.execute("SELECT event_id FROM events ORDER BY event_id")
        got = [int(r[0]) for r in cur.fetchall()]
        cur.execute("SELECT count(*) FROM jets WHERE event_id NOT IN "
                    "(SELECT event_id FROM events)")
        orphans = int(cur.fetchall()[0][0])
    finally:
        conn.close()
    want = np.concatenate([EXABYTE_LAYOUT.make_event_ids(fid, np.arange(n))
                           for fid, n in enumerate([400, 300, 500])])
    assert counts["events"] == 1200 and got == want.tolist() and orphans == 0
