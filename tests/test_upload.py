"""Bulk upload over the client connection (COPY ... ON CLIENT) for remote shards.

Most tests use a fake pymonetdb connection that plays the server's side of the
file-transfer protocol: it asks the registered Uploader for each named file and
decodes the bytes exactly as MonetDB would (little-endian columns for BINARY,
CSV for text). ``test_real_server_*`` run the same loads against a real MonetDB
server when AWKWARD_MONETIZER_TEST_DB is set, e.g.

    monetdb create hep_test && monetdb release hep_test
    AWKWARD_MONETIZER_TEST_DB=localhost/hep_test pytest tests/test_upload.py
"""
import io
import os
import re

import numpy as np
import pandas as pd
import pytest

from awkward_monetizer import db
from awkward_monetizer.datasets import DATASETS
from awkward_monetizer.ingest import (
    build_tables,
    ingest_root_chunked,
    load_tables,
    read_root,
)
from awkward_monetizer.keys import split_event_ids
from awkward_monetizer.upload import binary_columns, choose_method

NANO = DATASETS["nanoaod"]
_SQL_TO_NP = {"tinyint": "<i1", "smallint": "<i2", "int": "<i4", "bigint": "<i8",
              "real": "<f4", "double": "<f8", "boolean": "u1"}


# ------------------------------------------------------------ fake server
def _schema_types(name):
    """{table: [(column, sql_type)]} parsed from a packaged DDL file."""
    out = {}
    for table, body in re.findall(r"CREATE TABLE (\w+) \((.*?)\);",
                                  db.read_schema(name), re.DOTALL):
        cols = []
        for line in body.splitlines():
            line = line.split("--")[0].strip().rstrip(",")
            if line:
                col, typ = line.split()[:2]
                cols.append((col, typ.lower()))
        out[table] = cols
    return out


class _Upload:
    def __init__(self):
        self.raw = io.BytesIO()
        self.error = None
        self._text = None

    def binary_writer(self):
        return self.raw

    def text_writer(self):
        if self._text is None:
            self._text = io.TextIOWrapper(self.raw, encoding="utf-8", newline="\n",
                                          write_through=True)
        return self._text

    def send_error(self, msg):
        self.error = msg

    def data(self):
        if self._text is not None:
            self._text.flush()
        return self.raw.getvalue()


class _Target:
    def __init__(self, host):
        self.host = host


class _Mapi:
    def __init__(self, host):
        self.uploader = None
        self.target = _Target(host)


class FakeServer:
    """Duck-typed pymonetdb connection backed by DataFrames."""

    def __init__(self, schema="nanoaod", host="node7.example.org"):
        self.mapi = _Mapi(host)
        self.types = _schema_types(schema)
        self.tables = {t: [] for t in self.types}
        self.statements = []

    # DB-API bits used by the loader
    def set_uploader(self, uploader):
        self.mapi.uploader = uploader

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        pass

    def frame(self, table):
        cols = [c for c, _ in self.types[table]]
        parts = self.tables[table]
        return (pd.concat(parts, ignore_index=True) if parts
                else pd.DataFrame(columns=cols))

    def _upload(self, name, text_mode):
        up = _Upload()
        self.mapi.uploader.handle_upload(up, name, text_mode, 0)
        if up.error:
            raise RuntimeError(f"upload refused: {up.error}")
        return up.data()


class _Cursor:
    def __init__(self, server):
        self.s = server
        self.description = None
        self.rows = []

    def execute(self, sql, params=None):
        s = self.s
        s.statements.append(sql)
        if sql.startswith("SELECT c.name, c.type FROM sys.columns"):
            assert "CURRENT_SCHEMA" in sql and "ORDER BY c.number" in sql
            self.rows = list(s.types.get(params[0], []))
        elif m := re.fullmatch(
                r"COPY LITTLE ENDIAN BINARY INTO (\w+) FROM (.+) ON CLIENT", sql):
            table, files = m[1], re.findall(r"'([^']+)'", m[2])
            cols = s.types[table]
            assert len(files) == len(cols), "one file per column, in table order"
            data = {c: np.frombuffer(s._upload(f, False), dtype=_SQL_TO_NP[t])
                    for (c, t), f in zip(cols, files, strict=True)}
            assert len({len(v) for v in data.values()}) == 1, "ragged columns"
            s.tables[table].append(pd.DataFrame(data))
        elif m := re.fullmatch(
                r"COPY (\d+) RECORDS INTO (\w+) FROM '([^']+)' ON CLIENT "
                r"USING DELIMITERS ',', '\\n', '\"' NULL AS ''", sql):
            n, table, name = int(m[1]), m[2], m[3]
            cols = s.types[table]
            df = pd.read_csv(io.BytesIO(s._upload(name, True)), header=None,
                             names=[c for c, _ in cols],
                             dtype={c: _SQL_TO_NP[t] for c, t in cols})
            assert len(df) == n
            s.tables[table].append(df)
        elif m := re.fullmatch(r"DELETE FROM (\w+)(.*)", sql):
            assert not m[2], "range deletes not needed here"
            s.tables[m[1]] = []
        else:
            raise AssertionError(f"fake server can't run: {sql}")

    def fetchall(self):
        rows, self.rows = self.rows, []
        return rows

    def close(self):
        pass


@pytest.fixture
def tables(nano_file):
    events, _ = read_root(nano_file, NANO, None, entry_stop=600)
    return build_tables(events, NANO, file_id=3)


# ------------------------------------------------------------ binary
def test_binary_upload_is_bit_exact(tables):
    srv = FakeServer()
    load_tables(srv, tables, method="binary")
    assert any("BINARY" in q for q in srv.statements)
    for name, src in tables.items():
        got = srv.frame(name)
        assert list(got.columns) == list(src.columns)
        for c in src.columns:
            want = src[c].to_numpy()
            # float32 kinematics widen to DOUBLE exactly; ints keep their values
            np.testing.assert_array_equal(got[c].to_numpy(),
                                          want.astype(got[c].dtype), err_msg=c)
    fids, _ = split_event_ids(srv.frame("events")["event_id"])
    assert set(fids.tolist()) == {3}


def test_client_csv_upload_roundtrip(tables):
    srv = FakeServer()
    load_tables(srv, tables, method="client")
    assert not any("BINARY" in q for q in srv.statements)
    for name, src in tables.items():
        got = srv.frame(name)
        assert len(got) == len(src)
        for c in src.columns:
            np.testing.assert_allclose(got[c].to_numpy(), src[c].to_numpy(),
                                       rtol=1e-6, err_msg=c)


def test_binary_and_csv_agree_through_chunked_ingest(nano_file):
    got = {}
    for method in ("binary", "client"):
        srv = FakeServer()
        n = ingest_root_chunked(nano_file, NANO, srv, step_size=500, file_id=1,
                                method=method)
        assert n == 1500
        got[method] = srv
    for t in ("events", "jets", "muons"):
        a, b = got["binary"].frame(t), got["client"].frame(t)
        assert a["event_id"].tolist() == b["event_id"].tolist()
        np.testing.assert_allclose(a.iloc[:, 1:].to_numpy(float),
                                   b.iloc[:, 1:].to_numpy(float), rtol=1e-6)


def test_binary_falls_back_to_csv_when_not_numeric():
    srv = FakeServer()
    srv.types["events"] = [("event_id", "bigint"), ("label", "varchar")]
    df = pd.DataFrame({"event_id": np.arange(3), "label": ["a", "b", "c"]})
    _SQL_TO_NP["varchar"] = object          # let the fake server parse text
    try:
        load_tables(srv, {"events": df}, method="binary")
    finally:
        del _SQL_TO_NP["varchar"]
    assert srv.statements[-1].startswith("COPY 3 RECORDS INTO events")
    assert srv.frame("events")["label"].tolist() == ["a", "b", "c"]


def test_binary_columns_rules():
    cols = [("a", "int"), ("b", "double")]
    ok = binary_columns(pd.DataFrame({"a": np.array([1, 2], np.uint32),
                                      "b": np.array([0.5, 1.5], np.float32)}), cols)
    assert [x.dtype.str for _, x in ok] == ["<i4", "<f8"]
    assert binary_columns(pd.DataFrame({"b": [1.0], "a": [1]}), cols) is None  # order
    assert binary_columns(pd.DataFrame({"a": [1.5], "b": [1.0]}), cols) is None  # f->i
    with pytest.raises(ValueError, match="don't fit"):
        binary_columns(pd.DataFrame({"a": np.array([2**31], np.int64),
                                     "b": [1.0]}), cols)
    with pytest.raises(ValueError, match="don't fit"):          # NULL sentinel
        binary_columns(pd.DataFrame({"a": np.array([-(2**31)], np.int64),
                                     "b": [1.0]}), cols)
    big = pd.DataFrame({"a": np.array([2**63], np.uint64), "b": [1.0]})
    with pytest.raises(ValueError):
        binary_columns(big, [("a", "bigint"), ("b", "double")])


def test_uploader_only_serves_registered_names_and_is_restored(tables):
    srv = FakeServer()
    sentinel = object()
    srv.mapi.uploader = sentinel
    load_tables(srv, {"jets": tables["jets"]}, method="binary")
    assert srv.mapi.uploader is sentinel                 # previous handler back
    from awkward_monetizer.upload import MemoryUploader
    up, u = _Upload(), MemoryUploader()
    u.handle_upload(up, "../../etc/passwd", True, 0)
    assert up.error and up.data() == b""


def test_choose_method():
    assert choose_method(FakeServer(host="localhost")) == "copy"
    assert choose_method(FakeServer(host="")) == "copy"             # unix socket
    assert choose_method(FakeServer(host="node3.cluster")) == "binary"
    import sqlite3
    assert choose_method(sqlite3.connect(":memory:")) == "insert"


def test_non_server_connection_rejects_upload_methods(tables):
    import sqlite3
    with pytest.raises(ValueError, match="pymonetdb"):
        load_tables(sqlite3.connect(":memory:"), tables, method="binary")
    with pytest.raises(ValueError, match="unknown load method"):
        load_tables(FakeServer(), tables, method="parquet")


def test_shard_method_selection():
    from awkward_monetizer.shards import Shard, _IngestTask
    task = _IngestTask({}, NANO, 100, None, "auto", False)
    assert task._method(Shard("a", (0, 1), host="localhost")) == "copy"
    assert task._method(Shard("b", (0, 1), host="node1")) == "binary"
    assert task._method(Shard("c", (0, 1), backend="embedded", target="/x")) == "insert"
    forced = _IngestTask({}, NANO, 100, None, "client", False)
    assert forced._method(Shard("a", (0, 1), host="localhost")) == "client"
    with pytest.raises(ValueError):
        forced._method(Shard("c", (0, 1), backend="embedded", target="/x"))


# ------------------------------------------------------------ real server
_REAL = os.environ.get("AWKWARD_MONETIZER_TEST_DB")


def _real_conn():
    hostport, _, database = _REAL.partition("/")
    host, _, port = hostport.partition(":")
    return db.open_server(database=database or "hep_test", host=host or "localhost",
                          port=int(port or 50000))


@pytest.mark.skipif(not _REAL, reason="set AWKWARD_MONETIZER_TEST_DB=host/db "
                                      "(a scratch database; it is overwritten)")
@pytest.mark.parametrize("method", ["binary", "client", "copy", "insert"])
def test_real_server_load_methods_agree(nano_file, method):
    import awkward as ak

    from awkward_monetizer.adl import events_from_root
    from awkward_monetizer.shards import fetch_shard_events
    conn = _real_conn()
    try:
        db.create_schema(conn, db.read_schema("nanoaod"))
        n = ingest_root_chunked(nano_file, NANO, conn, step_size=400, method=method)
        conn.commit()
        assert n == 1500
        got = fetch_shard_events(conn)
    finally:
        conn.close()
    want = events_from_root(nano_file)
    assert ak.to_list(got.event_id) == ak.to_list(want.event_id)
    for coll in ("jets", "muons"):
        assert ak.to_list(ak.num(got[coll])) == ak.to_list(ak.num(want[coll]))
        # binary ships the widened float32 bits: exact. INSERT sends repr()
        # text, which MonetDB parses to within 1 ulp; CSV paths format float32
        # as short decimal text.
        tol = {"binary": 0, "insert": 1e-15}.get(method, 1e-6)
        np.testing.assert_allclose(ak.to_numpy(ak.flatten(got[coll].pt)),
                                   ak.to_numpy(ak.flatten(want[coll].pt)),
                                   rtol=tol, atol=0)
