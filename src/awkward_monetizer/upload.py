"""Bulk loading into MonetDB servers that are *not* on this machine.

``COPY INTO t FROM '/tmp/x.csv'`` makes the *server* open the file, so it only
works when client and server share a filesystem. For remote shards the data has
to travel over the existing MAPI connection instead. pymonetdb (>= 1.6)
supports this through ``COPY ... ON CLIENT``: the server asks the client for a
named "file" and a registered :class:`pymonetdb.Uploader` streams the bytes.
Here the "files" are produced from in-memory DataFrames, so nothing touches
disk on either side.

Load methods (``method=`` in :func:`~.ingest.load_tables`):

``"binary"``  ``COPY LITTLE ENDIAN BINARY INTO t FROM 'c1', 'c2', ... ON CLIENT``
              -- one raw little-endian array per column, cast to the column's
              SQL type. Fastest, no text formatting or parsing, and float32
              values widen to DOUBLE exactly. Needs every column to be a fixed
              width numeric/boolean type and the DataFrame columns to match the
              table's column order; otherwise falls back to ``"client"``.
``"client"``  ``COPY INTO t FROM 'name' ON CLIENT`` with CSV streamed from
              memory in row chunks. Works for any column types.
``"copy"``    the original same-host path: temp CSV + server-side COPY INTO.
``"insert"``  ``executemany`` INSERTs (any DB-API driver, incl. monetdbe/SQLite).

:func:`choose_method` picks ``copy`` for a server on localhost, ``binary`` for a
remote pymonetdb server, and ``insert`` for anything else.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

METHODS = ("auto", "binary", "client", "copy", "insert")
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")

# MonetDB SQL type (cursor.description type_code) -> little-endian numpy dtype
_BINARY_DTYPES = {
    "tinyint": "<i1", "smallint": "<i2", "int": "<i4", "bigint": "<i8",
    "real": "<f4", "double": "<f8", "boolean": "u1",
}
_CHUNK_BYTES = 16 * 1024 * 1024
_CSV_ROWS = 200_000


def is_pymonetdb(conn) -> bool:
    return hasattr(conn, "set_uploader") and hasattr(conn, "mapi")


def choose_method(conn, host: str | None = None) -> str:
    """Default load method for a connection (see module docstring)."""
    if not is_pymonetdb(conn):
        return "insert"
    if host is None:
        target = getattr(getattr(conn, "mapi", None), "target", None)
        host = getattr(target, "host", None) or "localhost"   # "" = unix socket
    return "copy" if host.rstrip(".") in LOCAL_HOSTS else "binary"


# --------------------------------------------------------------------------
# Uploader: serves named in-memory "files" to the server
# --------------------------------------------------------------------------
def _uploader_base():
    try:
        from pymonetdb import Uploader
        return Uploader
    except ImportError:  # tests without pymonetdb use a duck-typed connection
        return object


class MemoryUploader(_uploader_base()):
    """Answers ``ON CLIENT`` upload requests from registered producers.

    ``producers`` maps a file name used in the COPY statement to a callable
    ``producer(upload, text_mode)`` that writes the content. Only registered
    names are served -- the server cannot make us read anything else.
    """

    def __init__(self):
        self.producers = {}

    def handle_upload(self, upload, filename, text_mode, skip_amount):
        producer = self.producers.get(filename)
        if producer is None:
            upload.send_error(f"awkward-monetizer: unknown upload {filename!r}")
            return
        producer(upload, text_mode)

    def cancel(self):
        pass


class _Uploading:
    """Context manager: register a MemoryUploader, restore the previous one."""

    def __init__(self, conn):
        self.conn = conn
        self.uploader = MemoryUploader()

    def __enter__(self) -> MemoryUploader:
        self.previous = getattr(self.conn.mapi, "uploader", None)
        self.conn.set_uploader(self.uploader)
        return self.uploader

    def __exit__(self, *exc):
        self.conn.set_uploader(self.previous)


# --------------------------------------------------------------------------
# CSV ON CLIENT
# --------------------------------------------------------------------------
def copy_csv_on_client(conn, cur, table: str, df: pd.DataFrame) -> None:
    name = f"{table}.csv"

    def produce(upload, text_mode):
        w = upload.text_writer()
        for start in range(0, len(df), _CSV_ROWS):
            df.iloc[start:start + _CSV_ROWS].to_csv(
                w, index=False, header=False, lineterminator="\n")

    with _Uploading(conn) as up:
        up.producers[name] = produce
        cur.execute(
            f"COPY {len(df)} RECORDS INTO {table} FROM '{name}' ON CLIENT "
            "USING DELIMITERS ',', '\\n', '\"' NULL AS ''")


# --------------------------------------------------------------------------
# BINARY ON CLIENT
# --------------------------------------------------------------------------
_COLUMNS_SQL = (
    "SELECT c.name, c.type FROM sys.columns c "
    "JOIN sys.tables t ON c.table_id = t.id "
    "JOIN sys.schemas s ON t.schema_id = s.id "
    "WHERE t.name = %s AND s.name = CURRENT_SCHEMA ORDER BY c.number")


def table_columns(cur, table: str) -> list[tuple[str, str]]:
    """``[(column, sql_type), ...]`` in table order, from the MonetDB catalog.

    (A ``SELECT * ... LIMIT 0`` probe is rejected by the server with "Illegal
    argument" once the table has rows, so the catalog is used instead.)
    """
    cur.execute(_COLUMNS_SQL, (table,))
    cols = [(str(name), str(typ).lower()) for name, typ in cur.fetchall()]
    if not cols:
        raise ValueError(f"table {table!r} not found in the current schema")
    return cols


def binary_columns(df: pd.DataFrame, cols: list[tuple[str, str]]):
    """Cast each DataFrame column to its table column's binary dtype.

    Returns ``[(name, little-endian ndarray), ...]`` in table order, or None if
    the table can't be loaded in binary (non-numeric type, column mismatch).
    Raises ValueError if a value doesn't fit the SQL type.
    """
    if [c for c, _ in cols] != list(df.columns):
        return None
    out = []
    for col, sqltype in cols:
        dt = _BINARY_DTYPES.get(sqltype)
        if dt is None:
            return None
        src = df[col].to_numpy()
        if src.dtype.kind not in "biuf":
            return None
        target = np.dtype(dt)
        if src.dtype.kind in "iu" and target.kind == "i" and len(src):
            info = np.iinfo(target)
            lo, hi = int(src.min()), int(src.max())
            # the type's minimum is MonetDB's NULL sentinel -> not allowed
            if lo <= info.min or hi > info.max:
                raise ValueError(f"{col}: values [{lo}, {hi}] don't fit {sqltype}")
        if src.dtype.kind == "f" and target.kind in "iu":
            return None
        out.append((col, np.ascontiguousarray(src.astype(target, copy=False))))
    return out


def copy_binary_on_client(conn, cur, table: str, arrays) -> None:
    names = [f"{table}.{col}.bin" for col, _ in arrays]

    def producer(arr):
        def produce(upload, text_mode):
            if text_mode:
                upload.send_error("expected a binary upload")
                return
            w = upload.binary_writer()
            buf = memoryview(arr).cast("B")
            for start in range(0, len(buf), _CHUNK_BYTES):
                w.write(buf[start:start + _CHUNK_BYTES])
        return produce

    with _Uploading(conn) as up:
        for name, (_, arr) in zip(names, arrays, strict=True):
            up.producers[name] = producer(arr)
        files = ", ".join(f"'{n}'" for n in names)
        cur.execute(f"COPY LITTLE ENDIAN BINARY INTO {table} FROM {files} ON CLIENT")


def load_frame(conn, cur, table: str, df: pd.DataFrame, method: str) -> str:
    """Load one DataFrame with ``method`` ("binary"/"client"); returns the
    method actually used (binary falls back to client CSV when needed)."""
    if not is_pymonetdb(conn):
        raise ValueError(f"method {method!r} needs a pymonetdb server connection")
    if df.empty:
        return method
    if method == "binary":
        arrays = binary_columns(df, table_columns(cur, table))
        if arrays is not None:
            copy_binary_on_client(conn, cur, table, arrays)
            return "binary"
        print(f"  note: {table} not binary-loadable, using CSV ON CLIENT",
              file=sys.stderr)
    copy_csv_on_client(conn, cur, table, df)
    return "client"
