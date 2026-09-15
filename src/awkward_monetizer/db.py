"""MonetDB connection helpers and packaged-schema access.

Two backends behind one DB-API surface:
  * :func:`open_server`   -- a running MonetDB server via ``pymonetdb``
  * :func:`open_embedded` -- in-process MonetDB via ``monetdbe`` (no server)

Both return a DB-API 2.0 connection, so ``ingest.load_tables`` and
``reconstruct.fetch_tables`` work unchanged against either.
"""

from __future__ import annotations

import re
from importlib import resources


# --------------------------------------------------------------------------
# Connections
# --------------------------------------------------------------------------
def open_server(*, database: str = "hep", host: str = "localhost",
                port: int = 50000, user: str = "monetdb",
                password: str = "monetdb", autocommit: bool = False):
    """Open a connection to a running MonetDB server (pymonetdb)."""
    import pymonetdb
    return pymonetdb.connect(database=database, hostname=host, port=port,
                             username=user, password=password,
                             autocommit=autocommit)


def open_embedded(target: str = ":memory:"):
    """Open an in-process MonetDB (monetdbe). ``target`` is ':memory:' or a dir.

    Note: monetdbe has no cp312 macOS-arm64 wheel; it needs Python <= 3.10.
    """
    import monetdbe
    return monetdbe.connect(target)


# --------------------------------------------------------------------------
# Schema (packaged SQL under awkward_monetizer/schemas/)
# --------------------------------------------------------------------------
def read_schema(name: str) -> str:
    """Return the text of a packaged schema, e.g. ``read_schema('dimuon')``."""
    return (resources.files("awkward_monetizer")
            .joinpath("schemas", f"{name}.sql").read_text())


def split_statements(sql: str) -> list[str]:
    """Split a .sql string into statements, stripping ``--`` comments first so a
    stray ``;`` inside a comment can't break the split."""
    no_comments = re.sub(r"--[^\n]*", "", sql)
    return [s.strip() for s in no_comments.split(";") if s.strip()]


def apply_sql(conn, sql_text: str, split: bool = True) -> None:
    """Execute a SQL script (e.g. a UDF definition). With split=False the whole
    comment-stripped text runs as one statement — needed for CREATE FUNCTION,
    whose body contains its own semicolons."""
    cur = conn.cursor()
    if split:
        for stmt in split_statements(sql_text):
            cur.execute(stmt)
    else:
        cur.execute(re.sub(r"--[^\n]*", "", sql_text).strip())
    conn.commit()


def create_schema(conn, sql_text: str,
                  drop: tuple[str, ...] = ("muons", "jets", "events")) -> None:
    """Drop the named tables (if present) and (re)create from ``sql_text``."""
    cur = conn.cursor()
    for tbl in drop:
        cur.execute(f"DROP TABLE IF EXISTS {tbl}")
    for stmt in split_statements(sql_text):
        cur.execute(stmt)
    conn.commit()
