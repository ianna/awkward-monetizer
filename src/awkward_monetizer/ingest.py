"""Ingestion: ROOT -> Uproot -> Awkward -> flat pandas tables -> MonetDB.

Splits a ROOT TTree into flat relational tables (``events`` plus one table per
particle collection), keyed by ``event_id``, with a per-object ``*_index``
column so the nesting can be reconstructed losslessly (see
:mod:`awkward_monetizer.reconstruct`).
"""

from __future__ import annotations

import csv
import os
import tempfile

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from .datasets import Dataset, JaggedCollection, WideCollection


# --------------------------------------------------------------------------
# ROOT -> Awkward
# --------------------------------------------------------------------------
def open_tree(f, tree: str | None):
    """Return the requested TTree, or the first TTree in the file if None."""
    if tree:
        if tree in f:
            return f[tree]
        raise KeyError(f"tree {tree!r} not found; available: {list(f.keys())}")
    for name, cls in f.classnames().items():
        if cls == "TTree":
            return f[name]
    raise KeyError(f"no TTree found in file; keys: {list(f.keys())}")


def read_root(path: str, ds: Dataset, tree: str | None,
              entry_stop: int | None = None):
    with uproot.open(path) as f:
        t = open_tree(f, tree if tree is not None else ds.tree)
        wanted = ds.all_branches()
        available = set(t.keys())
        missing = [b for b in wanted if b not in available]
        if missing:
            raise KeyError(
                f"branches not found in {path}:{t.name}: {missing}\n"
                f"available: {sorted(available)}"
            )
        return t.arrays(wanted, entry_stop=entry_stop), t.name


# --------------------------------------------------------------------------
# Awkward -> flat pandas tables
# --------------------------------------------------------------------------
def build_events_table(events: ak.Array, ds: Dataset, id_offset: int = 0):
    """Return (events DataFrame, event_id array reused by the collections)."""
    n = len(events)
    if ds.event_id == "row":
        eid = np.arange(n, dtype=np.int64) + id_offset
    else:
        eid = np.asarray(events[ds.event_id])
    cols = {"event_id": eid}
    for dest, branch in ds.events.fields.items():
        cols[dest] = np.asarray(events[branch])
    for col, kind, ref in ds.derived:
        if kind == "count":
            cols[col] = np.asarray(ak.num(events[ref], axis=1))
        elif kind == "const":
            cols[col] = np.full(n, ref)
    return pd.DataFrame(cols), eid


def build_jagged(events: ak.Array, coll: JaggedCollection,
                 eid: np.ndarray) -> pd.DataFrame:
    """Explode a jagged collection: broadcast event_id down, flatten in lockstep."""
    counts = ak.num(events[coll.count_branch], axis=1)
    event_id = np.repeat(eid, np.asarray(counts))
    local_index = np.asarray(
        ak.flatten(ak.local_index(events[coll.count_branch], axis=1)))
    out = {"event_id": event_id, coll.index_col: local_index}
    for dest, branch in coll.branches.items():
        out[dest] = np.asarray(ak.flatten(events[branch], axis=1))
    return pd.DataFrame(out)


def build_wide(events: ak.Array, coll: WideCollection,
               eid: np.ndarray) -> pd.DataFrame:
    """Un-pivot fixed suffixed columns (pt1/pt2, ...) into one row per object."""
    n = len(eid)
    frames = []
    for i, slot in enumerate(coll.slots):
        cols = {"event_id": eid,
                coll.index_col: np.full(n, i, dtype=np.int64)}
        for dest, tmpl in coll.branches.items():
            cols[dest] = np.asarray(events[tmpl.format(slot)])
        frames.append(pd.DataFrame(cols))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["event_id", coll.index_col]).reset_index(drop=True)


def build_tables(events: ak.Array, ds: Dataset,
                 id_offset: int = 0) -> dict[str, pd.DataFrame]:
    events_df, eid = build_events_table(events, ds, id_offset=id_offset)
    tables = {"events": events_df}
    for coll in ds.collections:
        if isinstance(coll, JaggedCollection):
            tables[coll.table] = build_jagged(events, coll, eid)
        else:
            tables[coll.table] = build_wide(events, coll, eid)
    return tables


# --------------------------------------------------------------------------
# pandas -> MonetDB
# --------------------------------------------------------------------------
def load_tables(conn, tables: dict[str, pd.DataFrame], *,
                truncate: bool = False, use_copy_into: bool = True) -> None:
    """Load each DataFrame into its table using an existing DB-API connection.

    Works with any MonetDB DB-API connection — pymonetdb (a running server) or
    monetdbe (in-process/embedded). Tables must already exist.
    """
    cur = conn.cursor()
    for name, df in tables.items():
        if truncate:
            cur.execute(f"DELETE FROM {name}")
        if use_copy_into:
            _copy_into(cur, name, df)
        else:
            _insert_many(cur, name, df)
        print(f"  loaded {len(df):>8d} rows into {name}")
    conn.commit()


def load_monetdb(tables: dict[str, pd.DataFrame], *, database: str,
                 host: str = "localhost", port: int = 50000,
                 user: str = "monetdb", password: str = "monetdb",
                 truncate: bool = False, use_copy_into: bool = True) -> None:
    """Open a pymonetdb server connection and load the tables (must exist)."""
    from .db import open_server
    conn = open_server(database=database, host=host, port=port,
                       user=user, password=password)
    try:
        load_tables(conn, tables, truncate=truncate, use_copy_into=use_copy_into)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _copy_into(cur, table: str, df: pd.DataFrame) -> None:
    fd, csv_path = tempfile.mkstemp(prefix=f"{table}_", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            df.to_csv(fh, index=False, header=False, quoting=csv.QUOTE_MINIMAL)
        cur.execute(
            f"COPY {len(df)} RECORDS INTO {table} FROM %s "
            "USING DELIMITERS ',', '\\n', '\"' NULL AS ''",
            (csv_path,),
        )
    finally:
        os.unlink(csv_path)


def _insert_many(cur, table: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    cols = ",".join(df.columns)
    placeholders = ",".join(["%s"] * len(df.columns))
    sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
    rows = [tuple(v.item() if hasattr(v, "item") else v for v in row)
            for row in df.itertuples(index=False, name=None)]
    cur.executemany(sql, rows)


def ingest_root_chunked(path: str, ds: Dataset, conn, *, tree: str | None = None,
                        step_size="100 MB", entry_stop: int | None = None,
                        use_copy_into: bool = True) -> int:
    """Stream a (possibly huge) ROOT file into MonetDB in chunks with
    ``uproot.iterate``, so a multi-GB NanoAOD file never has to fit in memory.
    Reads only the dataset's branches; synthesizes globally-unique ``event_id``
    by offsetting each chunk. The tables must already exist. Returns the total
    number of events loaded.
    """
    total = 0
    with uproot.open(path) as f:
        t = open_tree(f, tree if tree is not None else ds.tree)
        for chunk in t.iterate(ds.all_branches(), step_size=step_size,
                               entry_stop=entry_stop):
            tables = build_tables(chunk, ds, id_offset=total)
            load_tables(conn, tables, use_copy_into=use_copy_into)
            total += len(chunk)
            print(f"  ... {total} events ingested")
    return total
