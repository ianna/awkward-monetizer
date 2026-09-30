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
from .keys import MAX_ENTRY, make_event_id, make_event_ids


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
def build_events_table(events: ak.Array, ds: Dataset, id_offset: int = 0,
                       file_id: int = 0):
    """Return (events DataFrame, event_id array reused by the collections).

    For synthetic ids (``ds.event_id == "row"``), ``id_offset`` is the entry
    number of the first row *within the file* and ``file_id`` identifies the
    file; ``event_id = (file_id << 40) | entry`` (see :mod:`.keys`). With the
    default ``file_id=0`` this is just ``id_offset + row``.
    """
    n = len(events)
    if ds.event_id == "row":
        eid = make_event_ids(file_id, np.arange(n, dtype=np.int64) + id_offset)
    else:
        if file_id:
            raise ValueError(
                f"dataset {ds.name!r} takes event_id from branch "
                f"{ds.event_id!r}; file_id only applies to synthetic ids")
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


def build_tables(events: ak.Array, ds: Dataset, id_offset: int = 0,
                 file_id: int = 0) -> dict[str, pd.DataFrame]:
    events_df, eid = build_events_table(events, ds, id_offset=id_offset,
                                        file_id=file_id)
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
                truncate: bool = False, use_copy_into: bool = True,
                method: str | None = None) -> None:
    """Load each DataFrame into its table using an existing DB-API connection.

    ``method``: "copy" (temp CSV, server on this host), "client" (CSV streamed
    over the connection), "binary" (per-column binary over the connection),
    "insert" (DB-API INSERTs), or "auto" (:func:`.upload.choose_method`).
    ``None`` keeps the old behaviour: "copy" if ``use_copy_into`` else
    "insert". "client"/"binary" work against remote servers; see
    :mod:`.upload`. Tables must already exist.
    """
    from .upload import METHODS, choose_method, load_frame
    if method is None:
        method = "copy" if use_copy_into else "insert"
    if method not in METHODS:
        raise ValueError(f"unknown load method {method!r}; one of {METHODS}")
    if method == "auto":
        method = choose_method(conn)
    cur = conn.cursor()
    for name, df in tables.items():
        if truncate:
            cur.execute(f"DELETE FROM {name}")
        if method == "copy":
            _copy_into(cur, name, df)
            used = "COPY INTO"
        elif method == "insert":
            _insert_many(cur, name, df)
            used = "INSERT"
        else:
            used = load_frame(conn, cur, name, df, method) + " upload"
        print(f"  loaded {len(df):>8d} rows into {name} ({used})")
    conn.commit()


def load_monetdb(tables: dict[str, pd.DataFrame], *, database: str,
                 host: str = "localhost", port: int = 50000,
                 user: str = "monetdb", password: str = "monetdb",
                 truncate: bool = False, use_copy_into: bool = True,
                 method: str | None = None) -> None:
    """Open a pymonetdb server connection and load the tables (must exist)."""
    from .db import open_server
    conn = open_server(database=database, host=host, port=port,
                       user=user, password=password)
    try:
        load_tables(conn, tables, truncate=truncate, use_copy_into=use_copy_into,
                    method=method)
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
                        step_size="100 MB", entry_start: int | None = None,
                        entry_stop: int | None = None, file_id: int = 0,
                        use_copy_into: bool = True, truncate: bool = False,
                        replace: bool = False, dry_run: bool = False,
                        method: str | None = None) -> int:
    """Stream a (possibly huge) ROOT file into MonetDB in chunks with
    ``uproot.iterate``, so a multi-GB NanoAOD file never has to fit in memory.
    Reads only the dataset's branches. The tables must already exist. Returns
    the total number of events loaded.

    ``event_id`` is ``(file_id << 40) | entry`` where ``entry`` is the event's
    entry number in the file (see :mod:`.keys`). It is therefore independent of
    ``step_size`` and of how the file is split, so parallel jobs -- different
    files with different ``file_id``, or disjoint ``[entry_start, entry_stop)``
    ranges of one file -- can load into the same tables without collisions.

    ``truncate`` empties the whole tables first (single-writer use only).
    ``replace`` deletes only this job's own ``event_id`` range -- this file,
    ``[entry_start, entry_stop)`` -- so a failed or repeated job can be re-run
    idempotently without touching rows loaded by other workers.

    With dry_run=True, build and summarize each chunk without using conn.
    Truncation happens once before loading, including for an empty input.
    """
    total = 0
    first = int(entry_start or 0)
    if first < 0:
        raise ValueError("entry_start must be >= 0")
    with uproot.open(path) as f:
        t = open_tree(f, tree if tree is not None else ds.tree)
        tables_to_clear = [c.table for c in ds.collections] + ["events"]
        if truncate and not dry_run:
            cur = conn.cursor()
            for name in tables_to_clear:
                cur.execute(f"DELETE FROM {name}")
        elif replace and not dry_run:
            if ds.event_id != "row":
                raise ValueError("replace needs synthetic (file_id, entry) ids")
            lo = make_event_id(file_id, first)
            hi = make_event_id(file_id, min(int(entry_stop), MAX_ENTRY)
                               if entry_stop is not None else MAX_ENTRY)
            # entry_stop is exclusive; with no stop, clear to the end of the file
            op = "<" if entry_stop is not None else "<="
            cur = conn.cursor()
            for name in tables_to_clear:
                cur.execute(f"DELETE FROM {name} WHERE event_id >= {lo} "
                            f"AND event_id {op} {hi}")
        for chunk in t.iterate(ds.all_branches(), step_size=step_size,
                               entry_start=first, entry_stop=entry_stop):
            tables = build_tables(chunk, ds, id_offset=first + total,
                                  file_id=file_id)
            if dry_run:
                for name, df in tables.items():
                    print(f"  built {name}: {len(df)} rows x {len(df.columns)} cols")
            else:
                load_tables(conn, tables, use_copy_into=use_copy_into,
                            method=method)
            total += len(chunk)
            print(f"  ... {total} events {'read' if dry_run else 'ingested'}")
        if (truncate or replace) and not dry_run and total == 0:
            conn.commit()
    return total
