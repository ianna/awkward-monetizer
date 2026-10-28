"""Reconstruction: flat relational tables -> Awkward NF2 (``ak.unflatten``).

The inverse of :mod:`awkward_monetizer.ingest`. A flat table sorted by
``event_id`` plus the per-event object counts reconstructs the jagged sublists
exactly. Sources: a live MonetDB connection (:func:`fetch_tables`) or a ROOT
file directly (:func:`tables_from_root`, no database).
"""

from __future__ import annotations

import warnings

import awkward as ak
import numpy as np
import pandas as pd

from .datasets import DATASETS
from .ingest import build_tables, read_root
from .timing import stage_time


# --------------------------------------------------------------------------
# Rebuild NF2 structure
# --------------------------------------------------------------------------
def _column_to_ak(series: pd.Series) -> ak.Array:
    """Numpy-backed for numeric/bool columns; list-backed for strings/objects.

    Uses a dtype-kind check rather than ``== object``: depending on the Awkward
    build, ``np.asarray`` on a string branch may yield an object array OR a
    fixed-width unicode ('<U') array. Awkward rejects both via from_numpy, so
    only genuinely numeric columns take the numpy path.

    Nullable extension columns (e.g. pandas ``boolean`` for an optional trigger
    flag that a file doesn't have) become an option-type Awkward array, with
    NA -> None.
    """
    is_numeric = (pd.api.types.is_numeric_dtype(series)
                  or pd.api.types.is_bool_dtype(series))
    if is_numeric and pd.api.types.is_extension_array_dtype(series):
        valid = series.notna().to_numpy()
        values = series.to_numpy(dtype=series.dtype.numpy_dtype, na_value=0)
        arr = ak.Array(values)
        return arr if valid.all() else ak.mask(arr, valid)
    if is_numeric:
        return ak.Array(series.to_numpy())
    return ak.Array(series.astype(object).tolist())


def reconstruct_multi(events_df: pd.DataFrame,
                      collections: dict[str, pd.DataFrame], *,
                      timings: dict[str, float] | None = None) -> ak.Array:
    """Rebuild an Awkward record array of events with one nested (jagged) sublist
    per collection. ``event_id`` is the join key.

    events_df:   one row per event (must contain ``event_id``).
    collections: ``{field_name: objects_df}``; each objects_df has ``event_id``
                 and a ``<field>_index`` column. Nested as ``events.<field_name>``.
    """
    with stage_time(timings, "sort_group"):
        events_df = events_df.sort_values("event_id").reset_index(drop=True)
    with stage_time(timings, "array_build"):
        events = ak.zip({c: _column_to_ak(events_df[c]) for c in events_df.columns},
                        depth_limit=1)
    for field, df in collections.items():
        with stage_time(timings, "sort_group"):
            index_cols = [c for c in df.columns if c.endswith("_index")]
            df = df.sort_values(["event_id"] + index_cols).reset_index(drop=True)
            counts = (df.groupby("event_id").size()
                      .reindex(events_df["event_id"], fill_value=0)
                      .to_numpy())
        if counts.sum() != len(df):
            raise ValueError(
                f"{field}: {len(df)} rows but counts sum to {counts.sum()} "
                "— some objects reference events not in the events table"
            )
        with stage_time(timings, "array_build"):
            obj_fields = [c for c in df.columns if c != "event_id"]
            flat = ak.zip({f: _column_to_ak(df[f]) for f in obj_fields})
            events = ak.with_field(events, ak.unflatten(flat, counts), field)
    return events


def reconstruct_events(events_df: pd.DataFrame, muons_df: pd.DataFrame,
                       collection: str = "muons", *,
                       timings: dict[str, float] | None = None) -> ak.Array:
    """Single-collection convenience wrapper around :func:`reconstruct_multi`."""
    return reconstruct_multi(events_df, {collection: muons_df}, timings=timings)


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
def tables_from_root(root_file: str, dataset: str = "dimuon",
                     tree: str | None = None) -> tuple[pd.DataFrame, ...]:
    """Build the flat tables from a ROOT file (via ingest), without a database.

    Returns ``(events_df, muons_df)`` for the single-muon-collection datasets;
    for datasets with more collections use :func:`tables_dict_from_root`.
    """
    _events_df, tables = tables_dict_from_root(root_file, dataset, tree)
    return tables["events"], tables["muons"]


def tables_dict_from_root(root_file: str, dataset: str = "dimuon",
                          tree: str | None = None):
    """Build all flat tables from a ROOT file. Returns (events_df, tables_dict)."""
    ds = DATASETS[dataset]
    events_ak, _ = read_root(root_file, ds, tree)
    tables = build_tables(events_ak, ds)
    return tables["events"], tables


def fetch_tables(conn, where: str | None = None, collection: str = "muons", *,
                 event_columns: tuple[str, ...] | None = None,
                 object_columns: tuple[str, ...] | None = None,
                 timings: dict[str, float] | None = None,
                 event_range: tuple[int, int] | None = None,
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch events and matching objects, optionally projecting named columns.

    The WHERE expression is evaluated in the events scope for both queries.
    Column arguments are schema column names, not SQL expressions. Defaults
    preserve the full-table API. Timings measure client execute/fetch/convert
    calls; execute may include transfer of the driver's initial result batch.

    ``event_range=(lo, hi)`` restricts both queries to ``lo <= event_id < hi``,
    applied to the object table directly as well, so a large table can be read
    in chunks without scanning a join over all of it each time.
    """
    def projection(columns, prefix=""):
        if columns is None:
            return prefix + "*"
        if not columns or "event_id" not in columns:
            raise ValueError("projections must include event_id")
        return ", ".join(prefix + '"' + c.replace('"', '""') + '"' for c in columns)

    ev_projection = projection(event_columns)
    obj_projection = projection(object_columns, "objects.")
    if event_range is not None:
        lo, hi = (int(v) for v in event_range)
        span = f"event_id >= {lo} AND event_id < {hi}"
        ev_where = f"{span} AND ({where})" if where else span
    else:
        span, ev_where = None, where
    selection = f" WHERE {ev_where}" if ev_where else ""
    ev_sql = f"SELECT {ev_projection} FROM events{selection}"
    obj_sql = f"SELECT {obj_projection} FROM {collection} AS objects"
    if where:
        obj_sql += (f" JOIN (SELECT event_id FROM events{selection}) AS selected"
                    " ON objects.event_id = selected.event_id")
    if span:
        obj_sql += f" WHERE objects.event_id >= {lo} AND objects.event_id < {hi}"

    cur = conn.cursor()

    def fetch(sql):
        with stage_time(timings, "execute"):
            cur.execute(sql)
        columns = [d[0] for d in cur.description]
        with stage_time(timings, "fetch"):
            rows = cur.fetchall()
        with stage_time(timings, "dataframe"):
            return pd.DataFrame(rows, columns=columns)

    try:
        return fetch(ev_sql), fetch(obj_sql)
    finally:
        cur.close()


# --------------------------------------------------------------------------
# NumPy-direct fetch (no pandas)
# --------------------------------------------------------------------------
_INT_TYPES = {"tinyint", "smallint", "int", "integer", "bigint", "hugeint",
              "oid", "serial", "bigserial"}
_FLOAT_TYPES = {"double", "real", "float", "double precision"}
_BOOL_TYPES = {"boolean", "bool"}


def _np_dtype(type_code):
    """NumPy dtype for a DB-API type code, or None when unknown (e.g. SQLite)."""
    t = str(type_code).lower() if type_code is not None else ""
    if t in _INT_TYPES:
        return np.dtype(np.int64)
    if t in _FLOAT_TYPES or t.startswith(("decimal", "numeric")):
        return np.dtype(np.float64)
    if t in _BOOL_TYPES:
        return np.dtype(np.bool_)
    return None


def _column(values, dtype):
    """One fetched column -> NumPy (or a masked/list Awkward array when it has
    NULLs or is not numeric), with the same types the pandas path produces."""
    if dtype is None:
        probe = np.asarray(values) if len(values) else np.empty(0)
        if probe.dtype.kind in "biuf":
            return probe.astype(np.int64 if probe.dtype.kind in "iu" else probe.dtype)
        if probe.dtype.kind != "O" or all(v is not None for v in values):
            return ak.Array(list(values))       # strings
        kinds = {type(v) for v in values if v is not None}
        if not kinds:                           # all NULL: type unknown
            return ak.Array(list(values))
        dtype = (np.dtype(np.bool_) if kinds <= {bool} else
                 np.dtype(np.int64) if kinds <= {int, bool} else
                 np.dtype(np.float64) if kinds <= {int, float, bool} else None)
        if dtype is None:
            return ak.Array(list(values))
    if any(v is None for v in values):
        valid = np.fromiter((v is not None for v in values), bool, len(values))
        filled = np.array([v if v is not None else 0 for v in values], dtype=dtype)
        return ak.mask(ak.Array(filled), valid)
    return np.array(values, dtype=dtype)


# MonetDB binary result columns (pymonetdb's ``Xexportbin``): type -> raw dtype
_BIN_TYPES = {"tinyint": "i1", "smallint": "i2", "int": "i4", "bigint": "i8",
              "real": "f4", "double": "f8", "float": "f8", "boolean": "i1"}
BINARY_BATCH = 2_000_000


def _decode_binary_block(block, kinds, endian: str) -> list[np.ndarray]:
    """Split one ``Xexportbin`` reply into raw NumPy columns (no copies).

    Layout, as pymonetdb reads it: column payloads, then a table of
    (start, length) int64 pairs per column, then the int64 offset of that
    table; a negative offset points to an error message instead.
    """
    order = "<" if endian == "little" else ">"
    i8 = np.dtype(order + "i8")
    toc = int(np.frombuffer(block, i8, 1, len(block) - 8)[0])
    if toc < 0:
        msg = bytes(block[toc:-8]).split(b"\x00", 1)[0].decode("utf-8", "replace")
        raise RuntimeError(f"MonetDB binary export failed: {msg}")
    cols = []
    for i, code in enumerate(kinds):
        start, length = np.frombuffer(block, i8, 2, toc + 16 * i)
        cols.append(np.frombuffer(block, np.dtype(order + code),
                                  int(length) // np.dtype(code).itemsize, int(start)))
    return cols


def _finish_binary(parts: list[np.ndarray], type_name: str):
    """Concatenate one column's raw batches; MonetDB NULL sentinels -> None."""
    raw = np.concatenate(parts) if parts else np.empty(0, _BIN_TYPES[type_name])
    if type_name == "boolean":
        valid, out = raw != -128, raw == 1
    elif raw.dtype.kind == "f":
        valid, out = ~np.isnan(raw), raw.astype(np.float64)
    else:
        valid, out = raw != np.iinfo(raw.dtype).min, raw.astype(np.int64)
    out = np.ascontiguousarray(out)
    return out if valid.all() else ak.mask(ak.Array(out), valid)


def _fetch_binary(cur, timings) -> dict | None:
    """pymonetdb fast path: read the executed result with MonetDB's binary
    export and ``np.frombuffer``, so no Python object is made per value.

    Returns None (use the generic path) unless the cursor is a pymonetdb one
    with binary support and every column is a fixed-width number or boolean.
    """
    conn = getattr(cur, "connection", None)
    mapi = getattr(conn, "mapi", None)
    qid = getattr(cur, "_query_id", None)
    first = getattr(cur, "_rows", None)
    if (qid is None or mapi is None or first is None
            or not hasattr(conn, "binary_command")
            or not getattr(getattr(conn, "_policy", None), "use_binary", lambda: False)()):
        return None
    names = [d[0] for d in cur.description]
    types = [str(d[1]).lower() for d in cur.description]
    if any(t not in _BIN_TYPES for t in types):
        return None
    total = int(cur.rowcount)
    if total <= len(first):            # everything came with the first reply
        return None
    kinds = [_BIN_TYPES[t] for t in types]
    parts: list[list[np.ndarray]] = [[] for _ in names]
    for start in range(0, total, BINARY_BATCH):
        n = min(BINARY_BATCH, total - start)
        with stage_time(timings, "fetch"):
            block = conn.binary_command(f"Xexportbin {qid} {start} {n}")
        with stage_time(timings, "numpy"):
            for i, col in enumerate(_decode_binary_block(block, kinds,
                                                         mapi.server_endian)):
                parts[i].append(col.copy())   # the block buffer is reused
    with stage_time(timings, "numpy"):
        return {n: _finish_binary(p, t) for n, p, t in zip(names, parts, types,
                                                             strict=True)}


def _fetch_columns(cur, sql, timings) -> dict:
    with stage_time(timings, "execute"):
        cur.execute(sql)
    try:
        fast = _fetch_binary(cur, timings)
    except Exception as exc:  # noqa: BLE001 (any surprise: use the portable path)
        warnings.warn(f"MonetDB binary fetch failed, using the slow path: {exc!r}",
                      RuntimeWarning, stacklevel=2)
        fast = None
    if fast is not None:
        return fast
    names = [d[0] for d in cur.description]
    dtypes = [_np_dtype(d[1]) for d in cur.description]
    with stage_time(timings, "fetch"):
        rows = cur.fetchall()
    with stage_time(timings, "numpy"):
        # Column by column, checking for NULLs first: a structured np.array
        # over the rows would be one pass, but it silently turns NULL into
        # False (boolean) or NaN (double) instead of a missing value.
        fast = {}
        if rows and all(d is not None for d in dtypes):
            try:   # all-numeric result: one C-level pass into a 2-D float array
                grid = np.array(rows, dtype=np.float64)
            except (TypeError, ValueError):
                grid = None
            if grid is not None and grid.ndim == 2:
                for i, (n, d) in enumerate(zip(names, dtypes, strict=True)):
                    col = grid[:, i]
                    if np.isnan(col).any():     # NULL (or NaN): exact path below
                        continue
                    if d.kind in "iu" and np.abs(col).max(initial=0) >= 2**53:
                        continue                # beyond float64's exact integers
                    fast[n] = col.astype(d) if d != np.float64 else col.copy()
        if len(fast) == len(names):
            return fast
        cols = list(zip(*rows, strict=True)) if rows else [()] * len(names)
        return {n: fast[n] if n in fast else _column(c, d)
                for n, c, d in zip(names, cols, dtypes, strict=True)}


def _take(cols: dict, order) -> dict:
    return {k: (v[order] if isinstance(v, np.ndarray) else v[ak.Array(order)])
            for k, v in cols.items()}


def fetch_events(conn, where: str | None = None, *,
                 event_columns: tuple[str, ...] | None = None,
                 collections: dict[str, tuple[str, ...] | None] | None = None,
                 event_range: tuple[int, int] | None = None,
                 timings: dict[str, float] | None = None) -> ak.Array:
    """Fetch events and their object collections straight into Awkward arrays.

    Same result as :func:`fetch_tables` + :func:`reconstruct_multi` (same
    fields, types and nesting), but rows go from the DB-API cursor directly into
    NumPy columns: no pandas DataFrames, no groupby. The events query runs once,
    whatever the number of collections. ``collections`` maps a collection table
    to the columns to fetch (None = all; must include ``event_id`` and the
    ``*_index`` column). Timings: execute, fetch, numpy, array_build.
    """
    collections = collections or {}

    def projection(columns, prefix=""):
        if columns is None:
            return prefix + "*"
        if "event_id" not in columns:
            raise ValueError("projections must include event_id")
        return ", ".join(prefix + '"' + c.replace('"', '""') + '"' for c in columns)

    span_ev = span_obj = None
    if event_range is not None:
        lo, hi = (int(v) for v in event_range)
        span_ev = f"event_id >= {lo} AND event_id < {hi}"
        span_obj = f"objects.event_id >= {lo} AND objects.event_id < {hi}"
    ev_where = " AND ".join(x for x in (span_ev, f"({where})" if where else None) if x)
    selection = f" WHERE {ev_where}" if ev_where else ""

    cur = conn.cursor()
    try:
        ev = _fetch_columns(
            cur, f"SELECT {projection(event_columns)} FROM events{selection}", timings)
        objs = {}
        for coll, cols in collections.items():
            sql = f"SELECT {projection(cols, 'objects.')} FROM {coll} AS objects"
            if where:
                sql += (f" JOIN (SELECT event_id FROM events{selection}) AS selected"
                        " ON objects.event_id = selected.event_id")
            if span_obj:
                sql += f" WHERE {span_obj}"
            objs[coll] = _fetch_columns(cur, sql, timings)
    finally:
        cur.close()

    with stage_time(timings, "array_build"):
        eid = np.asarray(ev["event_id"], dtype=np.int64)
        if eid.size > 1 and not np.all(eid[1:] > eid[:-1]):
            ev = _take(ev, np.argsort(eid, kind="stable"))
            eid = np.asarray(ev["event_id"], dtype=np.int64)
        events = ak.zip({c: (ak.Array(v) if isinstance(v, np.ndarray) else v)
                         for c, v in ev.items()}, depth_limit=1)
        for coll, cols in objs.items():
            oid = np.asarray(cols["event_id"], dtype=np.int64)
            index_cols = [c for c in cols if c.endswith("_index")]
            if oid.size > 1:
                keys = [np.asarray(cols[c]) for c in reversed(index_cols)] + [oid]
                d = np.diff(oid)
                ordered = d > 0
                if index_cols:
                    ordered |= (d == 0) & (np.diff(keys[-2]) > 0)
                if not np.all(ordered):
                    cols = _take(cols, np.lexsort(keys))
                    oid = np.asarray(cols["event_id"], dtype=np.int64)
            counts = (np.searchsorted(oid, eid, side="right")
                      - np.searchsorted(oid, eid, side="left"))
            if counts.sum() != oid.size:
                raise ValueError(
                    f"{coll}: {oid.size} rows but counts sum to {counts.sum()} "
                    "— some objects reference events not in the events table")
            flat = ak.zip({f: (ak.Array(v) if isinstance(v, np.ndarray) else v)
                           for f, v in cols.items() if f != "event_id"})
            events = ak.with_field(events, ak.unflatten(flat, counts), coll)
    return events


def fetch_dimuon_events(conn, where: str | None = None, *,
                        timings: dict[str, float] | None = None) -> ak.Array:
    """Fetch the dimuon analysis fields directly into typed NumPy/Awkward arrays.

    Uses the packaged dimuon schema (non-null event IDs, indices and charges).
    NumPy orders rows when needed; searchsorted/bincount restore per-event lists
    without pandas inference or groupby. Empty events and variable muon counts
    are retained; no assumption of exactly two muons is made here.
    """
    selection = f" WHERE {where}" if where else ""
    event_dtype = np.dtype([("event_id", np.int64)])
    muon_dtype = np.dtype([
        ("event_id", np.int64), ("muon_index", np.int64),
        ("e", np.float64), ("px", np.float64), ("py", np.float64),
        ("pz", np.float64), ("charge", np.int64),
    ])
    event_sql = f"SELECT event_id FROM events{selection}"
    muon_sql = (
        "SELECT m.event_id, m.muon_index, m.e, m.px, m.py, m.pz, m.charge "
        "FROM muons AS m "
        f"JOIN (SELECT event_id FROM events{selection}) AS selected "
        "ON m.event_id = selected.event_id"
    )
    cur = conn.cursor()

    def fetch(sql, dtype):
        with stage_time(timings, "execute"):
            cur.execute(sql)
        with stage_time(timings, "fetch"):
            rows = cur.fetchall()
        with stage_time(timings, "column_arrays"):
            return np.fromiter(rows, dtype=dtype, count=len(rows))

    try:
        event_ids = fetch(event_sql, event_dtype)["event_id"]
        muons = fetch(muon_sql, muon_dtype)
    finally:
        cur.close()

    with stage_time(timings, "sort_group"):
        if np.any(event_ids[1:] < event_ids[:-1]):
            event_ids.sort()
        ids, indices = muons["event_id"], muons["muon_index"]
        if np.any((ids[1:] < ids[:-1]) |
                  ((ids[1:] == ids[:-1]) & (indices[1:] < indices[:-1]))):
            muons = muons[np.lexsort((indices, ids))]
        positions = np.searchsorted(event_ids, muons["event_id"])
        if (np.any(positions >= len(event_ids)) or
                np.any(event_ids[positions] != muons["event_id"])):
            raise ValueError("muons reference events not in the selected events table")
        counts = np.bincount(positions, minlength=len(event_ids))
    with stage_time(timings, "array_build"):
        flat = ak.zip({name: np.ascontiguousarray(muons[name]) for name in muon_dtype.names
                       if name != "event_id"})
        return ak.zip({"event_id": event_ids,
                       "muons": ak.unflatten(flat, counts)}, depth_limit=1)


def fetch_from_monetdb(*, database: str = "hep", host: str = "localhost",
                       port: int = 50000, user: str = "monetdb",
                       password: str = "monetdb", where: str | None = None,
                       collection: str = "muons"
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Open a pymonetdb server connection and fetch the event/object slice."""
    from .db import open_server
    conn = open_server(database=database, host=host, port=port,
                       user=user, password=password)
    try:
        return fetch_tables(conn, where=where, collection=collection)
    finally:
        conn.close()
