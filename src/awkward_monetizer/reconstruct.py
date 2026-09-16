"""Reconstruction: flat relational tables -> Awkward NF2 (``ak.unflatten``).

The inverse of :mod:`awkward_monetizer.ingest`. A flat table sorted by
``event_id`` plus the per-event object counts reconstructs the jagged sublists
exactly. Sources: a live MonetDB connection (:func:`fetch_tables`) or a ROOT
file directly (:func:`tables_from_root`, no database).
"""

from __future__ import annotations

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
    """
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
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
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch events and matching objects, optionally projecting named columns.

    The WHERE expression is evaluated in the events scope for both queries.
    Column arguments are schema column names, not SQL expressions. Defaults
    preserve the full-table API. Timings measure client execute/fetch/convert
    calls; execute may include transfer of the driver's initial result batch.
    """
    def projection(columns, prefix=""):
        if columns is None:
            return prefix + "*"
        if not columns or "event_id" not in columns:
            raise ValueError("projections must include event_id")
        return ", ".join(prefix + '"' + c.replace('"', '""') + '"' for c in columns)

    ev_projection = projection(event_columns)
    obj_projection = projection(object_columns, "objects.")
    selection = f" WHERE {where}" if where else ""
    ev_sql = f"SELECT {ev_projection} FROM events{selection}"
    obj_sql = f"SELECT {obj_projection} FROM {collection} AS objects"
    if where:
        obj_sql += (f" JOIN (SELECT event_id FROM events{selection}) AS selected"
                    " ON objects.event_id = selected.event_id")

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
