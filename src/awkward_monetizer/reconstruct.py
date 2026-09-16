"""Reconstruction: flat relational tables -> Awkward NF2 (``ak.unflatten``).

The inverse of :mod:`awkward_monetizer.ingest`. A flat table sorted by
``event_id`` plus the per-event object counts reconstructs the jagged sublists
exactly. Sources: a live MonetDB connection (:func:`fetch_tables`) or a ROOT
file directly (:func:`tables_from_root`, no database).
"""

from __future__ import annotations

import awkward as ak
import pandas as pd

from .datasets import DATASETS
from .ingest import build_tables, read_root


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
                      collections: dict[str, pd.DataFrame]) -> ak.Array:
    """Rebuild an Awkward record array of events with one nested (jagged) sublist
    per collection. ``event_id`` is the join key.

    events_df:   one row per event (must contain ``event_id``).
    collections: ``{field_name: objects_df}``; each objects_df has ``event_id``
                 and a ``<field>_index`` column. Nested as ``events.<field_name>``.
    """
    events_df = events_df.sort_values("event_id").reset_index(drop=True)
    events = ak.zip({c: _column_to_ak(events_df[c]) for c in events_df.columns},
                    depth_limit=1)
    for field, df in collections.items():
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
        obj_fields = [c for c in df.columns if c != "event_id"]
        flat = ak.zip({f: _column_to_ak(df[f]) for f in obj_fields})
        events = ak.with_field(events, ak.unflatten(flat, counts), field)
    return events


def reconstruct_events(events_df: pd.DataFrame, muons_df: pd.DataFrame,
                       collection: str = "muons") -> ak.Array:
    """Single-collection convenience wrapper around :func:`reconstruct_multi`."""
    return reconstruct_multi(events_df, {collection: muons_df})


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
def tables_from_root(root_file: str, dataset: str = "dimuon",
                     tree: str | None = None) -> tuple[pd.DataFrame, ...]:
    """Build the flat tables from a ROOT file (via ingest), without a database.

    Returns ``(events_df, muons_df)`` for the single-muon-collection datasets;
    for datasets with more collections use :func:`tables_dict_from_root`.
    """
    events_df, tables = tables_dict_from_root(root_file, dataset, tree)
    return tables["events"], tables["muons"]


def tables_dict_from_root(root_file: str, dataset: str = "dimuon",
                          tree: str | None = None):
    """Build all flat tables from a ROOT file. Returns (events_df, tables_dict)."""
    ds = DATASETS[dataset]
    events_ak, _ = read_root(root_file, ds, tree)
    tables = build_tables(events_ak, ds)
    return tables["events"], tables


def fetch_tables(conn, where: str | None = None, collection: str = "muons"
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull ``events`` (optionally filtered by a SQL WHERE) and the matching
    ``<collection>`` rows from an existing MonetDB DB-API connection.

    The SQL-slice step: push a selection down to MonetDB, bring back only the
    surviving events and their objects. Works with pymonetdb or monetdbe.
    """
    cur = conn.cursor()

    ev_sql = "SELECT * FROM events"
    if where:
        ev_sql += f" WHERE {where}"
    cur.execute(ev_sql)
    events_df = pd.DataFrame(cur.fetchall(),
                             columns=[d[0] for d in cur.description])

    if where:
        ids = events_df["event_id"].tolist()
        if ids:
            id_list = ",".join(str(int(i)) for i in ids)
            cur.execute(f"SELECT * FROM {collection} WHERE event_id IN ({id_list})")
        else:
            cur.execute(f"SELECT * FROM {collection} WHERE 1=0")
    else:
        cur.execute(f"SELECT * FROM {collection}")
    objs_df = pd.DataFrame(cur.fetchall(),
                           columns=[d[0] for d in cur.description])
    return events_df, objs_df


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
