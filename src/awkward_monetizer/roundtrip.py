"""End-to-end live MonetDB round-trip.

    ROOT --ingest--> flat tables --> MonetDB (create + load)
         --SQL slice--> pandas --reconstruct--> Awkward (NF2) --> validate

Ties ingest + reconstruct together against a real MonetDB engine and checks that
the reconstructed dimuon mass still matches the file's stored ``M`` after a full
database round-trip.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from .datasets import DATASETS
from .db import create_schema, open_embedded, open_server, read_schema
from .ingest import build_tables, load_tables, read_root
from .physics import invariant_mass, opposite_charge_pair
from .reconstruct import fetch_tables, reconstruct_events

# dataset name -> packaged schema name
_SCHEMA_FOR = {"dimuon": "dimuon", "nanoaod": "nanoaod"}


def run(*, root_file: str, dataset: str = "dimuon", backend: str = "server",
        where: str = "mass BETWEEN 60 AND 120", use_copy_into: bool = True,
        conn_kwargs: dict | None = None) -> dict:
    """Run the full cycle and return a result dict. Prints progress."""
    conn_kwargs = conn_kwargs or {}
    ds = DATASETS[dataset]

    print(f"[1/5] ingest {root_file} (dataset={ds.name}) ...")
    events_ak, tree = read_root(root_file, ds, None)
    tables = build_tables(events_ak, ds)
    for name, df in tables.items():
        print(f"      {name}: {len(df)} rows")

    print(f"[2/5] connect ({backend}) + create schema ...")
    if backend == "embedded":
        conn = open_embedded(conn_kwargs.get("dbdir") or ":memory:")
        use_copy_into = False  # monetdbe can't prepare COPY INTO
    else:
        conn = open_server(**{k: v for k, v in conn_kwargs.items()
                              if k != "dbdir"})
    result: dict = {}
    try:
        create_schema(conn, read_schema(_SCHEMA_FOR[dataset]))

        method = "COPY INTO" if use_copy_into else "INSERT"
        print(f"[3/5] load tables into MonetDB ({method}) ...")
        load_tables(conn, tables, use_copy_into=use_copy_into)

        print(f"[4/5] SQL slice + fetch back (WHERE {where!r}) ...")
        events_df, muons_df = fetch_tables(conn, where=where)
        print(f"      fetched events: {len(events_df)} | muons: {len(muons_df)}")
    finally:
        conn.close()

    print("[5/5] reconstruct NF2 in Awkward + validate ...")
    events = reconstruct_events(events_df, muons_df)
    recon = ak.to_numpy(invariant_mass(events.muons))
    stored = ak.to_numpy(events.mass)
    diff = np.abs(recon - stored)
    n_oc = int(ak.sum(opposite_charge_pair(events)))

    result.update(
        n_events=len(events),
        max_abs_diff=float(np.nanmax(diff)),
        mean_abs_diff=float(np.nanmean(diff)),
        n_opposite_charge=n_oc,
        passed=bool(np.nanmax(diff) < 1e-2),
    )
    print("\n===== round-trip result =====")
    print(f"  events through DB round-trip : {result['n_events']}")
    print("  reconstructed m(μμ) vs stored M:")
    print(f"    max |Δ|  : {result['max_abs_diff']:.3e} GeV")
    print(f"    mean |Δ| : {result['mean_abs_diff']:.3e} GeV")
    print(f"  opposite-charge events in slice : {result['n_opposite_charge']}")
    print(f"\n  {'PASS' if result['passed'] else 'FAIL'}: "
          f"NF2 survives the SQL round-trip{' ✓' if result['passed'] else ''}")
    return result
