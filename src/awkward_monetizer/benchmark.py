"""Benchmark harness: compare analysis backends on the same data.

The reference analysis is the dimuon **Z→μμ** selection: opposite-charge muon
pair with 60 < m(μμ) < 120 GeV, computing the invariant-mass distribution.
Three backends run it on identical input and their results are cross-checked:

  * **awkward**     — pure Awkward: read ROOT (uproot) → reconstruct → analyze.
  * **hybrid**      — the MonetDB + Awkward path: data pre-loaded in MonetDB, the
                      event-level cut pushed down to SQL, survivors reconstructed
                      and finished in Awkward.
  * **rdataframe**  — ROOT ``RDataFrame`` computing the same quantity. Gated: it
                      only runs if ``import ROOT`` succeeds (skipped otherwise).

Timing model: the query phase is timed (repeats, median reported); one-time
*setup* (reading/reconstructing arrays for awkward, ingesting into MonetDB for
hybrid) is timed once and reported separately, because a database amortizes load
cost across many queries. Use ``--scale K`` to tile the input K× into a larger
physical ROOT file that every backend reads, so timings are meaningful.
"""

from __future__ import annotations

import statistics
import tempfile
import time

import awkward as ak
import numpy as np
import uproot

from .datasets import DATASETS
from .ingest import build_tables, load_tables, read_root
from .physics import invariant_mass, opposite_charge_pair
from .reconstruct import fetch_tables, reconstruct_events


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
def timed(fn, repeats: int = 5, warmup: int = 1):
    """Run fn warmup+repeats times; return (result, {median, min, n})."""
    result = None
    for _ in range(warmup):
        result = fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        ts.append(time.perf_counter() - t0)
    return result, {"median": statistics.median(ts), "min": min(ts), "n": repeats}


# --------------------------------------------------------------------------
# Data scaling — tile a ROOT file K times into a bigger physical file
# --------------------------------------------------------------------------
def scale_root(root_file: str, scale: int, out: str,
               dataset: str = "dimuon") -> str:
    """Write ``out`` = the input tiled ``scale`` times (same tree/branches)."""
    ds = DATASETS[dataset]
    with uproot.open(root_file) as f:
        tname = ds.tree
        if tname is None:
            for name, cls in f.classnames().items():
                if cls == "TTree":
                    tname = name.split(";")[0]
                    break
        # flat (dimuon) branches: numpy dict tiles cleanly
        arrays = f[tname].arrays(library="np")
    tiled = {k: np.tile(v, scale) for k, v in arrays.items()}
    # write a TTree explicitly (uproot's f[name]=... defaults to RNTuple in 5.7,
    # which also can't take the object-dtype string branch)
    branch_types = {k: (str if v.dtype == object else v.dtype)
                    for k, v in tiled.items()}
    with uproot.recreate(out) as f:
        f.mktree(tname, branch_types)
        f[tname].extend(tiled)
    return out


# --------------------------------------------------------------------------
# The reference analysis (Awkward core, shared by awkward + hybrid backends)
# --------------------------------------------------------------------------
def zmumu_masses(events: ak.Array) -> np.ndarray:
    """Invariant mass of opposite-charge dimuon events in 60–120 GeV."""
    m = invariant_mass(events.muons)
    mask = opposite_charge_pair(events) & (m > 60) & (m < 120)
    return ak.to_numpy(m[mask])


def _metric(masses: np.ndarray) -> dict:
    return {"n": int(masses.size),
            "mean": float(np.mean(masses)) if masses.size else float("nan")}


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
def backend_awkward(root_file: str, dataset: str, repeats: int) -> dict:
    """Pure Awkward: setup reads+reconstructs; query analyzes in memory."""
    events_df, muons_df = None, None

    def setup():
        nonlocal events_df, muons_df
        ds = DATASETS[dataset]
        events_ak, _ = read_root(root_file, ds, None)
        tables = build_tables(events_ak, ds)
        events_df, muons_df = tables["events"], tables["muons"]
        return reconstruct_events(events_df, muons_df)

    events, setup_t = timed(setup, repeats=1, warmup=0)
    result, query_t = timed(lambda: zmumu_masses(events), repeats=repeats)
    return {"setup": setup_t, "query": query_t, "metric": _metric(result)}


def backend_hybrid(root_file: str, dataset: str, repeats: int,
                   hybrid_backend: str = "embedded",
                   conn_kwargs: dict | None = None,
                   where: str = "mass BETWEEN 60 AND 120") -> dict:
    """MonetDB + Awkward: setup ingests+loads; query = SQL slice + reconstruct."""
    from . import db
    conn_kwargs = conn_kwargs or {}
    ds = DATASETS[dataset]
    schema_name = {"dimuon": "dimuon", "nanoaod": "nanoaod"}[dataset]

    events_ak, _ = read_root(root_file, ds, None)
    tables = build_tables(events_ak, ds)

    if hybrid_backend == "embedded":
        conn = db.open_embedded(conn_kwargs.get("dbdir") or ":memory:")
        use_copy = False
    else:
        conn = db.open_server(**{k: v for k, v in conn_kwargs.items()
                                 if k != "dbdir"})
        use_copy = True

    def setup():
        db.create_schema(conn, db.read_schema(schema_name))
        load_tables(conn, tables, use_copy_into=use_copy)
        return True

    _, setup_t = timed(setup, repeats=1, warmup=0)

    def query():
        events_df, muons_df = fetch_tables(conn, where=where)
        return zmumu_masses(reconstruct_events(events_df, muons_df))

    try:
        result, query_t = timed(query, repeats=repeats)
    finally:
        conn.close()
    return {"setup": setup_t, "query": query_t, "metric": _metric(result)}


def backend_rdataframe(root_file: str, dataset: str, repeats: int) -> dict:
    """ROOT RDataFrame computing the same Z→μμ selection. Requires ROOT."""
    import ROOT  # noqa: F401 — only imported when this backend is requested

    tree = DATASETS[dataset].tree or "events"
    mass_expr = ("sqrt((E1+E2)*(E1+E2) - ((px1+px2)*(px1+px2) "
                 "+ (py1+py2)*(py1+py2) + (pz1+pz2)*(pz1+pz2)))")

    def query():
        df = ROOT.RDataFrame(tree, root_file)
        df = (df.Define("m_mumu", mass_expr)
                .Filter("Q1 != Q2")
                .Filter("m_mumu > 60 && m_mumu < 120"))
        n = df.Count().GetValue()
        mean = df.Mean("m_mumu").GetValue() if n else float("nan")
        return {"n": int(n), "mean": float(mean)}

    result, query_t = timed(query, repeats=repeats)
    return {"setup": {"median": 0.0, "min": 0.0, "n": 0}, "query": query_t,
            "metric": result}


BACKENDS = {
    "awkward": backend_awkward,
    "hybrid": backend_hybrid,
    "rdataframe": backend_rdataframe,
}


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def run(root_file: str, *, dataset: str = "dimuon",
        backends=("awkward", "hybrid"), scale: int = 1, repeats: int = 5,
        hybrid_backend: str = "embedded", conn_kwargs: dict | None = None,
        rtol: float = 1e-6) -> dict:
    """Run the selected backends and return a results dict. Prints a table."""
    tmp = None
    if scale > 1:
        tmp = tempfile.NamedTemporaryFile(suffix=".root", delete=False)
        tmp.close()
        print(f"scaling {root_file} ×{scale} → temp file ...")
        root_file = scale_root(root_file, scale, tmp.name, dataset)

    results: dict = {}
    for name in backends:
        print(f"running backend: {name} ...")
        try:
            if name == "hybrid":
                results[name] = backend_hybrid(
                    root_file, dataset, repeats,
                    hybrid_backend=hybrid_backend, conn_kwargs=conn_kwargs)
            else:
                results[name] = BACKENDS[name](root_file, dataset, repeats)
        except ImportError as e:
            print(f"  skipped {name}: {e}")
            results[name] = {"skipped": str(e)}

    _print_table(results)
    _check_agreement(results, rtol)

    if tmp is not None:
        import os
        os.unlink(tmp.name)
    return results


def _print_table(results: dict) -> None:
    print("\n" + "=" * 74)
    print(f"{'backend':<12} {'setup (s)':>10} {'query median (ms)':>18} "
          f"{'n':>8} {'mean m':>10}")
    print("-" * 74)
    for name, r in results.items():
        if "skipped" in r:
            print(f"{name:<12} {'—':>10} {'skipped':>18}")
            continue
        setup = r["setup"]["median"]
        qms = r["query"]["median"] * 1e3
        print(f"{name:<12} {setup:>10.3f} {qms:>18.3f} "
              f"{r['metric']['n']:>8d} {r['metric']['mean']:>10.4f}")
    print("=" * 74)


def _check_agreement(results: dict, rtol: float) -> None:
    metrics = {k: v["metric"] for k, v in results.items() if "metric" in v}
    if len(metrics) < 2:
        return
    ref_name, ref = next(iter(metrics.items()))
    ok = True
    for name, m in metrics.items():
        if m["n"] != ref["n"] or (
            m["n"] and abs(m["mean"] - ref["mean"]) > rtol * abs(ref["mean"])
        ):
            ok = False
            print(f"  DISAGREE: {name} (n={m['n']}, mean={m['mean']:.6f}) "
                  f"vs {ref_name} (n={ref['n']}, mean={ref['mean']:.6f})")
    print(f"agreement across backends: {'OK ✓' if ok else 'FAILED'}")
