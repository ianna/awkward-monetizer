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

Two more analyses reuse the harness on the jagged ``nanoaod`` dataset, for real
CMS NanoAOD files: :func:`run_zmumu_nano` (Z→μμ over variable-multiplicity
``Muon_*``) and :func:`run_trijet` (ADL Q6). The flat-ntuple ``zmumu`` path above
assumes exactly two muons per event and only works on the teaching ntuple.
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
from .physics import invariant_mass, muons_p4, opposite_charge_pair
from .reconstruct import fetch_dimuon_events, reconstruct_events
from .timing import stage_time


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


# Whole Z→μμ selection expressed in SQL, using the dimuon_mass UDF: self-join the
# two muons per event, compute the mass in-database, keep opposite-charge pairs in
# the Z window. Returns just the surviving masses.
_DIMUON_SELECT_SQL = """SELECT m FROM (
  SELECT dimuon_mass(a.e, a.px, a.py, a.pz, b.e, b.px, b.py, b.pz) AS m,
         a.charge AS qa, b.charge AS qb
  FROM muons a JOIN muons b
    ON a.event_id = b.event_id AND a.muon_index = 0 AND b.muon_index = 1
) t
WHERE qa <> qb AND m BETWEEN 60 AND 120"""


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
    if dataset != "dimuon":
        raise ValueError("the hybrid dimuon analysis requires the dimuon dataset")
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

    samples = []

    def query():
        stages = {}
        events = fetch_dimuon_events(conn, where=where, timings=stages)
        with stage_time(stages, "analysis"):
            masses = zmumu_masses(events)
        samples.append(stages)
        return masses

    try:
        _, setup_t = timed(setup, repeats=1, warmup=0)
        query()  # Warm up without including it in stage statistics.
        samples.clear()
        result, query_t = timed(query, repeats=repeats, warmup=0)
    finally:
        conn.close()
    stages = {
        key: {"median": statistics.median(s[key] for s in samples),
              "min": min(s[key] for s in samples), "n": len(samples)}
        for key in samples[0]
    }
    return {"setup": setup_t, "query": query_t, "metric": _metric(result),
            "stages": stages}


def backend_monetdb(root_file: str, dataset: str, repeats: int,
                    hybrid_backend: str = "embedded",
                    conn_kwargs: dict | None = None) -> dict:
    """Whole analysis in the database via a SQL UDF — no Awkward reconstruction.

    setup ingests + loads + defines the `dimuon_mass` UDF; query runs the entire
    Z→μμ selection in SQL and fetches only the surviving masses.
    """
    if dataset != "dimuon":
        raise ValueError("the monetdb (in-DB UDF) backend is dimuon-only")
    from . import db
    conn_kwargs = conn_kwargs or {}
    ds = DATASETS[dataset]
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
        db.create_schema(conn, db.read_schema("dimuon"))
        load_tables(conn, tables, use_copy_into=use_copy)
        db.apply_sql(conn, db.read_schema("udf_dimuon"), split=False)
        return True

    _, setup_t = timed(setup, repeats=1, warmup=0)

    def query():
        cur = conn.cursor()
        cur.execute(_DIMUON_SELECT_SQL)
        return np.array([r[0] for r in cur.fetchall()], dtype=float)

    try:
        result, query_t = timed(query, repeats=repeats)
    finally:
        conn.close()
    return {"setup": setup_t, "query": query_t, "metric": _metric(result)}


def backend_rdataframe(root_file: str, dataset: str, repeats: int) -> dict:
    """ROOT RDataFrame computing the same Z→μμ selection. Requires ROOT."""
    import ROOT

    tree = DATASETS[dataset].tree or "events"
    mass_expr = ("sqrt((E1+E2)*(E1+E2) - ((px1+px2)*(px1+px2) "
                 "+ (py1+py2)*(py1+py2) + (pz1+pz2)*(pz1+pz2)))")

    def query():
        df = ROOT.RDataFrame(tree, root_file)
        df = (df.Define("m_mumu", mass_expr)
                .Filter("Q1 != Q2")
                .Filter("m_mumu > 60 && m_mumu < 120"))
        count_action = df.Count()
        mean_action = df.Mean("m_mumu")
        n = count_action.GetValue()
        mean = mean_action.GetValue() if n else float("nan")
        return {"n": int(n), "mean": float(mean)}

    result, query_t = timed(query, repeats=repeats)
    return {"setup": {"median": 0.0, "min": 0.0, "n": 0}, "query": query_t,
            "metric": result}


# ==========================================================================
# ADL Q6 trijet — the contrast case (jagged combinatorics)
# ==========================================================================
# The whole Q6 selection in SQL: a 3-way self-join over jet triples (the
# combinatorial blow-up that `ak.combinations(jets, 3)` expresses in one line),
# the trijet mass/pT computed by UDFs, and a window function to pick, per event,
# the triple whose mass is closest to 172.5 GeV.
_TRIJET_SELECT_SQL = """SELECT pt FROM (
  SELECT a.event_id,
         trijet_pt(a.pt, a.phi, b.pt, b.phi, c.pt, c.phi) AS pt,
         ROW_NUMBER() OVER (
           PARTITION BY a.event_id
           ORDER BY abs(trijet_mass(a.pt, a.eta, a.phi, a.mass,
                                    b.pt, b.eta, b.phi, b.mass,
                                    c.pt, c.eta, c.phi, c.mass) - 172.5)
         ) AS rn
  FROM jets a
  JOIN jets b ON a.event_id = b.event_id AND a.jet_index < b.jet_index
  JOIN jets c ON a.event_id = c.event_id AND b.jet_index < c.jet_index
) ranked
WHERE rn = 1"""


def _arr_metric(a: np.ndarray) -> dict:
    return {"n": int(a.size),
            "mean": float(np.mean(a)) if a.size else float("nan")}


def trijet_awkward(nano_file: str, repeats: int) -> dict:
    """ADL Q6 in Awkward: setup reconstructs; query = ak.combinations + argmin."""
    from .adl import events_from_root, q6_trijet_pt
    events = None

    def setup():
        nonlocal events
        events = events_from_root(nano_file)
        return events

    _, setup_t = timed(setup, repeats=1, warmup=0)
    result, query_t = timed(lambda: q6_trijet_pt(events), repeats=repeats)
    return {"setup": setup_t, "query": query_t, "metric": _arr_metric(result)}


def trijet_monetdb(nano_file: str, repeats: int, hybrid_backend: str = "embedded",
                   conn_kwargs: dict | None = None) -> dict:
    """ADL Q6 in the database: 3-way self-join + UDFs + window function."""
    from . import db
    conn_kwargs = conn_kwargs or {}
    ds = DATASETS["nanoaod"]
    events_ak, _ = read_root(nano_file, ds, None)
    tables = build_tables(events_ak, ds)

    if hybrid_backend == "embedded":
        conn = db.open_embedded(conn_kwargs.get("dbdir") or ":memory:")
        use_copy = False
    else:
        conn = db.open_server(**{k: v for k, v in conn_kwargs.items()
                                 if k != "dbdir"})
        use_copy = True

    def setup():
        db.create_schema(conn, db.read_schema("nanoaod"))
        load_tables(conn, tables, use_copy_into=use_copy)
        db.load_functions(conn, db.read_schema("udf_trijet"))
        return True

    _, setup_t = timed(setup, repeats=1, warmup=0)

    def query():
        cur = conn.cursor()
        cur.execute(_TRIJET_SELECT_SQL)
        return np.array([r[0] for r in cur.fetchall()], dtype=float)

    try:
        result, query_t = timed(query, repeats=repeats)
    finally:
        conn.close()
    return {"setup": setup_t, "query": query_t, "metric": _arr_metric(result)}


def run_trijet(nano_file: str, *, backends=("awkward", "monetdb"), repeats: int = 5,
               hybrid_backend: str = "embedded", conn_kwargs: dict | None = None,
               rtol: float = 1e-6) -> dict:
    """Run the Q6 trijet comparison (awkward vs in-DB SQL) and print a table."""
    results: dict = {}
    for name in backends:
        print(f"running trijet backend: {name} ...")
        if name == "awkward":
            results[name] = trijet_awkward(nano_file, repeats)
        elif name == "monetdb":
            results[name] = trijet_monetdb(nano_file, repeats,
                                           hybrid_backend=hybrid_backend,
                                           conn_kwargs=conn_kwargs)
        else:
            raise ValueError(f"trijet analysis supports awkward/monetdb, not {name!r}")
    _print_table(results)
    _check_agreement(results, rtol)
    return results


# ==========================================================================
# Z->mumu on real NanoAOD (jagged muons) -- the fix for real CMS NanoAOD files.
# The flat-ntuple zmumu path assumes exactly two muons per event; real NanoAOD
# stores Muon_* as per-event variable-length lists. Here the selection is every
# opposite-charge muon pair with 60 < m(mumu) < 120 GeV -- expressed identically
# in Awkward (ak.combinations) and in SQL (a self-join over the jagged muons
# table), so the backends agree exactly.
# ==========================================================================
def zmumu_masses_nano(events: ak.Array) -> np.ndarray:
    """Mass of every opposite-charge muon pair in 60-120 GeV, over jagged
    NanoAOD muons (any multiplicity >= 2). Returns a flat 1-D array of masses."""
    mu = muons_p4(events)
    pair = ak.combinations(mu, 2, fields=["a", "b"])
    m = (pair.a + pair.b).mass
    good = (pair.a.charge != pair.b.charge) & (m > 60) & (m < 120)
    return ak.to_numpy(ak.flatten(m[good]))


# The same selection in SQL. NOTE on performance: a scalar SQL UDF called once
# per muon pair is interpreted row-by-row and defeats MonetDB's vectorized
# operators. Instead we precompute each muon's Cartesian components ONCE (the CTE
# `mp`, vectorized over the whole muons column), then the self-join is pure bulk
# arithmetic on those columns. Same physics as dimuon_mass_nano, ~an order of
# magnitude faster. Strict window (> / <) to match the Awkward `(m>60)&(m<120)`.
# Shared CTE: precompute each muon's Cartesian components ONCE (vectorized over
# the whole muons column), then form same-event pairs and their masses. The full
# query streams the masses; the metric query aggregates count/avg in-database so
# only two numbers cross the client socket (the --metric-only path).
_ZMUMU_NANO_CTE = """WITH mp AS (
  SELECT event_id, muon_index, charge,
         pt*cos(phi)               AS px,
         pt*sin(phi)               AS py,
         pt*(exp(eta)-exp(-eta))/2 AS pz,
         sqrt(pt*pt + power(pt*(exp(eta)-exp(-eta))/2, 2) + mass*mass) AS e
  FROM muons
),
pairs AS (
  SELECT sqrt((a.e+b.e)*(a.e+b.e)
              - ((a.px+b.px)*(a.px+b.px)
                 + (a.py+b.py)*(a.py+b.py)
                 + (a.pz+b.pz)*(a.pz+b.pz))) AS mll
  FROM mp a JOIN mp b
    ON a.event_id = b.event_id AND a.muon_index < b.muon_index
       AND a.charge <> b.charge   -- prune same-charge pairs BEFORE the mass
)"""

_ZMUMU_NANO_WHERE = "WHERE mll > 60 AND mll < 120"

# Stream every surviving mass (client computes the metric -- matches how the
# awkward backend materializes the full array).
_ZMUMU_NANO_SELECT_SQL = f"{_ZMUMU_NANO_CTE}\nSELECT mll FROM pairs {_ZMUMU_NANO_WHERE}"

# Aggregate in the database: return only (count, mean).
_ZMUMU_NANO_METRIC_SQL = (
    f"{_ZMUMU_NANO_CTE}\n"
    f"SELECT count(*) AS n, avg(mll) AS mean FROM pairs {_ZMUMU_NANO_WHERE}")


def zmumu_nano_awkward(nano_file: str, repeats: int) -> dict:
    """Z->mumu on NanoAOD in pure Awkward: setup reconstructs jagged muons;
    query forms OS pairs and keeps the Z window."""
    from .adl import events_from_root
    events = None

    def setup():
        nonlocal events
        events = events_from_root(nano_file)   # reconstructs jets + muons
        return events

    _, setup_t = timed(setup, repeats=1, warmup=0)
    result, query_t = timed(lambda: zmumu_masses_nano(events), repeats=repeats)
    return {"setup": setup_t, "query": query_t, "metric": _metric(result)}


def zmumu_nano_monetdb(nano_file: str, repeats: int,
                       hybrid_backend: str = "embedded",
                       conn_kwargs: dict | None = None,
                       metric_only: bool = False) -> dict:
    """Z->mumu on NanoAOD entirely in the database: setup ingests the jagged
    muons; query is the self-join selection.

    metric_only=True aggregates count/avg in the database and ships only two
    numbers, instead of streaming every surviving mass to the client -- the
    apples-to-apples "how fast can the engine get the answer" number, without the
    result-serialization cost that dominates the full-fetch path at scale."""
    from . import db
    conn_kwargs = conn_kwargs or {}
    ds = DATASETS["nanoaod"]
    events_ak, _ = read_root(nano_file, ds, None)
    tables = build_tables(events_ak, ds)

    if hybrid_backend == "embedded":
        conn = db.open_embedded(conn_kwargs.get("dbdir") or ":memory:")
        use_copy = False
    else:
        conn = db.open_server(**{k: v for k, v in conn_kwargs.items()
                                 if k != "dbdir"})
        use_copy = True

    # Z->mumu touches only events + muons; skip the (large) jets table entirely.
    load = {k: tables[k] for k in ("events", "muons") if k in tables}

    def setup():
        db.create_schema(conn, db.read_schema("nanoaod"))
        load_tables(conn, load, use_copy_into=use_copy)
        return True

    _, setup_t = timed(setup, repeats=1, warmup=0)

    if metric_only:
        def query():
            cur = conn.cursor()
            cur.execute(_ZMUMU_NANO_METRIC_SQL)
            n, mean = cur.fetchall()[0]
            return {"n": int(n), "mean": float(mean) if n else float("nan")}
    else:
        def query():
            cur = conn.cursor()
            cur.execute(_ZMUMU_NANO_SELECT_SQL)
            return _metric(np.array([r[0] for r in cur.fetchall()], dtype=float))

    try:
        result, query_t = timed(query, repeats=repeats)
    finally:
        conn.close()
    return {"setup": setup_t, "query": query_t, "metric": result}


# ROOT RDataFrame over the jagged NanoAOD Muon_* branches. All-OS-pairs-in-window
# needs per-event combinatorics, done in a small JITted helper. Gated on ROOT.
_ZMUMU_NANO_CPP = r"""
#include "ROOT/RVec.hxx"
#include "Math/Vector4D.h"
using namespace ROOT::VecOps;
RVec<double> amz_osPairMassesInWindow(
        const RVec<float>& pt, const RVec<float>& eta, const RVec<float>& phi,
        const RVec<float>& mass, const RVec<int>& charge, double lo, double hi) {
    RVec<double> out;
    auto c = Combinations(pt, 2);
    for (size_t k = 0; k < c[0].size(); ++k) {
        auto i = c[0][k]; auto j = c[1][k];
        if (charge[i] == charge[j]) continue;
        ROOT::Math::PtEtaPhiMVector p1(pt[i], eta[i], phi[i], mass[i]);
        ROOT::Math::PtEtaPhiMVector p2(pt[j], eta[j], phi[j], mass[j]);
        double m = (p1 + p2).M();
        if (m > lo && m < hi) out.push_back(m);
    }
    return out;
}
"""
_ZMUMU_NANO_CPP_DECLARED = False


def zmumu_nano_rdataframe(nano_file: str, repeats: int,
                         threads: int = 1) -> dict:
    """ROOT RDataFrame computing the same NanoAOD Z->mumu selection. Requires
    ROOT; untested in CI (no ROOT), gated so it is skipped when ROOT is absent.

    The count and sum are reduced in C++ in a single event-loop pass (two lazy
    Sum actions triggered together). Do NOT Take the per-event mass vectors into
    Python -- iterating millions of RVecs in Python dwarfs the actual work.

    threads controls ROOT implicit multithreading: 1 = single-threaded (default,
    to match single-threaded uproot); 0 = all cores; N>1 = N cores. RDataFrame
    parallelizes the event loop across cores and merges the Sum reductions."""
    import ROOT
    if threads != 1 and not ROOT.IsImplicitMTEnabled():
        ROOT.EnableImplicitMT() if threads <= 0 else ROOT.EnableImplicitMT(threads)
    global _ZMUMU_NANO_CPP_DECLARED
    if not _ZMUMU_NANO_CPP_DECLARED:
        ROOT.gInterpreter.Declare(_ZMUMU_NANO_CPP)
        _ZMUMU_NANO_CPP_DECLARED = True

    tree = DATASETS["nanoaod"].tree or "Events"

    def query():
        df = (ROOT.RDataFrame(tree, nano_file)
              .Filter("Muon_pt.size() >= 2")
              .Define("amz_m",
                      "amz_osPairMassesInWindow(Muon_pt, Muon_eta, Muon_phi, "
                      "Muon_mass, Muon_charge, 60, 120)")
              .Define("amz_n", "(double) amz_m.size()")
              .Define("amz_sum", "Sum(amz_m)"))
        n_action = df.Sum["double"]("amz_n")     # both actions share one event
        s_action = df.Sum["double"]("amz_sum")   # loop (lazy, co-triggered)
        n = round(n_action.GetValue())
        s = float(s_action.GetValue())
        return {"n": n, "mean": (s / n) if n else float("nan")}

    result, query_t = timed(query, repeats=repeats)
    return {"setup": {"median": 0.0, "min": 0.0, "n": 0}, "query": query_t,
            "metric": result}


# --------------------------------------------------------------------------
# Columnar transport for the hybrid: pull the SQL result as COLUMNS, not rows,
# and build the jagged Awkward array directly (no pandas, no per-event groupby).
# monetdbe exposes numpy columns natively (fetchnumpy) -- the columnar/Arrow-style
# path; pymonetdb (server) has no columnar API, so we transpose fetchall once
# (still avoids the pandas rebuild that dominated the row-based hybrid).
def _fetch_columns(cur) -> dict:
    if hasattr(cur, "fetchnumpy"):
        return {k: np.asarray(v) for k, v in cur.fetchnumpy().items()}
    names = [d[0] for d in cur.description]
    rows = cur.fetchall()
    if not rows:
        return {n: np.array([]) for n in names}
    cols = list(zip(*rows, strict=True))
    return {n: np.asarray(cols[i]) for i, n in enumerate(names)}


def _muon_events_from_columns(cols: dict) -> ak.Array:
    """Build events.muons directly from columnar muon data ordered by event_id."""
    eid = np.asarray(cols["event_id"])
    if eid.size:
        bounds = np.nonzero(np.diff(eid))[0] + 1          # run-length, eid sorted
        counts = np.diff(np.concatenate(([0], bounds, [eid.size])))
    else:
        counts = np.array([], dtype="int64")
    rec = ak.zip({"pt": cols["pt"].astype("float64"),
                  "eta": cols["eta"].astype("float64"),
                  "phi": cols["phi"].astype("float64"),
                  "mass": cols["mass"].astype("float64"),
                  "charge": cols["charge"]})
    return ak.Array({"muons": ak.unflatten(rec, counts)})


# --------------------------------------------------------------------------
# Hybrid awkward + MonetDB: SQL does the cheap, selective event-level cut (a
# per-event columnar aggregate the DB is fast at); Awkward does the jagged pair
# combinatorics (which the DB is slow at). SQL keeps only events that CAN hold an
# opposite-charge muon pair -- >=1 positive and >=1 negative muon -- and ships
# just those muons; Awkward forms the pairs and applies the mass window.
_ZMUMU_NANO_HYBRID_SQL = """SELECT m.event_id, m.pt, m.eta, m.phi, m.mass, m.charge
FROM muons m
JOIN (
  SELECT event_id
  FROM muons
  GROUP BY event_id
  HAVING sum(CASE WHEN charge > 0 THEN 1 ELSE 0 END) >= 1
     AND sum(CASE WHEN charge < 0 THEN 1 ELSE 0 END) >= 1
) keep ON m.event_id = keep.event_id
ORDER BY m.event_id"""  # run-length counts need only event_id grouped


def zmumu_nano_hybrid(nano_file: str, repeats: int,
                      hybrid_backend: str = "embedded",
                      conn_kwargs: dict | None = None) -> dict:
    """Z->mumu, split across engines: SQL prunes events to those with an OS pair
    possible; Awkward reconstructs the survivors and does the combinatorics."""
    from . import db
    conn_kwargs = conn_kwargs or {}
    ds = DATASETS["nanoaod"]
    events_ak, _ = read_root(nano_file, ds, None)
    tables = build_tables(events_ak, ds)

    if hybrid_backend == "embedded":
        conn = db.open_embedded(conn_kwargs.get("dbdir") or ":memory:")
        use_copy = False
    else:
        conn = db.open_server(**{k: v for k, v in conn_kwargs.items()
                                 if k != "dbdir"})
        use_copy = True

    load = {k: tables[k] for k in ("events", "muons") if k in tables}

    def setup():
        db.create_schema(conn, db.read_schema("nanoaod"))
        load_tables(conn, load, use_copy_into=use_copy)
        return True

    _, setup_t = timed(setup, repeats=1, warmup=0)

    def query():
        cur = conn.cursor()
        cur.execute(_ZMUMU_NANO_HYBRID_SQL)
        events = _muon_events_from_columns(_fetch_columns(cur))
        return _metric(zmumu_masses_nano(events))

    try:
        result, query_t = timed(query, repeats=repeats)
    finally:
        conn.close()
    return {"setup": setup_t, "query": query_t, "metric": result}


_ZMUMU_NANO_BACKENDS = {
    "awkward": zmumu_nano_awkward,
    "monetdb": zmumu_nano_monetdb,
    "rdataframe": zmumu_nano_rdataframe,
}


def run_zmumu_nano(nano_file: str, *, backends=("awkward", "monetdb"),
                   repeats: int = 5, hybrid_backend: str = "embedded",
                   conn_kwargs: dict | None = None, rtol: float = 1e-6,
                   metric_only: bool = False, threads: int = 1) -> dict:
    """Run the NanoAOD Z->mumu comparison (awkward vs in-DB SQL, optional ROOT)
    and print a table. This is the benchmark path for real CMS NanoAOD files.

    metric_only pushes the count/mean aggregation into MonetDB (server-side),
    matching how RDataFrame already reduces in C++ and how Awkward reduces in
    process -- so each engine reports the metric in its native idiom instead of
    serializing the full result set."""
    results: dict = {}
    for name in backends:
        print(f"running zmumu-nano backend: {name} ...")
        try:
            if name == "monetdb":
                results[name] = zmumu_nano_monetdb(
                    nano_file, repeats, hybrid_backend=hybrid_backend,
                    conn_kwargs=conn_kwargs, metric_only=metric_only)
            elif name == "hybrid":
                results[name] = zmumu_nano_hybrid(
                    nano_file, repeats, hybrid_backend=hybrid_backend,
                    conn_kwargs=conn_kwargs)
            elif name == "rdataframe":
                results[name] = zmumu_nano_rdataframe(nano_file, repeats,
                                                      threads=threads)
            elif name in _ZMUMU_NANO_BACKENDS:
                results[name] = _ZMUMU_NANO_BACKENDS[name](nano_file, repeats)
            else:
                raise ValueError(
                    f"zmumu-nano supports {list(_ZMUMU_NANO_BACKENDS)}, "
                    f"not {name!r}")
        except ImportError as e:
            print(f"  skipped {name}: {e}")
            results[name] = {"skipped": str(e)}
    _print_table(results)
    _check_agreement(results, rtol)
    return results


BACKENDS = {
    "awkward": backend_awkward,
    "hybrid": backend_hybrid,
    "monetdb": backend_monetdb,
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
        with tempfile.NamedTemporaryFile(suffix=".root", delete=False) as tmp:
            pass
        print(f"scaling {root_file} ×{scale} → temp file ...")
        root_file = scale_root(root_file, scale, tmp.name, dataset)

    results: dict = {}
    for name in backends:
        print(f"running backend: {name} ...")
        try:
            if name in ("hybrid", "monetdb"):
                fn = backend_hybrid if name == "hybrid" else backend_monetdb
                results[name] = fn(
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

    for name, result in results.items():
        if "stages" not in result:
            continue
        print(f"\n{name} query stages (median ms; warmup excluded):")
        for stage, timing in result["stages"].items():
            print(f"  {stage:<16} {timing['median'] * 1e3:>10.3f}")
        print("  execute/fetch are client-call timings, not isolated server timings.")
        print("  Stage medians need not sum to the total query median.")


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
