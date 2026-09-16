# Project Roadmap
Awkward + MonetDB Hybrid HEP Analysis Engine

This roadmap outlines the development plan for the hybrid HEP analysis system
combining MonetDB, Awkward Array, and Apache Arrow.

**Status legend:** ✅ done · 🚧 in progress · ⬜ not started

## Current Status
Real CMS NanoAOD is now the primary target and works end-to-end. `ingest.py`
(ROOT → flat tables, with chunked `uproot.iterate` streaming) and `reconstruct.py`
(flat tables → Awkward NF2) are validated against both the `cms.root` dimuon
ntuple (reconstructed m(μμ) matches stored M to ~3e-8) and a **real 1.9 GB CMS
NanoAOD file** (2.15 M events / 4.81 M muons / 11.16 M jets). The ADL Q1–Q8
queries (`adl.py`) plus a **four-backend benchmark harness** (`benchmark.py`:
awkward · monetdb · rdataframe · hybrid) run on that real file with every backend
in exact agreement. A NanoAOD-native Z→μμ analysis (`bench --analysis zmumu-nano`)
exercises jagged muon combinatorics; the `zmumu` (flat ntuple) and `trijet`
(ADL Q6) paths cover the flat-scalar and jagged-jet cases. `roundtrip.py` still
runs the full ingest → SQL slice → reconstruct → validate cycle on both the
embedded engine and a live Homebrew server. See **Performance findings** below.

---

## Phase 1 — Foundations (Weeks 1–2)

### 🚧 Data Model
- ✅ Define MonetDB schema (`events`, `jets`, `muons`, `tracks`) — see `schema.sql`
- ⬜ Validate schema against CMS/ATLAS ROOT files
- ⬜ Document NF2 → flat mapping

### ✅ Ingestion Pipeline
- ✅ Implement ROOT → Awkward → flat tables → MonetDB — `ingest.py` (`COPY INTO`, `INSERT` fallback), verified loading into MonetDB via `roundtrip.py`
- ✅ Add batching (chunked uproot.iterate streaming for large/real NanoAOD) and CSV ingestion options
- ⬜ Validate ingestion correctness

### ✅ Basic Reconstruction
- ✅ MonetDB → Awkward round-trip — `reconstruct.py` + `roundtrip.py`; verified end-to-end on the embedded engine (server path via pymonetdb ready)
- ✅ Group objects by event (`ak.unflatten` in `reconstruct.py`)
- ✅ Confirm NF2 reconstruction fidelity (dimuon mass matches stored M to ~7e-9 GeV through a full SQL round-trip)

---

## Phase 2 — ADL Benchmark Implementation (Weeks 3–6)

All eight queries are implemented in `adl.py` (Awkward + scikit-hep `vector`) and validated against a synthetic NanoAOD file (`make_sample_nanoaod.py`): Q8 matches a hand calculation exactly, Q6/Q7 pass constructed spot-checks, Q1–Q5 match independent recomputation. The harness now also runs on a **real CMS NanoAOD file** (see Phase 3). Remaining ADL work: add b-tag (Q6) and electrons (Q8) to the schema for the full canonical query definitions.

### ✅ ADL Q1–Q3
- Simple projections and filters
- Validate performance vs pure Awkward

### ✅ ADL Q4
- Jet counting and event selection
- Compare SQL vs Awkward filtering

### ✅ ADL Q5
- Opposite-charge muon pair selection
- Invariant mass computation

### ✅ ADL Q6
- Trijet combinatorics
- Closest-mass selection
- Benchmark nested combinatorics

### ✅ ADL Q7–Q8
- ΔR matching
- Lepton veto logic
- Multi-object combinatorics

---

## Phase 3 — Performance & Optimization (Weeks 7–10)

### 🚧 SQL Optimization
- ✅ Inline, vectorizable mass expression (per-muon components precomputed in a
  CTE) instead of a per-pair scalar SQL UDF — scalar `LANGUAGE SQL` UDFs are
  interpreted row-by-row and defeat MonetDB's vectorized operators
- ✅ Push the opposite-charge cut into the `JOIN` (prune pairs before the mass)
- ✅ Load only the tables a query needs (skip the 11 M-row jets table for Z→μμ)
- ⬜ Index tuning · materialized views · preselection strategies

### 🚧 Awkward Optimization
- ✅ Compute 4-vectors in **float64** to match MonetDB's `DOUBLE` columns —
  float32 catastrophic cancellation in E²−p² caused a one-pair selection mismatch
  at the window edge on real data
- ✅ Columnar transport for the hybrid (monetdbe `fetchnumpy` / transposed
  `fetchall`) with a direct `ak.unflatten` build — removes the pandas rebuild
  (~16× on the embedded path)
- ⬜ ROOT→Awkward native read (drop the pandas detour in `reconstruct_multi` to
  cut setup)
- ⬜ Numba acceleration · Arrow zero-copy transport (see Long-Term Vision)

### ✅ Benchmarking
`benchmark.py` runs four backends on the same selection, cross-checked to exact
agreement and timed with `--repeats`. Flags: `--analysis {zmumu,zmumu-nano,trijet}`,
`--backends`, `--hybrid-backend {server,embedded}`, `--threads N` (ROOT implicit
MT), `--metric-only` (aggregate count/mean in-DB).

### Performance findings — real 1.9 GB CMS NanoAOD, Z→μμ (465,699 OS pairs, mean 87.06 GeV)
Final per-query medians (`--threads 0`, MonetDB server); all backends agree exactly:

| backend    | query   | evolution |
|------------|---------|-----------|
| awkward    | 471 ms  | reference — vectorized numpy, data resident in memory |
| rdataframe | 901 ms  | 14.9 s → 0.9 s (C++ stream-and-reduce, then all-core MT) |
| monetdb    | 1084 ms | 3.3 s → 1.1 s (inline SQL + charge-prune + jets-skip) |
| hybrid     | 1904 ms | 2.6 s → 1.9 s (SQL event cut + Awkward combinatorics, columnar fetch) |

**Crossover conclusion.** SQL wins flat/scalar cuts (the `zmumu` dimuon
selection, Q1–Q5-style); Awkward wins jagged per-combination work (Z→μμ pairs,
trijet Q6) on **resident** data; RDataFrame's stream-and-reduce wins a **single
cold scan** — it re-reads the file each query (≈0 setup, I/O per query), while
awkward/monetdb/hybrid pay a one-time setup (6.8 s / 12 s) and cheap queries
after. So the `query` column is not apples-to-apples across backends; the ranking
flips between one-shot and repeated use.

**Recurring lesson — materialization at the boundary, not the physics, is the
cost.** Shipping 465 k rows over the pymonetdb socket, `Take`-ing per-event RVecs
into Python, and rebuilding through pandas each dominated their backend. Every fix
had the same shape: reduce/aggregate **inside** the engine, or move data
**columnar**. Corollaries: precision must match the store (float32 vs `DOUBLE`
disagree at window edges); `--metric-only` did **not** help the jagged query,
which proved MonetDB's cost is the self-join + per-pair mass, not result
transport; and multithreading (`--threads 0`) closed most of RDataFrame's gap
(~3.4× here).

**Open bottleneck.** On the **server**, pymonetdb (1.9) exposes no columnar/Arrow
fetch, so the hybrid stays row-bound. Closing this needs an Arrow-Flight transport
(Long-Term Vision); the embedded path already gets columnar via monetdbe
`fetchnumpy`, but monetdbe has no arm64 / Python 3.12 wheel.

---

## Phase 4 — Integration & Tooling (Weeks 11–14)

### ⬜ Plotting
- mplhep
- boost-histogram
- hist

### ⬜ Coffea Integration
- Awkward-native transformations
- Columnar executors

### ⬜ ServiceX Integration
- Arrow Flight transport
- Remote slicing

### ⬜ Packaging
- Docker environment
- Python package
- Example notebooks (Marimo)

---

## Phase 5 — Publication & Outreach (Weeks 15–18)

### ⬜ Documentation
- Tutorials
- Architecture diagrams
- Performance report

### ⬜ Outreach
- Present results to HEP software groups
- Prepare a short paper or blog post
- Invite external contributors

---

## Long-Term Vision
- **Arrow-Flight transport (concrete next perf item)** — pymonetdb (server)
  ships results row-by-row with no columnar API, which is the hybrid's remaining
  bottleneck; an Arrow-Flight path would make MonetDB→Awkward zero-copy and let
  the server match the embedded columnar result.
- Distributed execution via Arrow Flight; GPU acceleration via CuPy + Awkward
- MonetDB UDFs for physics math — `dimuon_mass`, `dimuon_mass_nano`,
  `trijet_mass`/`trijet_pt`. Benchmarks showed a scalar per-row UDF is far slower
  than an inline vectorizable expression, so the hot paths now inline the math;
  UDFs remain the portable/readable reference. Python (`LANGUAGE PYTHON`) UDFs next.
- Full replacement for ROOT/RDataFrame workflows
