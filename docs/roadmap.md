# Project Roadmap
Awkward + MonetDB Hybrid HEP Analysis Engine

This roadmap outlines the development plan for the hybrid HEP analysis system
combining MonetDB, Awkward Array, and Apache Arrow.

**Status legend:** ✅ done · 🚧 in progress · ⬜ not started

## Current Status
The project is in early prototyping. The architecture, roadmap, and relational
schema is defined; `ingest.py` (ROOT → flat tables) and `reconstruct.py`
(flat tables → Awkward NF2) both work and are validated end-to-end against the
real `cms.root` dimuon file (reconstructed m(μμ) matches the stored M to ~3e-8).
A Marimo notebook (`analysis.py`) plots the Z→μμ spectrum, and `roundtrip.py`
runs the full ingest → MonetDB → SQL slice → reconstruct → validate cycle
(verified on both the embedded engine and a live Homebrew server via pymonetdb).
The ADL Q1–Q8 benchmark queries are implemented and validated (`adl.py`). Next:
run ADL on real NanoAOD data and build the ROOT/RDataFrame benchmark harness.

---

## Phase 1 — Foundations (Weeks 1–2)

### 🚧 Data Model
- ✅ Define MonetDB schema (`events`, `jets`, `muons`, `tracks`) — see `schema.sql`
- ⬜ Validate schema against CMS/ATLAS ROOT files
- ⬜ Document NF2 → flat mapping

### ✅ Ingestion Pipeline
- ✅ Implement ROOT → Awkward → flat tables → MonetDB — `ingest.py` (`COPY INTO`, `INSERT` fallback), verified loading into MonetDB via `roundtrip.py`
- ⬜ Add batching and CSV/Arrow ingestion options
- ⬜ Validate ingestion correctness

### ✅ Basic Reconstruction
- ✅ MonetDB → Awkward round-trip — `reconstruct.py` + `roundtrip.py`; verified end-to-end on the embedded engine (server path via pymonetdb ready)
- ✅ Group objects by event (`ak.unflatten` in `reconstruct.py`)
- ✅ Confirm NF2 reconstruction fidelity (dimuon mass matches stored M to ~7e-9 GeV through a full SQL round-trip)

---

## Phase 2 — ADL Benchmark Implementation (Weeks 3–6)

All eight queries are implemented in `adl.py` (Awkward + scikit-hep `vector`) and validated against a synthetic NanoAOD file (`make_sample_nanoaod.py`): Q8 matches a hand calculation exactly, Q6/Q7 pass constructed spot-checks, Q1–Q5 match independent recomputation. Remaining: run on a real CMS NanoAOD sample, add b-tag (Q6) and electrons (Q8) to the schema, and benchmark performance.

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

### ⬜ SQL Optimization
- Index tuning
- Materialized views
- Preselection strategies

### ⬜ Awkward Optimization
- Vectorized physics kernels
- Numba acceleration
- Arrow zero-copy validation

### 🚧 Benchmarking
- `benchmark.py` implemented: awkward vs hybrid (MonetDB) validated end-to-end and timed on scaled cms.root; ROOT/RDataFrame backend written and gated on `import ROOT`.
- Compare:
  - MonetDB only
  - Awkward only
  - Hybrid
  - ROOT/RDataFrame baseline

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
- Distributed execution via Arrow Flight
- GPU acceleration via CuPy + Awkward
- MonetDB UDFs for physics math
- Full replacement for ROOT/RDataFrame workflows
