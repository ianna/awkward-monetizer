# awkward-monetizer

A hybrid High-Energy Physics (HEP) analysis engine that combines:

- **[MonetDB](https://www.monetdb.org/)** — fast, columnar SQL over event-level data
- **[Awkward Array](https://awkward-array.org/)** — high-performance nested (NF2) physics analysis
- **[Apache Arrow](https://arrow.apache.org/)** — the bridge between the two

Interactive exploration is done in [Marimo](https://marimo.io/) notebooks; the aim
is a Python-native alternative to [ROOT/RDataFrame](https://root.cern/doc/master/classROOT_1_1RDataFrame.html).

The idea: flatten nested ROOT events into relational tables, push selections down
to SQL, then reconstruct the nested structure in Awkward for the physics. It's
motivated by *Evaluating Query Languages and Systems for High-Energy Physics
Data* (Graur et al., PVLDB vol. 15, 2022; [arXiv:2104.12615](https://arxiv.org/abs/2104.12615)),
which found SQL systems struggle with nested data while ROOT stays dominant
thanks to its NF2-friendly execution model.

## Pipeline

```
ROOT ──uproot──▶ Awkward ──flatten──▶ pandas ──▶ MonetDB
                                                     │
   Awkward (NF2) ◀──unflatten── pandas ◀──SQL slice──┘
        │
        └─▶ physics: invariant mass, ΔR, combinatorics ─▶ plots
```

## Layout

```
src/awkward_monetizer/
  datasets.py     dataset registry (dimuon, nanoaod) + collection types
  ingest.py       ROOT → Awkward → flat tables → MonetDB
  reconstruct.py  flat tables → Awkward NF2 (ak.unflatten)
  db.py           MonetDB connections (server / embedded) + packaged schema
  physics.py      4-vectors (scikit-hep vector), invariant mass, charge pairs
  adl.py          ADL Q1–Q8 benchmark queries
  roundtrip.py    full ingest → DB → slice → reconstruct → validate cycle
  sampledata.py   synthetic NanoAOD generator
  cli.py          `awkward-monetizer` command-line interface
  schemas/        packaged SQL (dimuon.sql, nanoaod.sql, jets.sql)
examples/         analysis.py (Marimo notebook), make_sample_nanoaod.py
tests/            pytest suite (reconstruction, round-trip, ADL)
data/             inputs (git-ignored) — see data/README.md
```

## Install

Development environment via **pixi** (builds Awkward from a local source
checkout; see *Awkward from source* below):

```bash
git clone --recursive https://github.com/scikit-hep/awkward.git
pixi install
```

Or plain pip (uses released Awkward):

```bash
pip install -e ".[notebook,dev]"
```

The MonetDB **server** is separate (not on PyPI/conda): `brew install monetdb`
on macOS. The embedded engine (`pip install ".[embedded]"`, i.e. `monetdbe`) needs
Python ≤ 3.10.

## Quickstart

```bash
awkward-monetizer ingest data/cms.root --dry-run    # ROOT → flat tables
awkward-monetizer reconstruct data/cms.root         # → Awkward NF2 + validate
awkward-monetizer make-nano                          # synthetic NanoAOD
awkward-monetizer adl                                # ADL Q1–Q8 on it
```

With pixi these are tasks: `pixi run ingest-dryrun`, `pixi run reconstruct-demo`,
`pixi run make-nano`, `pixi run adl`, `pixi run notebook`, `pixi run test`.

## Datasets

`event_id` is the join key across tables; `*_index` columns preserve each
object's position so nesting round-trips exactly. Two layouts (`--dataset`):

- **dimuon** (default; `data/cms.root`) — flat CMS dimuon ntuple: one row per
  event, two muons as parallel columns (`pt1/pt2`, `Q1/Q2`, …), precomputed mass
  `M`. Un-pivoted into a `muons` table. Schema: `schemas/dimuon.sql`.
- **nanoaod** — jagged per-event lists (`Jet_pt`, `Muon_pt`, …) exploded into
  `jets`/`muons`, plus MET. Schema: `schemas/nanoaod.sql`.

## Live MonetDB round-trip

```bash
brew install monetdb
monetdbd create ~/hep-dbfarm && monetdbd start ~/hep-dbfarm
monetdb create hep && monetdb release hep
awkward-monetizer roundtrip --root-file data/cms.root      # server backend
```

`roundtrip` ingests, creates the schema, loads (`COPY INTO`), pushes a
`WHERE mass BETWEEN 60 AND 120` slice to SQL, reconstructs, and checks the
reconstructed dimuon mass against the stored `M`. `--backend embedded` runs the
same cycle in-process via `monetdbe` (loads with `INSERT`).

## ADL benchmark

`adl.py` implements the eight IRIS-HEP ADL benchmark queries (Q1 MET, Q2/Q3 jet
pT, Q4 ≥2 jets, Q5 OS dimuon, Q6 trijet, Q7 lepton-cleaned HT, Q8 MT of MET +
lead lepton) on reconstructed NanoAOD events using `vector`. No real NanoAOD file
ships here — `make-nano` writes a synthetic one; point `--from-root` at a real
CMS Open Data NanoAOD file to run on data. Scope: only muons are present, so
Q7/Q8 use muons as leptons; Q6 omits the b-tag term (no b-tag branch yet).

## Awkward from source

`pixi.toml` builds Awkward from a local `./awkward` clone (path dependencies:
`awkward-cpp` compiled, `awkward` editable) rather than a wheel — a `{ git=... }`
spec resolves to the released wheel. Verify with `pixi run check-awkward`: the
`file:` path should resolve into `./awkward/src/awkward/`. On macOS the source
build pins `MACOSX_DEPLOYMENT_TARGET` so the compiled wheel matches conda-forge's
Python; if a build is rejected as `macosx_15_0` incompatible, run
`MACOSX_DEPLOYMENT_TARGET=11.0 pixi install`.

## Testing

```bash
pytest        # or: pixi run test
```

The ADL suite runs on generated synthetic data; the dimuon and round-trip tests
use `data/cms.root` and skip cleanly if it isn't present.

## Status

Ingestion, reconstruction, the live MonetDB round-trip, and ADL Q1–Q8 are
implemented and validated (round-trip verified on a real MonetDB server; the
reconstructed dimuon mass matches the stored `M` to ~1e-8). See `docs/roadmap.md`
for what's next (real NanoAOD data, ROOT/RDataFrame benchmarks, Arrow Flight).

## License

BSD-3-Clause. Contributions welcome — see `CONTRIBUTING.md`.
