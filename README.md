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
  `jets`/`muons`, plus MET. Schema: `schemas/nanoaod.sql`. Matches real CMS
  NanoAODv9 branch names/types; `event_id` is synthesized from a row index
  (because `event` alone isn't unique in NanoAOD — the key is run+lumi+event,
  kept as `run`/`lumi`/`event_number` columns).

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

## Benchmarks

`benchmark.py` compares backends on identical data for the dimuon Z→μμ selection
(opposite-charge pair, 60–120 GeV):

- **awkward** — pure Awkward (uproot read → reconstruct → analyze in memory)
- **hybrid** — MonetDB + Awkward: data pre-loaded, the event cut pushed to SQL,
  survivors reconstructed and finished in Awkward
- **monetdb** — the *whole* selection in SQL via a `dimuon_mass` user-defined
  function (`schemas/udf_dimuon.sql`): self-join the two muons per event, compute
  the mass and filter in-database, return only the surviving masses — no Awkward
  reconstruction. Runs on a server or embedded `monetdbe`.
- **rdataframe** — ROOT `RDataFrame` (gated: runs only if `import ROOT` works)

`--scale K` tiles the input into a K× larger physical ROOT file every backend
reads, so timings are meaningful; the query phase is timed (median of `--repeats`)
while one-time setup (reconstruction / DB ingest) is reported separately, since a
database amortizes load cost across many queries.

```bash
awkward-monetizer bench --root-file data/cms.root --scale 50 \\
    --backends awkward,hybrid,rdataframe --repeats 5
```

The backends cross-check on selected-event count and mean mass (the harness prints
`agreement across backends: OK`). Pushing the physics into the database with the UDF (`monetdb` backend) returns
only the final selection, so it can beat the fetch-and-reconstruct `hybrid` path
by a wide margin on this query. The embedded hybrid backend loads via `INSERT`
(slow); use `--hybrid-backend server` (a real MonetDB server with `COPY INTO`) to
benchmark a realistic load path.

### When SQL wins vs when Awkward wins

The harness ships two analyses (`--analysis`) that show the crossover:

- `zmumu` (flat: two muons/event, a scalar mass) **favors the database** — the
  in-DB `monetdb` backend runs the whole selection in SQL and beats
  fetch-and-reconstruct.
- `trijet` (ADL Q6, jagged: every 3-jet combination, keep the mass closest to
  172.5 GeV) **favors Awkward** — `ak.combinations(jets, 3)` is one vectorized
  line, whereas the SQL equivalent (`schemas/udf_trijet.sql`) needs two UDFs, a
  3-way self-join over jet triples, and a window function — and runs *slower* on
  the same data.

```bash
awkward-monetizer bench --analysis trijet --root-file data/nano_synth.root \\
    --backends awkward,monetdb
```

The relational store is the right tool for columnar filtering, materialization,
and scalar arithmetic; Awkward is the right tool for variable-length
combinatorics and nested (NF2) physics. The point of the hybrid design is to use
each where it wins, not to replace one with the other.

## Ingesting real CMS Open Data NanoAOD

The `nanoaod` dataset reads real CMS NanoAODv9 files directly (uproot pulls only
the branches we map, so a multi-GB file isn't fully read). Get a file — e.g. the
DoubleMuon sample from the [CMS NanoAOD guide](https://opendata.cern.ch/docs/cms-getting-started-nanoaod) —
via XRootD and stream it into a running MonetDB `hep` server in chunks:

```bash
xrdcp root://eospublic.cern.ch//eos/opendata/cms/Run2016H/DoubleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/2510000/127C2975-1B1C-A046-AABF-62B77E757A86.root .
awkward-monetizer ingest 127C2975-*.root --dataset nanoaod --database hep \\
    --create-schema --step-size "100 MB" --entry-stop 200000
```

`--step-size` streams with `uproot.iterate` (each chunk is offset so `event_id`
stays globally unique), keeping memory flat regardless of file size;
`--create-schema` (re)creates the tables first. Then the `nanoaod` ADL queries
and benchmarks run on real physics.

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
