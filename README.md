# awkward-monetizer

A hybrid High-Energy Physics (HEP) analysis engine that combines:

- **[MonetDB](https://www.monetdb.org/)** — fast, columnar SQL over event-level data
- **[Awkward Array](https://awkward-array.org/)** — high-performance nested (NF2) physics analysis
- **pandas** — the current table interchange between Python and MonetDB

[Apache Arrow](https://arrow.apache.org/) is a dependency and a planned transport
integration; the current database path uses pandas with CSV `COPY INTO` or SQL
`INSERT` for loading and DB-API rows for fetching. The dimuon hybrid benchmark
converts fetched rows directly to typed NumPy and Awkward arrays.

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
  benchmark.py    dimuon and trijet backend timing/comparison
  roundtrip.py    full ingest → DB → slice → reconstruct → validate cycle
  sampledata.py   synthetic NanoAOD generator
  cli.py          `awkward-monetizer` command-line interface
  schemas/        packaged SQL (dimuon.sql, nanoaod.sql, jets.sql)
examples/         analysis.py (Marimo notebook), make_sample_nanoaod.py
marimo/           ADL dashboard and individual Q4/Q6/Q7/Q8 notebooks
tests/            pytest suite (reconstruction, round-trip, ADL)
data/             inputs (git-ignored) — see data/README.md
.github/workflows/ CI and PyPI Trusted Publishing
```

## Install

Requires Python 3.10 or newer. For a published release:

```bash
python -m pip install awkward-monetizer
# Include plotting and notebook dependencies:
python -m pip install "awkward-monetizer[notebook]"
```

For development and the repository's notebooks, clone this repository first.
Use a virtual environment with pip (this uses released Awkward):

```bash
git clone https://github.com/ianna/awkward-monetizer.git
cd awkward-monetizer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebook,dev]"
```

Alternatively, from the repository root, use **pixi** to build Awkward from a
local source checkout (see *Awkward from source* below):

```bash
git clone --recursive https://github.com/scikit-hep/awkward.git awkward
pixi install
```

The MonetDB **server** is installed separately: `brew install monetdb` on macOS.
The optional embedded backend (`python -m pip install ".[embedded]"` from a
checkout) uses `monetdbe`. Its [published wheels](https://pypi.org/project/monetdbe/#files)
cover Python through 3.10; use Python 3.10 for that extra, or a server backend on
newer Python versions. A database is not needed for the quickstart below.

## Quickstart

Start with generated data; no ROOT files are bundled:

```bash
awkward-monetizer make-nano                         # synthetic NanoAOD
awkward-monetizer adl                               # ADL Q1–Q8 on it
awkward-monetizer ingest data/nano_synth.root --dataset nanoaod --dry-run
```

For the dimuon example, first provide `data/cms.root` as described in
[data/README.md](https://github.com/ianna/awkward-monetizer/blob/main/data/README.md):

```bash
awkward-monetizer ingest data/cms.root --dry-run
awkward-monetizer reconstruct data/cms.root
```

With pixi these are tasks: `pixi run ingest-dryrun`, `pixi run reconstruct-demo`,
`pixi run make-nano`, `pixi run adl`, `pixi run notebook`, `pixi run test`.
Prefix other commands with `pixi run` when using the pixi environment.

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

Round-trip mass validation supports only `dimuon`; unsupported datasets are
rejected before connecting to the database. If the SQL filter selects no events,
validation reports that it did not run and returns `passed=False`.

## ADL benchmark

`adl.py` implements the eight IRIS-HEP ADL benchmark queries (Q1 MET, Q2/Q3 jet
pT, Q4 ≥2 jets, Q5 OS dimuon, Q6 trijet, Q7 lepton-cleaned HT, Q8 MT of MET +
lead lepton) on reconstructed NanoAOD events using `vector`. No real NanoAOD file
ships here — `make-nano` writes a synthetic one; point `--from-root` at a real
CMS Open Data NanoAOD file to run on data. Scope: only muons are present, so
Q7/Q8 use muons as leptons; Q6 omits the b-tag term (no b-tag branch yet).

## Awkward from source

`pixi.toml` builds Awkward from a local `./awkward` clone (path dependencies:
`awkward-cpp` compiled, `awkward` editable). Verify with `pixi run check-awkward`: the
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

For the dimuon analysis, `--scale K` tiles the input into a K× larger physical
ROOT file. Query timings report the median of `--repeats`. Setup is reported
separately, but timing boundaries differ: Awkward setup includes reading and
reconstruction; database setup excludes the preceding ROOT read and flattening;
RDataFrame reads the file during its query. Interpret the results with those
differences in mind. Trijet analysis does not apply `--scale`.

The hybrid benchmark also reports median query-stage times for `execute`,
`fetch`, `column_arrays`, `sort_group`, `array_build`, and `analysis`, excluding
warmup. Execute/fetch measure client calls: the driver may receive some rows
during execute, so these are not isolated server and network timings. Stage
medians need not sum to the overall median. Hybrid selection uses a SQL join
and fetches only the event ID and muon fields required for dimuon analysis.
Typed NumPy conversion bypasses pandas dtype inference and groupby; sorting is
only performed when the returned rows are not already ordered. Contiguous muon
columns feed the same Awkward mass calculation. The general-purpose
`fetch_tables` API still returns pandas DataFrames.
The ROOT backend books count and mean together to compute both in one event
loop per query.

```bash
awkward-monetizer bench --root-file data/cms.root --scale 50 \
    --backends awkward,hybrid,rdataframe --hybrid-backend server --repeats 5
```

The backends cross-check on selected-event count and mean mass (the harness prints
`agreement across backends: OK`). Pushing the physics into the database with the UDF (`monetdb` backend) returns
only the final selection, so it can beat the fetch-and-reconstruct `hybrid` path
by a wide margin on this query. The embedded hybrid backend loads via `INSERT`
(slow); use `--hybrid-backend server` (a real MonetDB server with `COPY INTO`) to
benchmark a realistic load path.

### Comparing scalar selection and jagged combinatorics

The harness ships two analyses (`--analysis`) with different workloads:

- `zmumu` (flat: two muons/event, a scalar mass): the in-DB `monetdb` backend
  runs the whole selection in SQL and returns only the selected masses.
- `trijet` (ADL Q6, jagged: every 3-jet combination, keep the mass closest to
  172.5 GeV): Awkward uses `ak.combinations`; the SQL equivalent needs two UDFs,
  a 3-way self-join over jet triples, and a window function.

Relative performance depends on input size, multiplicities, backend, and
hardware; run the comparison on your data.

```bash
awkward-monetizer bench --analysis trijet --root-file data/nano_synth.root \
    --backends awkward,monetdb --hybrid-backend server
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
awkward-monetizer ingest 127C2975-*.root --dataset nanoaod --database hep \
    --create-schema --step-size "100 MB" --entry-stop 200000
```

`--step-size` streams with `uproot.iterate` (each chunk is offset so `event_id`
stays unique within that ingestion call), bounding the input read to a chunk;
`--create-schema` (re)creates the tables first. Then the `nanoaod` ADL queries
and benchmarks run on real physics.

`--step-size 50000` sets an entry count instead of a memory size. Add `--dry-run`
to build and summarize chunks without connecting to MonetDB, or `--truncate`
to clear the dataset's tables once before loading the chunks.

Each ingestion call starts event IDs at zero; use `--truncate` or
`--create-schema` when replacing a previous load. Appending multiple files with
separate CLI calls is not supported. Chunks are committed individually.

## Marimo notebooks

From a repository checkout with notebook dependencies installed:

```bash
marimo edit examples/analysis.py    # dimuon notebook; requires data/cms.root
```

For the ADL dashboard, start the MonetDB `hep` server described above, then load
the NanoAOD schema and data. `--create-schema` replaces the existing tables:

```bash
awkward-monetizer make-nano
awkward-monetizer ingest data/nano_synth.root --dataset nanoaod --create-schema
marimo edit marimo/dashboard.py
```

The dashboard offers Q4, Q6, Q7, and Q8. Individual notebooks are in `marimo/`.
They fetch the tables and reconstruct events in memory; the dashboard does not
stream the entire analysis in chunks.

## Testing

```bash
pytest        # or: pixi run test
```

ADL and CLI tests use generated synthetic data. Dimuon tests need
`data/cms.root`; embedded database tests also need `monetdbe`. Tests skip when
their prerequisites are unavailable. CI runs lint and tests on Python 3.10–3.12;
it does not run a live MonetDB server.

## Versioning and releases

`hatch-vcs` derives the package version from Git tags. A clean checkout tagged
`v0.1.1` builds version `0.1.1`; there is no static version to edit in
`pyproject.toml`. Commit and push the intended changes before tagging a release.

Publishing a GitHub release triggers `.github/workflows/release.yml`, which
builds the wheel and source archive, tests the wheel, checks its version against
the release tag, and uploads through PyPI Trusted Publishing. Pushing a tag
alone does not trigger publishing. The configured publisher must use owner
`ianna`, repository `awkward-monetizer`, workflow `release.yml`, and environment
`pypi`. See [CONTRIBUTING.md](https://github.com/ianna/awkward-monetizer/blob/main/CONTRIBUTING.md#releases)
for release setup.

## Status

Ingestion, reconstruction, the live MonetDB round-trip, and ADL Q1–Q8 are
implemented, along with chunked NanoAOD ingestion, backend benchmarks, Marimo
notebooks, and a PyPI release workflow. Arrow transport, electron collections,
and the Q6 b-tag quantity remain future work. The ROOT/RDataFrame backend is
implemented but requires a separately installed ROOT environment.

## License

BSD-3-Clause. Contributions welcome — see `CONTRIBUTING.md`.
