<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ianna/awkward-monetizer/main/docs/logo/awkward-monetizer-logo-dark.svg">
    <img alt="awkward-monetizer" src="https://raw.githubusercontent.com/ianna/awkward-monetizer/main/docs/logo/awkward-monetizer-logo.svg" width="640">
  </picture>
</p>

Hybrid analysis of nested (ragged) HEP data: flatten ROOT events into
[MonetDB](https://www.monetdb.org/) tables, push selections down to SQL, then
rebuild the nesting in [Awkward Array](https://awkward-array.org/) for the
physics. An LLM agent can drive the same pipeline through MCP.

```
ROOT ──uproot──▶ Awkward ──flatten──▶ MonetDB
                                         │
   Awkward (NF2) ◀──unflatten── SQL slice┘
        │
        └─▶ physics: invariant mass, ΔR, combinatorics ─▶ plots
```

## Install

```bash
python -m pip install awkward-monetizer             # core
python -m pip install "awkward-monetizer[agent]"    # + LLM agent and MCP server
```

Python 3.10+. The MonetDB server is installed separately (`brew install monetdb`
on macOS). For development (pip or pixi), see
[docs/usage.md](https://github.com/ianna/awkward-monetizer/blob/main/docs/usage.md#development-setup).

## Quickstart

```bash
awkward-monetizer make-nano          # synthetic NanoAOD file
awkward-monetizer adl                # IRIS-HEP ADL Q1–Q8 on it, no database needed
```

With a MonetDB server running, load real CMS Open Data:

```bash
monetdb create hep && monetdb release hep
awkward-monetizer ingest <NanoAOD file>.root --dataset nanoaod --database hep \
    --create-schema --step-size 50000 --entry-stop 100000
```

## Agentic analysis

An LLM agent gets six tools: schema discovery, read-only SQL, Awkward code on
events selected in SQL, submit, plot, and (over MCP) a check against the ADL
reference. To use it from Claude Code:

```bash
claude mcp add -s user awkward-monetizer -- awkward-monetizer agent mcp --database hep
```

On 100,000 CMS Open Data events (DoubleMuon Run2016H, record 30555), Claude
answered all eight ADL questions correctly on the first attempt, doing cuts in
SQL and combinatorics in Awkward. See
[docs/agent.md](https://github.com/ianna/awkward-monetizer/blob/main/docs/agent.md).

## Documentation

- [docs/usage.md](https://github.com/ianna/awkward-monetizer/blob/main/docs/usage.md):
  development setup, datasets, ingestion, sharding, benchmarks, notebooks, tests
- [docs/agent.md](https://github.com/ianna/awkward-monetizer/blob/main/docs/agent.md):
  the agent, its tools, evaluation and demo
- [CONTRIBUTING.md](https://github.com/ianna/awkward-monetizer/blob/main/CONTRIBUTING.md):
  how to contribute and how releases are made

## Status

A finished prototype in maintenance mode: bug fixes and releases continue, no
new features are planned. Arrow transport, electrons and b-tag are open for
anyone who wants to take them on.

## License

BSD-3-Clause.
