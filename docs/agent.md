# Agentic analysis: an LLM that routes SQL vs Awkward

`awkward_monetizer.agent` lets an LLM answer HEP analysis questions on the
hybrid stack. It runs a plain tool-use loop with four tools, and each step goes
to the engine that our benchmarks say suits it:

| tool | engine | used for |
|---|---|---|
| `describe_schema` | MonetDB catalog | discover tables, columns, types, row counts, units |
| `run_sql` | MonetDB | read-only SELECT preview: cuts, counts, aggregates, flat selections |
| `awkward_eval` | MonetDB → Arrow/pandas → Awkward | `where` is pushed into SQL, then agent-written Awkward code does combinatorics, ΔR, masses |
| `submit_answer` | either | final per-entry values, as a `result_id` or as a SQL query |

```
question ──► Claude ──tool_use──► AnalysisTools (SQL | Awkward) ──result──┐
               ▲                                                         │
               └──────────────────── tool_result (or error) ◄────────────┘
            ... until submit_answer
```

Errors (bad column, nested result, forbidden SQL) go back to the model as
`is_error` tool results, so it fixes its own call. Every call is traced with its
engine and wall time.

## Ground truth: ADL Q1–Q8

`agent eval` poses the eight IRIS-HEP ADL questions in plain English. It scores
the submitted values against the hand-written `adl.py` functions on the same
database contents, comparing them as a multiset with `allclose`. A pass means
the physics is right, not merely plausible. The `--scripted` expert (no LLM)
is the baseline. It routes Q1–Q4 to pure SQL and Q5–Q8 to SQL-cut + Awkward,
and passes 8/8 on `nano_bench.root` (15k events, embedded MonetDB).

```
query result route   turns  sql  awk  err  tokens   time  detail
Q1    PASS   sql         3    1    0    0       0   0.0s  n=15000 match
Q4    PASS   sql         3    1    0    0       0   0.0s  n=8089 match
Q6    PASS   awkward     3    0    1    0       0   0.4s  n=9425 match
Q8    PASS   awkward     3    0    1    0       0   0.3s  n=4790 match
...
8/8 passed
```

## Running it (macOS, Homebrew MonetDB)

```bash
pixi install                                  # adds anthropic + mcp
monetdb create hep_agent && monetdb release hep_agent
pixi run agent-db                             # ingest data/nano_bench.root
pixi run agent-baseline                       # scripted expert, expect 8/8
export ANTHROPIC_API_KEY=...
pixi run agent-eval                           # Claude on Q1–Q8; transcripts in output/agent_runs/
awkward-monetizer agent ask --database hep_agent --plot q6.png \
  "Find the three-jet combination closest to the top mass in each event and plot its pT"
```

Model: `--model` or `AWKWARD_MONETIZER_MODEL` (default `claude-sonnet-5-5`).
Without a server, `--from-root FILE` loads a throwaway embedded database
(monetdbe if installed, else SQLite, which is fine for trying it but is not
MonetDB).

## Offline / reproducible demo

Each run can be saved (`--transcript run.json`, or `--transcripts DIR` for
eval). `--replay` replays the model's turns while the tools run live against
the database. That gives the same plan and freshly computed numbers, with no
network or API key needed:

```bash
awkward-monetizer agent ask --database hep_agent --replay output/agent_runs/Q6.json
awkward-monetizer agent eval --database hep_agent --replay output/agent_runs
```

Record the runs the day before a talk, then replay them live.

## MCP: point any agent at it

```bash
claude mcp add awkward-monetizer -- awkward-monetizer agent mcp --database hep_agent
```

This exposes the same four tools to Claude Code, Claude Desktop or any MCP
client.

## Demo script (≈5 min)

1. `agent ask` with a Q6-style question, live or replayed. Narrate the trace:
   the agent reads the schema, pushes `n_jets >= 3` into MonetDB, writes
   `ak.combinations` in Awkward, then submits.
2. Show a self-correction. A run where it first asks for a nested result (or
   a wrong column) and the tool error steers it back.
3. Show the `agent eval` table: Claude vs the scripted baseline on Q1–Q8, with
   pass/fail, the route chosen, the turns and the tokens.
4. Close with the MCP line: any agent framework gets the same tools.

## Caveats

* `awkward_eval` executes model-written Python in-process. AST checks block
  imports, dunder access and file/exec builtins, but this is **not a security
  sandbox**. Use it on your own machine and data only.
* Scope matches `adl.py`: muons are the only leptons, and there is no b-tag
  (Q6 omits max b-tag).
* The tools run with autocommit on (they only read), so a failed agent query
  cannot leave the connection in an aborted transaction.
