# Agentic analysis: an LLM that routes SQL vs Awkward

`awkward_monetizer.agent` lets an LLM answer HEP analysis questions on the
hybrid stack. It runs a plain tool-use loop with six tools, and each step goes
to the engine that our benchmarks say suits it:

| tool | engine | used for |
|---|---|---|
| `describe_schema` | MonetDB catalog | discover tables, columns, types, row counts, units |
| `run_sql` | MonetDB | read-only SELECT preview: cuts, counts, aggregates, flat selections |
| `awkward_eval` | MonetDB → NumPy → Awkward | `where` is pushed into SQL, then agent-written Awkward code does combinatorics, ΔR, masses |
| `submit_answer` | either | final per-entry values, as a `result_id` or as a SQL query |
| `plot_result` | matplotlib | histogram PNG of a `result_id` or SQL column; returns the path (default `output/agent_plots/`). Without `x_range` the axis spans the minimum to the 99.5th percentile and the plot reports the count outside |
| `check_answer` | ADL reference | **MCP only**: score the last submitted answer against Q1–Q8; logged with attempt number |

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

This exposes the same tools, plus `check_answer`, to Claude Code, Claude
Desktop or any MCP client. It runs on a Claude subscription and needs no API
credits.

### Over HTTP instead of stdio

By default the client starts the server as a subprocess and talks to it over
stdio. The server can also run on its own and speak MCP streamable HTTP, so
several sessions (or a client on the same machine that cannot spawn
processes) connect to one running server:

```bash
awkward-monetizer agent mcp --database hep_agent --transport http
# awkward-monetizer MCP server: http://127.0.0.1:8765/mcp
claude mcp add --transport http awkward-monetizer http://127.0.0.1:8765/mcp
```

`--http-host` and `--http-port` change the address. It listens on 127.0.0.1
(this computer only) by default, on purpose: `awkward_eval` runs agent-written
Python behind an AST guard, which is not a sandbox, so anyone who can reach
the port can run code as you. Another address prints a warning; do not expose
the server beyond a network you trust until the Python tool is sandboxed. All
clients of one server share its tool state (stored results, the submitted
answer) and its database connection; tool calls are serialized.

On a large database, `check_answer` first needs the ADL reference answers,
which take one pass over every event (about 40 minutes for the full 49M-event
2016H record on a laptop). `agent mcp` therefore saves them to
`output/refs_DATABASE.npz` by default and reuses them on every later start
(`--ref-cache` picks another file, `--ref-cache ''` turns it off). At startup
the server says whether it found the file or will build it on the first check.

### Large databases

Above `--chunk-events` events (default 1,000,000), `awkward_eval` and the ADL
reference read MonetDB in `event_id`-range chunks and concatenate the results,
fetching only the columns the agent's code names. Agent code must therefore
work event by event (no normalising by global totals); the tool description
tells the model so. `run_sql` previews without transferring the whole result,
and SQL answers are read in binary batches.

On the full DoubleMuon Run2016H record (48.9 M events) the scripted baseline
reproduces all eight ADL answers exactly, in 0.7–2.7 s for Q1–Q4 and 31 s to
under 5 minutes for Q5–Q8 (`docs/usage.md` has the table). Over MCP, Claude
Code answered Q1 there correctly on the first attempt.

Two practical points at this size:

* Reference answers take one pass over every event (about 40 minutes for 49 M
  events). `agent mcp` saves them per database (`output/refs_DATABASE.npz`),
  and `agent eval` takes `--ref-cache` for the same purpose.
* An MCP client may give up on a tool call that stays silent too long. Claude
  Code stopped waiting after 5 minutes, which a full-record Q6
  (`awkward_eval` alone about 4–5 minutes) can reach: raise the client's MCP
  tool timeout before running all eight questions at this size.

### Scoring ADL Q1–Q8 interactively (no API credits)

```bash
awkward-monetizer agent questions     # copy-paste texts for Q1–Q8
```

In Claude Code, paste a question and add: *"Use only the awkward-monetizer
tools. Submit, then call check_answer for Q6."* Each check is appended to
`output/agent_checks.jsonl` with its attempt number, so retries after a FAIL
stay visible. Tabulate the session with:

```bash
awkward-monetizer agent checks        # first-check vs final pass, tries per query
```

`check_answer` is deliberately not given to the built-in `agent eval` loop,
which scores outside the agent's reach. In MCP it is a convenience for a human
driving the session. Report the first-check column, not the final one.

For a clean demo, keep the agent from reading the reference code in this repo:
register the server for your user and start Claude Code in an empty folder.

```bash
claude mcp add -s user awkward-monetizer -- \
  /Users/yana/Projects/Awkward-MonetDB-Marimo/.pixi/envs/default/bin/awkward-monetizer \
  agent mcp --database hep_agent \
  --plot-dir /Users/yana/Projects/Awkward-MonetDB-Marimo/output/agent_plots \
  --check-log /Users/yana/Projects/Awkward-MonetDB-Marimo/output/agent_checks.jsonl
mkdir -p ~/agent-demo && cd ~/agent-demo && claude
```

## Demo script (≈5 min)

1. `agent ask` with a Q6-style question, live or replayed. Narrate the trace:
   the agent reads the schema, pushes `n_jets >= 3` into MonetDB, writes
   `ak.combinations` in Awkward, then submits.
2. Show a self-correction. A run where it first asks for a nested result (or
   a wrong column) and the tool error steers it back.
3. Show the `agent eval` table: Claude vs the scripted baseline on Q1–Q8, with
   pass/fail, the route chosen, the turns and the tokens.
4. Close with the MCP line: any agent framework gets the same tools.

### Live scoreboard in marimo

`marimo/agent_demo.py` puts a verified score next to a live agent session.
Run the MCP server over HTTP with its own check log, open the notebook as an
app, and drive the agent from Claude Code in another window:

```bash
awkward-monetizer agent mcp --database hep_real --transport http --check-log output/agent_checks_demo.jsonl
marimo run marimo/agent_demo.py        # or: pixi run agent-demo
claude mcp add --transport http awkward-monetizer-http http://127.0.0.1:8765/mcp
```

- **Live scoreboard** refreshes every 2 s from the check log: Q1–Q8 with the
  first and latest check, attempts and route, the first-check score, and the
  agent's latest plot from this session.
- **Plots** shows every agent plot made since the notebook started, newest
  first, with file name and time; a toggle shows all plots in the plot folder.
- **Replay (no agent)** runs the reference solution for one question through
  the same tools, with no model and no network: the tool trace with engines
  and times, the histogram and the `check_answer` result. It is the fallback
  if the network fails. A `.root` file in the database box loads it into an
  in-memory database instead of MonetDB.

marimo's chat panel cannot connect to custom MCP servers yet, so the agent
runs in Claude Code (or any MCP client) and the notebook only watches. Clear
the demo check log before a talk so attempts start at 1.

## Caveats

* `awkward_eval` executes model-written Python in-process. AST checks block
  imports, dunder access, file/exec builtins and the numpy/Awkward functions
  that read or write files (`np.load`, `np.save`, `ak.to_parquet`, ...), but
  this is **not a security sandbox**: it limits neither memory nor CPU, and
  Python has indirect routes no syntax check can close. Use it on your own
  machine and data only; with `--transport http` keep the default 127.0.0.1
  address. `agent mcp --no-code-tool` leaves `awkward_eval` out entirely
  (SQL, plots and checks only). See [SECURITY.md](../SECURITY.md).
* Scope matches `adl.py`: muons are the only leptons, and there is no b-tag
  (Q6 omits max b-tag).
* The tools run with autocommit on (they only read), so a failed agent query
  cannot leave the connection in an aborted transaction.
