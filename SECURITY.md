# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub: on the
repository's **Security** tab, choose **Report a vulnerability**. Do not open a
public issue for a vulnerability. Reports are acknowledged as soon as possible;
the project is maintained by one person in maintenance mode, so please allow a
few working days.

Only the latest release receives security fixes.

## The code-execution tool, by design

The agent tool `awkward_eval` (`awkward_monetizer/agent/tools.py`) runs Python
code written by an AI agent, so that the agent can do combinatorics on Awkward
arrays. It is the only place in the package that executes code it was given
(one `exec`). Static analysers report it as CWE-94 (code injection); that is
this intended behaviour, for example
[GCVE-110-VULN-PYPI-2026-000656](https://www.vulnetix.com/vdb/GCVE-110-VULN-PYPI-2026-000656).

### Who can reach it

- **stdio** (the default): only the MCP client that started the server, on
  your own machine.
- **HTTP** (`agent mcp --transport http`): listens on 127.0.0.1 by default.
  Any other address prints a warning, because anyone who can reach the port can
  have code run as the user running the server.
- **The built-in loop** (`agent ask`, `agent eval`): only the model you call.

The realistic risk is **prompt injection**: text the agent reads (a question,
a document, data) steering it to write harmful code, which `awkward_eval` then
runs with your permissions.

### What the guard does and does not do

Before running agent code, an AST check rejects imports, `global`, class
definitions, `with`, `try`, `raise` and `del`; private and dunder attributes;
built-ins such as `open`, `eval`, `exec` and `__import__` (the code sees only a
small allow-list); and numpy and Awkward functions that read or write files
(`np.load`, `np.save`, `np.loadtxt`, `np.memmap`, `ak.to_parquet`,
`ak.from_parquet`, ...).

It is **not a sandbox**. It does not limit memory or CPU, and Python has
indirect routes that no syntax check can close. Treat agent code as code you
run yourself.

SQL from the agent (`run_sql`, SQL answers) is checked to be a single read-only
`SELECT`. That check is a pattern match, not a parser.

## Recommended use

- Run the server on your own machine, against data you are allowed to read.
- Keep HTTP on the default 127.0.0.1.
- Connect with a MonetDB user that has read-only access to the analysis
  tables, so the database itself refuses writes whatever the SQL check misses.
- If you do not want agent-written code to run at all, start the server with
  **`agent mcp --no-code-tool`**: `awkward_eval` is left out, and agents answer
  with SQL, plots and checks only.
- For a shared deployment (for example next to an analysis facility), run the
  server in an isolated container or VM with no secrets, read-only database
  credentials, and memory and CPU limits. The AST guard alone is not enough.
