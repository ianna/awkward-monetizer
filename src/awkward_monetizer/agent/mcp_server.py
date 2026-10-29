"""Expose the analysis tools as an MCP server, over stdio or HTTP.

Lets any MCP client — Claude Code, Claude Desktop, or another agent framework —
drive the same tools the built-in loop uses::

    awkward-monetizer agent mcp --database hep_agent              # stdio
    awkward-monetizer agent mcp --database hep_agent --transport http

stdio (the default): the client starts the server as a subprocess, e.g.
``claude mcp add awkward-monetizer -- awkward-monetizer agent mcp --database
hep_agent``. HTTP (MCP streamable HTTP): the server runs on its own and clients
connect to ``http://HOST:PORT/mcp``, e.g. ``claude mcp add --transport http
awkward-monetizer http://127.0.0.1:8765/mcp``. Needs ``pip install mcp`` (1.x or
2.x).

HTTP listens on 127.0.0.1 by default. ``awkward_eval`` runs agent-written
Python behind an AST guard, which is *not* a sandbox: anyone who can reach the
port can run code as this user. Bind to another address only on a network you
trust, and do not expose the server publicly until the Python tool is properly
sandboxed. All clients of one HTTP server share its tool state (results, the
submitted answer) and its database connection; calls are serialized.

Besides the loop's tools, the server offers ``check_answer``: after the agent
submits, ask it to score the answer against the ADL reference (Q1–Q8). Each
check is appended to a JSON-lines log (``--check-log``), so a session's
results, including retries, can be tabulated afterwards with
``awkward-monetizer agent checks``.
"""

from __future__ import annotations

import ipaddress
import sys
import threading

from .loop import SYSTEM_PROMPT
from .tools import CHECK_SPEC, CHUNK_EVENTS, TOOL_SPECS, AnalysisTools

# MCP clients never see the built-in loop's system prompt, only these server
# instructions and the tool descriptions. Without them, an agent driving the
# tools over MCP has no routing rule and may do combinatorics in SQL.
MCP_INSTRUCTIONS = SYSTEM_PROMPT + """

check_answer (ADL Q1-Q8 scoring) is for when the user asks for it, after \
submit_answer; every call is logged with its attempt number."""


def build_server(tools: AnalysisTools, instructions: str = MCP_INSTRUCTIONS):
    try:                                   # mcp >= 2
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:                    # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server

    mcp = Server("awkward-monetizer", instructions=instructions)
    desc = {s["name"]: s["description"] for s in [*TOOL_SPECS, CHECK_SPEC]}

    lock = threading.Lock()     # one DB connection: one tool call at a time

    def call(name, **kwargs):
        with lock:
            return tools.call(name, {k: v for k, v in kwargs.items()
                                     if v is not None})[0]

    @mcp.tool(description=desc["describe_schema"])
    def describe_schema() -> str:
        return call("describe_schema")

    @mcp.tool(description=desc["run_sql"])
    def run_sql(sql: str, max_rows: int = 20) -> str:
        return call("run_sql", sql=sql, max_rows=max_rows)

    @mcp.tool(description=desc["awkward_eval"])
    def awkward_eval(code: str, where: str | None = None) -> str:
        return call("awkward_eval", code=code, where=where or None)

    @mcp.tool(description=desc["submit_answer"])
    def submit_answer(explanation: str, result_id: str | None = None,
                      sql: str | None = None) -> str:
        return call("submit_answer", explanation=explanation,
                    result_id=result_id or None, sql=sql or None)

    @mcp.tool(description=desc["plot_result"])
    def plot_result(result_id: str | None = None, sql: str | None = None,
                    title: str = "", xlabel: str = "", bins: int = 50,
                    x_range: list[float] | None = None) -> str:
        return call("plot_result", result_id=result_id or None, sql=sql or None,
                    title=title, xlabel=xlabel, bins=bins, x_range=x_range)

    @mcp.tool(description=desc["check_answer"])
    def check_answer(query: str) -> str:
        return call("check_answer", query=query)

    return mcp


TRANSPORTS = ("stdio", "http")
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8765


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def run_server(server, transport: str = "stdio", *,
               host: str = DEFAULT_HTTP_HOST, port: int = DEFAULT_HTTP_PORT) -> None:
    """Run a built server over stdio or MCP streamable HTTP (mcp 1.x or 2.x)."""
    if transport not in TRANSPORTS:
        raise ValueError(f"transport must be one of {TRANSPORTS}, not {transport!r}")
    if transport == "stdio":
        server.run()
        return
    if not is_loopback(host):
        print(f"WARNING: serving on {host}, not only this computer. awkward_eval "
              "runs agent-written Python with an AST guard, not a sandbox: anyone "
              "who can reach this port can run code as this user.", file=sys.stderr)
    print(f"awkward-monetizer MCP server: http://{host}:{port}/mcp",
          file=sys.stderr, flush=True)
    settings = getattr(server, "settings", None)
    try:
        if settings is not None and hasattr(settings, "port"):   # mcp 1.x
            settings.host, settings.port = host, port
            server.run(transport="streamable-http")
        else:                                                     # mcp >= 2
            server.run("streamable-http", host=host, port=port)
    except KeyboardInterrupt:    # Ctrl-C is how an HTTP server is stopped
        print("awkward-monetizer MCP server stopped", file=sys.stderr)


def serve(conn, *, plot_dir: str = "output/agent_plots",
          check_log: str | None = "output/agent_checks.jsonl",
          chunk_events: int = CHUNK_EVENTS, ref_cache: str | None = None,
          transport: str = "stdio", host: str = DEFAULT_HTTP_HOST,
          port: int = DEFAULT_HTTP_PORT) -> None:
    tools = AnalysisTools(conn, plot_dir=plot_dir, check_log=check_log,
                          chunk_events=chunk_events, ref_cache=ref_cache)
    warn_reference_cost(tools)
    run_server(build_server(tools), transport, host=host, port=port)


def warn_reference_cost(tools: AnalysisTools) -> None:
    """On a database too big for one pass, tell the user up front whether the
    first check_answer will read every event to build the ADL reference."""
    if tools.ranges() is None:          # small: the reference takes seconds
        return
    import os
    cache = tools.ref_cache
    if cache and os.path.exists(os.path.expanduser(cache)):
        print(f"ADL reference answers: {cache} (reused if it matches this "
              "database, rebuilt otherwise)", file=sys.stderr)
    elif cache:
        print(f"ADL reference answers: none yet; the first check_answer reads "
              f"every event to build them (can take tens of minutes) and saves "
              f"them to {cache}", file=sys.stderr)
    else:
        print("WARNING: no --ref-cache. On this database every server start's "
              "first check_answer reads every event to build the ADL reference "
              "answers (can take tens of minutes). Pass --ref-cache FILE.npz to "
              "build them once and reuse them.", file=sys.stderr)
