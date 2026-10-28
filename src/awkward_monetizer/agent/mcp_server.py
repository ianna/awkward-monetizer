"""Expose the analysis tools as an MCP server (stdio).

Lets any MCP client — Claude Code, Claude Desktop, or another agent framework —
drive the same tools the built-in loop uses::

    awkward-monetizer agent mcp --database hep_agent

Claude Code: ``claude mcp add awkward-monetizer -- awkward-monetizer agent mcp
--database hep_agent``. Needs ``pip install mcp`` (1.x or 2.x).

Besides the loop's tools, the server offers ``check_answer``: after the agent
submits, ask it to score the answer against the ADL reference (Q1–Q8). Each
check is appended to a JSON-lines log (``--check-log``), so a session's
results, including retries, can be tabulated afterwards with
``awkward-monetizer agent checks``.
"""

from __future__ import annotations

from .tools import CHECK_SPEC, TOOL_SPECS, AnalysisTools


def build_server(tools: AnalysisTools):
    try:                                   # mcp >= 2
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:                    # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server

    mcp = Server("awkward-monetizer")
    desc = {s["name"]: s["description"] for s in [*TOOL_SPECS, CHECK_SPEC]}

    def call(name, **kwargs):
        return tools.call(name, {k: v for k, v in kwargs.items() if v is not None})[0]

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


def serve(conn, *, plot_dir: str = "output/agent_plots",
          check_log: str | None = "output/agent_checks.jsonl") -> None:
    build_server(AnalysisTools(conn, plot_dir=plot_dir, check_log=check_log)).run()
