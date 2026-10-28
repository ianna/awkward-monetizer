"""Expose the analysis tools as an MCP server (stdio).

Lets any MCP client — Claude Desktop, Claude Code, or another agent framework —
drive the same tools the built-in loop uses::

    awkward-monetizer agent mcp --database hep_agent

Claude Code: ``claude mcp add awkward-monetizer -- awkward-monetizer agent mcp
--database hep_agent``. Needs ``pip install mcp``.
"""

from __future__ import annotations

from .tools import AnalysisTools


def build_server(tools: AnalysisTools):
    try:                                   # mcp >= 2
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:                    # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server

    mcp = Server("awkward-monetizer")
    specs = {}
    from .tools import TOOL_SPECS
    for s in TOOL_SPECS:
        specs[s["name"]] = s["description"]

    @mcp.tool(description=specs["describe_schema"])
    def describe_schema() -> str:
        return tools.call("describe_schema", {})[0]

    @mcp.tool(description=specs["run_sql"])
    def run_sql(sql: str, max_rows: int = 20) -> str:
        return tools.call("run_sql", {"sql": sql, "max_rows": max_rows})[0]

    @mcp.tool(description=specs["awkward_eval"])
    def awkward_eval(code: str, where: str | None = None) -> str:
        inp = {"code": code}
        if where:
            inp["where"] = where
        return tools.call("awkward_eval", inp)[0]

    @mcp.tool(description=specs["submit_answer"])
    def submit_answer(explanation: str, result_id: str | None = None,
                      sql: str | None = None) -> str:
        inp = {"explanation": explanation}
        if result_id:
            inp["result_id"] = result_id
        if sql:
            inp["sql"] = sql
        return tools.call("submit_answer", inp)[0]

    return mcp


def serve(conn) -> None:
    build_server(AnalysisTools(conn)).run()
