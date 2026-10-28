"""Agentic analysis on the hybrid stack: an LLM routes SQL vs Awkward work.

Quick start (MonetDB server with the nanoaod schema loaded)::

    from awkward_monetizer import db
    from awkward_monetizer.agent import AnalysisTools, AnthropicClient, run_agent

    conn = db.open_server(database="hep_agent")
    run = run_agent("Histogram the pT of the trijet closest to the top mass",
                    AnalysisTools(conn), AnthropicClient(), echo=print)
    run.answer          # numpy array of submitted values
    run.save("q6.json") # transcript; replay with ReplayClient("q6.json")
"""

from __future__ import annotations

from .evaluate import TASKS, compare, evaluate, format_table, reference_answers
from .loop import (
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    AgentRun,
    AnthropicClient,
    ModelTurn,
    ReplayClient,
    ScriptedClient,
    run_agent,
)
from .scripted import expert_script
from .tools import TOOL_SPECS, AnalysisTools, ToolError

__all__ = [
    "DEFAULT_MODEL", "SYSTEM_PROMPT", "TASKS", "TOOL_SPECS", "AgentRun",
    "AnalysisTools", "AnthropicClient", "ModelTurn", "ReplayClient",
    "ScriptedClient", "ToolError", "compare", "evaluate", "expert_script",
    "format_table", "load_demo_db", "reference_answers", "run_agent",
]


def load_demo_db(root_file: str, backend: str = "auto", target: str = ":memory:"):
    """Ingest a NanoAOD file into a throwaway database and return the connection.

    ``backend``: "monetdbe" (embedded MonetDB, Python <= 3.10 wheels), "sqlite"
    (stdlib stand-in for quick tries where monetdbe is unavailable) or "auto".
    For the real demo, ingest into a MonetDB server with
    ``awkward-monetizer ingest FILE --dataset nanoaod --database NAME
    --create-schema`` and connect with ``db.open_server``.
    """
    import importlib.util
    import sqlite3

    from .. import db
    from ..datasets import DATASETS
    from ..ingest import build_tables, load_tables, read_root

    if backend == "auto":
        backend = ("monetdbe" if importlib.util.find_spec("monetdbe")
                   else "sqlite")
    if backend == "monetdbe":
        conn = db.open_embedded(target)
    elif backend == "sqlite":
        class _Cursor(sqlite3.Cursor):
            def execute(self, sql, params=()):
                return super().execute(sql.replace("%s", "?"), params)

            def executemany(self, sql, rows):
                return super().executemany(sql.replace("%s", "?"), rows)

        class _Conn(sqlite3.Connection):
            def cursor(self, factory=_Cursor):
                return super().cursor(factory=factory)

        conn = sqlite3.connect(target, factory=_Conn, check_same_thread=False)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    ds = DATASETS["nanoaod"]
    db.create_schema(conn, db.read_schema("nanoaod"))
    events, _ = read_root(root_file, ds, "Events")
    load_tables(conn, build_tables(events, ds), method="insert")
    return conn
