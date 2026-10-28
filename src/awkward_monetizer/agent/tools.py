"""Tools an LLM agent uses to run a HEP analysis on the hybrid stack.

The agent gets four tools, mirroring how the stack is meant to be used:

``describe_schema``
    Discover tables, columns, types and row counts (the database catalog),
    plus short notes on units and keys. Agents query what is there instead of
    guessing from ROOT branch names.
``run_sql``
    A read-only SELECT against MonetDB, returning a small preview. For
    event-level and flat selections, counts and aggregates (where SQL wins).
``awkward_eval``
    Fetch the events passing an optional SQL ``where`` (selection pushed to the
    database), rebuild them as nested Awkward arrays (``events.jets``,
    ``events.muons``) and run a short Python/Awkward snippet on them. For
    combinatorics, ΔR and invariant masses (where Awkward wins). The resulting
    array is stored under a ``result_id``.
``submit_answer``
    Declare the final per-entry values: either a stored ``result_id`` or a SQL
    query whose first column is the answer. The harness re-materialises the
    full answer, so an evaluation can check it against ground truth.

Every call is recorded in :attr:`AnalysisTools.trace` with its engine and
wall time, so a run can be shown as "which engine did what, and how long".

``awkward_eval`` runs agent-written code in-process. The AST checks below block
imports, dunder access and file/exec builtins, which keeps an honest model on
the rails, but it is *not* a security sandbox: use it on your own machine and
data, never as a service exposed to untrusted users.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np

from ..physics import TOP_MASS, Z_MASS, jets_p4, muons_p4
from ..reconstruct import (
    _fetch_binary,
    fetch_events,
    fetch_tables,
    reconstruct_multi,
)

COLLECTIONS = ("jets", "muons")
PREVIEW_ROWS = 50
# Above this many events, awkward_eval, SQL answers and the ADL reference are
# processed in event_id-range chunks, so the full record (tens of millions of
# events, hundreds of millions of jets) never has to be in memory at once.
CHUNK_EVENTS = 1_000_000
FETCH_BATCH = 1_000_000
# Columns jets_p4 / muons_p4 read (muons_p4 carries the charge along).
P4_COLUMNS = {"jets": ("pt", "eta", "phi", "mass"),
              "muons": ("pt", "eta", "phi", "mass", "charge")}

TABLE_NOTES = {
    "events": "one row per collision event. event_id is the primary key and the "
              "join key to every object table. met_pt/met_phi = missing "
              "transverse momentum (GeV, rad). n_jets/n_muons = stored "
              "multiplicities (count cuts need no GROUP BY). hlt_* are trigger "
              "flags, NULL if the file's menu lacks the path.",
    "jets": "one row per jet. (event_id, jet_index) identifies a jet; pt and "
            "mass in GeV, eta dimensionless, phi in rad.",
    "muons": "one row per muon. (event_id, muon_index) identifies a muon; pt "
             "and mass in GeV, charge = +1/-1. Muons are the only leptons.",
}

# --------------------------------------------------------------------------
# Tool specifications (Anthropic tool-use JSON schema)
# --------------------------------------------------------------------------
TOOL_SPECS = [
    {
        "name": "describe_schema",
        "description": (
            "List the tables in the analysis database with their columns, SQL "
            "types, row counts and short notes on units and keys. Call this "
            "first."),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_sql",
        "description": (
            "Run one read-only SQL SELECT (or WITH ... SELECT) on MonetDB and "
            f"get the first rows (at most {PREVIEW_ROWS}) plus the row count. "
            "Best for event-level cuts, counts, aggregates and flat per-object "
            "selections. For pairs or triplets of objects (self-joins), "
            "invariant masses or Delta-R, use awkward_eval instead: it is "
            "faster there. No DDL/DML, no semicolons."),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "a single SELECT"},
                "max_rows": {"type": "integer", "minimum": 1,
                             "maximum": PREVIEW_ROWS},
            },
            "required": ["sql"],
        },
    },
    {
        "name": "awkward_eval",
        "description": (
            "Fetch the events matching an optional SQL WHERE on the events table "
            "(the selection runs in MonetDB), rebuild them as nested Awkward "
            "arrays and run Python code on them. Best for combinatorics "
            "(ak.combinations / ak.cartesian), Delta-R, invariant masses and "
            "anything per-object-pair. In scope: events (fields: all events "
            "columns, plus events.jets[pt,eta,phi,mass,jet_index] and "
            "events.muons[pt,eta,phi,mass,charge,muon_index]), ak, np, "
            "jets_p4(events) and muons_p4(events) (scikit-hep vector Momentum4D "
            "arrays with .mass .pt .deltaR(), addable with +), Z_MASS, TOP_MASS. "
            "The code must assign a 1-D numeric array (one value per histogram "
            "entry, no None) to a variable named `result`. No imports. On a "
            "large database the code runs on consecutive chunks of events and "
            "the results are concatenated, so compute event by event (no "
            "normalising by global totals). Returns a summary and a result_id "
            "for submit_answer."),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string",
                         "description": "Python statements assigning `result`"},
                "where": {"type": "string",
                          "description": "optional SQL boolean expression on "
                                         "the events table, e.g. 'n_jets >= 3'"},
            },
            "required": ["code"],
        },
    },
    {
        "name": "submit_answer",
        "description": (
            "Submit the final answer: the list of per-entry values to histogram. "
            "Give exactly one of result_id (from awkward_eval) or sql (a SELECT "
            "whose first column is the values; it is run without a row limit). "
            "This ends the analysis."),
        "input_schema": {
            "type": "object",
            "properties": {
                "result_id": {"type": "string"},
                "sql": {"type": "string"},
                "explanation": {"type": "string",
                                "description": "one or two sentences on the method"},
            },
            "required": ["explanation"],
        },
    },
    {
        "name": "plot_result",
        "description": (
            "Save a histogram (PNG) of a stored result (result_id from "
            "awkward_eval) or of the first column of a SQL SELECT, and return "
            "the file path. Use it whenever the user asks for a plot."),
        "input_schema": {
            "type": "object",
            "properties": {
                "result_id": {"type": "string"},
                "sql": {"type": "string"},
                "title": {"type": "string"},
                "xlabel": {"type": "string", "description": "axis label with units"},
                "bins": {"type": "integer", "minimum": 5, "maximum": 200},
                "x_range": {"type": "array", "items": {"type": "number"},
                            "minItems": 2, "maxItems": 2},
            },
        },
    },
]

# Scoring is offered only to interactive clients (MCP), where a human asks for
# it after the agent has submitted. The built-in eval scores outside the loop,
# so this tool is deliberately not in TOOL_SPECS.
CHECK_SPEC = {
    "name": "check_answer",
    "description": (
        "Score the most recently submitted answer against the IRIS-HEP ADL "
        "reference for one benchmark query (Q1-Q8). Call it once, after "
        "submit_answer, and only when the user asks for scoring. Every check is "
        "logged with its attempt number, so retries are visible."),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string",
                                 "enum": [f"Q{i}" for i in range(1, 9)]}},
        "required": ["query"],
    },
}


class ToolError(Exception):
    """A tool failed in a way the agent should see and can fix."""


@dataclass
class ToolCall:
    name: str
    input: dict
    engine: str            # catalog | sql | awkward | submit
    ok: bool
    ms: float
    summary: str
    detail: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# SQL guard
# --------------------------------------------------------------------------
_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|create|alter|copy|call|grant|revoke|"
    r"truncate|merge|set|commit|rollback|start|declare|attach|pragma)\b",
    re.IGNORECASE)


def _strip_sql(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", "", sql).strip()
    return sql[:-1].rstrip() if sql.endswith(";") else sql


def check_select(sql: str) -> str:
    """Return a cleaned single read-only SELECT, or raise :class:`ToolError`."""
    s = _strip_sql(sql)
    if not s:
        raise ToolError("empty SQL")
    if ";" in s:
        raise ToolError("only one statement is allowed (remove ';')")
    if not re.match(r"^(select|with)\b", s, re.IGNORECASE):
        raise ToolError("only SELECT / WITH ... SELECT queries are allowed")
    # ignore keywords inside string literals
    bare = re.sub(r"'(?:[^']|'')*'", "''", s)
    m = _FORBIDDEN_SQL.search(bare)
    if m:
        raise ToolError(f"keyword {m.group(1).upper()!r} is not allowed "
                        "(read-only access)")
    return s


def check_where(where: str | None) -> str | None:
    if where is None or not str(where).strip():
        return None
    w = _strip_sql(str(where))
    if re.match(r"^where\b", w, re.IGNORECASE):
        w = w[5:].strip()
    check_select(f"SELECT event_id FROM events WHERE {w}")
    return w


# --------------------------------------------------------------------------
# Awkward code guard
# --------------------------------------------------------------------------
_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "len": len,
    "list": list, "max": max, "min": min, "range": range, "round": round,
    "sorted": sorted, "sum": sum, "tuple": tuple, "zip": zip,
    "True": True, "False": False, "None": None,
}


def check_code(code: str) -> ast.Module:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ToolError(f"syntax error: {exc.msg} (line {exc.lineno})") from None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ToolError("imports are not allowed; ak, np, jets_p4, muons_p4 "
                            "are already in scope")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef,
                             ast.AsyncFunctionDef, ast.Await, ast.With,
                             ast.AsyncWith, ast.Try, ast.Raise, ast.Delete)):
            raise ToolError(f"{type(node).__name__} statements are not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ToolError(f"private attribute {node.attr!r} is not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ToolError(f"name {node.id!r} is not allowed")
    if not any(isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign))
               and any(isinstance(t, ast.Name) and t.id == "result"
                       for t in (n.targets if isinstance(n, ast.Assign)
                                 else [n.target]))
               for n in ast.walk(tree)):
        raise ToolError("the code must assign the answer to `result`")
    return tree


def to_values(x: Any) -> np.ndarray:
    """Coerce a tool result to a 1-D float64 numpy array or raise ToolError."""
    if isinstance(x, ak.Array):
        if x.ndim != 1:
            raise ToolError(f"result is nested (ndim={x.ndim}, type {x.type}); "
                            "flatten it with ak.flatten(..., axis=1) or pick one "
                            "value per event")
        if ak.any(ak.is_none(x, axis=0)):
            raise ToolError("result contains None; drop or fill them "
                            "(e.g. ak.drop_none or ak.fill_none)")
        try:
            x = ak.to_numpy(x)
        except Exception as exc:  # noqa: BLE001 (becomes a ToolError)
            raise ToolError(f"could not convert result to numbers: {exc}") from None
    arr = np.asarray(x)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ToolError(f"result must be 1-D, got shape {arr.shape}")
    if arr.dtype.kind not in "biuf":
        raise ToolError(f"result must be numeric, got dtype {arr.dtype}")
    return arr.astype(np.float64)


def summarize(values: np.ndarray, bins: int = 20) -> dict:
    if values.size == 0:
        return {"n": 0}
    finite = values[np.isfinite(values)]
    out = {
        "n": int(values.size),
        "n_nonfinite": int(values.size - finite.size),
        "first_values": [round(float(v), 4) for v in values[:8]],
    }
    if finite.size:
        q = np.quantile(finite, [0.0, 0.25, 0.5, 0.75, 1.0])
        counts, edges = np.histogram(finite, bins=bins)
        out.update({
            "mean": round(float(finite.mean()), 4),
            "std": round(float(finite.std()), 4),
            "min/q25/median/q75/max": [round(float(v), 4) for v in q],
            "histogram": {"edges": [round(float(e), 3) for e in edges],
                          "counts": counts.tolist()},
        })
    return out


def _jsonable(v):
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, (bytes, bytearray)):
        return v.decode(errors="replace")
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def event_ranges(conn, chunk_events: int = CHUNK_EVENTS
                 ) -> list[tuple[int, int]] | None:
    """Split the events table into ``[lo, hi)`` event_id ranges of roughly
    ``chunk_events`` events, or return None when it fits in one chunk.

    Ranges follow the file part of the key (``event_id >> 40``), where entries
    are contiguous, so chunks are evenly filled; sparse ids just give more,
    smaller chunks. Every event falls in exactly one range.
    """
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM events")
        total = int(cur.fetchall()[0][0])
        if total <= chunk_events:
            return None
        cur.execute("SELECT f, MIN(event_id), MAX(event_id), COUNT(*) FROM "
                    "(SELECT event_id >> 40 AS f, event_id FROM events) AS t "
                    "GROUP BY f ORDER BY f")
        groups = cur.fetchall()
    finally:
        cur.close()
    out = []
    for _, lo, hi, n in groups:
        lo, hi, n = int(lo), int(hi) + 1, int(n)
        step = max(1, -(-chunk_events * (hi - lo) // n))   # ceil
        out.extend((a, min(a + step, hi)) for a in range(lo, hi, step))
    return out


def _fetch_values(cur, batch: int = FETCH_BATCH) -> np.ndarray:
    """First column of an executed query as float64, read in batches."""
    parts = []
    while True:
        rows = cur.fetchmany(batch)
        if not rows:
            break
        col = [r[0] for r in rows]
        if any(v is None for v in col):
            raise ToolError("first column contains NULL; filter them out")
        parts.append(np.asarray(col, dtype=np.float64))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)


def _names_used(tree: ast.Module) -> set[str]:
    """Attribute names, variable names and string constants in the code: a
    superset of the columns it can read, used to project what is fetched."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    return names


# --------------------------------------------------------------------------
# The tool set
# --------------------------------------------------------------------------
class AnalysisTools:
    """Agent-facing tools over one DB-API connection with the nanoaod schema."""

    def __init__(self, conn, *, tables: tuple[str, ...] = ("events",) + COLLECTIONS,
                 plot_dir: str = "output/agent_plots",
                 check_log: str | None = None,
                 chunk_events: int = CHUNK_EVENTS,
                 ref_cache: str | None = None):
        self.conn = conn
        self.chunk_events = int(chunk_events)
        self.ref_cache = ref_cache
        self._ranges: list | None = None
        self._ranges_known = False
        self._colcache: dict[str, list[str]] = {}
        self.tables = tables
        self.plot_dir = Path(plot_dir).expanduser().resolve()
        self.check_log = Path(check_log).expanduser().resolve() if check_log else None
        # Earlier checks from the same log count towards the attempt number,
        # so restarting the MCP server cannot reset a query's attempts.
        self.checks: list[dict] = []
        if self.check_log is not None and self.check_log.exists():
            for line in self.check_log.read_text().splitlines():
                try:
                    self.checks.append(json.loads(line))
                except ValueError:
                    continue
        self._refs: dict[str, np.ndarray] = {}
        self._n_plots = 0
        # The tools only read, so run in autocommit: a failed agent query then
        # cannot leave the connection in an aborted transaction (monetdbe does
        # not recover from ROLLBACK cleanly).
        self._autocommit = False
        if hasattr(conn, "set_autocommit"):
            try:
                conn.set_autocommit(True)
                self._autocommit = True
            except Exception:  # noqa: BLE001, S110 (best effort)
                pass
        self.results: dict[str, np.ndarray] = {}
        self.trace: list[ToolCall] = []
        self.answer: np.ndarray | None = None
        self.answer_info: dict | None = None
        self._events_cache: dict[str | None, ak.Array] = {}

    # -- dispatch ---------------------------------------------------------
    def call(self, name: str, tool_input: dict | None) -> tuple[str, bool]:
        """Run a tool by name. Returns (text for the model, is_error)."""
        tool_input = dict(tool_input or {})
        handler = {
            "describe_schema": (self.describe_schema, "catalog"),
            "run_sql": (self.run_sql, "sql"),
            "awkward_eval": (self.awkward_eval, "awkward"),
            "submit_answer": (self.submit_answer, "submit"),
            "plot_result": (self.plot_result, "plot"),
            "check_answer": (self.check_answer, "check"),
        }.get(name)
        if handler is None:
            text, ok, detail = f"unknown tool {name!r}", False, {}
            engine, ms = "?", 0.0
        else:
            fn, engine = handler
            t0 = time.perf_counter()
            try:
                try:
                    inspect.signature(fn).bind(**tool_input)
                except TypeError as exc:  # bad / missing arguments
                    raise ToolError(f"bad arguments: {exc}") from None
                detail = fn(**tool_input)
                ok, text = True, ""
            except ToolError as exc:
                ok, detail, text = False, {}, f"error: {exc}"
            except Exception as exc:  # noqa: BLE001 (SQL/Python errors go to the model)
                ok, detail = False, {}
                text = f"error: {type(exc).__name__}: {exc}"
                self._rollback()
            ms = (time.perf_counter() - t0) * 1e3
        summary = detail.get("_summary") if ok else text
        detail.pop("_summary", None)
        if ok:
            text = json.dumps(detail, default=_jsonable)
        self.trace.append(ToolCall(name, tool_input, engine, ok, round(ms, 2),
                                   (summary or "")[:200], detail))
        return text, not ok

    def _rollback(self):
        if self._autocommit:
            return
        try:
            self.conn.rollback()
        except Exception:  # noqa: BLE001, S110 (best effort)
            pass

    # -- tools ------------------------------------------------------------
    def describe_schema(self) -> dict:
        cur = self.conn.cursor()
        tables = {}
        try:
            for t in self.tables:
                cols = self._columns(cur, t)
                cur.execute(f"SELECT COUNT(*) FROM {t}")
                n = int(cur.fetchall()[0][0])
                tables[t] = {"rows": n, "columns": cols,
                             "notes": TABLE_NOTES.get(t, "")}
        finally:
            cur.close()
        return {"tables": tables,
                "_summary": ", ".join(f"{t}({v['rows']})" for t, v in tables.items())}

    def _columns(self, cur, table: str) -> dict:
        try:     # MonetDB catalog
            cur.execute(
                "SELECT c.name, c.type FROM sys.columns c JOIN sys.tables t "
                f"ON c.table_id = t.id WHERE t.name = '{table}' "
                "AND t.system = false ORDER BY c.number")
            rows = cur.fetchall()
            if rows:
                return {str(n): str(ty) for n, ty in rows}
        except Exception:  # noqa: BLE001 (try the next catalog)
            self._rollback()
        try:     # SQLite (tests, quick demos)
            cur.execute(f"PRAGMA table_info({table})")
            rows = cur.fetchall()
            if rows:
                return {str(r[1]): str(r[2]) for r in rows}
        except Exception:  # noqa: BLE001 (try the next catalog)
            self._rollback()
        cur.execute(f"SELECT * FROM {table} WHERE 1 = 0")
        return {d[0]: "?" for d in cur.description}

    def run_sql(self, sql: str, max_rows: int = 20) -> dict:
        s = check_select(sql)
        max_rows = max(1, min(int(max_rows), PREVIEW_ROWS))
        cur = self.conn.cursor()
        try:
            cur.execute(s)
            cols = [d[0] for d in cur.description]
            n = getattr(cur, "rowcount", -1)
            if n is not None and n >= 0:     # MonetDB knows the count up front:
                rows = cur.fetchmany(max_rows)  # do not transfer the rest
            else:                            # e.g. SQLite
                rows = cur.fetchall()
                n = len(rows)
        finally:
            cur.close()
        preview = [[_jsonable(v) for v in r] for r in rows[:max_rows]]
        return {"columns": cols, "row_count": n, "rows": preview,
                "truncated": n > len(preview),
                "_summary": f"{n} rows x {len(cols)} cols"}

    def ranges(self) -> list[tuple[int, int]] | None:
        """event_id chunks for this database (None: small enough for one)."""
        if not self._ranges_known:
            self._ranges = event_ranges(self.conn, self.chunk_events)
            self._ranges_known = True
        return self._ranges

    def _table_columns(self, table: str) -> list[str]:
        if table not in self._colcache:
            cur = self.conn.cursor()
            try:
                self._colcache[table] = list(self._columns(cur, table))
            finally:
                cur.close()
        return self._colcache[table]

    def _projection(self, tree: ast.Module) -> dict:
        """Columns to fetch per table for this code (None = all columns)."""
        used = _names_used(tree)
        ev_cols = [c for c in self._table_columns("events")
                   if c == "event_id" or c in used]
        out = {"events": ev_cols}
        for coll, p4 in (("jets", "jets_p4"), ("muons", "muons_p4")):
            if coll not in used and p4 not in used:
                continue          # collection never touched: do not fetch it
            cols = self._table_columns(coll)
            want = set(used) | (set(P4_COLUMNS[coll]) if p4 in used else set())
            out[coll] = [c for c in cols if c == "event_id"
                         or c.endswith("_index") or c in want]
        return out

    def _chunk(self, rng, where, proj) -> ak.Array:
        """One event_id range as nested Awkward arrays, fetched straight into
        NumPy (no pandas); ``proj`` None means every column."""
        if proj is None:
            return fetch_events(self.conn, where, event_range=rng,
                                collections={c: None for c in COLLECTIONS})
        return fetch_events(self.conn, where, event_columns=proj["events"],
                            collections={c: proj[c] for c in COLLECTIONS
                                         if c in proj},
                            event_range=rng)

    def _events(self, where: str | None) -> ak.Array:
        if where not in self._events_cache:
            ev_df, jets = fetch_tables(self.conn, where=where, collection="jets")
            _, muons = fetch_tables(self.conn, where=where, collection="muons")
            self._events_cache[where] = reconstruct_multi(
                ev_df, {"jets": jets, "muons": muons})
        return self._events_cache[where]

    def _run_code(self, tree: ast.Module, events: ak.Array) -> np.ndarray:
        scope = {"__builtins__": _SAFE_BUILTINS, "events": events, "ak": ak,
                 "np": np, "jets_p4": jets_p4, "muons_p4": muons_p4,
                 "Z_MASS": Z_MASS, "TOP_MASS": TOP_MASS}
        exec(compile(tree, "<agent>", "exec"), scope)   # noqa: S102 (guarded above)
        if "result" not in scope:
            raise ToolError("the code did not set `result`")
        return to_values(scope["result"])

    def awkward_eval(self, code: str, where: str | None = None) -> dict:
        tree = check_code(code)
        w = check_where(where)
        ranges = self.ranges()
        t_fetch = t_compute = 0.0
        n_events = 0
        if ranges is None:                     # small database: one pass, cached
            t0 = time.perf_counter()
            events = self._events(w)
            t_fetch = (time.perf_counter() - t0) * 1e3
            t1 = time.perf_counter()
            values = self._run_code(tree, events)
            t_compute = (time.perf_counter() - t1) * 1e3
            n_events = len(events)
        else:                                  # large: chunk by event_id range
            proj = self._projection(tree)
            parts = []
            for i, rng in enumerate(ranges):
                t0 = time.perf_counter()
                events = self._chunk(rng, w, proj)
                t_fetch += (time.perf_counter() - t0) * 1e3
                if len(events) == 0:
                    continue
                t1 = time.perf_counter()
                try:
                    parts.append(self._run_code(tree, events))
                except Exception:
                    if i or proj is None:
                        raise
                    # The projection may have missed a column the code reads
                    # indirectly: retry this chunk, and the rest, unprojected.
                    proj = None
                    events = self._chunk(rng, w, proj)
                    parts.append(self._run_code(tree, events))
                t_compute += (time.perf_counter() - t1) * 1e3
                n_events += len(events)
            values = (np.concatenate(parts) if parts
                      else np.empty(0, dtype=np.float64))
        rid = f"r{len(self.results) + 1}"
        self.results[rid] = values
        out = {"result_id": rid, "events_fetched": n_events,
               "where": w, "fetch_ms": round(t_fetch, 1),
               "compute_ms": round(t_compute, 1), "summary": summarize(values),
               "_summary": f"{rid}: n={values.size} from {n_events} events"}
        if ranges is not None:
            out["chunks"] = len(ranges)
        return out

    def sql_values(self, sql: str) -> np.ndarray:
        s = check_select(sql)
        cur = self.conn.cursor()
        try:
            cur.execute(s)
            if not cur.description:
                raise ToolError("query returned no columns")
            try:      # MonetDB binary export straight into NumPy, if possible
                fast = _fetch_binary(cur, None)
            except Exception:  # noqa: BLE001 (fall back to the portable path)
                fast = None
            if fast is not None:
                values = next(iter(fast.values()))
                if not isinstance(values, np.ndarray):   # masked: has NULLs
                    raise ToolError("first column contains NULL; filter them out")
            else:
                values = _fetch_values(cur)
        finally:
            cur.close()
        return to_values(values)

    def submit_answer(self, explanation: str, result_id: str | None = None,
                      sql: str | None = None) -> dict:
        values, route = self._values(result_id, sql)
        self.answer = values
        self.answer_info = {"route": route, "result_id": result_id, "sql": sql,
                            "explanation": explanation}
        return {"accepted": True, "route": route, "summary": summarize(values),
                "_summary": f"answer via {route}: n={values.size}"}

    def _values(self, result_id: str | None, sql: str | None):
        if (result_id is None) == (sql is None):
            raise ToolError("give exactly one of result_id or sql")
        if result_id is not None:
            if result_id not in self.results:
                raise ToolError(f"unknown result_id {result_id!r}; known: "
                                f"{sorted(self.results)}")
            return self.results[result_id], "awkward"
        return self.sql_values(sql), "sql"

    def plot_result(self, result_id: str | None = None, sql: str | None = None,
                    title: str = "", xlabel: str = "", bins: int = 50,
                    x_range: list | None = None) -> dict:
        values, route = self._values(result_id, sql)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise ToolError("matplotlib is not installed") from None
        bins = max(5, min(int(bins), 200))
        rng = None
        if x_range is not None:
            if len(x_range) != 2 or not x_range[0] < x_range[1]:
                raise ToolError("x_range must be [low, high] with low < high")
            rng = (float(x_range[0]), float(x_range[1]))
        finite = values[np.isfinite(values)]
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        try:
            ax.hist(finite, bins=bins, range=rng, histtype="stepfilled",
                    color="#3b6fb6", alpha=0.85, edgecolor="#26456e")
            ax.set_xlabel(xlabel or "value")
            ax.set_ylabel("entries")
            ax.set_title(title or "awkward-monetizer agent", fontsize=10)
            ax.text(0.98, 0.95, f"n = {finite.size}\nmean = {finite.mean():.2f}"
                    if finite.size else "n = 0", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9)
            fig.tight_layout()
            self.plot_dir.mkdir(parents=True, exist_ok=True)
            self._n_plots += 1
            slug = re.sub(r"[^a-z0-9]+", "_", (title or "plot").lower()).strip("_")
            stamp = time.strftime("%H%M%S")
            path = self.plot_dir / f"{slug[:40] or 'plot'}_{stamp}_{self._n_plots}.png"
            fig.savefig(path, dpi=120)
        finally:
            plt.close(fig)
        return {"path": str(path), "n": int(values.size), "route": route,
                "_summary": f"saved {path}"}

    def check_answer(self, query: str) -> dict:
        from .evaluate import TASKS, compare, reference_answers
        q = str(query).strip().upper()
        if q not in TASKS:
            raise ToolError(f"unknown query {query!r}; one of {list(TASKS)}")
        if self.answer is None:
            raise ToolError("no answer submitted yet; call submit_answer first")
        if q not in self._refs:
            # all eight in one pass: on a large database each pass reads
            # every event, so do it once (and cache it on disk if asked)
            self._refs.update(reference_answers(
                self.conn, chunk_events=self.chunk_events, cache=self.ref_cache))
        passed, reason = compare(self.answer, self._refs[q])
        attempt = 1 + sum(c["query"] == q for c in self.checks)
        entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "query": q, "passed": passed, "reason": reason,
                 "attempt": attempt, **(self.answer_info or {})}
        self.checks.append(entry)
        if self.check_log is not None:
            self.check_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.check_log, "a") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        return {"query": q, "passed": passed, "reason": reason, "attempt": attempt,
                "_summary": f"{q} {'PASS' if passed else 'FAIL'} ({reason}), "
                            f"attempt {attempt}"}

    # -- reporting --------------------------------------------------------
    def trace_dicts(self) -> list[dict]:
        out = []
        for c in self.trace:
            d = asdict(c)
            d["detail"] = {k: v for k, v in d["detail"].items()
                           if k in ("row_count", "result_id", "events_fetched",
                                    "fetch_ms", "compute_ms", "where", "route",
                                    "path", "passed", "attempt")}
            out.append(d)
        return out
