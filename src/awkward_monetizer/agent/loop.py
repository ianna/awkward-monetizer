"""The agent loop: a model plans, calls tools, reads results, and submits.

::

    question ──► model ──tool_use──► AnalysisTools (SQL │ Awkward) ──result──┐
                   ▲                                                         │
                   └──────────────────── tool_result ◄───────────────────────┘
                 ... until submit_answer (or the turn budget runs out)

The model client is pluggable:

* :class:`AnthropicClient` — Claude via the Messages API (needs
  ``ANTHROPIC_API_KEY`` and ``pip install anthropic``).
* :class:`ReplayClient` — replays the model turns of a saved transcript while
  the tools run live against the database. Use it for offline demos and
  reproducibility: same plan, fresh numbers.
* :class:`ScriptedClient` — fixed turns, for tests.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .tools import TOOL_SPECS, AnalysisTools

DEFAULT_MODEL = os.environ.get("AWKWARD_MONETIZER_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """\
You are a high-energy-physics analysis agent working on CMS NanoAOD-style \
collision data stored in a hybrid engine: a MonetDB columnar database (flat \
tables) and Awkward Array (nested per-event lists rebuilt from those tables).

How to work:
1. Call describe_schema first. Do not assume columns that are not listed.
2. Route each step to the engine that suits it (measured on this stack):
   - SQL (run_sql) is fastest for event-level cuts, counts, aggregates and flat \
per-object selections; an answer that is a plain column after a cut can be \
submitted directly as SQL.
   - Awkward (awkward_eval) is fastest for combinatorics: pairs/triplets, \
Delta-R matching, invariant masses, "best candidate per event". Push any \
event-level cut into its `where` argument so less data leaves the database.
3. Check intermediate numbers (counts, a few values) before submitting. If a \
tool returns an error, read it and fix the call.
4. Finish by calling submit_answer exactly once with one value per histogram \
entry, and a one-sentence explanation. Energies and momenta are in GeV.

Be concise in text; let the tool calls do the work."""


# --------------------------------------------------------------------------
# Model clients
# --------------------------------------------------------------------------
@dataclass
class ModelTurn:
    """One assistant message, normalised: content blocks as plain dicts."""
    content: list[dict]
    stop_reason: str
    usage: dict = field(default_factory=dict)

    def tool_uses(self) -> list[dict]:
        return [b for b in self.content if b.get("type") == "tool_use"]

    def text(self) -> str:
        return "\n".join(b["text"] for b in self.content if b.get("type") == "text")


class ModelClient(Protocol):
    def create(self, *, system: str, tools: list[dict],
               messages: list[dict]) -> ModelTurn: ...


class AnthropicClient:
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 4096,
                 client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def create(self, *, system, tools, messages) -> ModelTurn:
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                        system=system, tools=tools,
                                        messages=messages)
        blocks = []
        for b in r.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name,
                               "input": dict(b.input)})
        usage = {"input_tokens": getattr(r.usage, "input_tokens", 0),
                 "output_tokens": getattr(r.usage, "output_tokens", 0)}
        return ModelTurn(blocks, r.stop_reason or "end_turn", usage)


class ScriptedClient:
    """Returns pre-written turns in order (tests, deterministic demos)."""

    def __init__(self, turns: list[ModelTurn | dict]):
        self.turns = [t if isinstance(t, ModelTurn) else ModelTurn(**t)
                      for t in turns]
        self.i = 0
        self.seen_messages: list[list[dict]] = []

    def create(self, *, system, tools, messages) -> ModelTurn:
        self.seen_messages.append(json.loads(json.dumps(messages)))
        if self.i >= len(self.turns):
            return ModelTurn([{"type": "text", "text": "(script exhausted)"}],
                             "end_turn")
        t = self.turns[self.i]
        self.i += 1
        return t


class ReplayClient(ScriptedClient):
    """Replays the model turns stored in a transcript written by
    :meth:`AgentRun.save`."""

    def __init__(self, path: str):
        with open(path) as fh:
            data = json.load(fh)
        super().__init__(data["model_turns"])
        self.question = data.get("question")


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------
@dataclass
class AgentRun:
    question: str
    model: str
    answer: np.ndarray | None
    answer_info: dict | None
    final_text: str
    turns: int
    model_turns: list[dict]
    trace: list[dict]
    usage: dict
    wall_s: float
    stopped: str            # submitted | end_turn | max_turns

    def engine_summary(self) -> dict:
        out: dict[str, dict] = {}
        for c in self.trace:
            e = out.setdefault(c["engine"], {"calls": 0, "errors": 0, "ms": 0.0})
            e["calls"] += 1
            e["errors"] += 0 if c["ok"] else 1
            e["ms"] = round(e["ms"] + c["ms"], 1)
        return out

    def to_dict(self) -> dict:
        return {
            "question": self.question, "model": self.model,
            "stopped": self.stopped, "turns": self.turns, "wall_s": self.wall_s,
            "usage": self.usage, "answer_info": self.answer_info,
            "answer_n": None if self.answer is None else int(self.answer.size),
            "engines": self.engine_summary(), "trace": self.trace,
            "final_text": self.final_text, "model_turns": self.model_turns,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=1, default=str)


def _preview(text: str, n: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def run_agent(question: str, tools: AnalysisTools, client: ModelClient, *,
              max_turns: int = 16, system: str = SYSTEM_PROMPT,
              model_name: str | None = None, echo=None) -> AgentRun:
    """Run the tool-use loop until the agent submits an answer.

    ``echo``: optional ``print``-like callable for a live narration of the run.
    """
    say = echo or (lambda *a, **k: None)
    messages: list[dict] = [{"role": "user", "content": question}]
    model_turns: list[dict] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    final_text, stopped, turns = "", "max_turns", 0
    t0 = time.perf_counter()

    for turn_no in range(1, max_turns + 1):
        turns = turn_no
        turn = client.create(system=system, tools=TOOL_SPECS,
                             messages=list(messages))
        model_turns.append({"content": turn.content, "stop_reason": turn.stop_reason,
                            "usage": turn.usage})
        for k in usage:
            usage[k] += int(turn.usage.get(k, 0) or 0)
        messages.append({"role": "assistant", "content": turn.content})
        if turn.text().strip():
            final_text = turn.text()
            say(f"  [model] {_preview(turn.text())}")

        uses = turn.tool_uses()
        if not uses:
            stopped = "end_turn"
            break

        results, submitted = [], False
        for u in uses:
            text, is_error = tools.call(u["name"], u.get("input"))
            c = tools.trace[-1]
            arg = u.get("input", {})
            shown = arg.get("sql") or arg.get("code") or arg.get("result_id") or ""
            where = f" where={arg['where']!r}" if arg.get("where") else ""
            say(f"  [{c.engine:7s}] {u['name']}{where} {_preview(shown, 90)}")
            say(f"            -> {'ERROR ' if is_error else ''}{_preview(c.summary)}"
                f"  ({c.ms:.0f} ms)")
            block = {"type": "tool_result", "tool_use_id": u["id"], "content": text}
            if is_error:
                block["is_error"] = True
            results.append(block)
            submitted |= (u["name"] == "submit_answer" and not is_error)
        messages.append({"role": "user", "content": results})
        if submitted:
            stopped = "submitted"
            break

    return AgentRun(
        question=question,
        model=model_name or getattr(client, "model", type(client).__name__),
        answer=tools.answer, answer_info=tools.answer_info, final_text=final_text,
        turns=turns, model_turns=model_turns, trace=tools.trace_dicts(),
        usage=usage, wall_s=round(time.perf_counter() - t0, 2), stopped=stopped)
