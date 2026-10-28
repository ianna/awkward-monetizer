"""Evaluate the agent on the IRIS-HEP ADL benchmark (Q1–Q8).

Each task is the plain-English ADL question (with this dataset's scope notes:
muons are the only leptons, no b-tag). The reference answer is computed by the
hand-written :mod:`awkward_monetizer.adl` functions on the same database
contents, and the agent's submitted values must match it as a multiset
(sorted, ``allclose``). A pass therefore means the agent got the physics right,
not just produced plausible numbers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .. import adl
from ..reconstruct import fetch_tables, reconstruct_multi
from .loop import ModelClient, run_agent
from .tools import AnalysisTools

TASKS: dict[str, tuple[str, str]] = {
    "Q1": ("Q1: MET (all events)",
           ("Histogram the missing transverse energy (MET pT) of all events. "
           "Submit one value per event.")),
    "Q2": ("Q2: jet pT (all jets)",
           ("Histogram the transverse momentum of all jets in all events. Submit "
           "one value per jet.")),
    "Q3": ("Q3: jet pT, |eta|<1",
           "Histogram the pT of jets with |eta| < 1. Submit one value per jet."),
    "Q4": ("Q4: MET, >=2 jets pT>40",
           ("Histogram the MET of events that have at least two jets with "
           "pT > 40 GeV. Submit one value per selected event.")),
    "Q5": ("Q5: MET, OS dimuon 60-120",
           ("Histogram the MET of events that have an opposite-charge muon pair "
           "with an invariant mass between 60 and 120 GeV (any pair in the "
           "event may satisfy it). Submit one value per selected event.")),
    "Q6": ("Q6: trijet pT (m~172.5)",
           ("For events with at least three jets, find the trijet (three-jet "
           "combination) whose invariant mass is closest to 172.5 GeV and "
           "histogram the pT of that trijet system. Submit one value per "
           "selected event. (There is no b-tag information in this dataset.)")),
    "Q7": ("Q7: HT of cleaned jets",
           ("Histogram, for every event, the scalar sum of the pT of jets with "
           "pT > 30 GeV and |eta| < 2.4 that are not within Delta-R < 0.4 of "
           "any muon with pT > 10 GeV. Events with no such jet contribute 0. "
           "Submit one value per event (all events).")),
    "Q8": ("Q8: MT(MET, lead lepton)",
           ("In events with at least three muons and at least one opposite-"
           "charge muon pair, pick the opposite-charge pair whose invariant "
           "mass is closest to the Z mass (91.1876 GeV). Then compute the transverse mass of "
           "MET and the highest-pT muon not in that pair: "
           "MT = sqrt(2 pT MET (1 - cos(dphi))). Submit one value per selected "
           "event.")),
}


def reference_answers(conn, queries=None) -> dict[str, np.ndarray]:
    """Ground truth from the hand-written ADL functions on the DB contents."""
    ev_df, jets = fetch_tables(conn, collection="jets")
    _, muons = fetch_tables(conn, collection="muons")
    events = reconstruct_multi(ev_df, {"jets": jets, "muons": muons})
    out = {}
    for q in queries or TASKS:
        fn = adl.QUERIES[TASKS[q][0]]
        out[q] = np.asarray(fn(events), dtype=np.float64)
    return out


def compare(got: np.ndarray | None, ref: np.ndarray, *, rtol=1e-6,
            atol=1e-6) -> tuple[bool, str]:
    if got is None:
        return False, "no answer submitted"
    if got.size != ref.size:
        return False, f"n={got.size}, expected {ref.size}"
    a, b = np.sort(got), np.sort(ref)
    if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
        bad = int(np.sum(~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)))
        return False, f"{bad}/{ref.size} values differ"
    return True, f"n={ref.size} match"


@dataclass
class EvalRow:
    query: str
    passed: bool
    reason: str
    route: str
    turns: int
    sql_calls: int
    awkward_calls: int
    errors: int
    tokens: int
    wall_s: float
    run: object = None


def evaluate(conn, client_factory, *, queries=None, max_turns: int = 16,
             echo=None, transcripts_dir: str | None = None) -> list[EvalRow]:
    """Run each ADL task with a fresh agent and score it.

    ``client_factory(query_id)`` returns a :class:`ModelClient` (a fresh client
    per task lets a ReplayClient load that task's transcript).
    """
    import pathlib
    say = echo or (lambda *a, **k: None)
    queries = list(queries or TASKS)
    refs = reference_answers(conn, queries)
    rows = []
    for q in queries:
        say(f"\n=== {q}: {TASKS[q][1]}")
        tools = AnalysisTools(conn)
        client: ModelClient = client_factory(q)
        t0 = time.perf_counter()
        run = run_agent(TASKS[q][1], tools, client, max_turns=max_turns, echo=echo)
        ok, why = compare(run.answer, refs[q])
        eng = run.engine_summary()
        row = EvalRow(
            query=q, passed=ok, reason=why,
            route=(run.answer_info or {}).get("route", "-"), turns=run.turns,
            sql_calls=eng.get("sql", {}).get("calls", 0),
            awkward_calls=eng.get("awkward", {}).get("calls", 0),
            errors=sum(v["errors"] for v in eng.values()),
            tokens=run.usage["input_tokens"] + run.usage["output_tokens"],
            wall_s=round(time.perf_counter() - t0, 1), run=run)
        say(f"  => {'PASS' if ok else 'FAIL'} ({why}); answer via {row.route}")
        if transcripts_dir:
            d = pathlib.Path(transcripts_dir)
            d.mkdir(parents=True, exist_ok=True)
            run.save(str(d / f"{q}.json"))
        rows.append(row)
    return rows


def format_table(rows: list[EvalRow]) -> str:
    head = (f"{'query':5s} {'result':6s} {'route':7s} {'turns':>5s} {'sql':>4s} "
            f"{'awk':>4s} {'err':>4s} {'tokens':>7s} {'time':>6s}  detail")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r.query:5s} {'PASS' if r.passed else 'FAIL':6s} "
                     f"{r.route:7s} {r.turns:5d} {r.sql_calls:4d} "
                     f"{r.awkward_calls:4d} {r.errors:4d} {r.tokens:7d} "
                     f"{r.wall_s:5.1f}s  {r.reason}")
    n = sum(r.passed for r in rows)
    lines.append(f"\n{n}/{len(rows)} passed")
    return "\n".join(lines)
