"""A scripted "expert" for the ADL tasks: fixed tool calls, no LLM.

Used to (a) test the loop and the eval harness without an API key, and
(b) give an offline baseline / fallback demo. Each script routes the way the
benchmarks say it should: Q1–Q4 are submitted as pure SQL, Q5–Q8 push the
event-level cut into SQL and do the combinatorics in Awkward.
"""

from __future__ import annotations

from .loop import ModelTurn

_SQL = {
    "Q1": "SELECT met_pt FROM events",
    "Q2": "SELECT pt FROM jets",
    "Q3": "SELECT pt FROM jets WHERE abs(eta) < 1",
    "Q4": ("SELECT met_pt FROM events WHERE event_id IN ("
           "SELECT event_id FROM jets WHERE pt > 40 "
           "GROUP BY event_id HAVING COUNT(*) >= 2)"),
}

_AWK = {
    "Q5": ("n_muons >= 2", """\
mu = muons_p4(events)
pair = ak.combinations(mu, 2, fields=["a", "b"])
mass = (pair.a + pair.b).mass
good = (pair.a.charge != pair.b.charge) & (mass > 60) & (mass < 120)
result = events.met_pt[ak.any(good, axis=1)]"""),
    "Q6": ("n_jets >= 3", """\
jets = jets_p4(events)
tri = ak.combinations(jets, 3, fields=["a", "b", "c"])
p4 = tri.a + tri.b + tri.c
best = ak.argmin(abs(p4.mass - TOP_MASS), axis=1, keepdims=True)
result = ak.flatten(p4.pt[best])"""),
    "Q7": (None, """\
jets = jets_p4(events)
jets = jets[(jets.pt > 30) & (abs(jets.eta) < 2.4)]
mu = muons_p4(events)
mu = mu[mu.pt > 10]
pair = ak.cartesian({"j": jets, "l": mu}, nested=True)
min_dr = ak.fill_none(ak.min(pair.j.deltaR(pair.l), axis=2), np.inf)
result = ak.fill_none(ak.sum(jets[min_dr > 0.4].pt, axis=1), 0.0)"""),
    "Q8": ("n_muons >= 3", """\
mu = muons_p4(events)
pairs = ak.argcombinations(mu, 2, fields=["i", "j"])
mi, mj = mu[pairs.i], mu[pairs.j]
sfos = mi.charge != mj.charge
metric = abs(ak.where(sfos, (mi + mj).mass, np.inf) - Z_MASS)
best = ak.argmin(metric, axis=1, keepdims=True)
bi, bj = ak.flatten(pairs.i[best]), ak.flatten(pairs.j[best])
idx = ak.local_index(mu, axis=1)
other = mu[(idx != bi) & (idx != bj)]
lead = other[ak.argmax(other.pt, axis=1, keepdims=True)]
dphi = ak.flatten(lead.phi) - events.met_phi
mt = np.sqrt(2.0 * ak.flatten(lead.pt) * events.met_pt * (1.0 - np.cos(dphi)))
ok = ak.any(sfos, axis=1) & (ak.num(other, axis=1) >= 1)
result = mt[ok]"""),
}


def _use(i, name, inp):
    return {"type": "tool_use", "id": f"toolu_{i:02d}", "name": name, "input": inp}


def expert_script(query: str) -> list[ModelTurn]:
    turns = [ModelTurn([{"type": "text", "text": "Checking the schema."},
                        _use(1, "describe_schema", {})], "tool_use")]
    if query in _SQL:
        sql = _SQL[query]
        turns.append(ModelTurn([_use(2, "run_sql", {"sql": sql, "max_rows": 5})],
                               "tool_use"))
        turns.append(ModelTurn([_use(3, "submit_answer", {
            "sql": sql, "explanation": "Flat selection, done entirely in SQL."})],
            "tool_use"))
    else:
        where, code = _AWK[query]
        inp = {"code": code}
        if where:
            inp["where"] = where
        turns.append(ModelTurn([_use(2, "awkward_eval", inp)], "tool_use"))
        turns.append(ModelTurn([_use(3, "submit_answer", {
            "result_id": "r1",
            "explanation": "Event cut pushed to SQL; combinatorics in Awkward."})],
            "tool_use"))
    return turns
