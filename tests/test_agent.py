"""Agent loop, tools and ADL evaluation (no API key needed: scripted/fake models)."""
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest

from awkward_monetizer.agent import (
    TASKS,
    AnalysisTools,
    AnthropicClient,
    ModelTurn,
    ReplayClient,
    ScriptedClient,
    ToolError,
    compare,
    evaluate,
    expert_script,
    format_table,
    load_demo_db,
    reference_answers,
    run_agent,
)
from awkward_monetizer.agent.tools import check_code, check_select
from awkward_monetizer.cli import main

_has_monetdbe = importlib.util.find_spec("monetdbe") is not None
_has_mcp = importlib.util.find_spec("mcp") is not None
BACKENDS = ["sqlite"] + (["monetdbe"] if _has_monetdbe else [])


@pytest.fixture(scope="module", params=BACKENDS)
def conn(request, nano_file):
    c = load_demo_db(nano_file, backend=request.param)
    yield c
    c.close()


@pytest.fixture(scope="module")
def sqlite_conn(nano_file):
    c = load_demo_db(nano_file, backend="sqlite")
    yield c
    c.close()


def _use(i, name, inp):
    return {"type": "tool_use", "id": f"t{i}", "name": name, "input": inp}


# -- guards -----------------------------------------------------------------
@pytest.mark.parametrize("sql", [
    "DROP TABLE events", "SELECT 1; DROP TABLE events", "DELETE FROM jets",
    "select * from events where 1=1; select 2", "UPDATE events SET run = 0",
    "COPY INTO events FROM 'x'", "",
])
def test_check_select_rejects(sql):
    with pytest.raises(ToolError):
        check_select(sql)


def test_check_select_allows_reads():
    assert check_select("select pt from jets;") == "select pt from jets"
    assert check_select("WITH a AS (SELECT 1 AS x) SELECT x FROM a")
    # keywords inside string literals are fine
    assert check_select("SELECT 'drop table' AS s FROM events")


@pytest.mark.parametrize("code", [
    "import os\nresult = 1", "result = events.__class__",
    "result = __import__('os')", "x = 1",
    "def f(): global g\nresult = 1",
])
def test_check_code_rejects(code):
    with pytest.raises(ToolError):
        check_code(code)


# -- tools ------------------------------------------------------------------
def test_describe_schema_and_sql(conn):
    t = AnalysisTools(conn)
    text, err = t.call("describe_schema", {})
    assert not err
    schema = json.loads(text)["tables"]
    assert {"events", "jets", "muons"} <= set(schema)
    assert "met_pt" in schema["events"]["columns"]
    assert schema["events"]["rows"] == 1500

    text, err = t.call("run_sql", {"sql": "SELECT COUNT(*) AS n FROM jets"})
    assert not err and json.loads(text)["rows"][0][0] == schema["jets"]["rows"]

    text, err = t.call("run_sql", {"sql": "SELECT nope FROM events"})
    assert err and text.startswith("error:")
    # connection still usable after a failed query
    _, err = t.call("run_sql", {"sql": "SELECT 1 AS x FROM events"})
    assert not err
    assert [c.engine for c in t.trace] == ["catalog", "sql", "sql", "sql"]
    assert [c.ok for c in t.trace] == [True, True, False, True]


def test_awkward_eval_pushes_where_and_validates(conn):
    t = AnalysisTools(conn)
    text, err = t.call("awkward_eval", {"code": "result = ak.num(events.jets)",
                                        "where": "n_jets >= 3"})
    assert not err
    d = json.loads(text)
    assert d["result_id"] == "r1" and d["summary"]["min/q25/median/q75/max"][0] >= 3
    # nested result -> helpful error
    text, err = t.call("awkward_eval", {"code": "result = events.jets.pt"})
    assert err and "flatten" in text
    # python errors are reported, not raised
    text, err = t.call("awkward_eval", {"code": "result = events.nope"})
    assert err
    _, err = t.call("submit_answer", {"explanation": "x", "result_id": "r9"})
    assert err
    _, err = t.call("submit_answer", {"explanation": "x"})
    assert err
    _, err = t.call("submit_answer", {"explanation": "x", "result_id": "r1"})
    assert not err and t.answer_info["route"] == "awkward"


# -- loop -------------------------------------------------------------------
def test_loop_recovers_from_tool_error(sqlite_conn):
    turns = [
        ModelTurn([_use(1, "run_sql", {"sql": "SELECT met FROM events"})], "tool_use"),
        ModelTurn([{"type": "text", "text": "Column is met_pt."},
                   _use(2, "submit_answer", {"sql": "SELECT met_pt FROM events",
                                             "explanation": "MET"})], "tool_use"),
    ]
    client = ScriptedClient(turns)
    run = run_agent("MET of all events", AnalysisTools(sqlite_conn), client)
    assert run.stopped == "submitted" and run.turns == 2
    assert run.answer.size == 1500
    # the model saw the error as an is_error tool_result
    last_user = client.seen_messages[1][-1]
    assert last_user["content"][0]["is_error"] is True
    assert run.engine_summary()["sql"]["errors"] == 1


def test_loop_stops_without_tool_use_and_on_budget(sqlite_conn):
    run = run_agent("hi", AnalysisTools(sqlite_conn),
                    ScriptedClient([ModelTurn([{"type": "text", "text": "ok"}],
                                              "end_turn")]))
    assert run.stopped == "end_turn" and run.answer is None
    loop = [ModelTurn([_use(i, "describe_schema", {})], "tool_use") for i in range(5)]
    run = run_agent("x", AnalysisTools(sqlite_conn), ScriptedClient(loop), max_turns=3)
    assert run.stopped == "max_turns" and run.turns == 3


def test_anthropic_client_adapter(sqlite_conn):
    """The Messages-API adapter, driven by a fake SDK object."""
    def block(**kw):
        return SimpleNamespace(**kw)
    responses = iter([
        SimpleNamespace(content=[block(type="text", text="Look first."),
                                 block(type="tool_use", id="a", name="describe_schema",
                                       input={})],
                        stop_reason="tool_use",
                        usage=SimpleNamespace(input_tokens=10, output_tokens=5)),
        SimpleNamespace(content=[block(type="tool_use", id="b", name="submit_answer",
                                       input={"sql": "SELECT pt FROM jets",
                                              "explanation": "all jets"})],
                        stop_reason="tool_use",
                        usage=SimpleNamespace(input_tokens=20, output_tokens=7)),
    ])
    calls = []

    def create(**kw):
        calls.append(kw)
        return next(responses)
    fake = SimpleNamespace(messages=SimpleNamespace(create=create))
    run = run_agent("jet pT", AnalysisTools(sqlite_conn),
                    AnthropicClient(model="m", client=fake))
    assert run.stopped == "submitted" and run.usage == {"input_tokens": 30,
                                                        "output_tokens": 12}
    assert calls[0]["model"] == "m" and calls[0]["tools"][0]["name"] == "describe_schema"
    # second request carries assistant tool_use + user tool_result
    assert calls[1]["messages"][-1]["content"][0]["tool_use_id"] == "a"


def test_transcript_replay_reproduces_answer(sqlite_conn, tmp_path):
    run = run_agent(TASKS["Q6"][1], AnalysisTools(sqlite_conn),
                    ScriptedClient(expert_script("Q6")))
    path = tmp_path / "q6.json"
    run.save(str(path))
    replay = ReplayClient(str(path))
    assert replay.question == TASKS["Q6"][1]
    again = run_agent(replay.question, AnalysisTools(sqlite_conn), replay)
    np.testing.assert_array_equal(run.answer, again.answer)


# -- evaluation -------------------------------------------------------------
def test_scripted_expert_passes_all_adl_queries(conn):
    rows = evaluate(conn, lambda q: ScriptedClient(expert_script(q)))
    table = format_table(rows)
    assert all(r.passed for r in rows), table
    assert {r.query: r.route for r in rows} == {
        "Q1": "sql", "Q2": "sql", "Q3": "sql", "Q4": "sql",
        "Q5": "awkward", "Q6": "awkward", "Q7": "awkward", "Q8": "awkward"}
    assert "8/8 passed" in table


def test_eval_catches_wrong_physics(sqlite_conn):
    refs = reference_answers(sqlite_conn, ["Q4"])
    # >= 1 jet instead of >= 2: plausible-looking but wrong
    wrong = [ModelTurn([_use(1, "submit_answer", {
        "sql": "SELECT met_pt FROM events WHERE event_id IN (SELECT event_id "
               "FROM jets WHERE pt > 40)", "explanation": "x"})], "tool_use")]
    rows = evaluate(sqlite_conn, lambda q: ScriptedClient(wrong), queries=["Q4"])
    assert not rows[0].passed
    ok, why = compare(refs["Q4"][::-1].copy(), refs["Q4"])
    assert ok, why       # order does not matter


def test_cli_agent_eval_scripted(nano_file, capsys, tmp_path):
    main(["agent", "eval", "--scripted", "--from-root", nano_file,
          "--backend", "sqlite", "--queries", "q1,Q6",
          "--transcripts", str(tmp_path)])
    out = capsys.readouterr().out
    assert "2/2 passed" in out
    assert (tmp_path / "Q6.json").exists()
    # replay a saved transcript through `agent ask`
    main(["agent", "ask", "--from-root", nano_file, "--backend", "sqlite",
          "--replay", str(tmp_path / "Q6.json")])
    out = capsys.readouterr().out
    assert "stopped: submitted" in out and "answer via awkward" in out


@pytest.mark.skipif(not _has_mcp, reason="mcp not installed")
def test_mcp_server_lists_tools(sqlite_conn):
    import asyncio

    from awkward_monetizer.agent.mcp_server import build_server
    server = build_server(AnalysisTools(sqlite_conn))
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"describe_schema", "run_sql", "awkward_eval", "submit_answer"}
