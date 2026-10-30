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
    assert ok and why.startswith(f"all {refs['Q4'].size} values match"), why


def test_compare_counts_wrong_values_not_shifted_pairs():
    ref = np.arange(1000, dtype=float)
    got = ref.copy()
    got[10] = -5.0            # one wrong value shifts the whole sorted order
    ok, why = compare(got, ref)
    assert not ok and why.startswith("1/1000 ")


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


# -- plot_result, check_answer, CLI extras ----------------------------------
_has_mpl = importlib.util.find_spec("matplotlib") is not None


@pytest.mark.skipif(not _has_mpl, reason="matplotlib not installed")
def test_plot_result_writes_png(sqlite_conn, tmp_path):
    t = AnalysisTools(sqlite_conn, plot_dir=str(tmp_path / "plots"))
    t.call("awkward_eval", {"code": "result = events.met_pt"})
    text, err = t.call("plot_result", {"result_id": "r1", "title": "MET",
                                       "xlabel": "MET [GeV]", "bins": 30})
    assert not err
    path = json.loads(text)["path"]
    assert path.endswith(".png") and (tmp_path / "plots").exists()
    with open(path, "rb") as fh:
        assert fh.read(4) == b"\x89PNG"
    # SQL source and argument validation
    _, err = t.call("plot_result", {"sql": "SELECT pt FROM jets", "x_range": [0, 200]})
    assert not err
    _, err = t.call("plot_result", {"sql": "SELECT pt FROM jets", "x_range": [5, 1]})
    assert err
    assert t.trace[-1].engine == "plot"


def test_check_answer_scores_and_logs_attempts(sqlite_conn, tmp_path):
    log = tmp_path / "checks.jsonl"
    t = AnalysisTools(sqlite_conn, check_log=str(log))
    _, err = t.call("check_answer", {"query": "Q1"})
    assert err                                     # nothing submitted yet
    t.call("submit_answer", {"sql": "SELECT met_pt FROM events WHERE met_pt > 30",
                             "explanation": "wrong cut"})
    text, err = t.call("check_answer", {"query": "q1"})
    assert not err and json.loads(text)["passed"] is False
    t.call("submit_answer", {"sql": "SELECT met_pt FROM events", "explanation": "MET"})
    d = json.loads(t.call("check_answer", {"query": "Q1"})[0])
    assert d["passed"] is True and d["attempt"] == 2
    _, err = t.call("check_answer", {"query": "Q9"})
    assert err
    entries = [json.loads(x) for x in log.read_text().splitlines()]
    assert [(e["query"], e["passed"], e["attempt"]) for e in entries] == [
        ("Q1", False, 1), ("Q1", True, 2)]
    # a restarted server (new AnalysisTools on the same log) keeps counting
    t2 = AnalysisTools(sqlite_conn, check_log=str(log))
    t2.call("submit_answer", {"sql": "SELECT met_pt FROM events", "explanation": "MET"})
    assert json.loads(t2.call("check_answer", {"query": "Q1"})[0])["attempt"] == 3


def test_check_answer_not_offered_to_eval_loop():
    from awkward_monetizer.agent import TOOL_SPECS
    assert "check_answer" not in {s["name"] for s in TOOL_SPECS}


def test_cli_questions_and_checks_table(tmp_path, capsys):
    main(["agent", "questions"])
    out = capsys.readouterr().out
    assert out.count("\nQ") >= 7 and "172.5" in out
    log = tmp_path / "c.jsonl"
    rows = [{"query": "Q6", "passed": False, "reason": "n=1", "attempt": 1,
             "route": "awkward"},
            {"query": "Q6", "passed": True, "reason": "ok", "attempt": 1,
             "route": "awkward"},
            {"query": "Q1", "passed": True, "reason": "ok", "attempt": 1,
             "route": "sql"}]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    main(["agent", "checks", "--check-log", str(log)])
    out = capsys.readouterr().out
    assert "1/2 passed on first check" in out
    q6 = next(line for line in out.splitlines() if line.startswith("Q6"))
    assert q6.split()[1:4] == ["FAIL", "PASS", "2"]   # tries counted from the log


def test_cli_reports_api_errors_in_one_line(nano_file, monkeypatch):
    from awkward_monetizer import cli

    class BadRequestError(Exception):
        """Stand-in for an anthropic SDK error (same module, body shape)."""
        def __init__(self):
            super().__init__("Error code: 400")
            self.body = {"error": {"message": "Your credit balance is too low"}}
    BadRequestError.__module__ = "anthropic._exceptions"

    class _Failing:
        model = "m"

        def create(self, **kw):
            raise BadRequestError()
    monkeypatch.setattr(cli, "_agent_client", lambda args, q=None: _Failing())
    with pytest.raises(SystemExit,
                       match=r"^API error \(BadRequestError\): Your credit balance"):
        main(["agent", "ask", "MET", "--from-root", nano_file, "--backend", "sqlite"])


@pytest.mark.skipif(not _has_mcp, reason="mcp not installed")
def test_mcp_server_lists_all_tools(sqlite_conn):
    import asyncio

    from awkward_monetizer.agent.mcp_server import build_server
    server = build_server(AnalysisTools(sqlite_conn))
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"describe_schema", "run_sql", "awkward_eval", "submit_answer",
                     "plot_result", "check_answer"}
    # the routing rule reaches MCP clients as server instructions
    assert "combinatorics" in server.instructions and "check_answer" in server.instructions


# --------------------------------------------------------------------------
# Large databases: chunked by event_id range
# --------------------------------------------------------------------------
def test_event_ranges_cover_every_event_once(conn):
    from awkward_monetizer.agent.tools import event_ranges
    assert event_ranges(conn, 10**9) is None          # small: one pass
    ranges = event_ranges(conn, 200)
    cur = conn.cursor()
    total = 0
    for lo, hi in ranges:
        cur.execute(f"SELECT COUNT(*) FROM events WHERE event_id >= {lo} "
                    f"AND event_id < {hi}")
        total += cur.fetchall()[0][0]
    cur.execute("SELECT COUNT(*) FROM events")
    assert total == cur.fetchall()[0][0]
    assert len(ranges) > 1


def test_chunked_reference_and_scripted_expert_match_single_pass(conn):
    full = reference_answers(conn)
    chunked = reference_answers(conn, chunk_events=200)
    for q in TASKS:
        assert compare(chunked[q], full[q])[0], q
    rows = evaluate(conn, lambda q: ScriptedClient(expert_script(q)),
                    chunk_events=200)
    assert all(r.passed for r in rows), format_table(rows)


def test_chunked_awkward_eval_projects_and_falls_back(conn):
    big = AnalysisTools(conn)
    small = AnalysisTools(conn, chunk_events=200)
    code = "result = events.met_pt"
    a = json.loads(big.call("awkward_eval", {"code": code})[0])
    b = json.loads(small.call("awkward_eval", {"code": code})[0])
    assert b["chunks"] > 1 and b["events_fetched"] == a["events_fetched"]
    np.testing.assert_array_equal(small.results[b["result_id"]],
                                  big.results[a["result_id"]])
    assert set(small._projection(check_code(code))) == {"events"}  # no objects
    # subscripting by string still projects the right column
    c = json.loads(small.call("awkward_eval",
                              {"code": 'result = ak.num(events["jets"], axis=1)',
                               "where": "n_jets >= 1"})[0])
    assert c["summary"]["min/q25/median/q75/max"][0] >= 1
    # a column the projection cannot see is fetched by the unprojected retry
    d = small.call("awkward_eval", {"code": "f = ak.fields(events)\n"
                                    "result = events[f[f.index('met_pt')]]"})
    assert not d[1], d[0]


def test_reference_cache_roundtrip(sqlite_conn, tmp_path, monkeypatch):
    import sys
    ev = sys.modules["awkward_monetizer.agent.evaluate"]
    path = str(tmp_path / "refs.npz")
    first = reference_answers(sqlite_conn, cache=path)
    monkeypatch.setattr(ev, "fetch_tables", lambda *a, **k: 1 / 0)  # must not run
    again = reference_answers(sqlite_conn, ["Q2", "Q6"], cache=path)
    np.testing.assert_array_equal(again["Q6"], first["Q6"])


def test_projection_covers_p4_helpers(conn):
    t = AnalysisTools(conn, chunk_events=200)
    proj = t._projection(check_code("result = ak.num(muons_p4(events), axis=1)"))
    assert {"pt", "eta", "phi", "mass", "charge"} <= set(proj["muons"])
    assert "jets" not in proj
    # every scripted Awkward answer runs on its projection, without the retry
    from awkward_monetizer.agent.scripted import _AWK
    for where, code in _AWK.values():
        tree = check_code(code)
        p = t._projection(tree)
        for rng in t.ranges()[:2]:
            t._run_code(tree, t._chunk(rng, where, p))


@pytest.mark.parametrize("where", [None, "n_jets >= 3"])
def test_fetch_events_matches_pandas_path(conn, where):
    import awkward as ak

    from awkward_monetizer.agent.tools import event_ranges
    from awkward_monetizer.reconstruct import (
        fetch_events,
        fetch_tables,
        reconstruct_multi,
    )
    for rng in [None, *event_ranges(conn, 400)[:2]]:
        ev_df, jets = fetch_tables(conn, where=where, collection="jets",
                                   event_range=rng)
        _, muons = fetch_tables(conn, where=where, collection="muons",
                                event_range=rng)
        old = reconstruct_multi(ev_df, {"jets": jets, "muons": muons})
        new = fetch_events(conn, where, event_range=rng,
                           collections={"jets": None, "muons": None})
        # same values; MonetDB BOOLEAN trigger flags now arrive as bool
        # (pandas made them int64), every other field keeps its type
        assert ak.to_list(new) == ak.to_list(old)
        for f in ak.fields(old):
            if not f.startswith("hlt_"):
                assert str(new[f].type) == str(old[f].type), f
    # projected: only the named columns, same values
    proj = fetch_events(conn, where, event_columns=("event_id", "met_pt"),
                        collections={"jets": ("event_id", "jet_index", "pt")})
    assert ak.fields(proj) == ["event_id", "met_pt", "jets"]
    assert ak.array_equal(proj.jets.pt, old.jets.pt if rng is None else
                          fetch_events(conn, where,
                                       collections={"jets": None}).jets.pt)


def test_fetch_columns_handles_nulls_and_unknown_types():
    import awkward as ak

    from awkward_monetizer.reconstruct import _column
    a = _column((1.5, None, 2.0), np.dtype(np.float64))
    assert ak.to_list(a) == [1.5, None, 2.0]
    assert _column((1, 2, 3), None).dtype == np.int64
    assert ak.to_list(_column((True, None), None)) == [True, None]
    assert ak.to_list(_column((None, None), np.dtype(np.bool_))) == [None, None]
    assert ak.to_list(_column((None, None), None)) == [None, None]
    assert ak.to_list(_column(("a", "b"), None)) == ["a", "b"]


def test_binary_fast_path_decodes_monetdb_export(monkeypatch):
    """The pymonetdb fast path, against a fake server speaking the
    Xexportbin layout (payloads, then (start, length) pairs, then the offset)."""
    import awkward as ak

    from awkward_monetizer import reconstruct as rc
    data = {
        "event_id": np.array([5, 6, 7, 8, 9], "<i8"),
        "n": np.array([1, -2**31, 3, 4, 5], "<i4"),          # one NULL
        "pt": np.array([1.5, 2.5, np.nan, 4.5, 5.5], "<f8"),  # NaN = NULL
        "flag": np.array([1, 0, -128, 1, 0], "i1"),           # one NULL
    }
    types = ["bigint", "int", "double", "boolean"]

    def export(start, n):
        payload, toc = b"", []
        for col in data.values():
            raw = col[start:start + n].tobytes()
            toc.append((len(payload), len(raw)))
            payload += raw
        table = np.array(toc, "<i8").tobytes()
        return memoryview(payload + table + np.array([len(payload)], "<i8").tobytes())

    class Conn:
        mapi = SimpleNamespace(server_endian="little")
        _policy = SimpleNamespace(use_binary=lambda: True)

        def binary_command(self, cmd):
            _, _, start, n = cmd.split()
            return export(int(start), int(n))

    class Cur:
        connection = Conn()

        def execute(self, sql):
            self._query_id = "7"
            self.rowcount = 5
            self._rows = [(5, 1, 1.5, True)]        # first reply, text protocol
            self.description = [(k, t) for k, t in zip(data, types, strict=True)]

        def fetchall(self):
            raise AssertionError("the fast path must not fall back")

    monkeypatch.setattr(rc, "BINARY_BATCH", 2)       # several export batches
    cols = rc._fetch_columns(Cur(), "SELECT ...", None)
    assert cols["event_id"].dtype == np.int64
    assert cols["event_id"].tolist() == [5, 6, 7, 8, 9]
    assert ak.to_list(cols["n"]) == [1, None, 3, 4, 5]
    assert ak.to_list(cols["pt"]) == [1.5, 2.5, None, 4.5, 5.5]
    assert ak.to_list(cols["flag"]) == [True, False, None, True, False]
    # an unsupported type (varchar) falls back to the portable path
    types[1] = "varchar"
    with pytest.raises(AssertionError, match="must not fall back"):
        rc._fetch_columns(Cur(), "SELECT ...", None)


# --------------------------------------------------------------------------
# MCP over HTTP
# --------------------------------------------------------------------------
def test_run_server_transports(capsys):
    from awkward_monetizer.agent.mcp_server import is_loopback, run_server

    class V2:                         # mcp >= 2: run(transport, host=, port=)
        def run(self, *a, **k):
            self.got = (a, k)

    class V1:                         # mcp 1.x: host/port live in settings
        def __init__(self):
            self.settings = SimpleNamespace(host="0.0.0.0", port=1)

        def run(self, *a, **k):
            self.got = (a, k, self.settings.host, self.settings.port)

    s = V2()
    run_server(s)
    assert s.got == ((), {})
    run_server(s, "http", port=9001)
    assert s.got == (("streamable-http",), {"host": "127.0.0.1", "port": 9001})
    s1 = V1()
    run_server(s1, "http", host="localhost", port=9002)
    assert s1.got == ((), {"transport": "streamable-http"}, "localhost", 9002)
    err = capsys.readouterr().err
    assert "http://127.0.0.1:9001/mcp" in err and "WARNING" not in err
    run_server(s, "http", host="0.0.0.0")
    assert "WARNING" in capsys.readouterr().err
    assert is_loopback("::1") and not is_loopback("192.168.1.5")
    with pytest.raises(ValueError):
        run_server(s, "sse")

    class Interrupted:
        def run(self, *a, **k):
            raise KeyboardInterrupt

    run_server(Interrupted(), "http")          # Ctrl-C: no traceback
    assert "server stopped" in capsys.readouterr().err


@pytest.mark.skipif(not _has_mcp, reason="mcp not installed")
def test_mcp_server_over_http_lists_and_calls_tools(nano_file):
    import socket
    import subprocess
    import sys
    import time
    import urllib.request

    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, "-m", "awkward_monetizer.cli", "agent", "mcp",
         "--from-root", nano_file, "--backend", "sqlite", "--check-log", "",
         "--transport", "http", "--http-port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    url = f"http://127.0.0.1:{port}/mcp"

    def rpc(payload, session=None):
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream",
                   "MCP-Protocol-Version": "2025-06-18"}
        if session:
            headers["Mcp-Session-Id"] = session
        req = urllib.request.Request(url, json.dumps(payload).encode(), headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read().decode()
            sid = r.headers.get("Mcp-Session-Id") or session
        if not body.strip():
            return None, sid
        if body.lstrip().startswith("{"):
            return json.loads(body), sid
        data = [ln[5:].strip() for ln in body.splitlines() if ln.startswith("data:")]
        return json.loads(data[-1]), sid

    try:
        for _ in range(100):              # wait for the server to listen
            try:
                init, sid = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                 "params": {"protocolVersion": "2025-06-18",
                                            "capabilities": {},
                                            "clientInfo": {"name": "t",
                                                           "version": "0"}}})
                break
            except OSError:
                if proc.poll() is not None:
                    pytest.fail(proc.stderr.read())
                time.sleep(0.2)
        else:
            pytest.fail("server did not start")
        assert "routing" in init["result"].get("instructions", "").lower() or \
            "SQL" in init["result"].get("instructions", "")
        rpc({"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
        tools, _ = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, sid)
        names = {t["name"] for t in tools["result"]["tools"]}
        assert {"describe_schema", "run_sql", "awkward_eval", "submit_answer",
                "plot_result", "check_answer"} <= names
        res, _ = rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                      "params": {"name": "run_sql",
                                 "arguments": {"sql": "SELECT COUNT(*) FROM events"}}},
                     sid)
        text = res["result"]["content"][0]["text"]
        assert json.loads(text)["rows"] == [[1500]], text
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_mcp_warns_about_reference_cost(sqlite_conn, tmp_path, capsys):
    from awkward_monetizer.agent.mcp_server import warn_reference_cost
    warn_reference_cost(AnalysisTools(sqlite_conn))          # small: silent
    assert capsys.readouterr().err == ""
    warn_reference_cost(AnalysisTools(sqlite_conn, chunk_events=200))
    assert "WARNING: no --ref-cache" in capsys.readouterr().err
    cache = tmp_path / "refs.npz"
    warn_reference_cost(AnalysisTools(sqlite_conn, chunk_events=200,
                                      ref_cache=str(cache)))
    assert "none yet" in capsys.readouterr().err
    reference_answers(sqlite_conn, cache=str(cache))
    warn_reference_cost(AnalysisTools(sqlite_conn, chunk_events=200,
                                      ref_cache=str(cache)))
    assert "reused if it matches" in capsys.readouterr().err



@pytest.mark.parametrize("code", [
    "result = np.load('x.npy')",
    "np.save('x.npy', events.met_pt)\nresult = events.met_pt",
    "result = np.loadtxt('/etc/passwd', dtype=str)",
    "ak.to_parquet(events, 'x.parquet')\nresult = events.met_pt",
    "result = ak.from_parquet('x.parquet').met_pt",
    "f = np.save\nresult = events.met_pt",
    "ak.to_numpy(events.met_pt).tofile('x')\nresult = events.met_pt",
    "result = np.lib.npyio.load('x')",
    "result = np.memmap('x')",
])
def test_check_code_rejects_file_io(code):
    with pytest.raises(ToolError, match="reads or writes files"):
        check_code(code)


def test_file_io_guard_keeps_array_code(sqlite_conn):
    from awkward_monetizer.agent.scripted import _AWK
    for _where, code in _AWK.values():          # the reference solutions pass
        check_code(code)
    t = AnalysisTools(sqlite_conn)
    out, err = t.call("awkward_eval", {"code": (
        "pt = ak.flatten(events.jets.pt)\n"
        "result = np.sqrt(ak.to_numpy(pt)) + np.log1p(np.abs(np.sin(1.0)))")})
    assert not err, out
