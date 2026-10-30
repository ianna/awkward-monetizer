import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="awkward-monetizer agent demo")


@app.cell
def _():
    import json
    import os
    import pathlib
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from awkward_monetizer import db
    from awkward_monetizer.agent import (
        TASKS,
        AnalysisTools,
        ScriptedClient,
        expert_script,
        load_demo_db,
        run_agent,
    )

    started = time.time()   # the Plots tab shows plots made after this
    return (AnalysisTools, ScriptedClient, TASKS, db, expert_script, json,
            load_demo_db, mo, np, os, pathlib, plt, run_agent, started, time)


@app.cell
def _(mo):
    mo.md(r"""
    # Agents on Awkward + MonetDB: live scoreboard

    **Live scoreboard**: run Claude Code against the awkward-monetizer MCP server
    in another window; every `check_answer` it makes lands here, scored against
    the ADL reference. **Replay**: no agent and no network: run the reference
    solution for one ADL question through the very same tools.
    """)
    return


@app.cell
def _(mo, os):
    database = mo.ui.text(value=os.environ.get("AM_DEMO_DB", "hep_real"),
                          label="MonetDB database (or a .root file for an "
                                "in-memory demo)", full_width=True)
    check_log = mo.ui.text(value=os.environ.get(
        "AM_DEMO_CHECK_LOG", "output/agent_checks_demo.jsonl"),
        label="check log the MCP server writes (--check-log)", full_width=True)
    plot_dir = mo.ui.text(value="output/agent_plots",
                          label="plot folder (--plot-dir)", full_width=True)
    mo.accordion({"Settings": mo.vstack([database, check_log, plot_dir])})
    return check_log, database, plot_dir


@app.cell
def _(mo):
    refresh = mo.ui.refresh(options=["1s", "2s", "5s", "10s"],
                            default_interval="2s", label="refresh")
    return (refresh,)


@app.cell
def _(TASKS, check_log, json, mo, os, pathlib, plot_dir, refresh, started):
    refresh.value  # re-run on every tick

    def _read_checks(path):
        p = pathlib.Path(path).expanduser()
        if not p.exists():
            return []
        out = []
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    _checks = _read_checks(check_log.value)
    _rows, _first_pass = [], 0
    for _q in TASKS:
        _mine = [c for c in _checks if c.get("query") == _q]
        if not _mine:
            _rows.append({"query": _q, "first check": "–", "latest": "–",
                          "attempts": 0, "route": "–", "detail": TASKS[_q][0]})
            continue
        _first = min(_mine, key=lambda c: c.get("attempt", 1))
        _last = _mine[-1]
        _first_pass += bool(_first.get("passed"))
        _rows.append({
            "query": _q,
            "first check": "PASS" if _first.get("passed") else "FAIL",
            "latest": "PASS" if _last.get("passed") else "FAIL",
            "attempts": len(_mine),
            "route": _last.get("route") or "–",
            "detail": _last.get("reason", ""),
        })
    _checked = sum(r["attempts"] > 0 for r in _rows)

    # only plots made since the notebook started, so a fresh run starts empty
    _pngs = sorted((p for p in pathlib.Path(plot_dir.value).expanduser().glob("*.png")
                    if os.path.getmtime(p) >= started),
                   key=os.path.getmtime)
    _plot = (mo.vstack([mo.md(f"Latest agent plot: `{_pngs[-1].name}`"),
                        mo.image(src=str(_pngs[-1]), width=560)])
             if _pngs else mo.md("_No agent plot yet._"))

    scoreboard_view = mo.vstack([
        mo.hstack([
            mo.stat(value=f"{_first_pass}/8", label="passed on the first check",
                    caption=f"{_checked} of 8 checked so far"),
            refresh,
        ], justify="space-between"),
        mo.ui.table(_rows, selection=None, pagination=False),
        _plot,
        mo.md(f"Reading `{check_log.value}`. Start the server with the same "
              "`--check-log`, and clear the file before the talk so attempts "
              "start at 1."),
    ])
    return (scoreboard_view,)


@app.cell
def _(TASKS, mo):
    question = mo.ui.dropdown(
        options={f"{q}: {TASKS[q][0].split(': ', 1)[1]}": q for q in TASKS},
        value=f"Q6: {TASKS['Q6'][0].split(': ', 1)[1]}", label="ADL question")
    run = mo.ui.run_button(label="Run through the agent's tools")
    return question, run


@app.cell
def _(AnalysisTools, ScriptedClient, TASKS, database, db, expert_script, json,
      load_demo_db, mo, np, plot_dir, plt, question, run, run_agent):
    def _replay(q):
        if database.value.endswith(".root"):
            conn = load_demo_db(database.value, backend="sqlite")
            ref_cache = None
        else:
            conn = db.open_server(database=database.value)
            ref_cache = f"output/refs_{database.value}.npz"
        tools = AnalysisTools(conn, plot_dir=plot_dir.value,
                              check_log="output/agent_checks_replay.jsonl",
                              ref_cache=ref_cache)
        try:
            run_ = run_agent(TASKS[q][1], tools, ScriptedClient(expert_script(q)))
            check = json.loads(tools.call("check_answer", {"query": q})[0])
        finally:
            conn.close()
        trace = [{"tool": c["name"], "engine": c["engine"],
                  "ms": round(c["ms"]), "ok": c["ok"], "summary": c["summary"]}
                 for c in run_.trace]
        vals = run_.answer if run_.answer is not None else np.empty(0)
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        if vals.size:
            hi = float(np.quantile(vals, 0.995))
            ax.hist(vals, bins=50, range=(min(0.0, float(vals.min())), hi),
                    histtype="stepfilled", color="#2F5E9E", alpha=0.85,
                    edgecolor="#22477A")
        ax.set_title(TASKS[q][0], fontsize=10)
        ax.set_ylabel("entries")
        fig.tight_layout()
        verdict = "PASS" if check.get("passed") else "FAIL"
        return mo.vstack([
            mo.md(f"**{TASKS[q][1]}**"),
            mo.ui.table(trace, selection=None, pagination=False),
            mo.hstack([fig, mo.stat(value=verdict, label=f"check_answer {q}",
                                    caption=check.get("reason", ""))],
                      align="center"),
            mo.md(f"{vals.size:,} values · answered via "
                  f"{(run_.answer_info or {}).get('route', '–')} · "
                  f"{run_.wall_s:.1f} s"),
        ])

    if run.value:
        with mo.status.spinner(title=f"Running {question.value} through the tools ..."):
            replay_result = _replay(question.value)
    else:
        replay_result = mo.md("_Pick a question and press the button._")
    return (replay_result,)


@app.cell
def _(mo, question, run):
    replay_controls = mo.vstack([
        mo.md("The scripted reference solution, driven through the same tools the "
              "agent uses: no model, no network. On the full 49 M-event record, "
              "Q6 takes about 5 minutes."),
        mo.hstack([question, run], justify="start"),
    ])
    return (replay_controls,)


@app.cell
def _(mo):
    plots_scope = mo.ui.radio(options=["this session", "all"], value="this session",
                              label="show plots from", inline=True)
    return (plots_scope,)


@app.cell
def _(mo, os, pathlib, plot_dir, plots_scope, refresh, started, time):
    refresh.value  # re-run on every tick, like the scoreboard
    _since = started if plots_scope.value == "this session" else 0.0
    _pngs = sorted((p for p in pathlib.Path(plot_dir.value).expanduser().glob("*.png")
                    if os.path.getmtime(p) >= _since),
                   key=os.path.getmtime, reverse=True)
    _cards = [mo.vstack([
        mo.image(src=str(p), width=420),
        mo.md(f"`{p.name}` · {time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(p)))}"),
    ]) for p in _pngs]
    plots_view = mo.vstack([
        mo.hstack([mo.md(f"**{len(_pngs)}** agent plot(s), newest first"),
                   plots_scope], justify="space-between"),
        mo.hstack(_cards, wrap=True, justify="start", gap=1.5) if _cards
        else mo.md("_No agent plots yet. They appear here as the agent calls "
                   "plot_result._"),
    ])
    return (plots_view,)


@app.cell
def _(mo, plots_view, replay_controls, replay_result, scoreboard_view):
    mo.ui.tabs({"Live scoreboard": scoreboard_view,
                "Plots": plots_view,
                "Replay (no agent)": mo.vstack([replay_controls, replay_result])})
    return


if __name__ == "__main__":
    app.run()
