import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from awkward_monetizer import adl
    from awkward_monetizer.db import open_server
    from awkward_monetizer.reconstruct import reconstruct_multi

    return adl, mo, np, open_server, pd, plt, reconstruct_multi


@app.cell
def _(mo):
    mo.md(r"""
    # ADL Q4 — MET of events with ≥ 2 jets (pT > 40 GeV)

    Reads the `nanoaod` tables from the running MonetDB **`hep`** database,
    rebuilds the nested (NF2) event structure in Awkward with
    `reconstruct_multi` (per-event counts + `ak.unflatten` — there is no
    `ak.group_by`), and runs the validated `adl.q4_met_ge2jets40`.

    Populate the DB first, e.g.
    `awkward-monetizer ingest nano.root --dataset nanoaod --database hep`.
    """)
    return


@app.cell
def _(open_server, pd):
    conn = open_server(database="hep")

    def fetch(sql):
        cur = conn.cursor()
        cur.execute(sql)
        return pd.DataFrame(cur.fetchall(),
                            columns=[c[0] for c in cur.description])

    return (fetch,)


@app.cell
def _(fetch, reconstruct_multi):
    # events carries met_pt/met_phi; jets are exploded (one row per jet).
    events_df = fetch("SELECT * FROM events")
    jets_df = fetch("SELECT * FROM jets ORDER BY event_id, jet_index")
    # reconstruct_multi reindexes jet counts onto every event (jetless events
    # get an empty list), so no events are silently dropped.
    events = reconstruct_multi(events_df, {"jets": jets_df})
    return (events,)


@app.cell
def _(adl, events):
    met = adl.q4_met_ge2jets40(events)   # MET of events with >=2 jets pT>40
    return (met,)


@app.cell
def _(met, mo, np, plt):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(met, bins=50, range=(0, 200), histtype="stepfilled",
            color="#3b6fb6", alpha=0.85, edgecolor="#26456e")
    ax.set_xlabel("MET [GeV]")
    ax.set_ylabel("events")
    ax.set_title("Q4: MET, ≥2 jets pT > 40 GeV")
    fig.tight_layout()
    mo.vstack([
        mo.md(f"**{met.size:,}** selected events "
              f"(mean MET {np.mean(met):.1f} GeV)" if met.size else "no events"),
        fig,
    ])
    return


if __name__ == "__main__":
    app.run()
