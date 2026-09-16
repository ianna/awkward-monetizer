import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from awkward_monetizer import adl
    from awkward_monetizer.db import open_server
    from awkward_monetizer.reconstruct import reconstruct_multi
    return adl, mo, np, open_server, pd, plt, reconstruct_multi


@app.cell
def _(mo):
    mo.md(
        r"""
        # ADL dashboard — MonetDB → Awkward

        Pick an ADL query; the plot updates reactively. Reads the `nanoaod` tables
        from the running MonetDB **`hep`** server, rebuilds the nested event
        structure with `reconstruct_multi`, and runs the validated `adl` query
        (no `ak.group_by`, real invariant mass / ΔR / transverse mass). Populate
        the DB first: `awkward-monetizer ingest nano.root --dataset nanoaod
        --database hep --create-schema`.
        """
    )
    return


@app.cell
def _(open_server, pd, reconstruct_multi):
    conn = open_server(database="hep")

    def fetch(sql):
        cur = conn.cursor()
        cur.execute(sql)
        return pd.DataFrame(cur.fetchall(),
                            columns=[c[0] for c in cur.description])

    # Reconstruct once; every query below reuses this jagged `events` array.
    events = reconstruct_multi(
        fetch("SELECT * FROM events"),
        {"jets": fetch("SELECT * FROM jets ORDER BY event_id, jet_index"),
         "muons": fetch("SELECT * FROM muons ORDER BY event_id, muon_index")},
    )
    return conn, events, fetch


@app.cell
def _(adl, mo):
    # label -> (adl function, x-axis label, histogram upper edge)
    ANALYSES = {
        "Q4: MET, ≥2 jets pT>40":   (adl.q4_met_ge2jets40, "MET [GeV]", 200),
        "Q6: trijet pT (m~172.5)":  (adl.q6_trijet_pt, "trijet pT [GeV]", 400),
        "Q7: HT of cleaned jets":   (adl.q7_ht_cleaned, "HT [GeV]", 600),
        "Q8: MT(MET, lead lepton)": (adl.q8_mt_met_lepton, r"$M_T$ [GeV]", 200),
    }
    choice = mo.ui.dropdown(options=list(ANALYSES),
                            value="Q4: MET, ≥2 jets pT>40", label="Analysis")
    choice
    return ANALYSES, choice


@app.cell
def _(ANALYSES, choice, events, mo, np, plt):
    fn, xlabel, hi = ANALYSES[choice.value]
    result = fn(events)

    fig, ax = plt.subplots(figsize=(7, 4))
    if result.size:
        ax.hist(result, bins=50, range=(0, hi), histtype="stepfilled",
                color="#3b6fb6", alpha=0.85, edgecolor="#26456e")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("events")
    ax.set_title(choice.value)
    fig.tight_layout()

    summary = (f"**{result.size:,}** events (mean {np.mean(result):.1f})"
               if result.size else "no events")
    mo.vstack([mo.md(summary), fig])
    return ax, fig, fn, hi, result, summary, xlabel


if __name__ == "__main__":
    app.run()
