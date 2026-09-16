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
        # ADL Q7 — HT of jets far from any lepton

        Scalar sum of pT over jets with pT > 30, |η| < 2.4 that are **not** within
        ΔR < 0.4 of a muon with pT > 10. `adl.q7_ht_cleaned` builds the jet×lepton
        pairs with `ak.cartesian(..., nested=True)` and uses `vector`'s `.deltaR`
        (which wraps Δφ correctly) — the schema's lepton collection is `muons`,
        not a `leptons` table.
        """
    )
    return


@app.cell
def _(open_server, pd):
    conn = open_server(database="hep")

    def fetch(sql):
        cur = conn.cursor()
        cur.execute(sql)
        return pd.DataFrame(cur.fetchall(),
                            columns=[c[0] for c in cur.description])
    return conn, fetch


@app.cell
def _(fetch, reconstruct_multi):
    events_df = fetch("SELECT * FROM events")
    jets_df = fetch("SELECT * FROM jets ORDER BY event_id, jet_index")
    muons_df = fetch("SELECT * FROM muons ORDER BY event_id, muon_index")
    events = reconstruct_multi(events_df, {"jets": jets_df, "muons": muons_df})
    return events, events_df, jets_df, muons_df


@app.cell
def _(adl, events):
    ht = adl.q7_ht_cleaned(events)
    return (ht,)


@app.cell
def _(ht, mo, np, plt):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ht, bins=50, range=(0, 600), histtype="stepfilled",
            color="#3b6fb6", alpha=0.85, edgecolor="#26456e")
    ax.set_xlabel("HT [GeV]")
    ax.set_ylabel("events")
    ax.set_title("Q7: HT of lepton-cleaned jets (pT>30, |η|<2.4, ΔR>0.4)")
    fig.tight_layout()
    mo.vstack([
        mo.md(f"**{ht.size:,}** events (mean HT {np.mean(ht):.1f} GeV)"
              if ht.size else "no events"),
        fig,
    ])
    return ax, fig


if __name__ == "__main__":
    app.run()
