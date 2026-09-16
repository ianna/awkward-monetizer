import marimo

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
    mo.md(
        r"""
        # ADL Q8 — transverse mass of MET + lead lepton outside the best Z pair

        Events with ≥ 3 muons and a same-flavour opposite-charge (SFOS) pair:
        pick the SFOS pair with mass closest to the Z (91.19 GeV), then compute the
        **transverse mass** MT = √(2·pT_ℓ·MET·(1 − cos Δφ)) of MET and the
        highest-pT muon *not* in that pair. `adl.q8_mt_met_lepton` does the real
        invariant mass and MT (the earlier notebook summed pT as a placeholder).
        Only muons are available, so SFOS = opposite-charge muon pair.
        """
    )


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
    # events carries met_pt/met_phi (used by the transverse mass).
    events_df = fetch("SELECT * FROM events")
    muons_df = fetch("SELECT * FROM muons ORDER BY event_id, muon_index")
    events = reconstruct_multi(events_df, {"muons": muons_df})
    return events, events_df, muons_df


@app.cell
def _(adl, events):
    mt = adl.q8_mt_met_lepton(events)
    return (mt,)


@app.cell
def _(mo, mt, np, plt):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(mt, bins=50, range=(0, 200), histtype="stepfilled",
            color="#3b6fb6", alpha=0.85, edgecolor="#26456e")
    ax.set_xlabel(r"$M_T$(MET, lead lepton) [GeV]")
    ax.set_ylabel("events")
    ax.set_title("Q8: transverse mass, MET + lead lepton outside the Z pair")
    fig.tight_layout()
    mo.vstack([
        mo.md(f"**{mt.size:,}** events with ≥3 muons + an SFOS pair "
              f"(mean MT {np.mean(mt):.1f} GeV)" if mt.size else "no events"),
        fig,
    ])
    return ax, fig


if __name__ == "__main__":
    app.run()
