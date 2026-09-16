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
        # ADL Q6 — pT of the trijet with mass closest to 172.5 GeV

        For events with ≥ 3 jets, take every 3-jet combination
        (`ak.combinations(jets, 3)`), pick the one whose **invariant mass** —
        the mass of the summed 4-momenta, via scikit-hep `vector`, *not* the sum
        of the jet masses — is closest to 172.5 GeV, and plot its pT. Uses the
        validated `adl.q6_trijet_pt`.
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
    events_df = fetch("SELECT * FROM events")
    jets_df = fetch("SELECT * FROM jets ORDER BY event_id, jet_index")
    events = reconstruct_multi(events_df, {"jets": jets_df})
    return events, events_df, jets_df


@app.cell
def _(adl, events):
    trijet_pt = adl.q6_trijet_pt(events)
    return (trijet_pt,)


@app.cell
def _(mo, np, plt, trijet_pt):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(trijet_pt, bins=50, range=(0, 400), histtype="stepfilled",
            color="#3b6fb6", alpha=0.85, edgecolor="#26456e")
    ax.set_xlabel("trijet pT [GeV]")
    ax.set_ylabel("events")
    ax.set_title("Q6: pT of the trijet with m closest to 172.5 GeV")
    fig.tight_layout()
    mo.vstack([
        mo.md(f"**{trijet_pt.size:,}** events with ≥3 jets "
              f"(mean trijet pT {np.mean(trijet_pt):.1f} GeV)"
              if trijet_pt.size else "no events with ≥3 jets"),
        fig,
    ])
    return ax, fig


if __name__ == "__main__":
    app.run()
