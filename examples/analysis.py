import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import awkward as ak
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        import mplhep as hep
        plt.style.use(hep.style.CMS)
    except ImportError:
        hep = None

    from awkward_monetizer.physics import invariant_mass, opposite_charge_pair
    from awkward_monetizer.reconstruct import (
        fetch_from_monetdb,
        reconstruct_events,
        tables_from_root,
    )

    return (
        ak,
        fetch_from_monetdb,
        invariant_mass,
        mo,
        np,
        opposite_charge_pair,
        plt,
        reconstruct_events,
        tables_from_root,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Dimuon analysis — Awkward + MonetDB round-trip

    1. **Ingest** `data/cms.root` into flat tables (`events`, `muons`).
    2. **Reconstruct** the per-event nested (NF2) structure (`ak.unflatten`).
    3. **Analyze**: dimuon invariant mass, validated against the file's `M`,
       and the Z→μμ spectrum.

    Set `SOURCE = "monetdb"` (after loading the schema and running ingest) to
    use the SQL round-trip instead of reading the file directly.
    """)
    return


@app.cell
def _(fetch_from_monetdb, reconstruct_events, tables_from_root):
    SOURCE = "root"  # "root" or "monetdb"

    if SOURCE == "monetdb":
        events_df, muons_df = fetch_from_monetdb(
            database="hep", where="mass BETWEEN 60 AND 120"
        )
    else:
        events_df, muons_df = tables_from_root("data/cms.root", "dimuon")

    events = reconstruct_events(events_df, muons_df)
    return events, events_df, muons_df


@app.cell
def _(events_df, mo, muons_df):
    mo.md(
        f"Loaded **{len(events_df):,}** events and **{len(muons_df):,}** muons; "
        f"reconstructed into a jagged `events.muons` array."
    )
    return


@app.cell
def _(ak, events, invariant_mass, mo, np):
    _recon = ak.to_numpy(invariant_mass(events.muons))
    _stored = ak.to_numpy(events.mass)
    _max = np.max(np.abs(_recon - _stored))
    mo.md(
        f"**Validation** — reconstructed m(μμ) vs stored `M`: "
        f"max |Δ| = `{_max:.2e}` GeV across {len(_recon):,} events. ✓"
    )
    return


@app.cell
def _(mo):
    pt_cut = mo.ui.slider(0, 40, value=20, step=1, label="min muon pT [GeV]")
    charge_only = mo.ui.checkbox(value=True, label="opposite-charge pairs only")
    mo.hstack([pt_cut, charge_only], justify="start", gap=2)
    return charge_only, pt_cut


@app.cell
def _(
    ak,
    charge_only,
    events,
    invariant_mass,
    np,
    opposite_charge_pair,
    pt_cut,
):
    _both_pt = ak.all(events.muons.pt > pt_cut.value, axis=1)
    _sel = _both_pt
    if charge_only.value:
        _sel = _sel & opposite_charge_pair(events)
    mass_sel = ak.to_numpy(invariant_mass(events.muons))[ak.to_numpy(_sel)]
    n_sel = int(np.count_nonzero(ak.to_numpy(_sel)))
    return mass_sel, n_sel


@app.cell
def _(charge_only, mass_sel, mo, n_sel, plt, pt_cut):
    fig, (ax_full, ax_z) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (lo, hi, title) in zip(
        (ax_full, ax_z), [(0, 120, "Full range"), (60, 120, "Z region")], strict=False
    ):
        ax.hist(mass_sel, bins=100, range=(lo, hi),
                histtype="stepfilled", color="#3b6fb6", alpha=0.85,
                edgecolor="#26456e")
        if lo <= 91.19 <= hi:
            ax.axvline(91.19, ls="--", lw=1.2, color="#c0392b", label="Z: 91.19 GeV")
            ax.legend()
        ax.set_xlabel(r"$m(\mu\mu)$ [GeV]")
        ax.set_ylabel("events")
        ax.set_title(title)
    fig.tight_layout()

    _cut = f"both muons pT > {pt_cut.value} GeV"
    _q = "opposite-charge, " if charge_only.value else ""
    mo.vstack([mo.md(f"**{n_sel:,}** selected events ({_q}{_cut})"), fig])
    return


if __name__ == "__main__":
    app.run()
