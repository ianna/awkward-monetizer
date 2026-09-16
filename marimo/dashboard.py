# marimo: dashboard.py
import marimo

app = marimo.App()

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
@app.cell
def imports():
    import pymonetdb
    import awkward as ak
    import pyarrow as pa
    import pyarrow.parquet as pq
    import numpy as np
    return pymonetdb, ak, pa, pq, np

# ------------------------------------------------------------
# Database connection
# ------------------------------------------------------------
@app.cell
def connect(pymonetdb):
    conn = pymonetdb.connect(database="hep")
    cur = conn.cursor()
    cur

# ------------------------------------------------------------
# Dashboard Controls
# ------------------------------------------------------------
@app.cell
def controls():
    import marimo as mo

    analysis = mo.ui.dropdown(
        label="Select analysis",
        options=[
            "Q4: ≥2 jets with pt > 40",
            "Q6: Trijet closest to 172.5 GeV",
            "Q7: ΔR lepton veto",
            "Q8: Z-like lepton pair + MT",
        ],
        value="Q4: ≥2 jets with pt > 40",
    )

    run_button = mo.ui.button("Run analysis")

    analysis, run_button

# ------------------------------------------------------------
# Load events (Arrow → Awkward)
# ------------------------------------------------------------
@app.cell
def load_events(cur, pa, ak):
    cur.execute("SELECT event_id, met_pt, met_phi FROM events")
    rows = cur.fetchall()
    table = pa.Table.from_pylist(rows)
    events = ak.from_arrow(table)
    events

# ------------------------------------------------------------
# Load jets (Arrow → Awkward)
# ------------------------------------------------------------
@app.cell
def load_jets(cur, pa, ak):
    cur.execute("SELECT event_id, jet_index, pt, eta, phi, mass FROM jets")
    rows = cur.fetchall()
    table = pa.Table.from_pylist(rows)
    jets = ak.from_arrow(table)
    jets

# ------------------------------------------------------------
# Load leptons (Arrow → Awkward)
# ------------------------------------------------------------
@app.cell
def load_leptons(cur, pa, ak):
    try:
        cur.execute("SELECT event_id, lep_index, pt, eta, phi, charge FROM leptons")
        rows = cur.fetchall()
        table = pa.Table.from_pylist(rows)
        leps = ak.from_arrow(table)
    except:
        leps = ak.Array([])
    leps

# ------------------------------------------------------------
# Group jets & leptons by event
# ------------------------------------------------------------
@app.cell
def group_objects(jets, load_leptons, ak):
    jets_g = ak.group_by(jets, "event_id")
    jets_by_event = jets_g["values"]
    event_ids = jets_g["keys"]["event_id"]

    if len(load_leptons) > 0:
        leps_g = ak.group_by(load_leptons, "event_id")
        leps_by_event = leps_g["values"]
    else:
        leps_by_event = ak.Array([])

    jets_by_event, leps_by_event, event_ids

# ------------------------------------------------------------
# Physics helpers
# ------------------------------------------------------------
@app.cell
def physics_helpers(np):
    def delta_r(j, l):
        return np.sqrt((j.eta - l.eta)**2 + (j.phi - l.phi)**2)

    def invariant_mass_trijet(j):
        return ak.sum(j.mass, axis=-1)

    delta_r, invariant_mass_trijet

# ------------------------------------------------------------
# Unified Analysis Logic
# ------------------------------------------------------------
@app.cell
def run_analysis(analysis, run_button, jets_by_event, leps_by_event, event_ids,
                 physics_helpers, ak, np, load_events):
    if not run_button.value:
        return "Select an analysis and press Run."

    delta_r, invariant_mass_trijet = physics_helpers

    if analysis.value.startswith("Q4"):
        good = jets_by_event.pt > 40
        n_good = ak.sum(good, axis=1)
        selected = event_ids[n_good >= 2]
        return selected

    if analysis.value.startswith("Q6"):
        trijets = ak.combinations(jets_by_event, 3, axis=1)
        masses = invariant_mass_trijet(trijets)
        best_idx = ak.argmin(abs(masses - 172.5), axis=1)
        best = trijets[best_idx]
        return best

    if analysis.value.startswith("Q7"):
        if len(leps_by_event) == 0:
            return "No leptons table found."

        jets30 = jets_by_event.pt > 30
        leps10 = leps_by_event.pt > 10

        dr_matrix = delta_r(
            jets_by_event[:, None],
            leps_by_event[None, :]
        )

        veto = ak.any(dr_matrix < 0.4, axis=-1)
        good_jets = jets30 & (~veto)

        ht = ak.sum(jets_by_event.pt * good_jets, axis=1)
        return ht

    if analysis.value.startswith("Q8"):
        if len(leps_by_event) == 0:
            return "No leptons table found."

        pairs = ak.combinations(leps_by_event, 2, axis=1)
        opp = pairs["0"].charge != pairs["1"].charge
        mass = ak.sum(pairs["0"].pt + pairs["1"].pt, axis=-1)
        best = ak.argmin(abs(mass - 91.2), axis=1)

        max_lep = ak.argmax(leps_by_event.pt, axis=1)
        mt = load_events.met_pt + leps_by_event.pt[max_lep]
        return mt

    return "Unknown analysis."

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------
@app.cell
def output(run_analysis):
    run_analysis

app.run()

