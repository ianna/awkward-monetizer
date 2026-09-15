# marimo: analysis_q4.py
import marimo

app = marimo.App()

@app.cell
def imports():
    import pymonetdb
    import pandas as pd
    import pyarrow as pa
    import awkward as ak
    return pymonetdb, pd, pa, ak

@app.cell
def connect(pymonetdb):
    conn = pymonetdb.connect(database="hep")
    cur = conn.cursor()
    cur

@app.cell
def load_events(cur, pd, pa, ak):
    cur.execute("SELECT event_id, met_pt FROM events")
    df = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
    events = ak.from_arrow(pa.Table.from_pandas(df))
    events

@app.cell
def load_jets(cur, pd, pa, ak):
    cur.execute("SELECT event_id, jet_index, pt FROM jets")
    df = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
    jets = ak.from_arrow(pa.Table.from_pandas(df))
    jets

@app.cell
def group_jets(jets, ak):
    grouped = ak.group_by(jets, "event_id")
    jets_by_event = grouped["values"]
    event_ids = grouped["keys"]["event_id"]
    jets_by_event, event_ids

@app.cell
def q4_selection(jets_by_event, event_ids, ak):
    good = jets_by_event.pt > 40
    n_good = ak.sum(good, axis=1)
    mask = n_good >= 2
    selected = event_ids[mask]
    selected

@app.cell
def output(selected):
    selected
    "Q4 complete."

app.run()

