"""Where the time goes in the ADL Q1-Q8 run on a MonetDB database.

Runs the scripted expert's SQL and Awkward code the same way the agent tools
do, and splits each query's wall time into stages:

  query     MonetDB executes the query (plus the first batch of rows)
  transfer  remaining rows move from MonetDB into Python
  convert   rows become NumPy columns (no pandas)
  arrays    columns become nested Awkward arrays
  compute   the analysis code itself (Awkward / vector)

For Q1-Q4 it also times the same query answered inside MonetDB with only a
COUNT and SUM sent back, i.e. the cost of the selection without the transfer.

Usage: python examples/time_breakdown.py [--database hep2016h] [Q1 Q2 ...]
(default: all eight). Results are also written to output/time_breakdown.json.
"""
import json
import os
import sys
import time

from awkward_monetizer import db
from awkward_monetizer.agent.scripted import _AWK, _SQL
from awkward_monetizer.agent.tools import COLLECTIONS, AnalysisTools, check_code
from awkward_monetizer.reconstruct import _fetch_binary, fetch_events

OUT = "output/time_breakdown.json"
STAGES = ("query", "transfer", "convert", "arrays", "compute")


def sql_query(conn, q):
    sql = _SQL[q]
    cur = conn.cursor()
    t0 = time.perf_counter()
    col = sql.split()[1]                      # met_pt or pt
    cur.execute(f"SELECT COUNT(*), SUM({col}) FROM ({sql}) AS t")
    n_in_db, _ = cur.fetchall()[0]
    in_db = time.perf_counter() - t0
    t0 = time.perf_counter()
    cur.execute(sql)
    t1 = time.perf_counter()
    fast = _fetch_binary(cur, None)
    if fast is not None:
        n = len(next(iter(fast.values())))
    else:
        n = 0
        while True:
            rows = cur.fetchmany(1_000_000)
            if not rows:
                break
            n += len(rows)
    t2 = time.perf_counter()
    cur.close()
    assert n == n_in_db
    return {"engine": "SQL", "values": n, "events": None,
            "in_db_only": in_db,
            "stages": {"query": t1 - t0, "transfer": t2 - t1}}


def awkward_query(tools, q):
    where, code = _AWK[q]
    tree = check_code(code)
    proj = tools._projection(tree)
    st = {}
    n_values = n_events = 0
    for rng in tools.ranges():
        t = {}
        events = fetch_events(tools.conn, where, event_columns=proj["events"],
                              collections={c: proj[c] for c in COLLECTIONS
                                           if c in proj},
                              event_range=rng, timings=t)
        for k, v in (("query", "execute"), ("transfer", "fetch"),
                     ("convert", "numpy"), ("arrays", "array_build")):
            st[k] = st.get(k, 0) + t.get(v, 0)
        if len(events) == 0:
            continue
        t0 = time.perf_counter()
        vals = tools._run_code(tree, events)
        st["compute"] = st.get("compute", 0) + time.perf_counter() - t0
        n_values += vals.size
        n_events += len(events)
    return {"engine": "Awkward", "values": n_values, "events": n_events,
            "stages": st}


def main():
    args = sys.argv[1:]
    database = "hep2016h"
    if args[:1] == ["--database"]:
        database, args = args[1], args[2:]
    queries = [a.upper() for a in args] or [f"Q{i}" for i in range(1, 9)]
    conn = db.open_server(database=database)
    tools = AnalysisTools(conn)
    out = {}
    print(f"{'':4} {'engine':8} {'values':>12} " +
          " ".join(f"{s:>9}" for s in STAGES) + f" {'total':>9} {'in-DB only':>11}")
    for q in queries:
        t0 = time.perf_counter()
        r = sql_query(conn, q) if q in _SQL else awkward_query(tools, q)
        r["total"] = time.perf_counter() - t0 - r.get("in_db_only", 0)
        out[q] = r
        s = r["stages"]
        cells = " ".join(f"{s[k]:8.1f}s" if k in s else f"{'-':>9}" for k in STAGES)
        extra = f"{r['in_db_only']:10.2f}s" if "in_db_only" in r else f"{'-':>11}"
        print(f"{q:4} {r['engine']:8} {r['values']:12,} {cells} {r['total']:8.1f}s {extra}",
              flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
