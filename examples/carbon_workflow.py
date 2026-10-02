"""Energy and carbon footprint of the awkward-monetizer workflow (carbontracker).

Measures each stage of the pipeline as one carbontracker "epoch":

  idle      the machine doing nothing          -> baseline power, subtracted
  ingest    ROOT -> flat tables -> MonetDB     (shards ingest, idempotent)
  fetch     SQL slice + Awkward NF2 rebuild    (scatter-gather fetch_events)
  analysis  ADL Q1-Q8 on the rebuilt events    (pure Awkward)

Each stage is repeated until it has run for at least --min-seconds, because
power is sampled about once per second; results are also given per single
run. carbontracker measures the *whole machine* (Apple Silicon: `sudo
powermetrics`; Linux: Intel RAPL; otherwise a TDP estimate), so the MonetDB
server's work on this machine is included -- and so is anything else running,
which is what the idle baseline corrects for. Keep other apps quiet.

Setup (MonetDB running, pixi env active):

    awkward-monetizer manifest data/nano_synth.root data/nano_bench.root --out manifest.json --fresh
    awkward-monetizer shards init --manifest manifest.json --shard localhost/hep0 --shard localhost/hep1
    sudo -v            # macOS: powermetrics needs root; caches your password
    python examples/carbon_workflow.py --shards shards.json --manifest manifest.json

A single database works too: ``--db localhost/hep`` instead of ``--shards``.
Results are printed and written to --out (JSON); carbontracker's own logs go
to --log-dir. ``--pdf`` also renders carbontracker's PDF report (needs
reportlab >= 4; carbontracker's ``[pdfreport]`` extra pins reportlab 3.6.13,
which has no Python 3.12 wheels). In that report "Epoch N" is stage N, its bars
are gross (no idle subtraction) and it quotes carbontracker's PUE; the stage
table printed here is the one to quote.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass


# --------------------------------------------------------------------------
# Power-measurement access
# --------------------------------------------------------------------------
def check_power_access() -> str:
    """Return a description of how power will be measured; exit if the
    measurement would silently hang or fail."""
    system = platform.system()
    if system == "Darwin":
        # carbontracker runs `sudo powermetrics` every second with stderr
        # hidden: without cached credentials it would wait for a password.
        if subprocess.run(["sudo", "-n", "true"], capture_output=True,
                          check=False).returncode:
            sys.exit("macOS power readings need `sudo powermetrics`. Run `sudo -v` "
                     "first (caches your password), then re-run this script.")
        return "Apple Silicon powermetrics (whole CPU package)"
    if system == "Linux":
        rapl = glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj")
        if rapl and all(os.access(p, os.R_OK) for p in rapl):
            return "Intel RAPL (whole CPU package)"
        print("warning: Intel RAPL counters not readable -- carbontracker will "
              "fall back to a TDP-based *estimate*, not a measurement.\n"
              "  (sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj)",
              file=sys.stderr)
        return "TDP estimate (no RAPL access)"
    return "TDP estimate"


def write_pdf_report(log_dir: str, pdf_path: str) -> None:
    """Render carbontracker's PDF report for the most recent run in log_dir.

    carbontracker 2.4.7 logs "fetched every N s at detected location: LOC: [..]"
    but its report parser expects "... location LOC: [..]" (no colon) and then
    fails with "No carbon intensity data found". A normalised copy is used.
    """
    import re
    import tempfile

    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        from carbontracker.parser import get_most_recent_logs
        from carbontracker.report import REPORTLAB_AVAILABLE, generate_report_from_log
    except ImportError as e:
        print(f"PDF report skipped: {e}", file=sys.stderr)
        return
    if not REPORTLAB_AVAILABLE:
        print("PDF report skipped: install reportlab>=4 (pixi install picks it up)",
              file=sys.stderr)
        return
    std_log, _ = get_most_recent_logs(log_dir)
    with open(std_log) as fh:
        text = fh.read()
    text = re.sub(r"(fetched every \d+ s at detected location): ", r"\1 ", text)
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as tmp:
        tmp.write(text)
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # "Parsing Log" chatter
            generate_report_from_log(tmp.name, pdf_path)
    finally:
        os.unlink(tmp.name)
    print(f"wrote {pdf_path} (carbontracker report; Epoch N = stage N)")


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
@dataclass
class StageResult:
    stage: str
    runs: int
    seconds: float
    energy_kwh: float           # measured, whole machine, x PUE
    avg_power_w: float
    net_energy_kwh: float       # minus idle baseline power x duration
    net_energy_kwh_per_run: float
    co2_g: float                # of net energy
    co2_g_per_run: float


def repeat(fn, min_seconds: float, max_runs: int):
    """Call fn until min_seconds have elapsed (at least once)."""
    runs, start, out = 0, time.perf_counter(), None
    while True:
        out = fn()
        runs += 1
        if time.perf_counter() - start >= min_seconds or runs >= max_runs:
            return runs, out


def build_shard_map(args):
    from awkward_monetizer.shards import Shard, ShardMap
    if args.shards:
        return ShardMap.from_json(args.shards)
    hostport, _, database = args.db.partition("/")
    host, _, port = hostport.partition(":")
    return ShardMap([Shard("db", (0, 1 << 23), host=host or "localhost",
                           port=int(port or 50000), database=database or "hep")])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--shards", help="shards.json (see `awkward-monetizer shards init`)")
    src.add_argument("--db", help="single database: host[:port]/database")
    p.add_argument("--manifest", required=True, help="manifest.json of input files")
    p.add_argument("--dataset", default="nanoaod")
    p.add_argument("--where", default="met_pt > 25",
                   help="SQL selection for the fetch/analysis stages")
    p.add_argument("--load-method", default="auto",
                   choices=("auto", "binary", "client", "copy", "insert"))
    p.add_argument("--stages", default="idle,ingest,fetch,analysis")
    p.add_argument("--min-seconds", type=float, default=15.0,
                   help="minimum measured time per stage (repeats the stage)")
    p.add_argument("--max-runs", type=int, default=1000)
    p.add_argument("--interval", type=float, default=1.0,
                   help="power sampling interval (s)")
    p.add_argument("--pue", type=float, default=1.0,
                   help="power usage effectiveness applied to the energy: 1.0 for "
                        "a laptop/desktop; carbontracker's data-centre default is 1.58")
    p.add_argument("--components", default="cpu", help="carbontracker components")
    p.add_argument("--electricitymaps-key", default=os.environ.get("ELECTRICITYMAPS_KEY"),
                   help="live carbon intensity (else the country average)")
    p.add_argument("--log-dir", default="carbon_logs")
    p.add_argument("--out", default="carbon_workflow.json")
    p.add_argument("--pdf", nargs="?", const="carbon_report.pdf", default=None,
                   metavar="PATH", help="also write carbontracker's PDF report "
                                        "(default path carbon_report.pdf)")
    args = p.parse_args(argv)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - {"idle", "ingest", "fetch", "analysis"}
    if unknown:
        p.error(f"unknown stages: {sorted(unknown)}")
    if "analysis" in stages and "fetch" not in stages:
        p.error("the analysis stage needs the fetch stage (it analyses its events)")

    method = check_power_access()

    from carbontracker import constants
    from carbontracker.tracker import CarbonTracker

    from awkward_monetizer import adl, db
    from awkward_monetizer.datasets import DATASETS
    from awkward_monetizer.keys import Manifest
    from awkward_monetizer.shards import fetch_events, ingest_manifest

    ds = DATASETS[args.dataset]
    smap = build_shard_map(args)
    manifest = Manifest.from_json(args.manifest)

    # Unmeasured setup: fresh schema + one load, so fetch/analysis have data
    # even if ingest isn't measured, and the ingest stage measures steady-state
    # (idempotent replace) loads rather than DDL.
    print(f"setup: loading {len(manifest.files)} file(s) into {len(smap)} shard(s) ...")
    for sh in smap:
        conn = sh.connect()
        try:
            db.create_schema(conn, db.read_schema(ds.name))
        finally:
            conn.close()
    def ingest():
        with contextlib.redirect_stdout(io.StringIO()):   # ingest prints per chunk
            return ingest_manifest(manifest, smap, ds, method=args.load_method)

    n_events = sum(ingest().values())
    state = {}

    def fetch():
        state["events"] = fetch_events(smap, where=args.where)
        return len(state["events"])

    actions = {
        "idle": lambda: time.sleep(args.min_seconds),
        "ingest": ingest,
        "fetch": fetch,
        "analysis": lambda: adl.run_all(state["events"]),
    }

    os.makedirs(args.log_dir, exist_ok=True)
    api_keys = ({"electricitymaps": args.electricitymaps_key}
                if args.electricitymaps_key else None)
    # one carbontracker "epoch" per stage; monitor one more than we run so the
    # tracker stays alive until the per-stage numbers have been read
    tracker = CarbonTracker(epochs=len(stages) + 1, monitor_epochs=len(stages) + 1,
                            epochs_before_pred=0, components=args.components,
                            update_interval=args.interval, log_dir=args.log_dir,
                            api_keys=api_keys, verbose=1)
    runs = {}
    for stage in stages:
        print(f"measuring {stage} (>= {args.min_seconds:g} s) ...", flush=True)
        tracker.epoch_start()
        runs[stage], _ = repeat(actions[stage], args.min_seconds, args.max_runs)
        tracker.epoch_end()

    # carbontracker applies a data-centre PUE; replace it with --pue
    energy = list(tracker.tracker.total_energy_per_epoch()
                  / constants.PUE_2023 * args.pue)
    seconds = list(tracker.tracker.epoch_times)
    intensity = float(tracker.intensity_updater.average_carbon_intensity())
    tracker.stop()

    idle_power_kw = 0.0
    if "idle" in stages:
        i = stages.index("idle")
        idle_power_kw = energy[i] / (seconds[i] / 3600.0)

    results = []
    for i, stage in enumerate(stages):
        e, t, n = float(energy[i]), float(seconds[i]), runs[stage]
        net = e if stage == "idle" else max(e - idle_power_kw * t / 3600.0, 0.0)
        results.append(StageResult(
            stage=stage, runs=n, seconds=t, energy_kwh=e,
            avg_power_w=e * 1000.0 / (t / 3600.0) if t else 0.0,
            net_energy_kwh=net, net_energy_kwh_per_run=net / n,
            co2_g=net * intensity, co2_g_per_run=net * intensity / n))

    print(f"\npower measurement: {method}; carbon intensity {intensity:.1f} gCO2eq/kWh; "
          f"PUE {args.pue:g}; {n_events} events on {len(smap)} shard(s)")
    print(f"{'stage':9s} {'runs':>5s} {'time s':>8s} {'avg W':>7s} "
          f"{'net J/run':>11s} {'net mg CO2e/run':>16s}")
    for r in results:
        print(f"{r.stage:9s} {r.runs:5d} {r.seconds:8.1f} {r.avg_power_w:7.2f} "
              f"{r.net_energy_kwh_per_run * 3.6e6:11.2f} {r.co2_g_per_run * 1e3:16.4f}")
    if "idle" in stages:
        print("(idle row: baseline power; other rows have it subtracted)")
    if method.startswith("TDP"):
        print("NOTE: power is a fixed TDP *estimate*, identical for every stage, so "
              "net energy above idle is not meaningful here.")

    with open(args.out, "w") as fh:
        json.dump({
            "power_measurement": method, "platform": platform.platform(),
            "carbon_intensity_g_per_kwh": intensity, "pue": args.pue,
            "shards": len(smap), "events": n_events, "where": args.where,
            "load_method": args.load_method,
            "epoch_to_stage": {str(i + 1): s for i, s in enumerate(stages)},
            "stages": [asdict(r) for r in results],
        }, fh, indent=2)
    print(f"\nwrote {args.out}; carbontracker logs in {args.log_dir}/")
    if args.pdf:
        write_pdf_report(args.log_dir, args.pdf)


if __name__ == "__main__":
    main()
