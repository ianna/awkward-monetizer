"""Pilot measurement of the ingestion-client memory bound.

The proposal states, for one ingestion worker,

    M_peak <= M_fixed + k * B          (proposed k = 6)

where B is the decoded size of one input batch (the Awkward Array that
``uproot.iterate`` yields, nested buffers included) and M_fixed is the resident
memory after imports, opening the file and connecting, but before the first
batch. It also states that the increment M_peak - M_fixed does not grow with
the total input volume when the batch size is fixed.

This script measures both. Every configuration runs in a fresh process, so one
run's peak cannot leak into the next. Resident memory is sampled by a
background thread every --interval seconds. On macOS the measure is the
physical footprint (what Activity Monitor calls "Memory"), because macOS keeps
freed pages in the resident set size until it needs them, so RSS there only
grows; elsewhere it is RSS. ``--measure rss`` forces RSS; the RSS peak is
recorded in the JSON either way.

Two modes:

  build   (default, no database) read each batch and build the flat tables,
          then discard them. Measures decoding and flattening only.
  load    (--db HOST/DATABASE) also send every batch to a MonetDB server with
          --load-method. Rows are written under --file-id and deleted again at
          the end unless --keep is given. This is the number the proposal's
          bound refers to; the server's own memory is NOT included.

Examples (pixi env active):

    # batch-size scan and tenfold-volume test on a real NanoAOD file, no server
    python examples/memory_pilot.py 127C2975-1B1C-A046-AABF-62B77E757A86.root

    # the same through the binary upload path of a local server
    python examples/memory_pilot.py FILE.root --db localhost/hep0 --load-method binary

Columns: "incr" is M_peak - M_fixed, "kept" is the resident memory still held
after the last batch (also above M_fixed), and k = incr / B max. The memory
after every batch is in the JSON. --gc runs Python's cyclic garbage collector
after each batch: if that lowers the numbers, something is keeping finished
batches alive through a reference cycle.

Results are printed and written to --out (JSON).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time

MIB = 1024 * 1024


def _maxrss_bytes() -> int:
    import resource
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v if platform.system() == "Darwin" else v * 1024  # macOS: bytes


def _darwin_footprint():
    """Return a function giving this process's physical footprint on macOS.

    macOS keeps freed pages in the resident set size until the system needs
    them, so RSS only ever grows there. The physical footprint (the "Memory"
    column of Activity Monitor) leaves those reusable pages out.
    """
    import ctypes

    class RUsageInfoV2(ctypes.Structure):
        _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
            (name, ctypes.c_uint64) for name in (
                "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups",
                "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
                "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
                "ri_proc_exit_abstime", "ri_child_user_time",
                "ri_child_system_time", "ri_child_pkg_idle_wkups",
                "ri_child_interrupt_wkups", "ri_child_pageins",
                "ri_child_elapsed_abstime", "ri_diskio_bytesread",
                "ri_diskio_byteswritten")]

    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.proc_pid_rusage.restype = ctypes.c_int
    pid = os.getpid()
    info = RUsageInfoV2()

    def footprint() -> int:
        if lib.proc_pid_rusage(pid, 2, ctypes.byref(info)):   # RUSAGE_INFO_V2
            raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
        return int(info.ri_phys_footprint)

    footprint()
    return footprint


class _Sampler(threading.Thread):
    """Track the peak memory of this process: physical footprint on macOS,
    resident set size elsewhere (or with --measure rss)."""

    def __init__(self, interval: float, measure: str = "auto"):
        super().__init__(daemon=True)
        import psutil
        self._proc = psutil.Process()
        self._interval = interval
        self._halt = threading.Event()
        self.measure = "rss"
        self._footprint = None
        if measure != "rss" and platform.system() == "Darwin":
            try:
                self._footprint = _darwin_footprint()
                self.measure = "footprint"
            except (OSError, AttributeError) as e:
                print(f"physical footprint unavailable ({e}); using RSS",
                      file=sys.stderr)
        self.peak = self.rss()
        self.peak_rss = self.true_rss()

    def true_rss(self) -> int:
        return self._proc.memory_info().rss

    def rss(self) -> int:
        """The chosen measure (kept under this name for brevity)."""
        return self._footprint() if self._footprint else self.true_rss()

    def poll(self) -> int:
        r = self.rss()
        self.peak = max(self.peak, r)
        self.peak_rss = max(self.peak_rss, self.true_rss())
        return r

    def run(self) -> None:
        while not self._halt.wait(self._interval):
            self.poll()

    def stop(self) -> None:
        self._halt.set()
        self.join()
        self.poll()


def _open_db(spec: str, user: str, password: str):
    from awkward_monetizer import db
    host, _, database = spec.rpartition("/")
    host, _, port = (host or "localhost").partition(":")
    return db.open_server(database=database, host=host,
                          port=int(port) if port else 50000,
                          user=user, password=password)


def worker(args) -> dict:
    """One configuration, in this (fresh) process."""
    import gc

    import awkward as ak
    import uproot

    from awkward_monetizer.datasets import DATASETS
    from awkward_monetizer.ingest import build_tables, load_tables, open_tree

    ds = DATASETS[args.dataset]
    conn = _open_db(args.db, args.user, args.password) if args.db else None
    step = int(args.step) if args.step.isdigit() else args.step

    sampler = _Sampler(args.interval, args.measure)
    f = uproot.open(args.root_file)
    t = open_tree(f, args.tree if args.tree is not None else ds.tree)
    branches = ds.all_branches()
    entries = t.num_entries if args.entry_stop is None else min(
        t.num_entries, args.entry_stop)
    gc.collect()
    m_fixed = sampler.rss()          # imports + open file + connection
    sampler.peak = m_fixed
    sampler.start()

    total = batches = 0
    trace = []                       # resident memory after each batch
    b_max = b_sum = table_max = 0
    t0 = time.perf_counter()
    try:
        for chunk in t.iterate(branches, step_size=step, entry_stop=args.entry_stop):
            # uproot can hand back views into whole baskets; count only the
            # bytes this batch really uses (transient copy, freed before build)
            packed = ak.to_packed(chunk)
            b = int(packed.nbytes)
            del packed
            tables = build_tables(chunk, ds, id_offset=total, file_id=args.file_id)
            tb = sum(int(df.memory_usage(index=True).sum()) for df in tables.values())
            sampler.poll()
            if conn is not None:
                with open(os.devnull, "w") as null:      # "loaded N rows" chatter
                    stdout, sys.stdout = sys.stdout, null
                    try:
                        load_tables(conn, tables, method=args.load_method)
                    finally:
                        sys.stdout = stdout
            sampler.poll()
            total += len(chunk)
            batches += 1
            b_sum += b
            b_max = max(b_max, b)
            table_max = max(table_max, tb)
            del tables, chunk
            if args.gc:
                gc.collect()
            trace.append(sampler.poll())
    finally:
        seconds = time.perf_counter() - t0
        sampler.stop()
        if conn is not None:
            if not args.keep:
                from awkward_monetizer.keys import DEFAULT_LAYOUT
                lo, hi = DEFAULT_LAYOUT.file_id_range(args.file_id)
                cur = conn.cursor()
                for name in [c.table for c in ds.collections] + ["events"]:
                    cur.execute(f"DELETE FROM {name} WHERE event_id >= {lo} "
                                f"AND event_id < {hi}")
                conn.commit()
            conn.close()

    peak = max(sampler.peak, m_fixed)
    incr = peak - m_fixed
    return {
        "step_size": args.step, "entry_stop": args.entry_stop,
        "mode": "load:" + args.load_method if args.db else "build",
        "events": total, "entries_available": int(entries), "batches": batches,
        "decoded_bytes_total": b_sum, "batch_bytes_max": b_max,
        "tables_bytes_max": table_max,
        "m_fixed": m_fixed, "m_peak_sampled": peak, "m_peak_os": _maxrss_bytes(),
        "increment": incr, "k": (incr / b_max) if b_max else None,
        "measure": sampler.measure, "rss_peak_sampled": sampler.peak_rss,
        "gc_each_batch": bool(args.gc),
        "retained": (trace[-1] - m_fixed) if trace else 0,
        "rss_after_batch": trace,
        "seconds": seconds,
    }


def run_config(args, step: str, entry_stop: int | None) -> dict:
    cmd = [sys.executable, os.path.abspath(__file__), args.root_file,
           "--_worker", "--step", step, "--dataset", args.dataset,
           "--interval", str(args.interval), "--file-id", str(args.file_id),
           "--load-method", args.load_method, "--user", args.user,
           "--password", args.password]
    if entry_stop is not None:
        cmd += ["--entry-stop", str(entry_stop)]
    if args.tree is not None:
        cmd += ["--tree", args.tree]
    if args.db:
        cmd += ["--db", args.db]
    if args.keep:
        cmd += ["--keep"]
    if args.gc:
        cmd += ["--gc"]
    cmd += ["--measure", args.measure]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode:
        sys.exit(f"worker failed (step={step}, entry_stop={entry_stop}):\n{out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def _row(r: dict) -> str:
    k = f"{r['k']:.2f}" if r["k"] is not None else "-"
    return (f"{r['step_size']:>9} {r['events']:>10,d} {r['batches']:>5d} "
            f"{r['decoded_bytes_total'] / MIB:>9.0f} {r['batch_bytes_max'] / MIB:>8.1f} "
            f"{r['tables_bytes_max'] / MIB:>8.1f} {r['m_fixed'] / MIB:>8.1f} "
            f"{r['m_peak_sampled'] / MIB:>8.1f} {r['increment'] / MIB:>8.1f} "
            f"{r['retained'] / MIB:>8.1f} {k:>6} {r['seconds']:>7.1f}")


HEADER = (f"{'step':>9} {'events':>10} {'batch':>5} {'decoded':>9} {'B max':>8} "
          f"{'tables':>8} {'M_fixed':>8} {'M_peak':>8} {'incr':>8} {'kept':>8} {'k':>6} {'s':>7}\n"
          f"{'':>9} {'':>10} {'es':>5} {'MiB':>9} {'MiB':>8} {'MiB':>8} "
          f"{'MiB':>8} {'MiB':>8} {'MiB':>8} {'MiB':>8} {'':>6} {'':>7}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("root_file")
    p.add_argument("--dataset", default="nanoaod")
    p.add_argument("--tree", default=None)
    p.add_argument("--steps", default="25 MB,50 MB,100 MB,200 MB",
                   help="comma-separated uproot step sizes for the batch-size scan")
    p.add_argument("--volume-step", default="10 MB",
                   help="fixed step size for the volume test (small enough that "
                        "1/N of the file still holds several full batches)")
    p.add_argument("--volume-factor", type=int, default=10,
                   help="compare 1/N of the file with the whole file")
    p.add_argument("--scan-entries", type=int, default=None,
                   help="limit the batch-size scan to the first N entries")
    p.add_argument("--k", type=float, default=6.0, help="proposed factor on B")
    p.add_argument("--db", default=None, help="HOST[:PORT]/DATABASE; load mode")
    p.add_argument("--load-method", default="binary")
    p.add_argument("--file-id", type=int, default=4095,
                   help="scratch file id for rows written in load mode")
    p.add_argument("--keep", action="store_true", help="keep loaded rows")
    p.add_argument("--measure", choices=("auto", "rss"), default="auto",
                   help="auto: physical footprint on macOS, RSS elsewhere")
    p.add_argument("--gc", action="store_true",
                   help="run the cyclic garbage collector after every batch")
    p.add_argument("--user", default="monetdb")
    p.add_argument("--password", default="monetdb")
    p.add_argument("--interval", type=float, default=0.01)
    p.add_argument("--out", default="memory_pilot.json")
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--step", default="50 MB", help=argparse.SUPPRESS)
    p.add_argument("--entry-stop", type=int, default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    if args._worker:
        print(json.dumps(worker(args)))
        return

    import uproot

    from awkward_monetizer.datasets import DATASETS
    from awkward_monetizer.ingest import open_tree
    ds = DATASETS[args.dataset]
    with uproot.open(args.root_file) as f:
        n = open_tree(f, args.tree if args.tree is not None else ds.tree).num_entries
    mode = f"load via {args.load_method} into {args.db}" if args.db else "build only"
    print(f"{args.root_file}: {n:,d} entries; mode: {mode}\n")

    print(f"1. Batch-size scan (is M_peak - M_fixed <= {args.k:g} B?)")
    print(HEADER)
    scan = []
    for step in [s.strip() for s in args.steps.split(",") if s.strip()]:
        r = run_config(args, step, args.scan_entries)
        scan.append(r)
        print(_row(r), flush=True)

    print(f"\n2. Volume test at step {args.volume_step} "
          f"(1/{args.volume_factor} of the file, then all of it)")
    print(HEADER)
    small = run_config(args, args.volume_step, max(1, n // args.volume_factor))
    print(_row(small), flush=True)
    full = run_config(args, args.volume_step, None)
    print(_row(full), flush=True)

    print(f"\nMemory measure: {full['measure']}")
    ks = [r["k"] for r in scan + [small, full] if r["k"] is not None]
    change = (full["increment"] - small["increment"]) / small["increment"]
    ratio = full["decoded_bytes_total"] / small["decoded_bytes_total"]
    print(f"\nLargest k = increment / B_max: {max(ks):.2f} "
          f"({'within' if max(ks) <= args.k else 'EXCEEDS'} the proposed {args.k:g})")
    print(f"Volume x{ratio:.1f} -> increment changed by {change:+.1%} "
          f"({'within' if abs(change) < 0.10 else 'EXCEEDS'} the proposed 10%)")
    if full["batches"] and small["batch_bytes_max"] != full["batch_bytes_max"]:
        print("  note: the largest batch differs between the two runs "
              f"({small['batch_bytes_max'] / MIB:.1f} vs "
              f"{full['batch_bytes_max'] / MIB:.1f} MiB); compare k as well.")

    result = {"file": args.root_file, "entries": int(n), "mode": mode,
              "python": platform.python_version(), "platform": platform.platform(),
              "proposed_k": args.k, "scan": scan,
              "volume": {"small": small, "full": full,
                         "volume_ratio": ratio, "increment_change": change},
              "max_k": max(ks)}
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
