"""Stream a whole CMS Open Data record into MonetDB, several files at a time.

Reads a list of file URLs (one per line, e.g. from
``cernopendata-client get-file-locations --recid 30555 --protocol xrootd``),
assigns stable file_ids with a manifest, and runs one
``awkward-monetizer ingest ... --replace`` per file, a few in parallel. Files are
streamed (``uproot.iterate``), so nothing is copied to local disk; only the
mapped branches land in MonetDB.

Resumable: every finished file is appended to a JSON-lines log, and files
already logged as ``ok`` are skipped on the next run. ``--replace`` makes a
re-run of a failed or interrupted file safe (it first deletes that file's own
event_id range).

Example::

    cernopendata-client get-file-locations --recid 30555 --protocol xrootd \\
        > data/2016H_files.txt
    monetdb create hep2016h && monetdb release hep2016h
    python examples/ingest_record.py data/2016H_files.txt --database hep2016h \\
        --create-schema

Reading ``root://`` URLs needs XRootD support for uproot (``fsspec-xrootd`` and
the XRootD Python bindings; both are in the pixi environment). ``https://`` URLs
work without it, usually more slowly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

CLI = [sys.executable, "-m", "awkward_monetizer.cli"]


def read_urls(path: str) -> list[str]:
    with open(path) as fh:
        urls = [line.strip() for line in fh]
    return [u for u in urls if u and not u.startswith("#")]


def done_urls(log: str) -> set[str]:
    if not os.path.exists(log):
        return set()
    out = set()
    with open(log) as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("status") == "ok":
                out.add(e["url"])
    return out


def free_gb(path: str) -> float:
    return shutil.disk_usage(os.path.expanduser(path)).free / 1e9


def server_args(a) -> list[str]:
    return ["--database", a.database, "--host", a.host, "--port", str(a.port),
            "--user", a.user, "--password", a.password]


def ingest_cmd(a, url: str) -> list[str]:
    return CLI + ["ingest", url, "--dataset", "nanoaod", "--tree", "Events",
                  "--manifest", a.manifest, "--step-size", str(a.step_size),
                  "--replace", *server_args(a)]


def run_one(a, url: str, lock: threading.Lock) -> dict:
    t0 = time.time()
    entry = {"url": url, "start": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for attempt in range(1, a.retries + 2):
        if free_gb(a.disk_path) < a.min_free_gb:
            entry.update(status="skipped", reason=f"less than {a.min_free_gb} GB free")
            break
        p = subprocess.run(ingest_cmd(a, url), capture_output=True, text=True,
                           check=False)
        tail = (p.stdout + p.stderr).strip().splitlines()[-3:]
        entry.update(attempt=attempt, returncode=p.returncode, tail=tail)
        if p.returncode == 0:
            n = next((ln.split()[1] for ln in reversed(tail) if ln.startswith("done:")),
                     None)
            entry.update(status="ok", events=int(n) if n and n.isdigit() else None)
            break
        entry["status"] = "failed"
    entry["seconds"] = round(time.time() - t0, 1)
    with lock, open(a.log, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("url_list", help="file with one ROOT file URL per line")
    ap.add_argument("--database", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=50000)
    ap.add_argument("--user", default="monetdb")
    ap.add_argument("--password", default="monetdb")
    ap.add_argument("--manifest", default=None,
                    help="default: <url_list stem>_manifest.json")
    ap.add_argument("--log", default=None, help="default: <url_list stem>_ingest.jsonl")
    ap.add_argument("--jobs", type=int, default=1,
                    help="files loaded in parallel; MonetDB aborts concurrent "
                         "writers to the same tables at COMMIT, so keep 1")
    ap.add_argument("--step-size", default="200000",
                    help="entries per chunk (or a size such as '100 MB')")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--create-schema", action="store_true",
                    help="(re)create the nanoaod tables first -- deletes their data")
    ap.add_argument("--min-free-gb", type=float, default=8.0,
                    help="do not start a file when less disk than this is free")
    ap.add_argument("--disk-path", default="~",
                    help="where the MonetDB dbfarm lives, for the free-space check")
    ap.add_argument("--dry-run", action="store_true", help="print the plan only")
    a = ap.parse_args(argv)
    stem = os.path.splitext(a.url_list)[0]
    a.manifest = a.manifest or f"{stem}_manifest.json"
    a.log = a.log or f"{stem}_ingest.jsonl"

    urls = read_urls(a.url_list)
    todo = [u for u in urls if u not in done_urls(a.log)]
    print(f"{len(urls)} files, {len(urls) - len(todo)} already done, "
          f"{len(todo)} to load; {free_gb(a.disk_path):.0f} GB free")
    if a.dry_run:
        for u in todo:
            print(" ".join(ingest_cmd(a, u)))
        return
    if not todo:
        return

    # stable file_ids: the manifest only ever appends, so re-runs keep their ids
    subprocess.run(CLI + ["manifest", *urls, "--out", a.manifest, "--tree", "Events"],
                   check=True, stdout=subprocess.DEVNULL)
    if a.create_schema:
        if len(todo) != len(urls):
            sys.exit("--create-schema would delete files already loaded; drop it "
                     "to resume")
        from awkward_monetizer import db
        conn = db.open_server(database=a.database, host=a.host, port=a.port,
                              user=a.user, password=a.password)
        db.create_schema(conn, db.read_schema("nanoaod"))
        conn.close()

    lock = threading.Lock()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures = {pool.submit(run_one, a, u, lock): u for u in todo}
        for i, f in enumerate(as_completed(futures), 1):
            e = f.result()
            name = e["url"].rsplit("/", 1)[-1]
            print(f"[{i}/{len(todo)}] {e['status']:7s} {e['seconds']:7.0f} s  "
                  f"{e.get('events') or '':>9}  {name}  "
                  f"({free_gb(a.disk_path):.0f} GB free)")
    failed = len(todo) - len(done_urls(a.log) & set(todo))
    print(f"\nfinished in {(time.time() - t0) / 60:.1f} min; {failed} not done "
          f"(re-run the same command to retry them). Log: {a.log}")


if __name__ == "__main__":
    main()
