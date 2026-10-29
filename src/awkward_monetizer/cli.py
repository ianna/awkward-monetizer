"""Unified command-line interface: ``awkward-monetizer <subcommand>``.

Subcommands: ingest, manifest, shards, reconstruct, roundtrip, adl, make-nano, bench,
agent.
"""

from __future__ import annotations

import argparse

DEFAULT_DIMUON = "data/cms.root"
DEFAULT_NANO = "data/nano_synth.root"


# --------------------------------------------------------------------------
# Subcommand handlers
# --------------------------------------------------------------------------
def _cmd_ingest(args) -> None:
    from . import db
    from .datasets import DATASETS
    from .ingest import build_tables, ingest_root_chunked, load_monetdb, read_root

    ds = DATASETS[args.dataset]
    file_id, layout = _resolve_file_id(args)
    if (args.entry_start is not None or args.replace) and not args.step_size:
        raise SystemExit("--entry-start/--replace require --step-size")
    if args.replace and args.truncate:
        raise SystemExit("use either --truncate or --replace, not both")
    if file_id:
        print(f"file_id={file_id} -> event_id = "
              f"({file_id} << {layout.entry_bits}) | entry")

    # Chunked streaming path for large (real NanoAOD) files.
    if args.step_size:
        if args.dry_run:
            n = ingest_root_chunked(
                args.root_file, ds, None, tree=args.tree,
                step_size=args.step_size, entry_start=args.entry_start,
                entry_stop=args.entry_stop, file_id=file_id, dry_run=True,
                layout=layout,
            )
            print(f"dry run -- {n} events; not loading into MonetDB")
            return
        conn = db.open_server(database=args.database, host=args.host,
                              port=args.port, user=args.user,
                              password=args.password)
        try:
            if args.create_schema:
                db.create_schema(conn, db.read_schema(ds.name))
            print(f"streaming {args.root_file} (dataset={ds.name}) "
                  f"in chunks of {args.step_size} ...")
            n = ingest_root_chunked(args.root_file, ds, conn, tree=args.tree,
                                    step_size=args.step_size,
                                    entry_start=args.entry_start,
                                    entry_stop=args.entry_stop,
                                    file_id=file_id, layout=layout,
                                    truncate=args.truncate,
                                    replace=args.replace,
                                    use_copy_into=not args.no_copy_into,
                                    method=_load_method(args))
            conn.commit()
            print(f"done: {n} events.")
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return

    # One-shot path (small files).
    print(f"reading {args.root_file} (dataset={ds.name}) ...")
    events, tree = read_root(args.root_file, ds, args.tree,
                             entry_stop=args.entry_stop)
    print(f"  tree {tree!r}: {len(events)} events")
    tables = build_tables(events, ds, file_id=file_id, layout=layout)
    for name, df in tables.items():
        print(f"  built {name}: {len(df)} rows x {len(df.columns)} cols "
              f"[{', '.join(df.columns)}]")
    if args.dry_run:
        print("dry run -- not loading into MonetDB")
        for name, df in tables.items():
            print(f"\n=== {name} (head) ===")
            print(df.head().to_string(index=False))
        return
    if args.create_schema:
        c = db.open_server(database=args.database, host=args.host,
                           port=args.port, user=args.user, password=args.password)
        try:
            db.create_schema(c, db.read_schema(ds.name))
        finally:
            c.close()
    print(f"loading into MonetDB '{args.database}' ...")
    load_monetdb(tables, database=args.database, host=args.host, port=args.port,
                 user=args.user, password=args.password,
                 truncate=args.truncate, use_copy_into=not args.no_copy_into,
                 method=_load_method(args))
    print("done.")


def _cmd_reconstruct(args) -> None:
    import awkward as ak
    import numpy as np

    from .physics import invariant_mass, opposite_charge_pair
    from .reconstruct import reconstruct_events, tables_from_root

    print(f"building tables from {args.root_file} (dataset={args.dataset}) ...")
    events_df, muons_df = tables_from_root(args.root_file, args.dataset)
    print(f"  events: {len(events_df)} rows | muons: {len(muons_df)} rows")
    events = reconstruct_events(events_df, muons_df)
    print(f"  reconstructed {len(events)} events; muons/event (first 10): "
          f"{ak.to_list(ak.num(events.muons, axis=1)[:10])}")
    if "mass" in events.fields:
        recon = ak.to_numpy(invariant_mass(events.muons))
        stored = ak.to_numpy(events.mass)
        diff = np.abs(recon - stored)
        print("\n=== mass validation (reconstructed vs stored M) ===")
        print(f"  events compared : {diff.size}")
        print(f"  max |Δ|         : {np.nanmax(diff):.3e} GeV")
        print(f"  agree < 1e-2 GeV: {np.mean(diff < 1e-2) * 100:.2f}%")
    n_oc = int(ak.sum(opposite_charge_pair(events)))
    print(f"  opposite-charge events: {n_oc}")


def _cmd_roundtrip(args) -> None:
    from .roundtrip import run
    conn_kwargs = {"database": args.database, "host": args.host, "port": args.port,
                       "user": args.user, "password": args.password, "dbdir": args.dbdir}
    run(root_file=args.root_file, dataset=args.dataset, backend=args.backend,
        where=args.where, use_copy_into=not args.no_copy_into,
        conn_kwargs=conn_kwargs)


def _cmd_adl(args) -> None:
    import awkward as ak

    from .adl import events_from_root, run_all

    print(f"loading {args.from_root} ...")
    events = events_from_root(args.from_root)
    print(f"  {len(events)} events "
          f"(jets/event mean {ak.mean(ak.num(events.jets, axis=1)):.2f}, "
          f"muons/event mean {ak.mean(ak.num(events.muons, axis=1)):.2f})\n")
    results = run_all(events)
    _print_adl(results)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        for ax, (name, arr) in zip(axes.ravel(), results.items(), strict=False):
            if arr.size:
                ax.hist(arr, bins=40, histtype="stepfilled",
                        color="#3b6fb6", alpha=0.85, edgecolor="#26456e")
            ax.set_title(name, fontsize=9)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print(f"\nsaved figure -> {args.plot}")


def _print_adl(results) -> None:
    import numpy as np
    for name, arr in results.items():
        arr = np.asarray([] if arr is None else arr)
        s = ("n=0" if arr.size == 0 else
             f"n={arr.size:6d}  mean={np.nanmean(arr):8.2f}  "
             f"min={np.nanmin(arr):7.2f}  max={np.nanmax(arr):7.2f}")
        print(f"  {name:32s} {s}")


def _parse_shard_spec(spec: str, i: int, user: str, password: str) -> dict:
    """``host[:port][/database]`` or ``embedded:/path/to/dbdir``."""
    name = f"s{i}"
    if spec.startswith("embedded:"):
        return {"name": name, "backend": "embedded",
                "target": spec[len("embedded:"):]}
    hostport, _, database = spec.partition("/")
    host, _, port = hostport.partition(":")
    return {"name": name, "host": host or "localhost",
            "port": int(port) if port else 50000,
            "database": database or "hep", "user": user, "password": password}


def _cmd_shards(args) -> None:
    from .keys import Manifest
    from .shards import ShardMap, ingest_manifest, map_events, verify

    if args.shards_cmd == "init":
        manifest = Manifest.from_json(args.manifest)
        n_files = len(manifest.files)
        specs = [_parse_shard_spec(h, i, args.user, args.password)
                 for i, h in enumerate(args.shard)]
        smap = ShardMap.split(n_files, specs, layout=manifest.layout)
        smap.to_json(args.out)
        print(f"{args.out}: {len(smap)} shards for {n_files} files "
              f"({smap.layout.entry_bits} entry bits)")
        for sh in smap:
            where = sh.target if sh.backend == "embedded" else (
                f"{sh.host}:{sh.port}/{sh.database}")
            print(f"  {sh.name}: file_ids [{sh.file_ids[0]}, {sh.file_ids[1]})"
                  f"  -> {where}")
        return

    smap = ShardMap.from_json(args.shard_map)
    if args.shards_cmd == "ingest":
        from .datasets import DATASETS
        from .shards import ShardError
        try:
            counts = ingest_manifest(
                Manifest.from_json(args.manifest), smap, DATASETS[args.dataset],
                step_size=args.step_size, tree=args.tree,
                create_schema=args.create_schema,
                method="insert" if args.no_copy_into else args.load_method)
        except (ValueError, ShardError) as e:
            raise SystemExit(f"error: {e}") from e
        for name, n in counts.items():
            print(f"  {name}: {n} events")
        print(f"done: {sum(counts.values())} events on {len(smap)} shards.")
    elif args.shards_cmd == "migrate":
        from .datasets import DATASETS
        from .keys import KeyLayout, migrate_tables
        ds = DATASETS[args.dataset]
        tables = ["events"] + [c.table for c in ds.collections]
        old, new = smap.layout, KeyLayout(args.entry_bits)
        manifest = Manifest.from_json(args.manifest).with_layout(new)
        new_map = ShardMap(smap.shards, layout=new)      # validates file-id range
        try:
            for check_only in (True, False):   # check every shard before any change
                for sh in smap:
                    conn = sh.connect()
                    try:
                        counts = migrate_tables(conn, tables, old, new,
                                                check_only=check_only)
                    finally:
                        conn.close()
                    if not check_only:
                        print(f"  {sh.name}: converted "
                              + ", ".join(f"{t}={n}" for t, n in counts.items()))
        except ValueError as e:
            raise SystemExit(f"error: {e}") from e
        new_map.to_json(args.shard_map)
        manifest.to_json(args.manifest)
        print(f"done: {old.entry_bits} -> {new.entry_bits} entry bits; "
              f"updated {args.shard_map} and {args.manifest}.")
    elif args.shards_cmd == "verify":
        bad = False
        for r in verify(smap):
            status = "OK" if not r["problems"] else "PROBLEMS"
            counts = ", ".join(f"{k}={v}" for k, v in r.items()
                               if k not in ("shard", "problems"))
            print(f"  {r['shard']}: {status}  ({counts})")
            for prob in r["problems"]:
                print(f"      - {prob}")
            bad |= bool(r["problems"])
        if bad:
            raise SystemExit(1)
    elif args.shards_cmd == "adl":
        from .adl import run_all
        results = map_events(smap, run_all, where=args.where)
        _print_adl(results or {})


def _cmd_make_nano(args) -> None:
    import os

    from .sampledata import write_nanoaod
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_nanoaod(args.out, args.events, args.seed, fmt=args.format,
                  triggers=not args.no_triggers)
    print(f"wrote {args.out}: {args.events} events, {args.format} 'Events'")


def _cmd_bench(args) -> None:
    conn_kwargs = {"database": args.database, "host": args.host, "port": args.port,
                       "user": args.user, "password": args.password, "dbdir": args.dbdir}
    backends = tuple(b.strip() for b in args.backends.split(",") if b.strip())
    if args.analysis == "trijet":
        from .benchmark import run_trijet
        nano = args.root_file if args.root_file != DEFAULT_DIMUON else DEFAULT_NANO
        # trijet has no `hybrid` backend; default to awkward vs in-DB SQL
        bk = backends if args.backends != "awkward,hybrid" else ("awkward", "monetdb")
        run_trijet(nano, backends=bk, repeats=args.repeats,
                   hybrid_backend=args.hybrid_backend, conn_kwargs=conn_kwargs)
    elif args.analysis == "zmumu-nano":
        from .benchmark import run_zmumu_nano
        nano = args.root_file if args.root_file != DEFAULT_DIMUON else DEFAULT_NANO
        # jagged NanoAOD Z->mumu: awkward vs in-DB SQL (rdataframe optional)
        bk = backends if args.backends != "awkward,hybrid" else ("awkward", "monetdb")
        run_zmumu_nano(nano, backends=bk, repeats=args.repeats,
                       hybrid_backend=args.hybrid_backend, conn_kwargs=conn_kwargs,
                       metric_only=args.metric_only, threads=args.threads)
    else:
        from .benchmark import run
        run(args.root_file, dataset=args.dataset, backends=backends,
            scale=args.scale, repeats=args.repeats,
            hybrid_backend=args.hybrid_backend, conn_kwargs=conn_kwargs)


# --------------------------------------------------------------------------
def _load_method(args) -> str | None:
    if args.no_copy_into:
        return "insert"
    return args.load_method


def _resolve_file_id(args):
    """Return (file_id, key layout) from --manifest or --file-id/--entry-bits."""
    from .keys import DEFAULT_LAYOUT, KeyLayout, Manifest
    if args.manifest is not None and args.file_id is not None:
        raise SystemExit("use either --file-id or --manifest, not both")
    if args.manifest is not None:
        if args.entry_bits is not None:
            raise SystemExit("--entry-bits comes from the manifest; omit it")
        m = Manifest.from_json(args.manifest)
        return m.file_id(args.root_file), m.layout
    layout = (KeyLayout(args.entry_bits) if args.entry_bits is not None
              else DEFAULT_LAYOUT)
    try:
        return layout.check_file_id(args.file_id or 0), layout
    except ValueError as e:
        raise SystemExit(f"error: {e}") from e


def _cmd_manifest(args) -> None:
    import os

    from .keys import DEFAULT_LAYOUT, KeyLayout, Manifest, build_manifest
    if os.path.exists(args.out) and not args.fresh:
        m = Manifest.from_json(args.out)
        if args.entry_bits is not None and args.entry_bits != m.layout.entry_bits:
            raise SystemExit(
                f"{args.out} uses {m.layout.entry_bits} entry bits; a manifest's "
                "key layout cannot change when files are appended "
                "(see `shards migrate`)")
        before = len(m.files)
        m.add(args.root_files, count_entries=args.count_entries, tree=args.tree)
        print(f"{args.out}: kept {before} files, added {len(m.files) - before}")
        given = {os.path.abspath(p) for p in args.root_files}
        extra = [f.path for f in m.files
                 if "://" not in f.path and os.path.abspath(f.path) not in given]
        if extra:
            print(f"  note: also contains files not given here: {', '.join(extra)}"
                  "\n  (existing ids never change; use --fresh to start over)")
    else:
        layout = (KeyLayout(args.entry_bits) if args.entry_bits is not None
                  else DEFAULT_LAYOUT)
        m = build_manifest(args.root_files, count_entries=args.count_entries,
                           tree=args.tree, layout=layout)
        print(f"{args.out}: {len(m.files)} files")
    print(f"  key layout: {m.layout.file_bits} file bits, "
          f"{m.layout.entry_bits} entry bits")
    for f in m.files:
        lo, hi = m.layout.file_id_range(f.file_id)
        n = "" if f.n_entries is None else f"  {f.n_entries} entries"
        print(f"  {f.file_id:>6d}  [{lo}, {hi})  {f.path}{n}")
    m.to_json(args.out)

# --------------------------------------------------------------------------
# agent: LLM-driven analysis over SQL + Awkward tools
# --------------------------------------------------------------------------
def _agent_conn(args, quiet: bool = False):
    import contextlib
    import sys
    if args.from_root:
        from .agent import load_demo_db
        backend = "auto" if args.backend == "server" else args.backend
        out = sys.stderr if quiet else sys.stdout
        with contextlib.redirect_stdout(out):
            print(f"loading {args.from_root} into a throwaway {backend} database ...")
            return load_demo_db(args.from_root, backend=backend)
    from . import db
    return db.open_server(database=args.database, host=args.host, port=args.port,
                          user=args.user, password=args.password)


def _agent_client(args, query: str | None = None):
    import pathlib

    from .agent import AnthropicClient, ReplayClient, ScriptedClient, expert_script
    if getattr(args, "scripted", False):
        return ScriptedClient(expert_script(query))
    if args.replay:
        p = pathlib.Path(args.replay)
        return ReplayClient(str(p / f"{query}.json" if p.is_dir() else p))
    return AnthropicClient(model=args.model)


def _cmd_agent(args) -> None:
    if args.agent_cmd == "questions":
        from .agent import TASKS
        for q, (_name, text) in TASKS.items():
            print(f"{q}: {text}\n")
        return
    if args.agent_cmd == "checks":
        _agent_checks(args)
        return
    if args.agent_cmd == "mcp":
        from .agent.mcp_server import serve
        serve(_agent_conn(args, quiet=True), plot_dir=args.plot_dir,
              check_log=args.check_log or None,
              chunk_events=args.chunk_events, ref_cache=args.ref_cache,
              transport=args.transport, host=args.http_host, port=args.http_port)
        return

    conn = _agent_conn(args)
    try:
        if args.agent_cmd == "ask":
            _agent_ask(args, conn)
        else:
            _agent_eval(args, conn)
    except Exception as exc:
        # Model-API failures (no credits, bad key, network): one clear line,
        # not a traceback through the SDK.
        if type(exc).__module__.split(".")[0] == "anthropic":
            msg = getattr(exc, "message", None) or str(exc)
            body = getattr(exc, "body", None)
            if isinstance(body, dict):
                msg = body.get("error", {}).get("message", msg)
            raise SystemExit(f"API error ({type(exc).__name__}): {msg}") from None
        raise
    finally:
        conn.close()


def _agent_checks(args) -> None:
    import json
    import pathlib
    path = pathlib.Path(args.check_log)
    if not path.exists():
        raise SystemExit(f"no check log at {path}")
    entries = [json.loads(line) for line in path.read_text().splitlines() if line]
    first: dict[str, dict] = {}
    last: dict[str, dict] = {}
    tries: dict[str, int] = {}
    for e in entries:
        first.setdefault(e["query"], e)
        last[e["query"]] = e
        tries[e["query"]] = tries.get(e["query"], 0) + 1
    print(f"{'query':5s} {'first':6s} {'final':6s} {'tries':>5s} {'route':8s} detail")
    for q in sorted(last):
        f, la = first[q], last[q]
        print(f"{q:5s} {'PASS' if f['passed'] else 'FAIL':6s} "
              f"{'PASS' if la['passed'] else 'FAIL':6s} {tries[q]:5d} "
              f"{la.get('route', '-')!s:8s} {la['reason']}")
    n = sum(e["passed"] for e in first.values())
    print(f"\n{n}/{len(first)} passed on first check ({len(entries)} checks logged)")


def _agent_ask(args, conn) -> None:
    import json

    from .agent import AnalysisTools, ReplayClient, run_agent
    from .agent.tools import summarize

    client = _agent_client(args)
    question = args.question
    if question is None and isinstance(client, ReplayClient):
        question = client.question
    if not question:
        raise SystemExit("give a question (or --replay a transcript)")
    print(f"question: {question}\n")
    run = run_agent(question, AnalysisTools(conn, plot_dir=args.plot_dir,
                                             chunk_events=args.chunk_events),
                    client,
                    max_turns=args.max_turns, echo=print)
    print(f"\nstopped: {run.stopped} after {run.turns} turns, {run.wall_s}s, "
          f"{run.usage['input_tokens'] + run.usage['output_tokens']} tokens")
    for eng, s in run.engine_summary().items():
        print(f"  {eng:8s} calls={s['calls']} errors={s['errors']} time={s['ms']} ms")
    if run.answer is not None:
        print(f"\nanswer via {run.answer_info['route']}: "
              f"{run.answer_info['explanation']}")
        s = summarize(run.answer)
        print(json.dumps({k: v for k, v in s.items() if k != "histogram"}))
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(run.answer, bins=40, histtype="stepfilled", color="#3b6fb6",
                    alpha=0.85, edgecolor="#26456e")
            ax.set_title(question[:70] + ("…" if len(question) > 70 else ""),
                         fontsize=8)
            fig.tight_layout()
            fig.savefig(args.plot, dpi=120)
            print(f"saved figure -> {args.plot}")
    if args.transcript:
        run.save(args.transcript)
        print(f"transcript -> {args.transcript}")


def _agent_eval(args, conn) -> None:
    from .agent import TASKS, evaluate, format_table
    queries = ([q.strip().upper() for q in args.queries.split(",")]
               if args.queries else list(TASKS))
    unknown = [q for q in queries if q not in TASKS]
    if unknown:
        raise SystemExit(f"unknown queries {unknown}; choose from {list(TASKS)}")
    rows = evaluate(conn, lambda q: _agent_client(args, q), queries=queries,
                    max_turns=args.max_turns, echo=print,
                    transcripts_dir=args.transcripts,
                    chunk_events=args.chunk_events, ref_cache=args.ref_cache)
    print("\n" + format_table(rows))



# Parser
# --------------------------------------------------------------------------
def _add_server_args(sp) -> None:
    sp.add_argument("--database", default="hep")
    sp.add_argument("--host", default="localhost")
    sp.add_argument("--port", type=int, default=50000)
    sp.add_argument("--user", default="monetdb")
    sp.add_argument("--password", default="monetdb")


def _step_size(value: str) -> int | str:
    try:
        count = int(value)
    except ValueError:
        return value
    if count <= 0:
        raise argparse.ArgumentTypeError("step size must be a positive entry count")
    return count


def _nonneg_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="awkward-monetizer",
                                description="Hybrid Awkward + MonetDB HEP engine.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="ROOT -> flat tables -> MonetDB")
    ing.add_argument("root_file", nargs="?", default=DEFAULT_DIMUON)
    ing.add_argument("--dataset", default="dimuon")
    ing.add_argument("--tree", default=None)
    ing.add_argument("--entry-start", type=int, default=None,
                     help="first entry to read (chunked mode); with --entry-stop "
                          "lets several jobs split one file")
    ing.add_argument("--entry-stop", type=int, default=None)
    ing.add_argument("--file-id", type=_nonneg_int, default=None,
                     help="stable per-file id: event_id = (file_id << 40) | entry "
                          "(default 0 = plain entry number)")
    ing.add_argument("--entry-bits", type=int, default=None,
                     help="key layout with --file-id: bits for the entry number "
                          "(default 40; 31 allows 4.3 billion files)")
    ing.add_argument("--manifest", default=None, metavar="JSON",
                     help="look up --file-id for root_file in a manifest "
                          "(see `awkward-monetizer manifest`)")
    ing.add_argument("--step-size", type=_step_size, default=None,
                     help="stream in chunks of this size (e.g. '100 MB' or 50000) "
                          "for large files; requires a running server")
    ing.add_argument("--create-schema", action="store_true",
                     help="(re)create the dataset's schema before loading")
    ing.add_argument("--truncate", action="store_true",
                     help="empty the tables first (single writer only)")
    ing.add_argument("--replace", action="store_true",
                     help="delete only this file/entry range first, so a job "
                          "can be re-run safely alongside other workers")
    ing.add_argument("--no-copy-into", action="store_true",
                     help="same as --load-method insert")
    ing.add_argument("--load-method", default=None,
                     choices=("auto", "binary", "client", "copy", "insert"),
                     help="copy (temp CSV, server on this host; default), binary/"
                          "client (stream over the connection -- remote servers), "
                          "insert, or auto")
    ing.add_argument("--dry-run", action="store_true")
    _add_server_args(ing)
    ing.set_defaults(func=_cmd_ingest)

    mf = sub.add_parser("manifest",
                        help="assign stable file_ids to input files (JSON)")
    mf.add_argument("root_files", nargs="+")
    mf.add_argument("--out", default="manifest.json",
                    help="manifest to create, or extend if it exists")
    mf.add_argument("--entry-bits", type=int, default=None,
                    help="key layout for a new manifest: bits for the entry number "
                         "(default 40 = 8.4 million files; 31 = 4.3 billion files)")
    mf.add_argument("--fresh", action="store_true",
                    help="overwrite an existing manifest instead of extending it "
                         "(renumbers files -- only before anything is ingested)")
    mf.add_argument("--count-entries", action="store_true",
                    help="open each file and record its entry count")
    mf.add_argument("--tree", default=None)
    mf.set_defaults(func=_cmd_manifest)

    sh = sub.add_parser("shards", help="scatter-gather over several MonetDB shards")
    shs = sh.add_subparsers(dest="shards_cmd", required=True)
    si = shs.add_parser("init", help="split a manifest's file_ids over shards")
    si.add_argument("--manifest", required=True)
    si.add_argument("--shard", action="append", required=True, metavar="SPEC",
                    help="host[:port][/database] or embedded:/dbdir; repeat per shard")
    si.add_argument("--out", default="shards.json")
    si.add_argument("--user", default="monetdb")
    si.add_argument("--password", default="monetdb")
    sg = shs.add_parser("ingest", help="load each manifest file into its shard")
    sg.add_argument("shard_map")
    sg.add_argument("--manifest", required=True)
    sg.add_argument("--dataset", default="nanoaod")
    sg.add_argument("--tree", default=None)
    sg.add_argument("--step-size", type=_step_size, default="100 MB")
    sg.add_argument("--create-schema", action="store_true")
    sg.add_argument("--load-method", default="auto",
                    choices=("auto", "binary", "client", "copy", "insert"),
                    help="auto: copy for a local server, binary upload for remote "
                         "servers, insert for embedded")
    sg.add_argument("--no-copy-into", action="store_true",
                    help="same as --load-method insert")
    sm = shs.add_parser("migrate", help="convert stored keys to another layout")
    sm.add_argument("shard_map")
    sm.add_argument("--manifest", required=True)
    sm.add_argument("--entry-bits", type=int, required=True)
    sm.add_argument("--dataset", default="nanoaod")
    sv = shs.add_parser("verify", help="check shard ranges and orphan rows")
    sv.add_argument("shard_map")
    sa = shs.add_parser("adl", help="run ADL Q1-Q8 scatter-gather")
    sa.add_argument("shard_map")
    sa.add_argument("--where", default=None,
                    help="SQL selection on events, pushed to every shard")
    sh.set_defaults(func=_cmd_shards)

    rec = sub.add_parser("reconstruct", help="ROOT -> Awkward NF2 + validate (no DB)")
    rec.add_argument("root_file", nargs="?", default=DEFAULT_DIMUON)
    rec.add_argument("--dataset", default="dimuon")
    rec.set_defaults(func=_cmd_reconstruct)

    rt = sub.add_parser("roundtrip", help="full ingest -> MonetDB -> reconstruct")
    rt.add_argument("--root-file", default=DEFAULT_DIMUON)
    rt.add_argument("--dataset", choices=("dimuon",), default="dimuon")
    rt.add_argument("--backend", choices=("server", "embedded"), default="server")
    rt.add_argument("--where", default="mass BETWEEN 60 AND 120")
    rt.add_argument("--no-copy-into", action="store_true")
    rt.add_argument("--dbdir", default=None)
    _add_server_args(rt)
    rt.set_defaults(func=_cmd_roundtrip)

    adl = sub.add_parser("adl", help="run the ADL Q1-Q8 benchmark")
    adl.add_argument("--from-root", default=DEFAULT_NANO)
    adl.add_argument("--plot", default=None, metavar="PNG")
    adl.set_defaults(func=_cmd_adl)

    mn = sub.add_parser("make-nano", help="write a synthetic NanoAOD file")
    mn.add_argument("--out", default=DEFAULT_NANO)
    mn.add_argument("--events", type=int, default=2000)
    mn.add_argument("--seed", type=int, default=0)
    mn.add_argument("--format", choices=("rntuple", "ttree"), default="rntuple",
                    help="RNTuple (uproot's default) or classic NanoAOD TTree")
    mn.add_argument("--no-triggers", action="store_true",
                    help="omit the HLT_* branches (their columns load as NULL)")
    mn.set_defaults(func=_cmd_make_nano)

    bn = sub.add_parser("bench", help="benchmark backends (awkward/hybrid/rdataframe)")
    bn.add_argument("--root-file", default=DEFAULT_DIMUON)
    bn.add_argument("--dataset", default="dimuon")
    bn.add_argument("--analysis", choices=("zmumu", "zmumu-nano", "trijet"),
                    default="zmumu",
                    help="zmumu (flat dimuon ntuple), zmumu-nano (Z->mumu on real "
                         "jagged NanoAOD), or trijet (ADL Q6, nanoaod)")
    bn.add_argument("--backends", default="awkward,hybrid",
                    help="comma-separated: awkward,hybrid,monetdb,rdataframe")
    bn.add_argument("--scale", type=int, default=1)
    bn.add_argument("--threads", type=int, default=1,
                    help="rdataframe: ROOT implicit-MT threads (1=off, 0=all "
                         "cores, N=N cores). MonetDB server is already parallel; "
                         "awkward combinatorics is single-threaded numpy")
    bn.add_argument("--metric-only", action="store_true",
                    help="zmumu-nano: aggregate count/mean inside MonetDB and "
                         "ship only two numbers, instead of streaming every mass")
    bn.add_argument("--repeats", type=int, default=5)
    bn.add_argument("--hybrid-backend", choices=("server", "embedded"),
                    default="embedded")
    bn.add_argument("--dbdir", default=None)
    _add_server_args(bn)
    bn.set_defaults(func=_cmd_bench)

    ag = sub.add_parser("agent", help="LLM agent that routes SQL vs Awkward work")
    ags = ag.add_subparsers(dest="agent_cmd", required=True)

    def _agent_common(sp, model=True):
        sp.add_argument("--from-root", default=None, metavar="ROOT",
                        help="load this NanoAOD file into a throwaway embedded "
                             "database instead of connecting to a server")
        sp.add_argument("--backend", choices=("server", "monetdbe", "sqlite"),
                        default="server",
                        help="with --from-root: monetdbe or sqlite (default: "
                             "monetdbe if installed)")
        _add_server_args(sp)
        from .agent.tools import CHUNK_EVENTS
        sp.add_argument("--chunk-events", type=int, default=CHUNK_EVENTS,
                        help="above this many events, Awkward work and the ADL "
                             "reference run in event_id chunks of this size")
        sp.add_argument("--ref-cache", default=None, metavar="NPZ",
                        help="cache the ADL reference answers in this file "
                             "(worth it for large databases)")
        if model:
            from .agent.loop import DEFAULT_MODEL
            sp.add_argument("--model", default=DEFAULT_MODEL)
            sp.add_argument("--max-turns", type=int, default=16)
            sp.add_argument("--replay", default=None, metavar="JSON_OR_DIR",
                            help="replay saved model turns (offline demo); "
                                 "tools still run live")

    aa = ags.add_parser("ask", help="answer one analysis question")
    aa.add_argument("question", nargs="?", default=None)
    aa.add_argument("--transcript", default=None, metavar="JSON")
    aa.add_argument("--plot", default=None, metavar="PNG")
    aa.add_argument("--plot-dir", default="output/agent_plots",
                    help="where the plot_result tool saves PNGs")
    _agent_common(aa)

    ae = ags.add_parser("eval", help="score the agent on ADL Q1-Q8")
    ae.add_argument("--queries", default=None, help="e.g. Q1,Q6 (default all)")
    ae.add_argument("--scripted", action="store_true",
                    help="use the scripted expert instead of an LLM (baseline)")
    ae.add_argument("--transcripts", default=None, metavar="DIR",
                    help="save one transcript per query (replayable)")
    _agent_common(ae)

    am = ags.add_parser("mcp", help="serve the tools over MCP (stdio or HTTP)")
    _agent_common(am, model=False)
    am.add_argument("--plot-dir", default="output/agent_plots",
                    help="where the plot_result tool saves PNGs")
    am.add_argument("--check-log", default="output/agent_checks.jsonl",
                    help="JSON-lines log of check_answer results ('' = off)")
    am.add_argument("--transport", choices=("stdio", "http"), default="stdio",
                    help="stdio: started by the client (default); http: MCP "
                         "streamable HTTP at http://HTTP_HOST:HTTP_PORT/mcp")
    am.add_argument("--http-host", default="127.0.0.1",
                    help="address to listen on with --transport http (default "
                         "127.0.0.1, this computer only; awkward_eval is not "
                         "sandboxed, so do not expose it)")
    am.add_argument("--http-port", type=int, default=8765)

    aq = ags.add_parser("questions", help="print the ADL Q1-Q8 task texts")
    aq.set_defaults(agent_cmd="questions")
    ac = ags.add_parser("checks", help="tabulate check_answer results from MCP")
    ac.add_argument("--check-log", default="output/agent_checks.jsonl")
    ag.set_defaults(func=_cmd_agent)

    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
