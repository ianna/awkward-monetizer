"""Unified command-line interface: ``awkward-monetizer <subcommand>``.

Subcommands: ingest, reconstruct, roundtrip, adl, make-nano.
"""

from __future__ import annotations

import argparse

DEFAULT_DIMUON = "data/cms.root"
DEFAULT_NANO = "data/nano_synth.root"


# --------------------------------------------------------------------------
# Subcommand handlers
# --------------------------------------------------------------------------
def _cmd_ingest(args) -> None:
    import awkward as ak  # noqa: F401 (kept for parity / future use)
    from .datasets import DATASETS
    from .ingest import build_tables, load_monetdb, read_root

    ds = DATASETS[args.dataset]
    print(f"reading {args.root_file} (dataset={ds.name}) ...")
    events, tree = read_root(args.root_file, ds, args.tree,
                             entry_stop=args.entry_stop)
    print(f"  tree {tree!r}: {len(events)} events")
    tables = build_tables(events, ds)
    for name, df in tables.items():
        print(f"  built {name}: {len(df)} rows x {len(df.columns)} cols "
              f"[{', '.join(df.columns)}]")
    if args.dry_run:
        print("dry run -- not loading into MonetDB")
        for name, df in tables.items():
            print(f"\n=== {name} (head) ===")
            print(df.head().to_string(index=False))
        return
    print(f"loading into MonetDB '{args.database}' ...")
    load_monetdb(tables, database=args.database, host=args.host, port=args.port,
                 user=args.user, password=args.password,
                 truncate=args.truncate, use_copy_into=not args.no_copy_into)
    print("done.")


def _cmd_reconstruct(args) -> None:
    import numpy as np

    import awkward as ak
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
    conn_kwargs = dict(database=args.database, host=args.host, port=args.port,
                       user=args.user, password=args.password, dbdir=args.dbdir)
    run(root_file=args.root_file, dataset=args.dataset, backend=args.backend,
        where=args.where, use_copy_into=not args.no_copy_into,
        conn_kwargs=conn_kwargs)


def _cmd_adl(args) -> None:
    import numpy as np

    import awkward as ak
    from .adl import events_from_root, run_all

    print(f"loading {args.from_root} ...")
    events = events_from_root(args.from_root)
    print(f"  {len(events)} events "
          f"(jets/event mean {ak.mean(ak.num(events.jets, axis=1)):.2f}, "
          f"muons/event mean {ak.mean(ak.num(events.muons, axis=1)):.2f})\n")
    results = run_all(events)
    for name, arr in results.items():
        s = ("n=0" if arr.size == 0 else
             f"n={arr.size:6d}  mean={np.nanmean(arr):8.2f}  "
             f"min={np.nanmin(arr):7.2f}  max={np.nanmax(arr):7.2f}")
        print(f"  {name:32s} {s}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        for ax, (name, arr) in zip(axes.ravel(), results.items()):
            if arr.size:
                ax.hist(arr, bins=40, histtype="stepfilled",
                        color="#3b6fb6", alpha=0.85, edgecolor="#26456e")
            ax.set_title(name, fontsize=9)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print(f"\nsaved figure -> {args.plot}")


def _cmd_make_nano(args) -> None:
    import os
    from .sampledata import write_nanoaod
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_nanoaod(args.out, args.events, args.seed)
    print(f"wrote {args.out}: {args.events} events, tree 'Events'")


def _cmd_bench(args) -> None:
    conn_kwargs = dict(database=args.database, host=args.host, port=args.port,
                       user=args.user, password=args.password, dbdir=args.dbdir)
    backends = tuple(b.strip() for b in args.backends.split(",") if b.strip())
    if args.analysis == "trijet":
        from .benchmark import run_trijet
        nano = args.root_file if args.root_file != DEFAULT_DIMUON else DEFAULT_NANO
        # trijet has no `hybrid` backend; default to awkward vs in-DB SQL
        bk = backends if args.backends != "awkward,hybrid" else ("awkward", "monetdb")
        run_trijet(nano, backends=bk, repeats=args.repeats,
                   hybrid_backend=args.hybrid_backend, conn_kwargs=conn_kwargs)
    else:
        from .benchmark import run
        run(args.root_file, dataset=args.dataset, backends=backends,
            scale=args.scale, repeats=args.repeats,
            hybrid_backend=args.hybrid_backend, conn_kwargs=conn_kwargs)


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
def _add_server_args(sp) -> None:
    sp.add_argument("--database", default="hep")
    sp.add_argument("--host", default="localhost")
    sp.add_argument("--port", type=int, default=50000)
    sp.add_argument("--user", default="monetdb")
    sp.add_argument("--password", default="monetdb")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="awkward-monetizer",
                                description="Hybrid Awkward + MonetDB HEP engine.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="ROOT -> flat tables -> MonetDB")
    ing.add_argument("root_file", nargs="?", default=DEFAULT_DIMUON)
    ing.add_argument("--dataset", default="dimuon")
    ing.add_argument("--tree", default=None)
    ing.add_argument("--entry-stop", type=int, default=None)
    ing.add_argument("--truncate", action="store_true")
    ing.add_argument("--no-copy-into", action="store_true")
    ing.add_argument("--dry-run", action="store_true")
    _add_server_args(ing)
    ing.set_defaults(func=_cmd_ingest)

    rec = sub.add_parser("reconstruct", help="ROOT -> Awkward NF2 + validate (no DB)")
    rec.add_argument("root_file", nargs="?", default=DEFAULT_DIMUON)
    rec.add_argument("--dataset", default="dimuon")
    rec.set_defaults(func=_cmd_reconstruct)

    rt = sub.add_parser("roundtrip", help="full ingest -> MonetDB -> reconstruct")
    rt.add_argument("--root-file", default=DEFAULT_DIMUON)
    rt.add_argument("--dataset", default="dimuon")
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
    mn.set_defaults(func=_cmd_make_nano)

    bn = sub.add_parser("bench", help="benchmark backends (awkward/hybrid/rdataframe)")
    bn.add_argument("--root-file", default=DEFAULT_DIMUON)
    bn.add_argument("--dataset", default="dimuon")
    bn.add_argument("--analysis", choices=("zmumu", "trijet"), default="zmumu",
                    help="zmumu (dimuon) or trijet (ADL Q6, nanoaod)")
    bn.add_argument("--backends", default="awkward,hybrid",
                    help="comma-separated: awkward,hybrid,monetdb,rdataframe")
    bn.add_argument("--scale", type=int, default=1)
    bn.add_argument("--repeats", type=int, default=5)
    bn.add_argument("--hybrid-backend", choices=("server", "embedded"),
                    default="embedded")
    bn.add_argument("--dbdir", default=None)
    _add_server_args(bn)
    bn.set_defaults(func=_cmd_bench)

    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
