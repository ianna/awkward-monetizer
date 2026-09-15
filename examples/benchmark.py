#!/usr/bin/env python3
"""Run the backend benchmark (thin wrapper over awkward_monetizer.benchmark).

Equivalent to ``awkward-monetizer bench``. Example:

    python examples/benchmark.py --scale 100 --backends awkward,hybrid
"""

import argparse

from awkward_monetizer.benchmark import run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root-file", default="data/cms.root")
    p.add_argument("--dataset", default="dimuon")
    p.add_argument("--backends", default="awkward,hybrid",
                   help="comma-separated: awkward,hybrid,monetdb,rdataframe")
    p.add_argument("--scale", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--hybrid-backend", default="embedded",
                   choices=("server", "embedded"))
    args = p.parse_args()
    run(args.root_file, dataset=args.dataset,
        backends=tuple(b.strip() for b in args.backends.split(",")),
        scale=args.scale, repeats=args.repeats,
        hybrid_backend=args.hybrid_backend)


if __name__ == "__main__":
    main()
