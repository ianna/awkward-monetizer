"""Consume the pipeline from an ML training loop (thin wrapper over
``awkward_monetizer.ml``).

Examples
--------
Self-contained demo (no ROOT file, no database)::

    python examples/ml_training.py --source synthetic --n 100000 --mode train

Throughput on a ROOT file::

    python examples/ml_training.py --source root --root-file data/nano_bench.root \
        --mode throughput

From a running MonetDB `hep` database, MET-selected slice pushed down to SQL::

    python examples/ml_training.py --source monetdb --where "met_pt > 30" --mode train
"""

import argparse
import json

from awkward_monetizer import ml


def load_events(args):
    if args.source == "synthetic":
        return ml.synthetic_events(args.n, avg_jets=args.avg_jets, seed=args.seed)
    if args.source == "root":
        return ml.events_from_root(args.root_file)
    if args.source == "monetdb":
        return ml.events_from_monetdb(database=args.database, where=args.where)
    raise ValueError(args.source)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="synthetic",
                   choices=("synthetic", "root", "monetdb"))
    p.add_argument("--mode", default="train", choices=("train", "throughput"))
    p.add_argument("--root-file", default="data/nano_bench.root")
    p.add_argument("--database", default="hep")
    p.add_argument("--where", default=None, help="SQL predicate pushed to MonetDB")
    p.add_argument("--n", type=int, default=100_000, help="synthetic event count")
    p.add_argument("--avg-jets", type=float, default=4.0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    events = load_events(args)
    print(f"loaded {len(events):,} events from {args.source}")

    if args.mode == "throughput":
        result = ml.throughput(events, batch_size=args.batch_size,
                               epochs=max(args.epochs // 5, 1), device=args.device)
        print("\nevents/second:")
        for stage in ("bridge", "loader", "train"):
            print(f"  {stage:<8} {result[stage]['events_per_s']:>12,.0f}")
    else:
        result = ml.train(events, epochs=args.epochs, batch_size=args.batch_size,
                          device=args.device, seed=args.seed)
        print(f"\nfinal val loss {result['final_val_loss']:.4f}  "
              f"val R² {result['val_r2']:.3f}  "
              f"({result['train_events_per_s']:,.0f} train events/s)")
    print("\n" + json.dumps({k: v for k, v in result.items() if k != "history"},
                            indent=2, default=float))


if __name__ == "__main__":
    main()
