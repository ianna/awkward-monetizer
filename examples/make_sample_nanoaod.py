"""Write a synthetic NanoAOD ROOT file (thin wrapper over the package).

Equivalent to ``awkward-monetizer make-nano``.
"""

import argparse
import os

from awkward_monetizer.sampledata import write_nanoaod


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/nano_synth.root")
    p.add_argument("--events", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_nanoaod(args.out, args.events, args.seed)
    print(f"wrote {args.out}: {args.events} events, tree 'Events'")


if __name__ == "__main__":
    main()
