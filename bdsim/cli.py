"""
Command-line entry point: ``python -m bdsim [seed]``

Usage:
    python -m bdsim              # run with upstream defaults, seed from system
    python -m bdsim 42           # reproducible run with seed 42
    python -m bdsim --no-plots   # CSV only, no PNG figures
"""

from __future__ import annotations

import argparse
import sys

from .plots import plot_all, save_csv
from .simulation import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bdsim",
        description="Biodiesel process simulator (Python port of BDSIM).",
    )
    parser.add_argument("seed", nargs="?", type=int, default=None,
                        help="RNG seed for reproducible runs.")
    parser.add_argument("--outdir", default="results",
                        help="Output directory for CSVs and PNGs.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip PNG figure generation (CSV only).")
    args = parser.parse_args(argv)

    results = run(seed=args.seed)
    files = save_csv(results, outdir=args.outdir)
    print("Saved CSVs:")
    for name, path in files.items():
        print(f"  {name}: {path}")

    if not args.no_plots:
        paths = plot_all(results, outdir=args.outdir, show=False)
        print(f"Saved {len(paths)} figures to {args.outdir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())