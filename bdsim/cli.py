"""
Command-line entry point: ``python -m bdsim [seed]``

Usage:
    python -m bdsim              # run with upstream defaults, seed from system
    python -m bdsim 42           # reproducible run with seed 42
    python -m bdsim --no-plots   # Parquet only, no figures
"""

from __future__ import annotations

import argparse
import sys

from .io_parquet import save_parquet
from .plots import plot_all
from .simulation import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bdsim",
        description="Biodiesel process simulator (Python port of BDSIM).",
    )
    parser.add_argument("seed", nargs="?", type=int, default=None,
                        help="RNG seed for reproducible runs.")
    parser.add_argument("--outdir", default="data/results",
                        help="Output directory for Parquet tables and HTML figures.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip HTML figure generation (Parquet only).")
    args = parser.parse_args(argv)

    results = run(seed=args.seed)
    files = save_parquet(results, outdir=args.outdir)
    print("Saved Parquet:")
    for name, path in files.items():
        print(f"  {name}: {path}")

    if not args.no_plots:
        paths = plot_all(results, outdir=args.outdir, show=False)
        print(f"Saved {len(paths)} figures to {args.outdir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())