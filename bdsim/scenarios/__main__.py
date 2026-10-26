"""CLI: ``python -m bdsim.scenarios S00|…|all [--npoints N] [--outdir data/results_py]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .gold import SCENARIO_IDS, run_gold_scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bdsim.scenarios",
        description="Run Octave-gold scenarios S00–S07 (Parquet export).",
    )
    parser.add_argument(
        "scenarios",
        nargs="+",
        help="Scenario id(s) S00–S07, or 'all' / 'SMOKE'.",
    )
    parser.add_argument(
        "--outdir",
        default="data/results_py",
        help="Root directory for per-scenario Parquet folders (default: data/results_py).",
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=None,
        help="Truncate like Octave BD_NPOINTS (SMOKE uses 100).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42).")
    parser.add_argument("--tf", type=float, default=None, help="Override final time [s].")
    parser.add_argument("--dt", type=float, default=30.0, help="Sample interval [s].")
    parser.add_argument("-q", "--quiet", action="store_true", help="Less driver logging.")
    args = parser.parse_args(argv)

    npoints = args.npoints
    names: list[str] = []
    for tok in args.scenarios:
        u = tok.upper()
        if u in ("ALL", "SMOKE"):
            names = list(SCENARIO_IDS)
            if u == "SMOKE" and npoints is None:
                npoints = 100
            break
        names.append(u)

    root = Path(args.outdir)
    tf = 604800.0 if args.tf is None else args.tf
    for sid in names:
        out = root / sid
        print(f"=== {sid} → {out} (npoints={npoints}) ===")
        run_gold_scenario(
            sid,
            seed=args.seed,
            npoints=npoints,
            tf=tf,
            dt=args.dt,
            outdir=out,
            verbose=not args.quiet,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
