#!/usr/bin/env python3
"""Compare Python gold Parquet vs Octave gold ASCII, per scenario.

Usage:
    .venv/bin/python scripts/compare_gold.py                 # all scenarios
    .venv/bin/python scripts/compare_gold.py S08 S15         # subset
    GOLD_ROOT=data/results_octave/gold_dataset_ext PY_ROOT=data/results_py

For each scenario dir under GOLD_ROOT/PY_ROOT, per table and per column:
RMSE, max |err|, mean bias, and the gold signal scale. Scenarios flagged
RNG-dependent in the scenario review (S08 noise, S13 intermittence masks,
S14 new-filter draws) are reported but marked fingerprint-level: Octave's
legacy rand/randn streams are not reproducible in NumPy at the same seed
(Port_execution.md §5), so per-sample agreement there is NOT expected —
judge those on statistics (means/scales), not RMSE.

Exit code 0 always (report tool, not an assertion gate); assertions live in
tests/test_gold_parity.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bdsim.io_parquet import (  # noqa: E402
    INPUT_COLS,
    MEASUREMENT_COLS,
    SETPOINT_COLS,
    STATE_COLS,
    load_ascii_gold,
    load_parquet_dir,
)

# Scenarios whose randomness makes sample-wise RMSE meaningless (RNG strategy
# note, Port_execution.md §5): compare statistics, not trajectories.
RNG_SCENARIOS = {"S06", "S08", "S13", "S14"}

TABLE_COLS = {
    "measurements": MEASUREMENT_COLS,
    "setpoints": SETPOINT_COLS,
    "states": STATE_COLS,
    "inputs": INPUT_COLS,
}


def _compare_one_table(py_df, gold_df, cols):
    g = gold_df.set_index("t_s")
    p = py_df.set_index("t_s")
    common = g.index.intersection(p.index)
    if len(common) < 10:
        n = min(len(g), len(p))
        g = g.iloc[:n]
        p = p.iloc[:n]
    else:
        g = g.loc[common]
        p = p.loc[common]
    rows = []
    for c in cols:
        if c == "t_s" or c not in g.columns or c not in p.columns:
            continue
        gv = np.asarray(g[c], dtype=float)
        pv = np.asarray(p[c], dtype=float)
        diff = pv - gv
        rows.append((
            c,
            float(np.sqrt(np.nanmean(diff ** 2))),
            float(np.nanmax(np.abs(diff))),
            float(np.nanmean(diff)),
            float(np.nanstd(gv)),
        ))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenarios", nargs="*", help="Scenario ids (default: all found in GOLD_ROOT)")
    ap.add_argument("--gold-root", default=os.environ.get("GOLD_ROOT", str(ROOT / "data/results_octave/gold_dataset_ext")))
    ap.add_argument("--py-root", default=os.environ.get("PY_ROOT", str(ROOT / "data/results_py")))
    args = ap.parse_args(argv)

    gold_root = Path(args.gold_root)
    py_root = Path(args.py_root)

    ids = args.scenarios or sorted(d.name for d in gold_root.iterdir() if d.is_dir())
    for sid in ids:
        gdir, pdir = gold_root / sid, py_root / sid
        if not (gdir / "measurements").is_file():
            print(f"{sid}: no Octave gold under {gdir} — skipped")
            continue
        if not (pdir / "measurements.parquet").is_file():
            print(f"{sid}: no Python parquet under {pdir} — skipped")
            continue
        gold = load_ascii_gold(gdir)
        py = load_parquet_dir(pdir)
        rng_note = " [RNG-dependent: fingerprint-level only]" if sid in RNG_SCENARIOS else ""
        print(f"== {sid}{rng_note} ==")
        for name, cols in TABLE_COLS.items():
            if name not in gold or name not in py:
                continue
            rows = _compare_one_table(py[name], gold[name], cols)
            if not rows:
                continue
            print(f"  {name}:")
            for c, rmse, mx, bias, gstd in rows:
                print(f"    {c:<22s} RMSE={rmse:<12.5g} max|e|={mx:<12.5g} "
                      f"bias={bias:<12.5g} gold_std={gstd:<12.5g}")
        if "tclean" in gold and "tclean" in py:
            gt = np.asarray(gold["tclean"]["t_s"])
            pt = np.asarray(py["tclean"]["t_s"])
            print(f"  tclean: gold n={len(gt)} first={gt[0] if len(gt) else '-'} | "
                  f"py n={len(pt)} first={pt[0] if len(pt) else '-'}")
        elif "tclean" in gold or "tclean" in py:
            print(f"  tclean: gold={'yes' if 'tclean' in gold else 'no'} "
                  f"py={'yes' if 'tclean' in py else 'no'} (MISMATCH)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
