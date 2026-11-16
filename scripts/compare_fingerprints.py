"""Fingerprint-level comparison for stochastic / phase-sensitive scenarios.

Computes statistics (mean/std/min/max, event counts) for selected signals of
the extended scenarios where sample-wise RMSE is not the right metric:
S08/S13/S14 (Octave RNG streams) and the stiction stem channels of S12.
"""
import os
import sys

os.chdir('/home/eugeniu/repos/bdsim-python')
sys.path.insert(0, '/home/eugeniu/repos/bdsim-python')
import numpy as np
from bdsim.io_parquet import load_ascii_gold, load_parquet_dir

GOLD = 'data/results_octave/gold_dataset_ext'
PY = 'data/results_py'


def stats(v):
    v = np.asarray(v, dtype=float)
    return dict(n=len(v), mean=float(np.mean(v)), std=float(np.std(v)),
                min=float(np.min(v)), max=float(np.max(v)))


def zero_cross_rate(v):
    v = np.asarray(v, dtype=float)
    s = np.sign(v - np.mean(v))
    return float(np.mean(np.abs(np.diff(s)) > 0))


def show(sid, table, col, label=None):
    g = load_ascii_gold(os.path.join(GOLD, sid))
    p = load_parquet_dir(os.path.join(PY, sid))
    if table not in g or table not in p:
        print(f"{sid}/{table}: missing")
        return
    gv = np.asarray(g[table][col])
    pv = np.asarray(p[table][col])
    n = min(len(gv), len(pv))
    sg, sp = stats(gv[:n]), stats(pv[:n])
    print(f"{sid} {table}.{col} ({label or col}):")
    print(f"   gold: mean={sg['mean']:.5g} std={sg['std']:.5g} min={sg['min']:.5g} max={sg['max']:.5g}")
    print(f"   py  : mean={sp['mean']:.5g} std={sp['std']:.5g} min={sp['min']:.5g} max={sp['max']:.5g}")


if __name__ == '__main__':
    print("== S08 noise (RNG): PV statistics ==")
    show('S08', 'measurements', 'TR_C', 'TR noise')
    show('S08', 'measurements', 'TD_C', 'TD noise')
    show('S08', 'measurements', 'Foil_kg_h', 'Foil noise')
    show('S08', 'measurements', 'DPfilter_Pa', 'DP noise')

    print("== S13 intermittent masks (RNG): burst duty ==")
    g = load_ascii_gold(os.path.join(GOLD, 'S13'))
    p = load_parquet_dir(os.path.join(PY, 'S13'))
    t = np.asarray(p['measurements']['t_s'])
    win = (t >= 1.5*86400) & (t <= 3.5*86400)
    for sid, df in (('gold', g), ('py', p)):
        mm = np.asarray(df['measurements']['TR_C'])
        ss = np.asarray(df['states']['TR_C'])
        d = mm[win] - ss[win]
        print(f"   {sid}: TR fault-active fraction (|meas-state|>2K) = {np.mean(np.abs(d) > 2):.3f}")

    print("== S14 cleaning variability (RNG): event statistics ==")
    for sid, df in (('gold', load_ascii_gold(os.path.join(GOLD, 'S14'))),
                    ('py', load_parquet_dir(os.path.join(PY, 'S14')))):
        if 'tclean' in df:
            tt = np.asarray(df['tclean']['t_s'])
            print(f"   {sid}: n={len(tt)} period mean={np.mean(np.diff(tt)):.0f} s "
                  f"period std={np.std(np.diff(tt)):.0f} s")

    print("== S12 stiction phase-sensitive channels (fingerprint) ==")
    g = load_ascii_gold(os.path.join(GOLD, 'S12'))
    p = load_parquet_dir(os.path.join(PY, 'S12'))
    for table, col in (('states', 'liftH_pct'), ('states', 'lifto_pct'),
                       ('inputs', 'order_lift_H_pct'), ('inputs', 'order_lift_oil_pct')):
        gv, pv = np.asarray(g[table][col]), np.asarray(p[table][col])
        n = min(len(gv), len(pv))
        print(f"   {table}.{col}: zcr gold={zero_cross_rate(gv[:n]):.3f} py={zero_cross_rate(pv[:n]):.3f} "
              f"std gold={np.std(gv[:n]):.4f} py={np.std(pv[:n]):.4f}")
