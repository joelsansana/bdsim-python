# Extended scenarios S08–S15 — port alignment

Status: mirrors Octave PR (BDSIM `feat/new-scenarios-gaps`) one-for-one.
Authoritative Octave spec: `scenarios/GOLD_SCENARIOS.md` §"Extended scenarios
S08–S15" and `docs/scenario_coverage.md` (finding F0) in the BDSIM repo.

## What this adds

`bdsim/scenarios/gold.py` now builds and runs S08–S15 with the **same overrides and
the same per-scenario time base** as the Octave scenario files:

| Id | Gap | Horizon | dt / nic | Deviations from S00 defaults |
|----|-----|---------|----------|------------------------------|
| S08 | G1 | 1 week | 30 s / 1 | `noise_std = [0.1, 0.1, 5e-3, 5, 300]` |
| S09 | G2 | 1 week | 30 s / 1 | loop 4 MANUAL; `armax.eta = [1,1,1,0,0,0]`; `d(:,1) = 40 + 5·H(t−2d) − 5·H(t−4d)`; `d(:,2) += 3·sin(2π t/T_DAY)` |
| S10 | G3 | 1 week | 30 s / 1 | SP steps: TR +2 K [d1,d3), TD +2 K [d2,d5), hH +0.1 m [d4,d6) |
| S11 | G4 | 1 month | 60 s / 1 | fouling `e = 6e-7` (Qheat pins at the 40 kW bound from ~day 13.4) |
| S12 | G5 | 1 week | 30 s / 1 | `vfaults.S = [2,3]`, `J = [2,0]`; hH/Foil SP steps at d1/d4 |
| S13 | G7 | 1 week | 30 s / 1 | TR scaling [d1,d4), TD drift from d2, Foil stuck from d5; `isIntermit = [1,1,0,1,0]`, `tmaxInterm = [3600,3600,0,3600,0]` |
| S14 | G6 | 1 month | 60 s / 1 | clogging + `filter_std = 6e10` (~1 % clean-filter jitter, ~14 cleanings) |
| S15 | G8 | 6 months | 300 s / 1 | clogging + fouling `e = 3e-7` + `ratio_robs_r = 0.9`; loop 4 MANUAL, oil order 40 % via `eta(1) = 1` |

Time bases live in `SCENARIO_TIMEBASE` (`tf`, `dt`, `nic`); `run_gold_scenario`
resolves `tf`/`dt` from it when not passed explicitly, so

```bash
./scripts/run_gold_scenarios.sh S08 S15     # uses each scenario's own time base
./scripts/run_gold_scenarios.sh SMOKE       # BD_NPOINTS=100 truncate, all 16
```

The runner scripts no longer force `--dt 30` (that would have overridden the
60 s / 300 s scenarios); `BD_DT` is only forwarded when explicitly set.

## RNG policy (Port_execution.md §5)

S08 (sensor noise), S13 (intermittence masks) and S14 (new-filter draws) consume
random streams. Octave legacy `rand`/`randn` ≠ NumPy at the same seed, so:

- S08/S13/S14 parity is **fingerprint-level** (signal scales, means, burst counts,
  cleaning counts), never sample-wise RMSE.
- S09, S10, S11, S12 and S15 are fully deterministic and must agree with Octave
  gold to the usual solver-drift tolerance (RK45 vs `lsode`).

## Comparison tool

`scripts/compare_gold.py` (this repo) prints per-table, per-column
RMSE / max|err| / bias / gold-scale for any (Octave gold dir, Python parquet dir)
pair and flags the RNG-dependent scenarios. Assertion gates remain in
`tests/test_gold_parity.py`.

## Results (2026-11-16)

See `data/results_octave/gold_dataset_ext/` (Octave full-horizon runs of S08–S15,
copied from the BDSIM `results_ref/` suite) vs `data/results_py/S08–S15/`; the
comparison tables are recorded in `docs/30-engine/Extended-scenarios-S08-S15.md`
and the progress file of the scenario-review chain.
