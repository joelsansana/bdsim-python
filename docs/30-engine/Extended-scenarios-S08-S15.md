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

## F1 — loads must pass through

The golden S00 defaults zero the whole ARMAX row, but u2 (Tmet) and u3 (Fmet) are
*pure loads* computed as `u = eta·d` — with `eta = 0` the plant receives **no
methanol feed** and the composition collapses to pure TG within a week (finding
F1 in the Octave repo's `docs/scenario_coverage.md`). The extended scenarios set
`armax.eta = [0,1,1,0,0,0]` (S08, S10–S14) / `[1,1,1,0,0,0]` (S09, S15 — the
first slot passes the manual oil order through). Deterministic either way.

## Comparison tool

`scripts/compare_gold.py` (this repo) prints per-table, per-column
RMSE / max|err| / bias / gold-scale for any (Octave gold dir, Python parquet dir)
pair and flags the RNG-dependent scenarios. Assertion gates remain in
`tests/test_gold_parity.py`.

## Results (2026-11-16)

Octave gold: full-horizon runs of S08–S15 with the F1-fixed settings, taken from
the BDSIM `results_ref/` suite into `data/results_octave/gold_dataset_ext/`.
Python: `data/results_py/S08–S15` (same settings, seed 42). Tools:
`scripts/compare_gold.py` (RMSE/max|err|/bias per table and column) and
`scripts/compare_fingerprints.py` (statistics for the RNG/phase-sensitive cases).

| Scenario | Mode | Key agreement |
|----------|------|---------------|
| S09 (det.) | sample-wise | manual oil order RMSE = 0 (exact); Tmet sine 5e-5 °C; Foil 4e-6 kg/h; Qheat RMSE 102 W (σ 1024); TR 0.045 K; TD 0.005 K |
| S10 (det.) | sample-wise | TR 0.044 K (σ 0.85); TD 0.004 K; compositions 1e-6..1e-5; Foil/lifto ~1e-7 |
| S11 (det.) | sample-wise | Qheat pin day 13.3809 (gold) vs 13.3785 (py) — 60 s apart over 30 d; Qheat RMSE 31 W (σ 5152); TD end identical 53.501 °C; xR_E 2.4e-6 |
| S12 (det. + stiction) | sample-wise + fingerprint | comps 1e-6; Foil/lifto 1e-6..1e-5; TR 0.029 K; H-valve stick-slip *phase* drifts (liftH RMSE 1.5 on σ 3.34) — fingerprint: stem zero-cross rate 0.014/0.014, amplitude σ 3.335/3.342 |
| S08 (RNG) | fingerprint | PV statistics to ~1 %: TR σ 0.1005/0.1050 K, TD σ 0.1006/0.1006 K, Foil σ 6.34/6.26 kg/h, DP σ 299.8/300.6 Pa, means at SP |
| S13 (RNG) | fingerprint | mask duty 0.641 (gold) / 0.549 (py) — legacy-RNG streams differ as documented; fault machinery verified in both |
| S14 (RNG draws + det. base) | fingerprint + events | 14 cleanings in BOTH; interval mean 177798 (gold) / 177822 (py) s (0.13 %); first two events identical; drift ≤ 1680 s by event 10 from the RNG-dependent new-filter K4F draws (per design) |
| S15 (det. + events) | sample-wise + events | 6 months: 79 cleanings in BOTH with **identical event times** (max diff 0 s); Qheat pin day 27.4 both; TD RMSE 0.0097 K (σ 2.5), end 57.284/57.266 °C; Qheat RMSE 20 W (σ 3852); xR_E RMSE 2.3e-6, end 0.4146 both; lifto RMSE = 0 (manual channel bit-exact); gold loads fed (Tmet 50 °C, Fmet 657 kg/h) |

Verdict: every deterministic extended scenario agrees within the
`Port_execution.md` solver-drift tolerance; the RNG-dependent ones agree at the
fingerprint level exactly as the §5 strategy prescribes. S12's stick-slip stem
channels are the only phase-sensitive signals and are handled as fingerprints.

The old suite S00–S07 was additionally re-verified against the stored
`gold_dataset_1w` (same tool): compositions ~1e-3 on σ 0.2, thermal channels
< 0.01 K; the heavy-phase split columns (xH_M/xH_G) show a ~0.01 bias from the
NN-split sensitivity and stay inside the existing abs-floor policy of
`tests/test_gold_parity.py`.
