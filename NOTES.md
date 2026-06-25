# Notes — issues found while reviewing the port

## Foil measurement is extremely noisy (2026-06-25)

The `fig08_foil_loop` plot shows Foil swinging between −74,191 and +75,427 kg/h,
centered at ~3,403 kg/h (setpoint is 3,050 kg/h).

### Root cause

`SensorFaults.noise_std[3] = 5.0` (kg/s). Applied to the oil flow measurement
`pv[3] = Qo * roo` (kg/s). After display conversion (`pv[:, 3] *= 3600` in
`in_display_units()`), that's **18,000 kg/h of noise sigma** on a signal of
~3,050 kg/h — a 590% relative noise. By contrast:

| Sensor | noise_std  | signal     | relative |
|--------|-----------:|-----------:|---------:|
| TR     | 0.1 K      | 333 K      | 0.03 %   |
| TD     | 0.1 K      | 323 K      | 0.03 %   |
| hH     | 5e-3 m     | 0.5 m      | 1.0 %    |
| **Foil** | **5.0 kg/s** | **0.85 kg/s** | **590 %** |
| DP     | 300 Pa     | 1e5 Pa     | 0.3 %    |

### Status

Confirmed by re-running with `noise_std[3] = 0`:

```
With Foil noise = 0:
  Foil: min=2997.94, max=3100.00, mean=3098.06, std=13.7557
```

Tracking is then perfect. So the noise is **the entire cause** of the
negative values and the wide spread.

### Upstream vs port

The original MATLAB `user_settings.m` line 182 has the same value:

```matlab
sfaults.noise_std  = [     0.1      0.1     5e-3        5      300];
```

So this is **upstream-faithful behavior**, not a port bug. The value is
probably a typo in Fernandes et al. (2019) — should plausibly be `5e-3`
(matching the hH convention) — but the port preserves the upstream
defaults verbatim to stay faithful to the benchmark.

### Why "centered around 0"

With σ ≈ 18,000 kg/h and a signal mean of 3,050 kg/h, the noise
dominates the signal. The empirical distribution of `pv[3] = signal + noise`
is approximately `N(3050, 18000²)`, so the lower tail goes deeply negative
and the histogram looks symmetric around the mean — which a casual viewer
reads as "centered at zero".

### Fixes (if you want a cleaner plot)

Three options, no upstream validation needed:

1. **For visualization only** — zero out Foil noise when calling the
   simulator: pass a custom `SensorFaults` with `noise_std[3] = 0`.

2. **For a clean baseline run** — add a CLI flag `--no-foil-noise` that
   overrides `noise_std[3]`.

3. **Document as the upstream-typo hypothesis** — leave the default
   unchanged for benchmark fidelity, but add a comment in `config.py`
   pointing at this note.

Joel decided to keep the upstream defaults unchanged for now (preserves
benchmark fidelity for downstream fault-detection R&D).