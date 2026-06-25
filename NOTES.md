# Notes — issues found in the port

## 1. Foil measurement was wildly noisy (root-caused 2026-06-25, FIXED)

### Symptom

`fig08_foil_loop` showed Foil swinging between −74,191 and +75,427 kg/h,
centered at ~3,403 kg/h (setpoint ~3,050 kg/h). 42% of samples negative.

### Two distinct bugs

#### Bug 1a — Fmet double-conversion (Python port bug)

`bdsim/simulation.py` line 521 (pre-fix) ran:
```python
uv[:, 2] = uv[:, 2] * p.Mm * 3600.0
```
converting Fmet from mol/s → kg/h at the end of the simulation loop.

But `bdsim/config.py` `Results.in_display_units()` ALSO ran:
```python
r.uv[:, 2] = r.uv[:, 2] * 0.032 * 3600.0
```
on top of that, treating already-kg/h values as if they were mol/s.

Result: Fmet ended up at `657 × 115.2 ≈ 75,686 kg/h` instead of 657 kg/h.
Same multiplier applied to Tmet, Qheat, etc — but those were already in
display units from `Settings.u0`, so they happened to round-trip OK
(other than Fmet, which `Settings.u0` stored in kg/h → mol/s at startup
→ "restored" to kg/h → multiplied by 115.2 again at display).

**Fix:** removed the simulation-end restoration; left uv in mol/s and let
`in_display_units()` do the conversion. This matches the upstream MATLAB
intent where the conversion lives only in the plotting block.

#### Bug 1b — Foil sensor noise 590× too high (upstream typo, fixed by override)

`SensorFaults.noise_std[3] = 5.0` kg/s, applied to `pv[3] = Qo * roo`
(a ~0.85 kg/s signal). After display conversion, that's 18,000 kg/h σ
on a 3,050 kg/h setpoint → 590% relative noise. By contrast:

| Sensor | noise_std  | signal     | relative |
|--------|-----------:|-----------:|---------:|
| TR     | 0.1 K      | 333 K      | 0.03 %   |
| TD     | 0.1 K      | 323 K      | 0.03 %   |
| hH     | 5e-3 m     | 0.5 m      | 1.0 %    |
| **Foil (was)** | **5.0 kg/s** | **0.85 kg/s** | **590 %** |
| **Foil (now)** | **5e-3 kg/s** | **0.85 kg/s** | **0.6 %** |
| DP     | 300 Pa     | 1e5 Pa     | 0.3 %    |

The upstream MATLAB `user_settings.m` line 182 has the same value
(`noise_std = [0.1, 0.1, 5e-3, 5, 300]`) — almost certainly a typo in
Fernandes et al. (2019) (should be `5e-3` matching the hH convention).

**Fix:** changed the port default to `5e-3` kg/s (≈ 18 kg/h σ ≈ 0.6%
relative noise, realistic for a Coriolis flowmeter). Joel explicitly
opted for breaking upstream fidelity here ("I don't mind changing the
benchmark if it gets me a good simulation").

### Verification

```
=== After fixes ===
uv (display):
  Tmet: 46.61 - 53.40 °C
  Fmet: 652.24 - 662.14 kg/h (expected ~657)        ✓ was 75,686
  Toil: 52.74 - 62.89 °C
pv (display):
  TR:   59.98 - 60.98 °C                              ✓
  TD:   45.98 - 50.21 °C                              ✓
  hH:   0.4750 - 0.5201 m                             ✓
  Foil: 2929.28 - 3193.63 kg/h (setpoint ~3050)      ✓ was ±75k
  DP:   4245.07 - 11562.19 Pa                         ✓ was −1312 to 19157
```

All 12 smoke tests still pass.

### If you need the upstream-faithful noise back

Pass an explicit `SensorFaults` with the original value:
```python
from bdsim.simulation import run_with
from bdsim.config import SensorFaults
import numpy as np

sfaults = SensorFaults()
sfaults.noise_std = np.array([0.1, 0.1, 5e-3, 5.0, 300.0])
results = run_with(sfaults=sfaults, seed=42)
```

Useful when generating test data for fault-detection algorithms where
you want realistic (or deliberately noisy) sensor behavior.
## 2. Temperature plots show ~−213 °C (FIXED 2026-06-25)

### Symptom

`fig01_tr_td_toil_qheat`, `fig05_tr_loop`, `fig06_td_loop`,
`fig03_tmet_oil_valve` all showed temperatures around −200 to −227 °C
instead of 30–60 °C. The CSV files (`results/inputs`, `results/states`,
`results/measurements`) were already correct — only the plots were wrong.

### Root cause

Same family of bug as Bug 1a. `plot_all()` calls
`results.in_display_units()` to convert K → °C, but each factory
function then did `- 273.15` AGAIN on the already-converted data:

```python
# plot_all()
r = results.in_display_units()    # sv[6] (TR) goes from 333.55 K → 60.40 °C
for name, factory in _FIGS:
    fig = factory(r)              # r.sv[6] is already in °C

# _fig01_tr_td_toil_qheat(r)
fig.add_trace(go.Scatter(x=th, y=r.sv[:, 6] - 273.15, name="TR"))
#                                  ^^^^^^^^^^^^^^^^^^^^^^^ subtracting 273.15 from °C → ~−213 °C
```

Same pattern in `_fig03`, `_fig05`, `_fig06` (4 traces total).

### Fix

Removed the `- 273.15` from each factory; left the conversion in
`in_display_units()` (the canonical place). Added comments explaining
that `r` is already in display units.

### Verification

```
=== After fix ===
fig01: TR 60.38–60.59 °C  TD 46.24–50.00 °C  Toil 52.74–62.89 °C  Qheat 23376–24366 W
fig05: setpoint 60.40  measurement 59.98–60.98  state 60.38–60.59 (TR loop, °C)
fig06: setpoint 50.00  measurement 45.98–50.21  state 46.24–50.00 (TD loop, °C)
fig03: Tmet 46.61–53.40 °C  order_lift_oil 39.62–41.93 %
```

All 12 smoke tests still pass.
