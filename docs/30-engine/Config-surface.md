---
tags: [engine, config]
aliases: [ProcessFaults, Settings, Parameters]
---

# Config-surface

All typed knobs live in `bdsim/config.py`. Public re-exports: `bdsim/__init__.py`. Operator recipes: [`USER.md`](../../USER.md).

## Main dataclasses

| Type | Role |
|------|------|
| `Parameters` | Physical constants, kinetics packs, feature rate constants |
| `Settings` | Horizon `ti/tf/dt`, `sv0`/`u0`, loop wiring, setpoints, disturbance profiles |
| `ProcessFaults` | Process feature gates and amplitudes |
| `SensorFaults` | Noise, bias templates; live bias/stuck/dropouts |
| `ValveFaults` | Stiction `S`, `J` per valve |
| `ARMAX` | Input disturbance ARMAX coeffs |
| `PIDController` | Gains and saturation |
| `Results` / `StepResult` | Batch trajectories / live sample |

## ProcessFaults field table

Defaults shown are exactly what `ProcessFaults()` produces as of 1.2.0
(many feature flags default to **`True`** — see footnote).

| Feature | Field | Type | Default | Purpose |
|-------|-------|------|---------|---------|
| (legacy) | `clog_fraction` | `float` | `5.95e-7` | filter clogging rate constant |
| (legacy) | `DPclean` | `float` | `1e5` | clean-filter ΔP (Pa) |
| (legacy) | `filter_std` | `float` | `5e-15` | filter pore-radius std |
| (legacy) | `ratio_robs_r` | `float` | `0.9` | side-reaction deactivation factor |
| (legacy) | `fouling` | `int` | `1` | 0 = off, 1 = on (pre-dynamic fouling series) |
| (legacy) | `foulingpar` | `np.ndarray` | `[3e-7]` | fouling rate parameter |
| **2.5** | `fouling_dynamic` | `bool` | **`True`** | α evolves as an ODE state; turn off for legacy fingerprint |
| **2.1** | `quality_state` | `bool` | **`True`** | master switch — adds 6 sv slots, latched QA channels; turn off for legacy fingerprint |
| **2.1** | `quality_lag_mode` | `str` | `"lab"` | `"lab"` (15 min) or `"online"` (60 s) |
| **2.1** | `lab_cycle_s` | `float` | `900.0` | lab sampling period |
| **2.1** | `online_cycle_s` | `float` | `60.0` | NIR sampling period |
| **2.1** | `lab_noise_fame` | `float` | `0.3` | FAME% noise (1σ, %) |
| **2.1** | `lab_noise_water` | `float` | `20.0` | water-ppm noise (1σ) |
| **2.1** | `lab_noise_iv` | `float` | `1.0` | IV noise (1σ, g I₂/100g) |
| **2.6** | `ambient_t_mean_k` | `float` | `293.15` | ambient baseline (K) |
| **2.6** | `ambient_t_amplitude_k` | `float` | `0.0` | daily sinusoid amplitude (K) |
| **2.6** | `ambient_t_period_s` | `float` | `86400.0` | sinusoid period (s) |
| **2.6** | `cw_t_mean_k` | `float` | `288.15` | CW inlet baseline (K) |
| **2.6** | `cw_t_amplitude_k` | `float` | `0.0` | seasonal sinusoid amplitude (K) |
| **2.6** | `cw_t_period_s` | `float` | `604800.0` | seasonal period (s) |
| **2.6** | `cw_p_nominal_pa` | `float` | `4.0e5` | nominal CW pressure (Pa) |
| **2.6** | `cw_p_drift_pa_per_h` | `float` | `0.0` | slow drift (Pa/h) |
| **2.6** | `cw_p_noise_pa` | `float` | `0.0` | jitter (1σ, Pa) |
| **2.6** | `met_cw_track` | `float` | `0.3` | Tmet shift per K of CW deviation |
| **2.6** | `oil_ambient_track` | `float` | `0.7` | Toil shift per K of ambient deviation |
| **2.6** | `qheat_cw_scaling` | `bool` | `True` | Qheat ∝ Pwater_cw / cw_p_nominal_pa |
| **2.6b** | `cw_pump_low_factor` | `float` | `0.3` | pressure floor during trip |
| **2.6b** | `cw_pump_ramp_s` | `float` | `30.0` | ramp down + ramp up (s) |
| **2.6b** | `cw_pump_default_duration_s` | `float` | `600.0` | default trip duration when FaultSpec omits one |
| **2.7** | `live_ambient_mean_k` | `float \| None` | `None` | operator override for `ambient_t_mean_k` |
| **2.7** | `live_ambient_amplitude_k` | `float \| None` | `None` | operator override for `ambient_t_amplitude_k` |
| **2.7** | `live_cw_t_mean_k` | `float \| None` | `None` | operator override for `cw_t_mean_k` |
| **2.7** | `live_cw_p_drift_pa_per_h` | `float \| None` | `None` | operator override for `cw_p_drift_pa_per_h` |
| **2.4** | `pump_wear` | `bool` | **`True`** | enables pump_health sv slot |
| **2.4** | `valve_wear` | `bool` | **`True`** | enables valve_stiction_pct sv slot |
| **2.4** | `pump_health_initial` | `float` | `1.0` | 1.0 = brand new, 0.05 = floor |
| **2.4** | `pump_wear_rate_per_h` | `float` | `0.01` | dh/dt baseline at nominal flow |
| **2.4** | `pump_wear_flow_exponent` | `float` | `1.5` | dh/dt ∝ (Q/Qnom)^p |
| **2.4** | `pump_wear_floor` | `float` | `0.05` | post-integration clamp |
| **2.4** | `pump_health_trip_threshold` | `float` | `0.25` | scenario-side trip likely below this |
| **2.4** | `valve_stiction_initial_pct` | `float` | `0.0` | 0 % = pristine, 100 % = full-stroke stuck |
| **2.4** | `valve_stiction_rate_pct_per_h` | `float` | `0.05` | grows proportional to `\|dlift/dt\|` |
| **2.4** | `valve_stiction_floor_pct` | `float` | `0.0` | lower bound |
| **2.4** | `valve_stiction_ceiling_pct` | `float` | `60.0` | above this → loop unstable |
| **2.8a** | `spectrum_enabled` | `bool` | **`True`** | master switch for NIR/IR sensor |
| **2.8a** | `spctr_t` | `float` | `3600.0` | spectrum sampling period (s) |
| **2.8a** | `spctr_cs` | `int` | `2` | Skoog photometric noise level (0..3) |
| **2.8a** | `spctr_snr_db` | `float` | `30.0` | AWGN SNR |
| **2.8a** | `spctr_k` | `float` | `0.03` | photometric noise scale |
| **2.8a** | `spctr_drift_a` | `float` | `0.01` | scatter baseline |
| **2.8a** | `spctr_drift_b` | `float` | `0.0001` | scatter linear term |
| **2.8a** | `spctr_drift_c` | `float` | `1.05` | scatter scaling term |
| **2.8a** | `spectra_ref_path` | `str \| None` | `None` | override reference spectra CSV |
| **2.8b** | `fouling_mode` | `int` | `0` | 0..5 — global mode selector (matches upstream) |
| **2.8b** | `fouling_mode_xRG_weight` | `bool` | `True` | mode 4 couples to glycerol mole fraction |
| **2.8b** | `fouling_ar_eps_std` | `float` | `5e-4` | ARMAX innovation σ (modes 4/5) |
| **2.8b** | `fouling_mode_default_window_s` | `float` | `3600.0` | default fault-window length |
| **2.8b** | `fouling_mode_active_mode` | `int` | `0` | runtime overlay: 0 = off, 4 / 5 = ARMAX |
| **2.8b** | `fouling_mode_active_end_t` | `float` | `-1.0` | sim time at which the active window expires |
| **2.8b** | `fouling_mode_active_seed` | `int \| None` | `None` | optional seed for ARMAX RNG (reproducibility) |

> [!note]
> **Default on (1.2.0+):** dynamic fouling, quality latching, pump + valve wear, and the NIR/IR spectrum sensor all default to **on**. Bare `ProcessFaults()` enables them all → `sv` width 30, hash `691cf51b…`. Tests and demos that need a narrower state vector must pass the relevant `False` overrides explicitly. Named profiles: [Byte-identical-contract](../00-orientation/Byte-identical-contract.md#canonical-profiles).

## Settings horizon

Default ~72 h: `tf=260000`, `dt=5` → ~52k samples (drivers drop the last point like upstream).

Related: [Config-surface](../30-engine/Config-surface.md), [Channel-indices](../40-helpers/Channel-indices.md), [Batch-driver](../30-engine/Batch-driver.md), [Live-simulator](../30-engine/Live-simulator.md), [Home](../Home.md)
