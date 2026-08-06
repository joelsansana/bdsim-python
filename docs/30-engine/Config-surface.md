---
tags: [engine, config]
aliases: [ProcessFaults, Settings, Parameters]
---

# Config-surface

All typed knobs live in `bdsim/config.py`. Public re-exports: `bdsim/__init__.py`. Operator recipes: [`USER.md`](../../USER.md).

## Main dataclasses

| Type | Role |
|------|------|
| `Parameters` | Physical constants, kinetics packs, Layer rate constants |
| `Settings` | Horizon `ti/tf/dt`, `sv0`/`u0`, loop wiring, setpoints, disturbance profiles |
| `ProcessFaults` | Process / Layer feature gates and amplitudes |
| `SensorFaults` | Noise, bias templates; live bias/stuck/dropouts |
| `ValveFaults` | Stiction `S`, `J` per valve |
| `ARMAX` | Input disturbance ARMAX coeffs |
| `PIDController` | Gains and saturation |
| `Results` / `StepResult` | Batch trajectories / live sample |

## ProcessFaults Layer flags (selected)

| Flag | Layer | Default (code today) | Effect |
|------|-------|----------------------|--------|
| `quality_state` | 2.1 | `False` | QA states + latching; widens `sv` |
| `fouling_dynamic` | 2.5 | **`True`** | Continuous α at `sv[21]` |
| `pump_wear` / `valve_wear` | 2.4 | `False` | Wear states |
| `spectrum_enabled` | 2.8a | `False` | NIR/IR samples |
| disturbance amplitudes | 2.6 | `0.0` | Tamb / Tcw / Pcw tracks |
| `fouling_mode*` | 2.8b | off | Windowed ARMAX fouling |

> [!note]
> **Exception:** Layer 2.5 is **default-on** (`fouling_dynamic=True`). Bare `ProcessFaults()` ≠ legacy fingerprint profile. Named profiles: [[Byte-identical-contract#Canonical profiles]].

## Settings horizon

Default ~72 h: `tf=260000`, `dt=5` → ~52k samples (drivers drop the last point like upstream).

Related: [[Layers-roadmap]], [[Channel-indices]], [[Batch-driver]], [[Live-simulator]], [[Home]]
