---
tags: [engine, roadmap]
aliases: [Layers, Feature flags, Roadmap]
---

# Layers-roadmap

Numbered feature packages gated on [[Config-surface|ProcessFaults]]. New Layer flags should default **off / zero**, except the documented exception: **Layer 2.5 `fouling_dynamic=True`** (runtime default). See [[Byte-identical-contract#Canonical profiles]].

## Status (from AGENTS)

| Layer | Topic | Master knobs | Default |
|-------|-------|--------------|---------|
| 2.1 | Quality latching | `quality_state` | off |
| 2.4 | Pump / valve wear | `pump_wear`, `valve_wear` | off |
| 2.5 | HEX fouling continuous α | `fouling_dynamic` | **on** |
| 2.6 | External disturbances | ambient / CW amplitudes & drift | zero amplitude |
| 2.6b | CW pump trip | live override / trip helpers | no active trip |
| 2.7 | Operator disturbance knobs | `live_*` overlays | `None` |
| 2.8a | NIR/IR spectra | `spectrum_enabled` | off |
| 2.8b | Five-mode fouling windows | `fouling_mode*`, `activate_fouling_mode_window` | no window |

Dashboard-side: MQTT publish, scenario catalog (not in this repo).

## Fouling path priority

```text
continuous α (2.5)  >  windowed mode 4/5 (2.8b)  >  static legacy factor
```

Implemented across `ode.py` + `fouling_modes.py` + drivers.

## Spectra (2.8a)

Post-process Beer–Lambert mix of reference spectra (`bdsim/data/spectra_ref.csv`) + photometric noise + AWGN + drift. Fires every `spctr_t` seconds when enabled. Does not replace the ODE.

## Quality (2.1)

Continuous QA states in extended `sv`; published lab sample latched on cycle with noise → `Results.quality` / `StepResult.quality_latched`.

Related: [[Channel-indices]], [[Config-surface]], [[ODE-and-AE]], [[Home]]
