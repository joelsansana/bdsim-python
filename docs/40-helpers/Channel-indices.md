---
tags: [helper, reference]
aliases: [sv indices, pv indices, Channel map]
---

# Channel-indices

Cheat sheet for juniors. Source of truth: `ode.py` header, `_measurements`, `config.Results`. Update this note when widths change.

> [!tip]
> **Runtime default** (`ProcessFaults()`): `fouling_dynamic=True` → **`sv` width 22** with α at `sv[21]`. The 21-column table below is the **legacy fingerprint profile** (`fouling_dynamic=False`). See [Byte-identical-contract](../00-orientation/Byte-identical-contract.md#canonical-profiles).

## State `sv` — baseline 21 (legacy / pin profile)

| Index | Name |
|------:|------|
| 0–5 | xR (TG, DG, MG, M, E, G) |
| 6 | TR |
| 7–12 | xL |
| 13–15 | xH (M, E, G) |
| 16 | hH |
| 17 | TD |
| 18 | r (filter pore) |
| 19 | lifto |
| 20 | liftH |

## Optional extensions (order depends on flags)

Drivers grow `sv` when flags are on (see `live_simulator` / `simulation` construction):

| Feature | Typical slot | Flag | Notes |
|---------|--------------|------|-------|
| HEX fouling α | `sv[21]` | `fouling_dynamic` | **On by default** (runtime profile) |
| Quality states | six slots after α / baseline | `quality_state` | Default off |
| Pump health | wear slot | `pump_wear` | Default off |
| Valve stiction % | wear slot | `valve_wear` | Default off |

> [!warning]
> Exact indices when **multiple** feature flags combine depend on construction order in the drivers. Read the allocator in `LiveSimulator` / batch setup when writing fusion code — do not hardcode `28` without checking flags.

## Inputs `uv` / `u` (6)

| Index | Name |
|------:|------|
| 0 | vinputo (oil valve order, %) |
| 1 | Tmet (K) |
| 2 | Fmet (mol/s in ODE path; watch unit conversions) |
| 3 | Toil (K) |
| 4 | Qheat (W) |
| 5 | vinputH (%) |

## Measurements `pv` (5)

| Index | Name |
|------:|------|
| 0 | TR |
| 1 | TD |
| 2 | hH |
| 3 | Foil (kg/s) |
| 4 | Filter DP |

## Setpoints `sp` (4)

TR, TD, hH, Foil (engine often stores Foil in kg/s; display converts to kg/h).

## Disturbances (external disturbances)

Shape `(…, 3)`: `[Tamb_K, Tcw_K, Pcw_Pa]`.

## Quality latched (quality latching)

`[FAME%, water_ppm, IV]`.

Related: [ODE-and-AE](../20-math/ODE-and-AE.md), [Sensors-and-control](../10-plant/Sensors-and-control.md), [Config-surface](../30-engine/Config-surface.md), [Acronyms](../Acronyms.md)
