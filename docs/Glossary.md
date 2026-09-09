---
tags: [helper, reference]
aliases: [Domain terms, Definitions]
---

# Glossary

Domain and engine terms. Abbreviations: [Acronyms](./Acronyms.md).

## Byte-identical

Given the **same seed** and the **same** [ProcessFaults](./30-engine/Config-surface.md) (and related settings), batch and live paths produce trajectories whose truncated SHA-256 fingerprints match pinned values in `tests/`. See [Byte-identical-contract](./00-orientation/Byte-identical-contract.md).

## Fingerprint

A short hash (16 hex chars of SHA-256) of a trajectory array (`sv`, `pv`, …) used as a regression pin. Drift fails tests loudly. Intentional math/schema changes require documenting old → new pins. See [Fingerprints-and-tests](./30-engine/Fingerprints-and-tests.md).

## ProcessFaults

Typed dataclass of process-side knobs: clogging, fouling, feature gates, disturbance amplitudes, spectrum, wear, etc. Default-mode feature flags are intended to be off/zero for upstream parity — but always verify code defaults against docs (see [Config-surface](./30-engine/Config-surface.md)).

## Quality latching

quality latching: continuous quality states evolve in `sv`, but published QA channels update only on a lab (or online) cycle with noise — like a slow lab sample. Enabled with `quality_state=True`. See [Config-surface](./30-engine/Config-surface.md), [Channel-indices](./40-helpers/Channel-indices.md).

## Fouling / fouling factor

Heat-exchanger efficiency penalty. Legacy path uses a static series; dynamic fouling evolves continuous α in `sv[21]`; fouling-mode windows injects windowed ARMAX modes 4/5. Priority: continuous α > windowed mode > static. See [Config-surface](./30-engine/Config-surface.md), `bdsim/fouling_modes.py`.

## Valve stiction

Nonlinear valve stick-slip (Kano model). `ValveFaults.S` / `J`; actuator wear can grow stiction % as a state when `valve_wear=True`. See [Sensors-and-control](./10-plant/Sensors-and-control.md).

## Numba kernel

A function decorated `@njit(cache=True)` that runs as compiled machine code. **Do not reorder statements** for style — fingerprints depend on numerical path. See [ODE-and-AE](./20-math/ODE-and-AE.md), [Practical dev workflow](./../AGENTS.md).

## State vector (`sv`)

Plant dynamic state: compositions, temperatures, levels, valve lifts, optional feature-flagged slots (α, quality, wear). Baseline width 21. See [Channel-indices](./40-helpers/Channel-indices.md), [ODE-and-AE](./20-math/ODE-and-AE.md).

## Measurement vector (`pv`)

Five noisy plant sensors after fault injection. See [Sensors-and-control](./10-plant/Sensors-and-control.md), [Channel-indices](./40-helpers/Channel-indices.md).

## Input vector (`uv` / `u`)

Six exogenous / manipulated inputs into the ODE (valve orders, feed T/F, Qheat, …). See [Channel-indices](./40-helpers/Channel-indices.md).

## Live vs batch

[Batch](./30-engine/Batch-driver.md) (`run` / `run_with`) integrates the full horizon and returns [Results](./Glossary.md#fingerprint). [Live](./30-engine/Live-simulator.md) (`LiveSimulator`) yields one [StepResult](./Acronyms.md#opc) per `dt` for the dashboard. Same physics contract when configured the same way.

## Feature groups

Feature groups (quality latching, actuator wear, dynamic fouling, external disturbances, NIR/IR spectrum sensor, fouling-mode windows). Gated on `ProcessFaults` flags. See [Config-surface](./30-engine/Config-surface.md).

Related: [Acronyms](./Acronyms.md), [Home](./Home.md), [Byte-identical-contract](./00-orientation/Byte-identical-contract.md)
