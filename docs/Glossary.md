---
tags: [helper, reference]
aliases: [Domain terms, Definitions]
---

# Glossary

Domain and engine terms. Abbreviations: [[Acronyms]].

## Byte-identical

Given the **same seed** and the **same** [[Config-surface|ProcessFaults]] (and related settings), batch and live paths produce trajectories whose truncated SHA-256 fingerprints match pinned values in `tests/`. See [[Byte-identical-contract]].

## Fingerprint

A short hash (16 hex chars of SHA-256) of a trajectory array (`sv`, `pv`, …) used as a regression pin. Drift fails tests loudly. Intentional math/schema changes require documenting old → new pins. See [[Fingerprints-and-tests]].

## ProcessFaults

Typed dataclass of process-side knobs: clogging, fouling, Layer 2.x feature gates, disturbance amplitudes, spectrum, wear, etc. Default-mode Layer features are intended to be off/zero for upstream parity — but always verify code defaults against docs (see [[Config-surface]]).

## Quality latching

Layer 2.1: continuous quality states evolve in `sv`, but published QA channels update only on a lab (or online) cycle with noise — like a slow lab sample. Enabled with `quality_state=True`. See [[Layers-roadmap]], [[Channel-indices]].

## Fouling / fouling factor

Heat-exchanger efficiency penalty. Legacy path uses a static series; Layer 2.5 evolves continuous α in `sv[21]`; Layer 2.8b injects windowed ARMAX modes 4/5. Priority: continuous α > windowed mode > static. See [[Layers-roadmap]], `bdsim/fouling_modes.py`.

## Valve stiction

Nonlinear valve stick-slip (Kano model). `ValveFaults.S` / `J`; Layer 2.4 can grow stiction % as a state when `valve_wear=True`. See [[Sensors-and-control]].

## Numba kernel

A function decorated `@njit(cache=True)` that runs as compiled machine code. **Do not reorder statements** for style — fingerprints depend on numerical path. See [[ODE-and-AE]], [[How-to-work-here]].

## State vector (`sv`)

Plant dynamic state: compositions, temperatures, levels, valve lifts, optional Layer slots (α, quality, wear). Baseline width 21. See [[Channel-indices]], [[ODE-and-AE]].

## Measurement vector (`pv`)

Five noisy plant sensors after fault injection. See [[Sensors-and-control]], [[Channel-indices]].

## Input vector (`uv` / `u`)

Six exogenous / manipulated inputs into the ODE (valve orders, feed T/F, Qheat, …). See [[Channel-indices]].

## Live vs batch

[[Batch-driver|Batch]] (`run` / `run_with`) integrates the full horizon and returns [[Glossary#Fingerprint|Results]]. [[Live-simulator|Live]] (`LiveSimulator`) yields one [[Acronyms#OPC|StepResult]] per `dt` for the dashboard. Same physics contract when configured the same way.

## Layer (roadmap)

Numbered feature packages (2.1 quality, 2.4 wear, 2.5 fouling dynamic, 2.6 disturbances, 2.8 spectra / fouling modes, …). Gated on `ProcessFaults` flags. See [[Layers-roadmap]].

Related: [[Acronyms]], [[Home]], [[Byte-identical-contract]]
