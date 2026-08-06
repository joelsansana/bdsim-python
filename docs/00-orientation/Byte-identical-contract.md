---
tags: [orientation, contract]
aliases: [Reproducibility, Fingerprint contract, Canonical profiles]
---

# Byte-identical-contract

**Hard rule:** same seed + same `ProcessFaults` (and matching settings) → **byte-identical** trajectory. Pins live in `tests/`; drift fails loud.

See also [[Glossary#Byte-identical]], [[Fingerprints-and-tests]], root [`AGENTS.md`](../../AGENTS.md).

## Why it exists

Two callers need trust:

1. **Dashboard** — deterministic live control demos
2. **ML / fault-detection** — comparable batch runs across machines and time

Without a pin contract, “faithful port” becomes hand-wavy.

## What must stay stable

- [[ODE-and-AE|Numba kernel]] statement order and numerical path
- Named **profiles** below (especially the legacy fingerprint profile used by pins)
- RNG seeding on the driver path

## Canonical profiles

Source of truth for defaults: `bdsim/config.py` (`ProcessFaults`). **Bare `ProcessFaults()` is not the legacy fingerprint path.**

| Profile | How to get it | `sv` width | Role |
|---------|---------------|------------|------|
| **Runtime default** | `ProcessFaults()` | **22** | Layer 2.5 on (`fouling_dynamic=True`); other Layer masters off/zero. What `run()` / default live construction use. |
| **Legacy fingerprint** | Explicit `fouling_dynamic=False` (+ other Layer masters off/zero) | **21** | Upstream-parity pins: batch `sv=6f61eb53…`, live `sv=f37fb5e0…`. Tests pass this override. |
| **Layer 2.5 fingerprint** | `fouling_dynamic=True` with other extras off | **22** | Dynamic-α pin family: batch `sv=1938fec8…`. |

Other Layer gates (`quality_state`, `pump_wear`, `valve_wear`, `spectrum_enabled`, disturbance amplitudes, `live_*`, fouling-mode windows) default **off / zero / `None`** and stay that way in the profiles above unless a test enables them.

> [!note]
> Changing `fouling_dynamic`’s code default requires Joel’s agreement plus coordinated pin/docs updates. Docs here describe **current** code behavior.

## Pseudocode

```text
1. Build Settings, ProcessFaults, SensorFaults, ValveFaults
2. np.random.seed(seed)   # (and Generator where Layer code uses it)
3. Run batch run_with(...) OR LiveSimulator.step() loop
4. Hash trajectory arrays (e.g. sv) → compare to pinned hex
```

## When fingerprints must change

Intentional trajectory change → update pins in the same PR/commit with an explicit “this is intentional” note: old pin ↔ new pin. Never reorder `@njit` lines for readability without that.

Related: [[Home]], [[Config-surface]], [[Fingerprints-and-tests]], [[How-to-work-here]]
