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

Source of truth for defaults: `bdsim/config.py` (`ProcessFaults`). As
of **bdsim 1.2.0**, the runtime default enables Layers 2.5 + 2.1 + 2.4
(both) + 2.8a — it is **its own pinned profile**, not the legacy path.

| Profile | How to get it | `sv` width | Role |
|---------|---------------|------------|------|
| **Runtime default** (1.2.0+) | `ProcessFaults()` | **30** | Layers 2.5 + 2.1 + 2.4 (both) + 2.8a all ON. Pin: batch `sv=691cf51b…`, live `sv=80f6f046…`. What `run()` / default live construction use. |
| **Legacy fingerprint** | Explicit `fouling_dynamic=False` (and every other Layer master off/zero) | **21** | Upstream-parity pins: batch `sv=c8807b23…`, live `sv=23c3c885…`. Tests pass these overrides. |
| **Layer 2.5 fingerprint** | `fouling_dynamic=True` with other Layer masters off/zero | **22** | Dynamic-α pin family: batch `sv=696531c4…`. |
| **Layer 2.5 + Layer 2.1** | `fouling_dynamic=True quality_state=True` (no Layer 2.4, no spectra) | **27** | `sv=d663e17d…`. |

Other Layer gates (disturbance amplitudes, `live_*` overlays,
fouling-mode windows) default **off / zero / `None`** and stay that
way in the profiles above unless a test enables them.

> [!note]
> Changing `fouling_dynamic`’s code default requires Joel’s agreement plus coordinated pin/docs updates. Docs here describe **current** code behavior.

## How to reproduce pins

Pins match the committed [`uv.lock`](../../uv.lock) numerical stack (bdsim **1.2.0** reference: Python **3.10** → `numpy==2.2.6`, `scipy==1.15.3`, `numba==0.66.0`).

```bash
uv sync --extra test
uv run python -c "import numpy,scipy,numba; print(numpy.__version__, scipy.__version__, numba.__version__)"
uv run python -m pytest tests/ -q
```

Do not use bare `pip install -e .` for fingerprint work — it ignores the lockfile. Full procedure and caveats (other OS/BLAS/Python): root [`ADMIN.md`](../../ADMIN.md) — **Fingerprint-aligned install**. See also [[Fingerprints-and-tests]], [[How-to-work-here]].

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
