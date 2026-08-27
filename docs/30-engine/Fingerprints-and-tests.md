---
tags: [engine, testing]
aliases: [Fingerprint pins, Regression tests]
---

# Fingerprints-and-tests

## What a fingerprint is

Truncated SHA-256 (16 hex) over a trajectory array — a **regression ID**, not a full integrity hash. See [[Glossary#Fingerprint]], [[Byte-identical-contract]].

## Where pins live

`tests/test_live_simulator.py` and the per-Layer test files pin SHA-256
fingerprints over the full trajectory. `tests/test_smoke.py` checks
shapes, physical ranges, and determinism only — no SHA pins. Profile
names match [[Byte-identical-contract#Canonical profiles]]:

| Profile | Construction | `sv` width | Pin family (full SHA-256[:16]) |
|---------|--------------|-------------|--------------------------------|
| Runtime default | `ProcessFaults()` | 22 | Layer 2.5 path; **no pinned family** |
| Legacy fingerprint (batch) | `fouling_dynamic=False`, all masters off | 21 | `sv=c8807b23…` `pv=77def506…` `uv=17e62051…` |
| Legacy fingerprint (live) | matching legacy knobs | 21 | `sv=23c3c885…` |
| Layer 2.5 fingerprint (batch) | `fouling_dynamic=True`, all masters off | 22 | `sv=696531c4…` `pv=3c96ca4f…` `uv=0d9a9673…` |
| Layer 2.5 + Layer 2.1 (batch) | `fouling_dynamic=True`, `quality_state=True` | 28 | (Layer 2.1 review pending — see Open Questions in `Progress.md`) |
| Layer 2.4 pump-only (batch) | `pump_wear=True`, `valve_wear=False`, `fouling_dynamic=True` | 23 | `sv=7ddd7aaa…` `pv=e0dba881…` `uv=86c2704f…` |
| Layer 2.4 valve-only (batch) | `pump_wear=False`, `valve_wear=True`, `fouling_dynamic=True` | 23 | `sv=9425d007…` `pv=02296ffa…` `uv=aa146ea3…` |
| Layer 2.4 both (batch) | `pump_wear=True`, `valve_wear=True`, `fouling_dynamic=True` | 24 | `sv=c092fe08…` `pv=2d26f03f…` `uv=4ef50b9f…` |
| Layer 2.6 active (batch) | `fouling_dynamic=True`, amplitudes > 0 | 22 | `sv=8865a8c3…` |
| Layer 2.6 active (live) | matching knobs | 22 | `sv=bb763a9b…` |

(Exact strings are in the tests — always trust the test file over this note if they diverge.) Bare `ProcessFaults()` is the **runtime default**, not the legacy pin row.

## Running tests

**Do not use bare `pip install` for fingerprint work** — it ignores `uv.lock`. Use a fingerprint-aligned install first ([`ADMIN.md`](../../ADMIN.md)):

```bash
uv sync --extra test
uv run python -c "import numpy,scipy,numba; print(numpy.__version__, scipy.__version__, numba.__version__)"
# on Python 3.10 expect: 2.2.6 1.15.3 0.66.0
uv run python -m pytest tests/ -q
```

Smoke tests check shapes, physical ranges, and fingerprints. They do **not** currently compare to MATLAB golden CSVs (future work noted in repo review). See [[Byte-identical-contract#How to reproduce pins]].

## When you change pins

1. Confirm the trajectory change is intentional and approved if ODE-related
2. Update pins in the same commit
3. Commit body: old pin ↔ new pin (“this is intentional”)

Related: [[How-to-work-here]], [[Batch-driver]], [[Live-simulator]], [[Config-surface]]
