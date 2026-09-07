---
tags: [engine, testing]
aliases: [Fingerprint pins, Regression tests]
---

# Fingerprints-and-tests

## What a fingerprint is

Truncated SHA-256 (16 hex) over a trajectory array — a **regression ID**, not a full integrity hash. See [Glossary](../Glossary.md#fingerprint), [Byte-identical-contract](../00-orientation/Byte-identical-contract.md).

## Where pins live

`tests/test_live_simulator.py` and the per-feature test files pin SHA-256
fingerprints over the full trajectory. `tests/test_smoke.py` checks
shapes, physical ranges, and determinism only — no SHA pins. Profile
names match [Byte-identical-contract](../00-orientation/Byte-identical-contract.md#canonical-profiles):

| Profile | Construction | `sv` width | Pin family (full SHA-256[:16]) |
|---------|--------------|-------------|--------------------------------|
| Runtime default (1.2.0+) | `ProcessFaults()` | 30 | batch `sv=691cf51b…` `pv=baedc29f…` `uv=4e4134e4…`; live `sv=80f6f046…` |
| Legacy fingerprint (batch) | `fouling_dynamic=False`, all masters off | 21 | `sv=c8807b23…` `pv=77def506…` `uv=17e62051…` |
| Legacy fingerprint (live) | matching legacy knobs | 21 | `sv=23c3c885…` |
| dynamic-fouling fingerprint (batch) | `fouling_dynamic=True`, all other masters off | 22 | `sv=696531c4…` `pv=3c96ca4f…` `uv=0d9a9673…` |
| dynamic-fouling + quality-latching (batch) | `fouling_dynamic=True`, `quality_state=True` (no actuator wear, no spectra) | 27 | `sv=d663e17d…` |
| pump-only actuator wear (batch) | `pump_wear=True`, `valve_wear=False`, `fouling_dynamic=True`, `quality_state=False` | 23 | `sv=7ddd7aaa…` `pv=e0dba881…` `uv=86c2704f…` |
| valve-only actuator wear (batch) | `pump_wear=False`, `valve_wear=True`, `fouling_dynamic=True`, `quality_state=False` | 23 | `sv=9425d007…` `pv=02296ffa…` `uv=aa146ea3…` |
| pump + valve wear (batch) | `pump_wear=True`, `valve_wear=True`, `fouling_dynamic=True`, `quality_state=False` | 24 | `sv=c092fe08…` `pv=2d26f03f…` `uv=4ef50b9f…` |
| external disturbances active (batch) | `fouling_dynamic=True`, amplitudes > 0 (all other masters off) | 22 | `sv=8865a8c3…` |
| external disturbances active (live) | matching knobs | 22 | `sv=bb763a9b…` |

(Exact strings are in the tests — always trust the test file over this note if they diverge.) Bare `ProcessFaults()` is the **runtime default** as of 1.2.0 — it's a new pinned profile (`sv=691cf51b…`), not the legacy pin row.

## Running tests

**Do not use bare `pip install` for fingerprint work** — it ignores `uv.lock`. Use a fingerprint-aligned install first ([`ADMIN.md`](../../ADMIN.md)):

```bash
uv sync --extra test
uv run python -c "import numpy,scipy,numba; print(numpy.__version__, scipy.__version__, numba.__version__)"
# on Python 3.10 expect: 2.2.6 1.15.3 0.66.0
uv run python -m pytest tests/ -q
```

Smoke tests check shapes, physical ranges, and fingerprints. They do **not** currently compare to MATLAB golden CSVs (future work noted in repo review). See [Byte-identical-contract](../00-orientation/Byte-identical-contract.md#how-to-reproduce-pins).

## When you change pins

1. Confirm the trajectory change is intentional and approved if ODE-related
2. Update pins in the same commit
3. Commit body: old pin ↔ new pin (“this is intentional”)

Related: [Practical dev workflow](../../AGENTS.md), [Batch-driver](../30-engine/Batch-driver.md), [Live-simulator](../30-engine/Live-simulator.md), [Config-surface](../30-engine/Config-surface.md)
