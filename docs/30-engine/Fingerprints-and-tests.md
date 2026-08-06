---
tags: [engine, testing]
aliases: [Fingerprint pins, Regression tests]
---

# Fingerprints-and-tests

## What a fingerprint is

Truncated SHA-256 (16 hex) over a trajectory array — a **regression ID**, not a full integrity hash. See [[Glossary#Fingerprint]], [[Byte-identical-contract]].

## Where pins live

Primarily `tests/test_smoke.py`, `tests/test_live_simulator.py`, and Layer-specific tests. Profile names match [[Byte-identical-contract#Canonical profiles]]:

| Canonical profile | How tests build it | Example pin family |
|-------------------|--------------------|--------------------|
| Legacy fingerprint (batch) | `fouling_dynamic=False` etc. | `sv=6f61eb53…` |
| Legacy fingerprint (live) | matching legacy knobs | `sv=f37fb5e0…` |
| Layer 2.5 fingerprint (batch) | `fouling_dynamic=True`, extras off | `sv=1938fec8…` |
| Layer 2.6 batch / live | disturbance amplitudes on | `e2a29849…` / `13ea81f3…` |

(Exact strings are in the tests — always trust the test file over this note if they diverge.) Bare `ProcessFaults()` is the **runtime default**, not the legacy pin row.

## Running tests

```bash
python3 -m pytest tests/ -q
```

Smoke tests check shapes, physical ranges, and fingerprints. They do **not** currently compare to MATLAB golden CSVs (future work noted in repo review).

## When you change pins

1. Confirm the trajectory change is intentional and approved if ODE-related
2. Update pins in the same commit
3. Commit body: old pin ↔ new pin (“this is intentional”)

Related: [[How-to-work-here]], [[Batch-driver]], [[Live-simulator]], [[Config-surface]]
