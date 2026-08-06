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
| Legacy fingerprint (batch) | `fouling_dynamic=False` etc. | `sv=c8807b23…` |
| Legacy fingerprint (live) | matching legacy knobs | `sv=23c3c885…` |
| Layer 2.5 fingerprint (batch) | `fouling_dynamic=True`, extras off | `sv=696531c4…` |
| Layer 2.6 batch / live | disturbance amplitudes on | `8865a8c3…` / `bb763a9b…` |

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
