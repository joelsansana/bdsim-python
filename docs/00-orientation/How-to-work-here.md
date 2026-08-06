---
tags: [orientation, howto]
aliases: [Contributing, Dev workflow]
---

# How-to-work-here

Practical rules for juniors editing this repo. Hard constraints: [`AGENTS.md`](../../AGENTS.md).

## Do

- Prefer drivers, config knobs, tests, and **this vault** over touching ODE statement order
- Gate new trajectory-affecting behavior on a `ProcessFaults` flag defaulting off/zero (exception today: `fouling_dynamic=True` — see [[Byte-identical-contract#Canonical profiles]])
- Add/update tests under `tests/`; keep ruff clean on new Python
- Update vault notes ([[Config-surface]], [[Channel-indices]], [[Layers-roadmap]]) in the **same** change when you add knobs or widen `sv`/`pv`

## Do not (without asking)

- Change ODE math “because MATLAB looks wrong” — capture with a test, then ask ([`NOTES.md`](../../NOTES.md) has prior agreed fixes)
- Add a new external dependency
- Relax the [[Byte-identical-contract]]
- Bump version in only one of `pyproject.toml` / `bdsim/__init__.py` / dashboard pin

## Tests

Fingerprint-aligned env first (see root [`ADMIN.md`](../../ADMIN.md) — **Fingerprint-aligned install**):

```bash
uv sync --extra test
uv run python -m pytest tests/ -q
```

Full suite ~6:40. Smoke + fingerprint tests are the first safety net. See [[Fingerprints-and-tests]]. Do **not** rely on bare `pip install` for pin parity — it ignores `uv.lock`.

## Sacred hot path

Anything `@njit(cache=True)` in `ode.py`, measurement/stiction/PID helpers, etc. is **sacred**. Style refactors that reorder statements break pins.

## Dashboard coordination

Dashboard builds `ProcessFaults` via `SimRunner._build_pfaults()`. New knobs may need a dashboard PR too. Editable install: `uv sync --extra test` from this repo root ([`ADMIN.md`](../../ADMIN.md)).

Related: [[Home]], [[Repo-map]], [[Byte-identical-contract]], [[Pseudocode-conventions]]
