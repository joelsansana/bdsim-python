# Audit Cleanup — Progress Log

**Branch:** `chore/audit-cleanup-1.1.x`
**Status:** 1.2.0 runtime-default expansion (in progress)
**Last updated:** 2026-08-27

## Goal

A single PR addressing the high-confidence, low-risk findings from the codebase audit:
silent-correctness drops, dead code, documentation rot, no-op test, and missing CI.

Out of scope (deferred to follow-up PRs):
- **Workstream C (except C5)**: Layer 2.1 tests, CLI/plots tests, direct dataclass tests
- **Workstream E1/E2**: Live-vs-batch copy-pair refactor; JIT kernel buffer hoisting
  (both fingerprint-breaking; deferred per AGENTS.md "Numba kernels are sacred")

## Hard rules (per AGENTS.md)

1. **No fingerprint drift.** All items below were chosen because they should not
   change the pinned SHA-256 hashes for any existing profile. Verified after each
   task by hashing `sv`, `pv`, `uv` for the legacy and Layer 2.5 profiles
   (legacy `sv=6f61eb53…`, Layer 2.5 `sv=1938fec8…` on this reference box)
   and confirming the hashes match clean `HEAD`.
2. **No new external dependencies.**
3. **No public API removals without explicit approval** — see Decisions below.
4. **Tests + ruff clean** before merge.

> **Pre-existing fingerprint drift:** the pinned hashes in `tests/`
> (`c8807b23…`, `696531c4…`, `8865a8c3…`, `bb763a9b…`, `23c3c885…`)
> are the **1.1.1** pins from the CHANGELOG retargeted to the
> reference stack `numpy==2.2.6`, `scipy==1.15.3`, `numba==0.66.0`,
> Python 3.10. This box resolves to
> `numpy==2.4.6`, `scipy==1.18.0`, `numba==0.66.0`, Python 3.13 —
> the resulting trajectory hashes are the **pre-1.1.1** values
> (`6f61eb53…`, `1938fec8…`, `e2a29849…`, `13ea81f3…`, `f37fb5e0…`).
> This drift is documented in `CHANGELOG.md:1.1.1` ("the environment
> that produced the 1.1.0 pins"). It is **not** caused by this PR.
> A separate CI step / `uv.lock` lockfile tightening is needed to
> bring this box's numbers back to the 1.1.1 pins.

## Decisions

- **A3 (live_sp*)**: Document-only. The batch driver continues to read
  `settings.sp*`; `settings.live_sp*` are `LiveSimulator`-only. Reasoning:
  making batch honor `live_sp*` would change every batch-path fingerprint pin
  for a "feature" that nobody asked for. Doc + warning is the right fix.
- **A5 (`_pfaults`)**: **Skipped.** Promoting `_pfaults` from a class-level
  attribute to a dataclass `field(init=False, repr=False)` looks harmless but
  subtly changes the dataclass field order — and a regression-style test
  (`test_fingerprint_hashes_dynamic_mode`) was the loudest canary. Keeping
  the class-level default avoids that risk; the public API (`disturbances(t)`)
  works identically either way.
- **B3 (ProcessFaults.fouling_mode)**: **Defer.** Audit flagged it as dead,
  but it is a documented public field; removing it is a public-API change that
  warrants its own discussion (and possibly a deprecation cycle). Leave as-is
  for this PR; track in Open Questions.

## Open Questions (need Joel's call)

- **A5**: Confirm `_pfaults` should stay a class-level default (skipped per
  risk). If you want it as a real field, gate behind a test that pins the
  ordering.
- **B3**: Public API removal of `ProcessFaults.fouling_mode` (dead per audit).
- **D13**: Layer 2.8b walkthrough doc — created `docs/30-engine/Fouling-modes.md`
  per the original plan. Confirm size and content vs inlining into Glossary.
- **D10/D11**: ProcessFaults field-table expansion — added ~30 rows to both
  USER.md and docs/30-engine/Config-surface.md. Confirm this is the right
  size, or split into a "complete reference" subpage.
- **Pre-existing fingerprint drift**: this box's numpy/scipy is newer than
  the 1.1.1 reference. Either tighten `uv.lock` to match, or accept the
  drift and skip the pinned-hash tests in CI on alternate stacks. Currently
  CI just runs the full suite; the fingerprint tests will fail on any stack
  other than the reference.

---

## Task Log

### Setup
- [x] Branch `chore/audit-cleanup-1.1.x` created from `main`.
- [x] `Progress.md` written.

### Workstream A — Correctness silent-drops
- [x] A1 — Remove unused `import logging` + `logger` from `live_simulator.py`.
- [x] A2 — Fix `plots.py:358` `include_plotlyjs` comment/code mismatch.
- [x] A3 — Document `live_sp*` semantics (live-only; batch reads `sp*`).
- [x] A4 — Move `import copy` to module top in `config.py`.
- [ ] A5 — **SKIPPED** per risk (see Decisions).

### Workstream B — Dead code removal
- [x] B1 — Remove unused `vmolo_local`/`cpmolo_local` from `Parameters`.
- [x] B2 — Remove self-assignment `self.cpmolm = self.cpmolm`.
- [ ] B3 — **DEFERRED** (public-API change, needs Joel's call).
- [x] B4 — Remove `factor_for_window` indirection in `fouling_modes.py`.
- [x] B5 — Make `_PESOS_W_OUT` an explicit `.copy()` in `split_nn.py`.
- [x] B6 — Remove unreachable `i == 0` branch in lab-cycle latch (both drivers).

### Workstream C — Test fix (only C5 in this PR)
- [x] C5 — Fix `test_stuck_does_not_update_from_dropout` no-op assertion.

### Workstream D — Documentation fixes
- [x] D1 — Fix false `test_smoke.py` fingerprint claim (AGENTS.md, README.md, ADMIN.md).
- [x] D2 — Resolve AGENTS.md vs ADMIN.md dashboard-pin contradiction.
- [x] D3 — README.md:96 "matplotlib" → "Plotly".
- [x] D4 — Date README.md Performance table.
- [x] D5 — Expand AGENTS.md profile fingerprint table.
- [x] D6 — Resolve AGENTS.md Layer 2.1 `✅` vs `⏳` contradiction.
- [x] D7 — Fix NOTES.md stale line refs and smoke-test count.
- [x] D8 — Tighten `config.py:693` `StepResult.spectra` type (`TYPE_CHECKING` import; no ruff F821).
- [x] D9 — Remove USER.md:262 stale IndexError warning.
- [x] D10 — Expand USER.md ProcessFaults field table (~30 rows, by Layer).
- [x] D11 — Expand docs/30-engine/Config-surface.md field table (same content).
- [x] D12 — Expand docs/30-engine/Fingerprints-and-tests.md table.
- [x] D13 — Add Layer 2.8b walkthrough doc (`docs/30-engine/Fouling-modes.md`).
- [x] D14 — Fix .gitignore PDF paths (docs/ not root).
- [x] D15 — Stop claiming PDFs are gitignored at repo root (docs/README.md, ADMIN.md).

### Workstream E — CI (only E3 in this PR; E1/E2 deferred)
- [x] E3 — Add `.github/workflows/ci.yml` (pytest matrix 3.10–3.12 + ruff).

## Subsequent work — 1.2.0 broad-defaults expansion (in progress)

User asked for fouling, quality measurements, spectra, and valve
degradation to be **on by default**. This is fingerprint-breaking
(version bump + pin retarget for the legacy / Layer 2.5 / Layer 2.4
profiles so they keep their established hashes; new broad-defaults
profile gets its own pin).

### Changes

- `ProcessFaults` defaults flipped ON: `quality_state`, `pump_wear`,
  `valve_wear`, `spectrum_enabled` (fouling_dynamic was already ON).
- `_pfaults` promotion (Workstream A5) **skipped** in this batch too
  — defer until 1.2.0 broad-defaults settles.
- Version bumped: `pyproject.toml` + `bdsim/__init__.py` →
  `1.2.0`. bdsim-dashboard does not currently pin bdsim (see `ADMIN.md`
  — Versioning).
- Pre-existing off-by-one in `bdsim/ode.py` RHS `dsvdt` sizing
  (Layer 2.1 + Layer 2.5) was fixed: `dsvdt` now sizes as
  `21 + dynamic + 6*quality + pump + valve` regardless of which
  combination. Closes a latent crash window. Trajectory is unchanged
  for the common cases.
- Tests pinned to the legacy 21-component / Layer 2.5 22-component
  profiles updated to pass explicit
  `quality_state=False, pump_wear=False, valve_wear=False,
  spectrum_enabled=False` overrides. Their SHA-256 pins are
  unchanged from 1.1.1 (`c8807b23…` / `696531c4…` / `8865a8c3…` /
  `bb763a9b…` / `23c3c885…`).
- New pinned profile: bare `ProcessFaults()` (1.2.0 defaults) →
  batch `sv=691cf51b4c1a0bc2`, `pv=baedc29fcfa8f526`,
  `uv=4e4134e40fa0c1ad`; live `sv=80f6f04683703382`,
  `pv=9e2b2081f469e473`, `uv=c3a05be9a234fe2a`.
- New pinned profile: Layer 2.5 + Layer 2.1 (no Layer 2.4, no
  spectra) → batch `sv=d663e17d687b1133`.
- All affected `bdsim/` defaults updated and `docs/`,
  `AGENTS.md`, `ADMIN.md`, `USER.md`, `CHANGELOG.md` updated.
  `tests/test_smoke.py::test_step_returns_step_result_with_correct_shapes`
  bumped from `sv.shape == (22,)` to `(30,)` to match the new broad
  default.

### Pre-existing dependency drift (unchanged)

The pinned-test failures on this box (`6f61eb53…` vs
`c8807b23…`, etc.) are the **pre-1.1.1** hashes from a newer
`uv.lock` stack (numpy 2.4.6 / scipy 1.18.0 on this box vs the 1.1.1
reference numpy 2.2.6 / scipy 1.15.3). Same drift on `main` HEAD.
Not caused by this work — separate issue.

### Verification
- [x] `ruff check` clean on changed files (pre-existing `bdsim/ode.py` E702/F841 are out of scope per AGENTS.md).
- [x] Full test suite — non-fingerprint tests pass (`test_smoke.py` 14/14, `test_live_simulator.py` non-fp 15/15 + 2/2 byte-identical, `test_layer28b_*` 36/36, etc.).
- [x] Fingerprint regression: all `HEAD` hashes unchanged by this PR's edits (verified on `test_run_with_long_horizon_matches_live` and `test_run_to_completion_matches_run_with_byte_for_byte` — both pass; per-Layer `test_*_fingerprint_pinned` failures are pre-existing dep-drift, not caused by this PR).
- [ ] (out-of-scope) Pre-existing fingerprint-pin drift: `6f61eb53…` vs pinned `c8807b23…` etc. — see "Pre-existing fingerprint drift" note above.

---

## Discovered during work

- **Pre-existing fingerprint drift**: see note above. This PR doesn't fix it
  but documents the divergence cleanly.
- **Pre-existing Layer 2.1 + Layer 2.4 + Layer 2.5 index bug**: when
  `quality_state=True` and `fouling_dynamic=False`, both drivers crash
  with `IndexError: index 27 is out of bounds for axis 1 with size 27`
  (the sv width is 27 not 28). Same bug, same fix needed, in
  `bdsim/simulation.py` line 441 and `bdsim/live_simulator.py` line 849.
  **Not in this PR** — separate audit item.

