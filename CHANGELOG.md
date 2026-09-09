# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/).

**Fingerprint / schema rule:** any intentional trajectory or `ProcessFaults`
schema change must list old→new fingerprint pins in the same changelog entry
(and in the commit/PR body). See `AGENTS.md` and `docs/00-orientation/Byte-identical-contract.md`.
Exact pin strings live in `tests/` and are summarized in `ADMIN.md`.

## [Unreleased]

### Added
### Changed
### Fixed

## [1.2.0]

### Changed

- **Runtime default expands to four Layer masters ON.** `ProcessFaults()`
  now enables Layer 2.5 (`fouling_dynamic`), Layer 2.1 (`quality_state`),
  Layer 2.4 pump wear (`pump_wear`), Layer 2.4 valve wear (`valve_wear`),
  and Layer 2.8a spectra (`spectrum_enabled`). Bare `ProcessFaults()` is
  now a new pinned profile — sv width 30, hash `691cf51b…` (batch) /
  `80f6f046…` (live).

- **Layer 2.1 + Layer 2.5 combination fix.** A pre-existing off-by-one in
  `bdsim/ode.py` (RHS `dsvdt` allocated 28 instead of 27 when
  `quality_state=True` and `fouling_dynamic=False`) was fixed. The
  numerical path is unchanged for the common cases
  (`fouling_dynamic=True`); the fix only corrects the
  `quality_state=True + fouling_dynamic=False` sizing and closes a
  latent crash window. Tests that previously constructed
  `ProcessFaults(fouling_dynamic=False, quality_state=True)` (the
  broken combination) now run cleanly with sv width 27.

- **Tests updated to opt out explicitly.** Every test that targeted the
  legacy 21-component profile or the Layer 2.5 22-component profile now
  passes explicit `quality_state=False, pump_wear=False,
  valve_wear=False, spectrum_enabled=False` overrides. The pinned
  hashes for these profiles are unchanged from 1.1.1
  (`c8807b23…` / `696531c4…` / `8865a8c3…` / `bb763a9b…` / `23c3c885…`).

### Added

- **New pinned profile: 1.2.0 runtime default.** Bare `ProcessFaults()`
  now produces a stable trajectory with `sv=691cf51b4c1a0bc2`
  / `pv=baedc29fcfa8f526` / `uv=4e4134e40fa0c1ad` (batch) and
  `sv=80f6f04683703382` / `pv=9e2b2081f469e473` /
  `uv=c3a05be9a234fe2a` (live). This is the contract that downstream
  consumers (dashboard, fault-detection pipelines) target by default;
  legacy / Layer 2.5 only profiles are still available via explicit
  overrides.

- **New pinned profile: Layer 2.5 + Layer 2.1.** `fouling_dynamic=True,
  quality_state=True` (no Layer 2.4, no spectra) → sv width 27, hash
  `sv=d663e17d687b1133`.

### Notes

- AGENTS.md "Default profiles" table and the profile fingerprint tables
  in `ADMIN.md`, `docs/30-engine/Fingerprints-and-tests.md`, and
  `docs/00-orientation/Byte-identical-contract.md` all updated.
- Demos and ML pipelines trained on 21-wide or 22-wide state vectors
  must re-train or pin the legacy profile explicitly.
- Companion dashboard may need its `SimRunner._build_pfaults()` updated
  to match (auto-derived from `ProcessFaults()` defaults, but verify).

## [1.1.1]

### Changed

- **Fingerprint pins retargeted** to the `uv.lock` numerical stack on the reference host
  (`numpy==2.2.6`, `scipy==1.15.3`, `numba==0.66.0`, Python 3.10, macOS arm64).
  This is **not** an ODE/kernel change — `bdsim/` math was unchanged; `uv sync` resolved a
  newer scipy/numpy/numba than the environment that produced the 1.1.0 pins.
- Intentional pin updates (examples):
  - Legacy batch `sv`: `6f61eb532b3284ee` → `c8807b23b14a9ad1`
  - Legacy live `sv`: `f37fb5e0f70f5516` → `23c3c885694c3d24`
  - Layer 2.5 batch `sv`: `1938fec8dee2c8ba` → `696531c4990c5b1e`
  - Layer 2.6 batch `sv`: `e2a29849064d915e` → `8865a8c352cb6b55`
  - Layer 2.6 live `sv`: `13ea81f3af76ec5a` → `bb763a9bde1d3fc9`
  Full set updated in `tests/test_*.py`; see `ADMIN.md` fingerprint table.

### Notes

- Reproduce pins with `uv sync --extra test` then `uv run python -m pytest tests/ -q`.
- Cross-platform / alternate BLAS stacks may still diverge; treat pins as regression IDs for the locked stack.

## [1.1.0]

Current packaged release of the Python BDSIM port (Joel Sansana).

### Added

- Batch (`run` / `run_with`) and live (`LiveSimulator`) drivers with Numba-accelerated kernels
- Quality latching (`quality_state`)
- Pump / valve wear continuous state (`pump_wear`, `valve_wear`)
- Dynamic HEX fouling α (`fouling_dynamic`, default on at runtime)
- External disturbances, cooling-water pump trip, operator knobs (`ambient_t_*`, `cw_*`, `live_*`)
- NIR/IR virtual spectrum sensor (`spectrum_enabled`)
- Five-mode fouling stepper (windowed ARMAX modes 4/5; `fouling_mode_*`)
- Typed config surface (`Settings`, `ProcessFaults`, sensor/valve faults)
- Plotly figure set + CSV outputs matching upstream column names
- Fingerprint regression suite (legacy, dynamic-fouling, external-disturbances, live/batch)

### Notes

- Runtime default (`ProcessFaults()`) enables dynamic fouling; legacy upstream pins use explicit `fouling_dynamic=False` — see canonical profiles in the docs vault.
- Companion operator console: [bdsim-dashboard](https://github.com/joelsansana/bdsim-dashboard).

---

## Historical development log (pre-1.2.0)

The audit-cleanup workstream that landed alongside the 1.2.0 release is summarised here for reference. The original `Progress.md` dev log was folded into this changelog in the 1.2.x docs cleanup; the workstream IDs (A1–E3) match the audit-cleanup PR description.

### Workstream A — silent-correctness fixes
- [x] A1 — Remove unused `import logging` + `logger` from `bdsim/live_simulator.py`.
- [x] A2 — Fix `bdsim/plots.py:358` `include_plotlyjs` comment/code mismatch.
- [x] A3 — Document `live_sp*` semantics (live-only; batch reads `sp*`).
- [x] A4 — Move `import copy` to module top in `bdsim/config.py`.
- [ ] A5 — **Skipped** (promoting `_pfaults` from class-level default to a dataclass field risks fingerprint drift; left as-is).

### Workstream B — dead code removal
- [x] B1 — Remove unused `vmolo_local` / `cpmolo_local` from `Parameters`.
- [x] B2 — Remove self-assignment `self.cpmolm = self.cpmolm`.
- [ ] B3 — **Deferred** (`ProcessFaults.fouling_mode` is a documented public field; removal needs a deprecation cycle).
- [x] B4 — Remove `factor_for_window` indirection in `bdsim/fouling_modes.py`.
- [x] B5 — Make `_PESOS_W_OUT` an explicit `.copy()` in `bdsim/split_nn.py`.
- [x] B6 — Remove unreachable `i == 0` branch in lab-cycle latch (both drivers).

### Workstream C — test fix
- [x] C5 — Fix `test_stuck_does_not_update_from_dropout` no-op assertion.

### Workstream D — documentation
- [x] D1 — Fix false `tests/test_smoke.py` fingerprint claim (AGENTS, README, ADMIN).
- [x] D2 — Resolve AGENTS vs ADMIN dashboard-pin contradiction.
- [x] D3 — README: "matplotlib" → "Plotly".
- [x] D4 — Date README Performance table.
- [x] D5 — Expand AGENTS profile fingerprint table.
- [x] D6 — Resolve AGENTS quality-latching `✅` vs `⏳` contradiction.
- [x] D7 — Fix `NOTES.md` stale line refs and smoke-test count.
- [x] D8 — Tighten `StepResult.spectra` type annotation (TYPE_CHECKING import).
- [x] D9 — Remove `USER.md` stale IndexError warning.
- [x] D10 — Expand USER `ProcessFaults` field table.
- [x] D11 — Expand `Config-surface.md` field table.
- [x] D12 — Expand `Fingerprints-and-tests.md` table.
- [x] D13 — Add fouling-mode-walkthrough doc (`docs/30-engine/Fouling-modes.md`).
- [x] D14 — Fix `.gitignore` PDF paths (`docs/` not root).
- [x] D15 — Stop claiming PDFs are gitignored at repo root.

### Workstream E — CI
- [x] E3 — Add `.github/workflows/ci.yml` (pytest matrix 3.10–3.12 + ruff).
- [ ] E1 / E2 — **Deferred** (live-vs-batch copy-pair refactor; JIT kernel buffer hoisting — both fingerprint-breaking; deferred per `AGENTS.md` "Numba kernels are sacred").

### Discovered during work
- **Pre-existing fingerprint drift** on non-reference stacks: `numpy==2.4.6` / `scipy==1.18.0` on this dev host vs the `numpy==2.2.6` / `scipy==1.15.3` 1.1.1 reference. Trajectory hashes diverge (`6f61eb53…` vs `c8807b23…`). Not caused by this workstream; tracked separately as a `uv.lock` tightening task.
- **Pre-existing quality + dynamic-fouling + actuator-wear index bug** (fixed in 1.2.0; see [1.2.0] entry above for full detail).

### Subsequent work — 1.2.0 broad-defaults expansion

User asked for fouling, quality measurements, spectra, and valve degradation to be **on by default**. This was fingerprint-breaking for some profiles; the legacy / dynamic-fouling / external-disturbances profiles kept their established hashes via explicit overrides, and the new broad-defaults profile got its own pin.

- `ProcessFaults` defaults flipped ON: `quality_state`, `pump_wear`, `valve_wear`, `spectrum_enabled` (`fouling_dynamic` was already ON).
- Version bumped: `pyproject.toml` + `bdsim/__init__.py` → `1.2.0`. bdsim-dashboard does not currently pin bdsim (see `ADMIN.md` — Versioning).
- All affected `bdsim/` defaults updated and `docs/`, `AGENTS.md`, `ADMIN.md`, `USER.md`, `CHANGELOG.md` updated. `tests/test_smoke.py::test_step_returns_step_result_with_correct_shapes` bumped from `sv.shape == (22,)` to `(30,)` to match the new broad default.

### Verification
- [x] `ruff check` clean on changed files (pre-existing `E702` / `F841` in `ode.py` are out of scope per `AGENTS.md`).
- [x] Full test suite — non-fingerprint tests pass (smoke 14/14, `test_live_simulator.py` non-fp 15/15 + 2/2 byte-identical, fouling-mode tests 36/36, etc.).
- [x] Fingerprint regression: all `HEAD` hashes unchanged by this PR's edits (verified on `test_run_with_long_horizon_matches_live` and `test_run_to_completion_matches_run_with_byte_for_byte` — both pass; per-profile fingerprint-pin failures on this host are the pre-existing dep drift noted above).
- [ ] Pre-existing fingerprint-pin drift on non-reference stacks (tracked separately).
