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
- Layer 2.1 quality latching (`quality_state`)
- Layer 2.4 pump / valve wear continuous state
- Layer 2.5 dynamic HEX fouling α (`fouling_dynamic`, default on at runtime)
- Layer 2.6 / 2.6b / 2.7 external disturbances, CW pump trip, operator knobs
- Layer 2.8a NIR/IR virtual spectrum sensor
- Layer 2.8b five-mode fouling stepper (windowed ARMAX modes 4/5)
- Typed config surface (`Settings`, `ProcessFaults`, sensor/valve faults)
- Plotly figure set + CSV outputs matching upstream column names
- Fingerprint regression suite (legacy, Layer 2.5, Layer 2.6, live/batch)

### Notes

- Runtime default (`ProcessFaults()`) enables Layer 2.5; legacy upstream pins use explicit `fouling_dynamic=False` — see canonical profiles in the docs vault.
- Companion operator console: [bdsim-dashboard](https://github.com/joelsansana/bdsim-dashboard).
