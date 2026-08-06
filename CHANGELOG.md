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
