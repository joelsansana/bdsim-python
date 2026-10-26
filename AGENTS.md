# AGENTS.md — bdsim

Quick orientation for any agent (human or AI) opening this repo cold.

## What this is

A Python port of the BDSIM MATLAB/Octave simulator by Natércia C. P. Fernandes (University of Coimbra, 2019). Models a biodiesel plant (filter → reactor → heat exchanger → decanter → washer → dryer) with sensors, PID controllers, valve stiction, and a decanter split neural network.

**This is the simulation engine.** It is the "plant" the dashboard wraps as a live, MQTT-bridged process. This repo is the math.

## Hard rule

**Faithful port + reproducible trajectories.** Two callers:

1. **bdsim-dashboard** (operator console, separate repo) — runs the sim live, exposes it over HTTP/MQTT. Needs deterministic, step-by-step control.
2. **External fault-detection / correlation models** — feed sim trajectories into ML / statistical models. Need batch runs with reproducible seeds and clean fingerprint pins.

Both callers depend on the **byte-identical trajectory contract**: given the same seed and the same `ProcessFaults`, the trajectory is reproducible. Fingerprints are pinned in `tests/` — drift fails loud. Don't reorder statements in the Numba kernels without updating the pinned hashes and explaining why.

## What NOT to do without asking Joel

- **Don't change the ODE math.** If you think the upstream MATLAB is wrong, write a test that captures the bug, then ask before "fixing" it. See `NOTES.md` for two prior upstream-faithful fixes (Foil noise typo, temperature display bug) — those were agreed, and documented.
- **Don't add a new external dependency** without asking. The current stack (numpy/scipy/numba/torch) is intentional. New deps = new deploy surface.
- **Don't relax the byte-identical fingerprint contract.** If you need different behavior, gate it behind a `ProcessFaults` flag (default off/zero, except documented exceptions — today `fouling_dynamic=True`).
- **Don't touch `pyproject.toml` version without also updating `bdsim/__init__.py:__version__`.** Both are pinned to the same value today (`1.2.0`). The bdsim-dashboard does not currently pin a minimum bdsim version; add the third pin when the dashboard gains an installation contract (see `ADMIN.md` — Versioning).

## File map

```
bdsim/
├── __init__.py         # public API: run, run_with, StepResult, Results
├── config.py           # Parameters, Settings, ProcessFaults, SensorFaults, ValveFaults, ARMAX, PIDController
├── thermo.py           # Qoil, Vmolar, Mmx, cpmx, side_reactions
├── kinetics.py         # rxrates (transesterification kinetics)
├── split_nn.py         # DecanterSplitNet (PyTorch MLP) + numpy split()
├── ode.py              # ODEmodel, AEmodel — Numba-JIT compiled
├── spectra.py          # NIR/IR virtual spectrum sensor (comp_spectrum)
├── data/
│   └── spectra_ref.csv # 6 species × 631 NIR channels, GPL-3 (Fernandes/Strelet 2019)
├── simulation.py       # run, run_with — the main driver (batch)
├── live_simulator.py   # LiveSimulator — stateful, per-step driver for the dashboard
├── fouling_modes.py    # five-mode fouling factor stepper (modes 0-3 time functions, modes 4-5 ARMAX noise)
├── plots.py            # 9-figure Plotly block + CSV writer
└── cli.py              # `python -m bdsim` entry point

tests/
├── test_smoke.py            # 14 smoke tests (shapes, physical ranges, determinism; no SHA pins)
├── test_live_simulator.py   # LiveSimulator + byte-identical contract to run_with
├── test_disturbances.py     # external disturbances track
├── test_actuator_wear.py    # pump_health + valve_stiction_pct continuous-state dynamics
├── test_cw_pump_trip.py     # cooling-water pump trip mid-run override
├── test_operator_knobs.py   # operator-driven disturbance knobs
├── test_spectrum_sensor.py  # NIR/IR virtual spectrum sensor
├── test_fouling_modes.py    # five-mode fouling stepper (offline)
└── test_fouling_modes_live.py # LiveSimulator wiring + priority for fouling-mode windows

results/                # default output dir (created on first run)
docs/                   # Obsidian knowledge base (docs/Home.md); see docs/README.md
scripts/                # optional maintainer utilities (not part of the package)
NOTES.md                # historical: upstream-faithful bugs we found and fixed
```

## Build conventions

- **Numba kernels are sacred.** Don't refactor for readability if it costs a fingerprint pin update. The ODE RHS, reaction kinetics, valve stiction, PID step, ARMAX noise update, sensor measurements, and AE model are all `@njit(cache=True)`. The 72h default sim runs in ~60s on a single core.
- **Default profiles (read carefully).** As of 1.2.0, the four major feature flag switches default to **`True`** (dynamic fouling, quality latching, pump wear, valve wear, NIR/IR spectrum sensor). Bare `ProcessFaults()` enables all of them → `sv` width 30, hash `691cf51b…`. Tests and demos that need a narrower state vector must pass the relevant `quality_state=False` / `pump_wear=False` / `valve_wear=False` / `spectrum_enabled=False` overrides explicitly. The legacy / dynamic-fouling-only profiles below remain canonical — they are the byte-identical contracts to the upstream MATLAB baseline and the 1.0 dynamic-fouling release.

  | Profile | Construction | `sv` width | Pins (full SHA-256[:16]) |
  |--------|--------------|-------------|----------------------------|
| Runtime default (1.2.0+) | `ProcessFaults()` | **30** | batch `sv=65c8e648…` `pv=49d36bb6…` `uv=936439b7…` (updated: Octave-aligned PID `nic` tick) |
 | Legacy fingerprint | `fouling_dynamic=False`, all masters off | 21 | batch `sv=ad3d55bf…` `pv=fbdcb0d4…` `uv=6c182226…` |
 | dynamic-fouling fingerprint | `fouling_dynamic=True`, all other masters off | 22 | batch `sv=07c2e836…` `pv=2ebd9f02…` `uv=7fe600c7…` |
 | dynamic-fouling + quality-latching | `fouling_dynamic=True`, `quality_state=True` (no actuator wear, no spectra) | 27 | (re-pin with suite after nic fix — see tests) |
 | pump-only actuator wear | `pump_wear=True`, `valve_wear=False`, `fouling_dynamic=True`, `quality_state=False` | 23 | batch `sv=736a8032…` `pv=fc29a420…` `uv=6865d7d6…` |
 | valve-only actuator wear | `pump_wear=False`, `valve_wear=True`, `fouling_dynamic=True`, `quality_state=False` | 23 | batch `sv=56c453e9…` `pv=3e8783ba…` `uv=bf7ff824…` |
 | pump + valve wear | `pump_wear=True`, `valve_wear=True`, `fouling_dynamic=True`, `quality_state=False` | 24 | batch `sv=2290bca3…` `pv=694dab99…` `uv=8231d3c9…` |
 | external disturbances active | amplitudes > 0 (legacy masters off) | 21 | batch `sv=9b102224…` `pv=1bb38dc9…` `uv=2e5a1b88…` |
 | cooling-water pump trip live | `LiveSimulator` cw_pump_trip mid-run override | — | live short-horizon pins in `tests/test_cw_pump_trip.py` |

  Full write-up: `docs/00-orientation/Byte-identical-contract.md`.
- **Tests run from the bdsim root:** `python3 -m pytest tests/ -q`. Full suite: ~6:40.
- **Ruff clean on new code.** Pre-existing `E702`/`E401` in `ode.py` are the Numba multi-statement lines and are out of scope.

## Git workflow

- Feature branches, PRs to `main`. Remote: `git@github.com:joelsansana/bdsim-python.git`.
- Branch naming: `feature/<name>`, `fix/<name>`, `layerN/<name>` for roadmap steps.
- Commit messages: `feat(<scope>)`, `fix(<scope>)`, `chore(<scope>)` prefix; body explains *why* the change is needed and *what fingerprint update* it caused, if any.
- Tests + ruff clean required before merge.
- **Fingerprint update = a "this is intentional" line in the commit body** with the old pin and the new pin, side by side.

## Feature groups

This repo's coverage:

- ✅ Step 1 (live sim driver), Step 2 (MQTT publish is on the dashboard side), Step 4 (live fault injection), Step 5 (sensor failure modes), Step 8 (scenario runner is on the dashboard side)
- ✅ dynamic fouling (HEX fouling as continuous state, `fouling_dynamic=True` mode), external disturbances (ambient / CW sinusoid amplitudes & drift), cooling-water pump trip (cw_pump_trip mid-run override), operator disturbance knobs (operator-driven disturbance schedule), actuator wear (pump_health + valve_stiction_pct as continuous state — `pump_wear=True` / `valve_wear=True`), NIR/IR spectrum sensor (NIR/IR virtual spectrum sensor — `spectrum_enabled=True`), **fouling-mode windows** (five-mode fouling stepper — modes 4/5 stochastic ARMAX windowed injection, port of upstream `fouling.m`)
- 🟡 quality latching (`quality_state=True` mode): shipped, but `quality_latched` code review (Joel owes) is still pending. Marked 🟡 rather than ✅ until that review closes.

## Practical dev workflow

*Migrated from `docs/00-orientation/How-to-work-here.md` (deleted in 1.2.x docs cleanup).*

### Do

- Prefer drivers, config knobs, tests, and the docs vault over touching ODE statement order
- Gate new trajectory-affecting behaviour on a `ProcessFaults` flag defaulting off/zero (exception today: `fouling_dynamic=True` — see [`Byte-identical-contract`](docs/00-orientation/Byte-identical-contract.md#canonical-profiles))
- Add/update tests under `tests/`; keep ruff clean on new Python
- Update the vault notes ([Config-surface](docs/30-engine/Config-surface.md), [Channel-indices](docs/40-helpers/Channel-indices.md)) in the **same** change when you add knobs or widen `sv`/`pv`

### Do not (without asking)

- Change ODE math "because MATLAB looks wrong" — capture with a test, then ask (`NOTES.md` has prior agreed fixes)
- Add a new external dependency
- Relax the byte-identical contract
- Bump version in only one of `pyproject.toml` / `bdsim/__init__.py` / dashboard pin

### Tests

Fingerprint-aligned env first (see [`ADMIN.md`](ADMIN.md) — **Fingerprint-aligned install**):

```bash
uv sync --extra test
uv run python -m pytest tests/ -q
```

Full suite ~6:40. Smoke + fingerprint tests are the first safety net. See [`Fingerprints-and-tests`](docs/30-engine/Fingerprints-and-tests.md). Do **not** rely on bare `pip install` for pin parity — it ignores `uv.lock`.

### Sacred hot path

Anything `@njit(cache=True)` in `ode.py`, measurement/stiction/PID helpers, etc. is **sacred**. Style refactors that reorder statements break pins.

### Dashboard coordination

Dashboard builds `ProcessFaults` via `SimRunner._build_pfaults()`. New knobs may need a dashboard PR too. Editable install: `uv sync --extra test` from this repo root ([`ADMIN.md`](ADMIN.md)).

## Coordination with bdsim-dashboard

The dashboard calls into bdsim via `from bdsim import LiveSimulator` (live path) or `from bdsim.simulation import run_with` (batch path). When you change a `ProcessFaults` knob, the dashboard's `SimRunner._build_pfaults()` may need a corresponding update to surface it. When you change a `Settings` field, check the dashboard's `RunnerConfig` for a parallel knob.

When you change the bdsim install, re-run from this repo's root (**fingerprint-aligned**):
```bash
uv sync --extra test
# verify: uv run python -c "import numpy,scipy,numba; print(numpy.__version__, scipy.__version__, numba.__version__)"
```
The dashboard depends on the editable install from that same env, not on `sys.path` hacks. Bare `pip install -e .` is **run-only** (ignores `uv.lock`; fingerprint tests may fail). See `ADMIN.md` — Fingerprint-aligned install.
