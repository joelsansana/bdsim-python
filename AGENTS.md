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
- **Don't touch `pyproject.toml` version without also updating `bdsim/__init__.py:__version__` and the dashboard's minimum-required-version pin in lockstep.** All three are pinned to the same value today (`1.1.1`).

## File map

```
bdsim/
├── __init__.py         # public API: run, run_with, StepResult, Results
├── config.py           # Parameters, Settings, ProcessFaults, SensorFaults, ValveFaults, ARMAX, PIDController
├── thermo.py           # Qoil, Vmolar, Mmx, cpmx, side_reactions
├── kinetics.py         # rxrates (transesterification kinetics)
├── split_nn.py         # DecanterSplitNet (PyTorch MLP) + numpy split()
├── ode.py              # ODEmodel, AEmodel — Numba-JIT compiled
├── spectra.py          # Layer 2.8 NIR/IR virtual spectrum sensor (comp_spectrum)
├── data/
│   └── spectra_ref.csv # 6 species × 631 NIR channels, GPL-3 (Fernandes/Strelet 2019)
├── simulation.py       # run, run_with — the main driver (batch)
├── live_simulator.py   # LiveSimulator — stateful, per-step driver for the dashboard
├── fouling_modes.py    # Layer 2.8b: five-mode fouling factor stepper (modes 0-3 time functions, modes 4-5 ARMAX noise)
├── plots.py            # 9-figure Plotly block + CSV writer
└── cli.py              # `python -m bdsim` entry point

tests/
├── test_smoke.py            # 14 smoke tests, fingerprint regression
├── test_live_simulator.py   # LiveSimulator + byte-identical contract to run_with
├── test_disturbances.py     # Layer 2.6 external disturbance track
├── test_layer24_degradation.py # Layer 2.4 pump_health + valve_stiction_pct
├── test_layer26b_cw_pump.py # Layer 2.6b cw_pump_trip mid-run override
├── test_layer27_knobs.py    # Layer 2.7 operator-driven disturbance knobs
├── test_layer28a_spectra.py # Layer 2.8 NIR/IR spectrum sensor
├── test_layer28b_fouling_modes.py # Layer 2.8b: 5-mode fouling stepper (offline)
└── test_layer28b_live_fouling.py # Layer 2.8b: LiveSimulator wiring + priority

results/                # default output dir (created on first run)
docs/                   # Obsidian knowledge base (docs/Home.md); see docs/README.md
scripts/                # optional maintainer utilities (not part of the package)
NOTES.md                # historical: upstream-faithful bugs we found and fixed
```

## Build conventions

- **Numba kernels are sacred.** Don't refactor for readability if it costs a fingerprint pin update. The ODE RHS, reaction kinetics, valve stiction, PID step, ARMAX noise update, sensor measurements, and AE model are all `@njit(cache=True)`. The 72h default sim runs in ~60s on a single core.
- **Default profiles (read carefully).** New Layer flags should default **off / zero**, except documented exceptions. Today `ProcessFaults.fouling_dynamic` defaults to **`True`** (Layer 2.5 on → `sv` width 22). Bare `ProcessFaults()` is **not** the upstream legacy pin path.

  | Profile | Construction | Pins (examples) |
  |--------|--------------|-----------------|
  | Runtime default | `ProcessFaults()` | Layer 2.5 path; not the legacy hash |
  | Legacy fingerprint | `fouling_dynamic=False` (+ other masters off/zero) | batch `sv=c8807b23…`, live `sv=23c3c885…` |
  | Layer 2.5 fingerprint | `fouling_dynamic=True`, extras off | batch `sv=696531c4…` |
  | Layer 2.6 active | disturbance amplitudes on | batch `sv=8865a8c3…`, live `sv=bb763a9b…` |

  Full write-up: `docs/00-orientation/Byte-identical-contract.md`.
- **Tests run from the bdsim root:** `python3 -m pytest tests/ -q`. Full suite: ~6:40.
- **Ruff clean on new code.** Pre-existing `E702`/`E401` in `ode.py` are the Numba multi-statement lines and are out of scope.

## Git workflow

- Feature branches, PRs to `main`. Remote: `git@github.com:joelsansana/bdsim-python.git`.
- Branch naming: `feature/<name>`, `fix/<name>`, `layerN/<name>` for roadmap steps.
- Commit messages: `feat(<scope>)`, `fix(<scope>)`, `chore(<scope>)` prefix; body explains *why* the change is needed and *what fingerprint update* it caused, if any.
- Tests + ruff clean required before merge.
- **Fingerprint update = a "this is intentional" line in the commit body** with the old pin and the new pin, side by side.

## Layer / roadmap status

This repo's coverage:

- ✅ Step 1 (live sim driver), Step 2 (MQTT publish is on the dashboard side), Step 4 (live fault injection), Step 5 (sensor failure modes), Step 8 (scenario runner is on the dashboard side)
- ✅ Layer 2.1 (quality latching — `quality_state=True` mode), Layer 2.5 (HEX fouling as continuous state, `fouling_dynamic=True` mode), Layer 2.6 (external disturbances), Layer 2.6b (cw_pump_trip mid-run override), Layer 2.7 (operator-driven disturbance schedule), Layer 2.4 (pump_health + valve_stiction_pct as continuous state — `pump_wear=True` / `valve_wear=True`), Layer 2.8a (NIR/IR virtual spectrum sensor — `spectrum_enabled=True`), **Layer 2.8b** (five-mode fouling stepper — modes 4/5 stochastic ARMAX windowed injection, port of upstream `fouling.m`)
- ⏳ Layer 2.1 `quality_latched` code review (Joel owes) — pending

## Coordination with bdsim-dashboard

The dashboard calls into bdsim via `from bdsim import LiveSimulator` (live path) or `from bdsim.simulation import run_with` (batch path). When you change a `ProcessFaults` knob, the dashboard's `SimRunner._build_pfaults()` may need a corresponding update to surface it. When you change a `Settings` field, check the dashboard's `RunnerConfig` for a parallel knob.

When you change the bdsim install, re-run from this repo's root (**fingerprint-aligned**):
```bash
uv sync --extra test
# verify: uv run python -c "import numpy,scipy,numba; print(numpy.__version__, scipy.__version__, numba.__version__)"
```
The dashboard depends on the editable install from that same env, not on `sys.path` hacks. Bare `pip install -e .` is **run-only** (ignores `uv.lock`; fingerprint tests may fail). See `ADMIN.md` — Fingerprint-aligned install.
