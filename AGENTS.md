# AGENTS.md — bdsim

Quick orientation for any agent (human or AI) opening this repo cold.

## What this is

A Python port of the BDSIM MATLAB/Octave simulator by Natércia C. P. Fernandes (University of Coimbra, 2019). Models a biodiesel plant (filter → reactor → heat exchanger → decanter → washer → dryer) with sensors, PID controllers, valve stiction, and a decanter split neural network.

**This is the simulation engine.** It is the "plant" the dashboard and the Lepanto FDE model both interface with. The bdsim-dashboard repo wraps it as a live, MQTT-bridged process. This repo is the math.

## Hard rule

**Faithful port + Lepanto's fault-detection research.** Two callers:

1. **bdsim-dashboard** (operator console) — runs the sim live, exposes it over HTTP/MQTT. Needs deterministic, step-by-step control.
2. **Lepanto R&D** (Joel + Eugeniu, off-the-books) — feeds sim trajectories into correlation / fault-detection models. Needs batch runs with reproducible seeds and clean fingerprint pins.

Both callers depend on the **byte-identical trajectory contract**: given the same seed and the same `ProcessFaults`, the trajectory is reproducible. Fingerprints are pinned in `tests/` — drift fails loud. Don't reorder statements in the Numba kernels without updating the pinned hashes and explaining why.

## What NOT to do without asking Joel

- **Don't change the ODE math.** If you think the upstream MATLAB is wrong, write a test that captures the bug, then ask before "fixing" it. See `NOTES.md` for two prior upstream-faithful fixes (Foil noise typo, temperature display bug) — those were agreed, and documented.
- **Don't add a new external dependency** without asking. The current stack (numpy/scipy/numba/torch) is intentional. New deps = new deploy surface.
- **Don't relax the byte-identical fingerprint contract.** If you need different behavior, gate it behind a `ProcessFaults` flag (default off).
- **Don't touch `pyproject.toml` version without also updating `bdsim/__init__.py:__version__` and the dashboard's minimum-required-version pin in lockstep.** All three are pinned to the same value today (`1.1.0`).

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
├── plots.py            # 9-figure Plotly block + CSV writer
└── cli.py              # `python -m bdsim` entry point

tests/
├── test_smoke.py            # 14 smoke tests, fingerprint regression
├── test_live_simulator.py   # LiveSimulator + byte-identical contract to run_with
├── test_disturbances.py     # Layer 2.6 external disturbance track
├── test_layer24_degradation.py # Layer 2.4 pump_health + valve_stiction_pct
├── test_layer26b_cw_pump.py # Layer 2.6b cw_pump_trip mid-run override
├── test_layer27_knobs.py    # Layer 2.7 operator-driven disturbance knobs
└── test_layer28a_spectra.py # Layer 2.8 NIR/IR spectrum sensor

results/                # default output dir (created on first run)
docs/                   # Fernandes 2019 PDF + manual PDF (upstream refs)
NOTES.md                # historical: upstream-faithful bugs we found and fixed
```

## Build conventions

- **Numba kernels are sacred.** Don't refactor for readability if it costs a fingerprint pin update. The ODE RHS, reaction kinetics, valve stiction, PID step, ARMAX noise update, sensor measurements, and AE model are all `@njit(cache=True)`. The 72h default sim runs in ~60s on a single core.
- **Default-mode must be byte-identical to the upstream MATLAB numbers.** All Layer 2.5+ features are gated on a `ProcessFaults` flag that defaults to **off / zero**. The legacy `sv=6f61eb53...` fingerprint pins the legacy batch path; `sv=f37fb5e0...` pins the legacy live path. Active profiles have their own pins (Layer 2.6: `sv=e2a29849...` batch, `sv=13ea81f3...` live; Layer 2.5: `sv=1938fec8...` batch).
- **Tests run from the bdsim root:** `python3 -m pytest tests/ -q`. Full suite: ~6:40.
- **Ruff clean on new code.** Pre-existing `E702`/`E401` in `ode.py` are the Numba multi-statement lines and are out of scope.

## Git workflow

- Feature branches, PRs to `main`. No remote yet.
- Branch naming: `feature/<name>`, `fix/<name>`, `layerN/<name>` for roadmap steps.
- Commit messages: `feat(<scope>)`, `fix(<scope>)`, `chore(<scope>)` prefix; body explains *why* the change is needed and *what fingerprint update* it caused, if any.
- Tests + ruff clean required before merge.
- **Fingerprint update = a "this is intentional" line in the commit body** with the old pin and the new pin, side by side.

## Layer / roadmap status (mirror of dashboard + Lepanto vault)

See `~/Documents/Notas/Lepanto/BDSIM_Roadmap.md` for the authoritative roadmap. This repo's coverage:

- ✅ Step 1 (live sim driver), Step 2 (MQTT publish is on the dashboard side), Step 4 (live fault injection), Step 5 (sensor failure modes), Step 8 (scenario runner is on the dashboard side)
- ✅ Layer 2.1 (quality latching — `quality_state=True` mode), Layer 2.5 (HEX fouling as continuous state, `fouling_dynamic=True` mode), Layer 2.6 (external disturbances), Layer 2.6b (cw_pump_trip mid-run override), Layer 2.7 (operator-driven disturbance schedule), Layer 2.4 (pump_health + valve_stiction_pct as continuous state — `pump_wear=True` / `valve_wear=True`), Layer 2.8a (NIR/IR virtual spectrum sensor — `spectrum_enabled=True`)
- ⏳ Layer 2.1 `quality_latched` code review (Joel owes) — pending

## Coordination with bdsim-dashboard

The dashboard calls into bdsim via `from bdsim import LiveSimulator` (live path) or `from bdsim.simulation import run_with` (batch path). When you change a `ProcessFaults` knob, the dashboard's `SimRunner._build_pfaults()` may need a corresponding update to surface it. When you change a `Settings` field, check the dashboard's `RunnerConfig` for a parallel knob.

When you change the bdsim install, re-run:
```bash
pip install --user --break-system-packages -e ~/Documents/projects/bdsim
```
The dashboard depends on the editable install, not on `sys.path` hacks.
