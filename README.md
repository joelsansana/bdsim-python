# bdsim — Biodiesel Process Simulator (Python port)

A Python port of the BDSIM MATLAB/Octave simulator by Natércia C. P. Fernandes
(University of Coimbra, 2019). The original models a biodiesel plant
(filter → reactor → heat exchanger → decanter → washer → dryer) with sensors,
PID controllers, valve stiction, and a decanter split neural network.

**Current version: 1.2.0** (`bdsim/__init__.py:__version__`, mirrored in `pyproject.toml`). See [`CHANGELOG.md`](CHANGELOG.md). A fingerprint bump always requires a version bump — see `AGENTS.md` for the rule. As of 1.2.0, the runtime default (`ProcessFaults()`) enables dynamic fouling, quality latching, pump + valve wear, and the NIR/IR spectrum sensor — `sv` width 30. The legacy 21-wide and dynamic fouling 22-wide profiles remain canonical but require explicit `False` overrides.

The port is faithful to the MATLAB semantics and uses modern Python idioms:

- **NumPy** for vectorised math
- **scipy.integrate.solve_ivp** with RK45 (`max_step=dt`) — the upstream default solver is `lsode` (LSODA) in Octave and `ode45` (RK45) in MATLAB; we use RK45 with `max_step=settings.dt` to keep per-interval cost predictable
- **Numba JIT** (`@njit(cache=True)`) on every per-step hot kernel — the ODE RHS, reaction kinetics, valve stiction, PID step, ARMAX noise update, sensor measurements, and AE model. The 72h upstream-default simulation runs in ~60s on a single CPU core.
- **PyTorch** for the decanter split MLP — trainable on new data
- **Plotly** for all nine upstream figures — saved as standalone interactive HTML (`index.html` collects all nine in one page).
- **Typed dataclasses** for every configuration block

## Install

**Fingerprint-aligned (contributors / pin tests)** — uses committed `uv.lock`:

```bash
uv python install 3.10          # if needed; 3.10 is the 1.2.0 pin reference
uv sync --extra test
uv run python -c "import numpy,scipy,numba; print(numpy.__version__, scipy.__version__, numba.__version__)"
# expect on 3.10: 2.2.6 1.15.3 0.66.0
```

Full procedure (verify path, caveats, health checks): [`ADMIN.md`](ADMIN.md) — section **Fingerprint-aligned install**.

**Run-only** (sim works; fingerprint tests may fail — pip ignores `uv.lock`):

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[test]"
```

PEP 668 fallback only: `pip install --user --break-system-packages -e .`

## Run

```bash
python -m bdsim                          # upstream defaults, random seed
python -m bdsim 42                       # reproducible run
python -m bdsim --no-plots               # CSV files only
python -m bdsim 42 --outdir my_results   # custom output directory
```

Outputs (4 CSVs + 9 HTML figures + index.html) are written to `results/`
by default. CSV column headers match the upstream `BDsim.m` byte-for-byte
so existing MATLAB plotting code can consume them unchanged. Open
`results/index.html` in any browser to see all nine figures.

## Documentation

| Doc | Purpose |
|-----|---------|
| [USER.md](USER.md) | On-ramp for developers / researchers; recipes for common tasks |
| [ADMIN.md](ADMIN.md) | Install, packaging, fingerprint-aligned env, health checks |
| [AGENTS.md](AGENTS.md) | Agent hard rules, dev workflow, fingerprint profiles |
| [docs/Home.md](docs/Home.md) | Knowledge vault (plant, math, engine) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to add a knob, bump a fingerprint, open a PR |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Community standards |
| [SECURITY.md](SECURITY.md) | How to report vulnerabilities privately |
| [CHANGELOG.md](CHANGELOG.md) | Release history and pinned-profile announcements |
| [CITATION.cff](CITATION.cff) | How to cite |

## Programmatic use

```python
from bdsim import run, run_with
from bdsim.config import ProcessFaults
import numpy as np

# Runtime default: fouling_dynamic=True → sv width 22 (dynamic fouling α at sv[21])
res = run(seed=42)
print(res.t.shape, res.sv.shape)         # (51999,) (51999, 22)

# Legacy fingerprint profile (upstream 21-wide state): pass fouling_dynamic=False
res_legacy = run_with(
    pfaults=ProcessFaults(fouling_dynamic=False),
    seed=42,
)

# Custom faults: turn clogging off (still dynamic fouling on unless you override)
res = run_with(
    pfaults=ProcessFaults(clog_fraction=0.0, fouling=0),
    seed=42,
)
```

`ProcessFaults()` is the **runtime default** — as of 1.2.0 it enables dynamic fouling, quality latching, pump + valve wear, and the NIR/IR spectrum sensor. The **legacy fingerprint** suite uses an explicit all-False profile, and the **dynamic fouling** suite uses `fouling_dynamic=True` with everything else `False` — see [`docs/00-orientation/Byte-identical-contract.md`](docs/00-orientation/Byte-identical-contract.md) and [`AGENTS.md`](AGENTS.md).

## Files

```
bdsim/
├── __init__.py         # public API
├── config.py           # Parameters, Settings, *Faults, ARMAX, PIDController, Results
├── thermo.py           # Qoil, Vmolar, Mmx, cpmx, side_reactions
├── kinetics.py         # rxrates (transesterification kinetics)
├── split_nn.py         # DecanterSplitNet (PyTorch MLP) + numpy split()
├── ode.py              # ODEmodel, AEmodel
├── spectra.py          # NIR/IR virtual spectrum sensor (comp_spectrum)
├── data/
│   └── spectra_ref.csv # 6 species × 631 NIR channels, GPL-3 (Fernandes/Strelet 2019)
├── fouling_modes.py    # fouling-mode windows: five-mode fouling factor stepper
├── simulation.py       # run, run_with — the main driver
├── live_simulator.py   # LiveSimulator — per-step driver for the dashboard
├── plots.py            # 9-figure Plotly block + CSV writer
└── cli.py              # `python -m bdsim` entry point
tests/
├── test_smoke.py             # smoke tests (run with `pytest`)
├── test_live_simulator.py    # LiveSimulator + byte-identical contract to run_with
├── test_disturbances.py       # external disturbances track
├── test_actuator_wear.py      # pump_health + valve_stiction_pct continuous-state dynamics
├── test_cw_pump_trip.py       # cooling-water pump trip mid-run override
├── test_operator_knobs.py     # operator-driven disturbance knobs
├── test_spectrum_sensor.py    # NIR/IR virtual spectrum sensor
├── test_fouling_modes.py      # five-mode fouling stepper (offline)
└── test_fouling_modes_live.py # LiveSimulator wiring + priority for fouling-mode windows
results/                # default output directory (created on first run)
```

## Differences from the MATLAB upstream

| Aspect | MATLAB | Python port |
|---|---|---|
| ODE solver | `lsode` (Octave) / `ode45` (MATLAB) | `solve_ivp` RK45 with `max_step=dt` |
| Configuration | Script-style `.m` files with workspace variables | `@dataclass` typed objects |
| Persistent state | `persistent` keyword in nested functions | Explicit state objects on driver |
| Hot kernels | MEX / interpreted MATLAB | **Numba `@njit(cache=True)`** — ~3× speedup |
| ANN | Hardcoded matrix operations | `torch.nn.Module`, trainable |
| Plotting | `plot`, `plotyy` | **Plotly** interactive HTML (zoom, pan, hover) |
| RNG | `rand('shuffle')` (MATLAB) | `np.random.seed(seed)` |
| Sensor intermittence | In-place mutation of growing arrays | Pre-allocated arrays |

## Performance

72-hour upstream-default simulation, single CPU core
(benchmarks captured 2026-07-13 on the bdsim-dashboard reference host,
Python 3.10 / macOS arm64 / `uv.lock`-resolved stack):

| Stack | Wall time | Speedup |
|---|---|---|
| Pure Python + matplotlib | 173.7 s | 1.0× (baseline) |
| **Numba JIT + Plotly** | **~60 s** | **~2.9×** |

The Numba win is in the per-step RHS evaluation. A 5× speedup would
require a custom integrator with Numba-compiled step-size control
(currently `solve_ivp` is Python-bound for adaptive stepping).

## License

GPLv3+, matching upstream BDSIM. Original copyright 2019 Natércia C. P. Fernandes,
natercia@eq.uc.pt.

Cite this software (and upstream BDSIM / split-NN references) via [`CITATION.cff`](CITATION.cff).

## Archival DOI (Zenodo)

A versioned, citable Zenodo DOI is set up at first release (issue #21).
Procedure: in Zenodo, link the GitHub repo (`joelsansana/bdsim-python`),
then create a GitHub release — Zenodo mints a DOI per release. The DOI
goes into `CITATION.cff` and a Zenodo badge is added to this README
on the first cut. Until then, cite via the version in
`pyproject.toml` / `bdsim/__init__.py:__version__`.

## Applications

The package was developed to provide a reproducible biodiesel-plant simulation
for:

- **Control-loop tuning and PID studies.** Sweep `Settings.live_sp1..4`, watch
  the controller chase them, and inspect the 9 standard figures.
- **Fault-detection / anomaly-detection R&D.** Inject sensor bias, dropout,
  or stuck-at faults via `SensorFaults`, or run the actuator wear pump/valve
  degradation paths, then train statistical or ML models on the trajectories.
- **Operator training and scenario rehearsal.** Drive the sim live via
  `LiveSimulator` or the companion [`bdsim-dashboard`](https://github.com/joelsansana/bdsim-dashboard)
  operator console to rehearse a fault response before going near a real plant.
- **Process-control coursework.** A faithful reimplementation of upstream
  Fernandes (2019) with the same inputs/outputs, suitable as a teaching
  reference for a unit on transesterification kinetics and PID loops.

## Contributing

Issues and pull requests are welcome. Before opening a PR, please:

1. **Read [`AGENTS.md`](AGENTS.md)** for build conventions and the
   fingerprint-regression contract.
2. **Don't change the ODE math without a failing test first.** The Numba-JIT
   kernels pin a SHA-256 fingerprint per profile (`tests/test_live_simulator.py`
   and the per-feature test files; `tests/test_smoke.py` checks shapes and
   determinism only — no SHA pins). Any numerical change updates the pin and
   must be called out in the PR description.
3. **Don't add a new top-level dependency without asking.** The current stack
   (numpy/scipy/numba/torch) is intentional.
4. **Tests + `ruff check bdsim/ tests/` must be clean** before requesting
   review.

The companion [`bdsim-dashboard`](https://github.com/joelsansana/bdsim-dashboard)
repo wraps this package as a live operator console (HTTP + MQTT +
Streamlit). When you change a `ProcessFaults` knob here, check the
dashboard's `SimRunner._build_pfaults()` for a corresponding update.