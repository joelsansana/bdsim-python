# bdsim — Biodiesel Process Simulator (Python port)

A Python port of the BDSIM MATLAB/Octave simulator by Natércia C. P. Fernandes
(University of Coimbra, 2019). The original models a biodiesel plant
(filter → reactor → heat exchanger → decanter → washer → dryer) with sensors,
PID controllers, valve stiction, and a decanter split neural network.

**Current version: 1.1.0** (`bdsim/__init__.py:__version__`, mirrored in `pyproject.toml`). A fingerprint bump always requires a version bump — see `AGENTS.md` for the rule.

The port is faithful to the MATLAB semantics and uses modern Python idioms:

- **NumPy** for vectorised math
- **scipy.integrate.solve_ivp** with RK45 (`max_step=dt`) — the upstream default solver is `lsode` (LSODA) in Octave and `ode45` (RK45) in MATLAB; we use RK45 with `max_step=settings.dt` to keep per-interval cost predictable
- **Numba JIT** (`@njit(cache=True)`) on every per-step hot kernel — the ODE RHS, reaction kinetics, valve stiction, PID step, ARMAX noise update, sensor measurements, and AE model. The 72h upstream-default simulation runs in ~60s on a single CPU core.
- **PyTorch** for the decanter split MLP — trainable on new data
- **Plotly** for all nine upstream figures — saved as standalone interactive HTML (`index.html` collects all nine in one page).
- **Typed dataclasses** for every configuration block

## Install

```bash
cd ~/Documents/projects/bdsim
pip install --break-system-packages -e .   # numpy, scipy, plotly, numba, torch
```

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

## Programmatic use

```python
from bdsim import run, run_with
from bdsim.config import ProcessFaults
import numpy as np

# Default settings
res = run(seed=42)
print(res.t.shape, res.sv.shape)         # (51999,) (51999, 21)

# Custom faults: turn clogging off
res = run_with(
    pfaults=ProcessFaults(clog_fraction=0.0, fouling=0),
    seed=42,
)
```

## Files

```
bdsim/
├── __init__.py         # public API
├── config.py           # Parameters, Settings, *Faults, ARMAX, PIDController, Results
├── thermo.py           # Qoil, Vmolar, Mmx, cpmx, side_reactions
├── kinetics.py         # rxrates (transesterification kinetics)
├── split_nn.py         # DecanterSplitNet (PyTorch MLP) + numpy split()
├── ode.py              # ODEmodel, AEmodel
├── spectra.py          # Layer 2.8 NIR/IR virtual spectrum sensor (comp_spectrum)
├── data/
│   └── spectra_ref.csv # 6 species × 631 NIR channels, GPL-3 (Fernandes/Strelet 2019)
├── fouling_modes.py    # Layer 2.8b: five-mode fouling factor stepper
├── simulation.py       # run, run_with — the main driver
├── live_simulator.py   # LiveSimulator — per-step driver for the dashboard
├── plots.py            # 9-figure matplotlib block + CSV writer
└── cli.py              # `python -m bdsim` entry point
tests/
├── test_smoke.py             # smoke tests (run with `pytest`)
├── test_live_simulator.py    # LiveSimulator + byte-identical contract to run_with
├── test_disturbances.py      # Layer 2.6 external disturbance track
├── test_layer24_degradation.py # Layer 2.4 pump_health + valve_stiction_pct
├── test_layer26b_cw_pump.py  # Layer 2.6b cw_pump_trip mid-run override
├── test_layer27_knobs.py     # Layer 2.7 operator-driven disturbance knobs
├── test_layer28a_spectra.py  # Layer 2.8 NIR/IR spectrum sensor
├── test_layer28b_fouling_modes.py # Layer 2.8b fouling stepper (offline)
└── test_layer28b_live_fouling.py  # Layer 2.8b LiveSimulator wiring + priority
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

72-hour upstream-default simulation, single CPU core:

| Stack | Wall time | Speedup |
|---|---|---|
| Pure Python + matplotlib | 173.7 s | 1.0× (baseline) |
| **Numba JIT + Plotly** | **~60 s** | **~2.9×** |

The Numba win is in the per-step RHS evaluation. A 5× speedup would
require a custom integrator with Numba-compiled step-size control
(currently `solve_ivp` is Python-bound for adaptive stepping).

## Licence

GPLv3+, matching upstream BDSIM. Original copyright 2019 Natércia C. P. Fernandes,
natercia@eq.uc.pt.

## Use case in Lepanto

This port is intended as a **TRL 5/6 simulation environment** for testing
fault-detection algorithms before deploying to the Bioadvance biodiesel
plant. See `~/Documents/Notas/Lepanto/Lepanto_Road_Map.md` for context.