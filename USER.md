# USER.md — bdsim

You, the developer or researcher, want to use bdsim. This is the on-ramp.

For a junior-oriented deep dive (plant flow, equations, channel maps, Obsidian wikilinks), start at [`docs/Home.md`](docs/Home.md).

## Who you probably are

- A **data scientist** building fault-detection models on plant trajectories. You want clean, reproducible batch runs with `np.random.seed(seed)` and pinned fingerprints.
- A **control engineer** tuning the PID loops. You want to sweep `Settings.live_sp1..4` and see how `TR-101` / `FICA-401` respond.
- A **dashboard / app developer** driving a real-time UI. You want `LiveSimulator` and `StepResult` to deliver one sample at a time without surprises.
- A **demo / training person** writing a scenario script. You want to inject a fault at sim_t=600s and watch the response.

## Quick start (5 minutes)

```python
# Batch run, default upstream settings, reproducible
from bdsim import run_with
res = run_with(seed=42)
print(res.t.shape, res.sv.shape, res.pv.shape)   # (51999,) (51999, 22) (51999, 5) — runtime default includes Layer 2.5 α
print(res.t[0], res.t[-1])                       # 0.0, 259980.0 (72h)
```

```python
# Live step-by-step
from bdsim import LiveSimulator
from bdsim.config import Settings, ProcessFaults
sim = LiveSimulator(settings=Settings(), seed=42, pfaults=ProcessFaults())
while not sim.done:
    sample = sim.step()
    print(sample.t, sample.pv)
```

```python
# Inject a fault mid-run
from bdsim import LiveSimulator
from bdsim.config import Settings, ProcessFaults, SensorFaults
sim = LiveSimulator(settings=Settings(), seed=42, pfaults=ProcessFaults())
# Pre-bias TR-101 by +5 K for the rest of the run
sim.sensor_faults.bias[0] = 5.0
while not sim.done:
    sim.step()
```

## Common tasks

### Run a reproducible batch experiment

```python
import numpy as np
from bdsim import run_with
from bdsim.config import ProcessFaults

pf = ProcessFaults(
    fouling_dynamic=True,        # Layer 2.5: HEX fouling as ODE state
    quality_state=True,           # Layer 2.1: latched QA measurements
    cw_p_drift_pa_per_h=-50.0,    # Layer 2.6: slow CW pressure drift
)
res = run_with(pfaults=pf, seed=42, verbose=False)
np.savez("experiment_01.npz", t=res.t, sv=res.sv, pv=res.pv, uv=res.uv, disturbances=res.disturbances)
```

`res.disturbances` is `(51999, 3)` columns `[Tamb_K, Tcw_K, Pcw_Pa]` (or `None` if no disturbance amplitudes set).

### Run a quality-latched experiment (Layer 2.1)

```python
from bdsim import LiveSimulator
from bdsim.config import ProcessFaults

# quality_state=True: the QA-101/102/103 measurements are latched
# at the lab-cycle rate (1 hour default), with realistic noise. The
# state vector widens to 25 components (was 21/22).
sim = LiveSimulator(
    pfaults=ProcessFaults(quality_state=True, fouling_dynamic=True),
    seed=42,
)
while not sim.done:
    s = sim.step()
    if s.quality_latched is not None:
        print(s.t, s.quality_latched)         # [FAME%, water_ppm, IV]
```

### Trigger a cw_pump_trip mid-run (Layer 2.6b)

```python
from bdsim import LiveSimulator
from bdsim.config import ProcessFaults

# Layer 2.6 ambient/cw knobs must be set for the kernel to apply
# the override. If amplitudes are zero the published PCW snapshot
# still drops (operator visibility) but the ODE perturbation path
# is skipped.
sim = LiveSimulator(
    pfaults=ProcessFaults(
        fouling_dynamic=True,
        quality_state=False,
        ambient_t_amplitude_k=2.0,        # nonzero so the kernel runs
        cw_t_amplitude_k=1.0,
    ),
    seed=42,
)
# Fire the trip at sim_t=600s, hold 300s, recover
sim._disturbance_override = {
    "channel": "pcw",
    "start_t": 600.0,
    "end_t": 900.0,
    "low_factor": 0.3,                   # 70% pressure drop
    "ramp_s": 30.0,
}
while not sim.done:
    s = sim.step()
    print(s.t, s.disturbances[2] / 1e5)  # PCW in bar
```

### Trigger a fouling-mode window (Layer 2.8b)

```python
from bdsim import LiveSimulator
from bdsim.config import ProcessFaults

# Layer 2.8b: windowed five-mode fouling. Modes 4 and 5 are
# stochastic ARMAX noise injections on the heat-exchanger
# efficiency factor. Activate a window mid-run via the live
# mutator API; the kernel applies the ARMAX stepper during the
# window and falls back to Layer 2.5 continuous α (or the static
# legacy series) afterwards.
sim = LiveSimulator(
    pfaults=ProcessFaults(
        fouling_dynamic=True,           # Layer 2.5 α path stays active
        fouling_ar_eps_std=5e-4,        # ARMAX innovation σ
    ),
    seed=42,
)
# Fire mode-4 (mean-reverting ARMAX) for 600 s starting now
info = sim.activate_fouling_mode_window(mode=4, duration_s=600.0, seed=11)
print(f"window: start={info['start_t']}, end={info['end_t']}")
while not sim.done:
    s = sim.step()
    # Results.factor (returned by run_to_completion) records the
    # actual factor the kernel applied each step. During the window
    # you'll see ARMAX noise; after the window it returns to α.
    print(s.t, sim.get_fouling_mode_state())
```

To turn off the window early: ``sim.clear_fouling_mode_window()``.
The snapshot/restore API is also exposed via
``sim.get_fouling_mode_state()`` for replay scenarios.

### Read the live disturbance track

```python
sim = LiveSimulator(
    pfaults=ProcessFaults(
        ambient_t_amplitude_k=8.0, cw_t_amplitude_k=4.0,
    ),
    seed=42,
)
sim.step()                                # initial sample
print(sim._disturbance_track[:5])        # (5, 3) [Tamb_K, Tcw_K, Pcw_Pa]
```

### Capture a NIR/IR spectrum sample (Layer 2.8a)

```python
from bdsim import LiveSimulator
from bdsim.config import ProcessFaults

sim = LiveSimulator(
    pfaults=ProcessFaults(
        spectrum_enabled=True,
        spctr_t=900.0,                  # fire every 15 minutes
    ),
    seed=42,
)
while not sim.done:
    s = sim.step()
    if s.spectra is not None:
        print(s.spectra.t, s.spectra.sim_t)
        print('  qc_MG :', s.spectra.qc_reactor_mg)
        print('  qc_TG :', s.spectra.qc_reactor_tg)
        print('  reactor absorbance:', s.spectra.reactor.shape)  # (631,)
```

The spectrum sensor is **post-process** — it reads the state vector
after each ODE step but never writes back. The legacy fingerprint
(`sv=c8807b23...`) is preserved with `spectrum_enabled=True`.

```

## API reference (essentials)

### `run_with(settings, pfaults, sfaults, vfaults, armax, pid, seed, verbose) -> Results`

Batch driver. Runs the full 72h sim in ~60s (Numba JIT). Returns `Results` with `t, sv, pv, uv, sp, quality, disturbances, ...`.

### `LiveSimulator(settings, pfaults, seed, ...) -> LiveSimulator`

Stateful, per-step driver. The dashboard uses this. Has the same `ProcessFaults` knobs but additionally supports mid-run mutation of `sensor_faults`, `valve_faults`, and (Layer 2.6b) `sim._disturbance_override`.

### `StepResult` (live only)

```python
@dataclass
class StepResult:
    t: float                        # sim time, seconds
    i: int                          # step index
    sv: np.ndarray                  # state vector
    pv: np.ndarray                  # sensor values (with bias/dropout/stuck applied)
    uv: np.ndarray                  # inputs
    sp: np.ndarray                  # setpoints
    quality: dict[int, str]         # sensor index → "good"/"uncertain"/"bad"
    quality_latched: np.ndarray | None  # Layer 2.1 latched QA values
    disturbances: np.ndarray | None     # Layer 2.6 [Tamb_K, Tcw_K, Pcw_Pa]
    spectra: SpectrumSample | None       # Layer 2.8 NIR/IR sample at fire times
    xLend: np.ndarray
    yLend: np.ndarray
```

### `ProcessFaults` (the main config block)

```python
@dataclass
class ProcessFaults:
    fouling: int = 1                  # 0/1 — pre-Layer 2.5 fouling (constant series when dynamic off)
    fouling_dynamic: bool = True      # Layer 2.5 — α evolves as an ODE state (default ON)
    quality_state: bool = False       # Layer 2.1 — latched QA measurements
    quality_lag_mode: str = "lab"     # "lab" / "online"
    ambient_t_amplitude_k: float = 0  # Layer 2.6 ambient sinusoid
    cw_t_amplitude_k: float = 0       # Layer 2.6 CW temperature sinusoid
    cw_p_drift_pa_per_h: float = 0    # Layer 2.6 CW pressure slow drift
    cw_p_noise_pa: float = 0          # Layer 2.6 CW pressure noise (1σ)
    met_cw_track: float = 0.3
    oil_ambient_track: float = 0.7
    qheat_cw_scaling: bool = True
    cw_pump_low_factor: float = 0.3   # Layer 2.6b trip pressure floor
    cw_pump_ramp_s: float = 30.0      # Layer 2.6b trip ramp duration
    cw_pump_default_duration_s: float = 600.0
    # ----- Layer 2.4: pump/valve degradation as continuous state -----
    pump_wear: bool = False                       # Layer 2.4: pump_health sv slot
    valve_wear: bool = False                      # Layer 2.4: valve_stiction_pct sv slot
    pump_health_initial: float = 1.0              # 1.0 = brand-new, floor 0.05 = end-state
    pump_wear_rate_per_h: float = 0.01            # dh/dt baseline
    pump_wear_flow_exponent: float = 1.5          # dh/dt ∝ (Q/Qnom)^p
    pump_wear_floor: float = 0.05                 # lower bound, post-integration clamp
    pump_health_trip_threshold: float = 0.25      # scenario-side trip becomes likely below this
    valve_stiction_initial_pct: float = 0.0
    valve_stiction_rate_pct_per_h: float = 0.05    # grows proportional to |dlift/dt|
    valve_stiction_floor_pct: float = 0.0
    valve_stiction_ceiling_pct: float = 60.0
```

> Tip: named profiles (runtime default vs legacy fingerprint) are in [`docs/00-orientation/Byte-identical-contract.md`](docs/00-orientation/Byte-identical-contract.md).

## Common gotchas

- **Runtime default includes Layer 2.5.** `ProcessFaults()` has `fouling_dynamic=True` → `sv.shape[1] == 22` (`α` at `sv[21]`). The legacy 21-wide vector needs `fouling_dynamic=False`. Layer 2.1 quality mode and Layer 2.4 wear flags widen further. If you trained a model on a 21-wide vector, re-train or pin the legacy profile explicitly.
- **Layer 2.4 `pump_health` multiplies the published PCW track.** When `pump_wear=True`, the `disturbances[i, 2]` channel reads 0.7× baseline when `pump_health=0.7`. The kernel applies the same factor on `u[4]` (Qheat) so the reactor temperature responds. Two independent multiplicative effects can stack: the Layer 2.6b `cw_pump_trip` override and the wear multiplier both act on the PCW channel.
- **Layer 2.4 valve stiction only grows when valves move.** Idle valves (`dlift = 0`) accumulate zero stiction per second. To see stiction grow in a demo, drive the PID loop with a Qheat dip or feedstock change — the control valves chasing the new setpoint is what builds stiction.
- **`res.disturbances` is `None` unless you set disturbance amplitudes.** The kernel skips the path entirely when all amplitudes are zero (legacy byte-identical contract). Set at least one to nonzero.
- **Live path and batch path have different fingerprints** even at the same seed. The legacy batch pin `sv=c8807b23...` requires `fouling_dynamic=False`. The live legacy baseline is `sv=23c3c885...`. Both are pinned.
- **`LiveSimulator.t` raises IndexError after the run completes** if you haven't installed the post-run fix (`v0.4.1+`). It returns `settings.tf` instead. The dashboard depends on this — make sure your install is current.
- **Numba caches are in `__pycache__/`** and `bdsim/*.nbi`. After major kernel changes, delete the cache: `find . -name "*.nbi" -delete && find . -name "__pycache__" -exec rm -rf {} +`.

## Where to look next

- `tests/test_smoke.py` — minimal usage examples
- `bdsim/live_simulator.py` — the live driver with extensive docstrings
- `bdsim/config.py` — every config dataclass, heavily commented
- `NOTES.md` — historical: upstream-faithful bugs we found and fixed
- `AGENTS.md` — build conventions and roadmap status
- Companion repo [`bdsim-dashboard`](https://github.com/joelsansana/bdsim-dashboard) for the live operator console
