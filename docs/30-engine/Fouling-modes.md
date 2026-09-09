---
tags: [engine, fouling, fouling-modes]
aliases: [Fouling modes, Five-mode stepper]
---

# Fouling modes (fouling-mode windows)

The five-mode fouling stepper is the Python port of Fernandes 2019 /
Strelet Dec 2019 `fouling.m` from the BDSIM_spectr reference
distribution. It exposes a `FoulingModeStepper` (in
`bdsim/fouling_modes.py`) that maintains the ARMAX state explicitly
(`RfOld`, `epsilon_OLD`) — the upstream uses `persistent` MATLAB
variables, which is unsafe for multi-instance or restart scenarios. Our
port makes this state explicit and bounded.

## What it produces

A scalar `factor ∈ (0, 1]` that multiplies `Qheat` in the reactor
energy balance:

```
Theat = TR - factor * Qheat / (NR * cpmolR)
```

`factor = 1` means no fouling; smaller values mean the heat exchanger
is less effective at delivering heat to the reactor.

## Three-path cooperation

Three paths cooperate to produce the factor:

1. **dynamic fouling (continuous α)** — when `pfaults.fouling_dynamic=True`
   the state vector carries `sv[21]` (α ∈ [0, 1]) and `factor` follows
   the dynamics. This is the "physics-based slow fouling" story.
2. **fouling-mode windows (windowed modes 4/5, this module)** — during an active
   fault event the stepper overrides `factor` with the ARMAX-mode
   output. This is the "fast, stochastic, fault-injection" story:
   intermittent feedstock-impurity spikes that look like ARMAX noise
   to a downstream correlation engine.
3. **dynamic fouling fallback (static)** — when neither above applies,
   `factor = 1 / (1 + foulingpar * t)` (the legacy pre-baked series)
   is used.

Priority when all three are configured:
**continuous α > windowed mode 4/5 > static**.

## Modes 0–5

| `FoulingMode` | Integer | Behaviour | Notes |
|---------------|---------|-----------|-------|
| `OFF` | 0 | `factor = 1` | trivial |
| `LINEAR` | 1 | `factor = 1 - k * t` (saturates at 0) | sanity-check |
| `EXPONENTIAL` | 2 | exponential recovery | flagged "weird" in upstream; kept for parity |
| `CHAIBAKHSH` | 3 | `k * t * xRG` (linear-in-t, glycerol-coupled) | sanity-check |
| `ARMAX_NOISE` | 4 | ARMAX(1,1) + ε noise, mean-reverting | **dashboard-fault-injection path** |
| `ARMAX_PURE_NOISE` | 5 | pure ARMAX(1,1) noise (no mean reversion) | **dashboard-fault-injection path** |

Modes 0–3 are ported for completeness. The bdsim-dashboard scenario
catalog only exercises modes 4 and 5; modes 1 and 3 are wired into the
`FoulingModeStepper` because they're well-defined and serve as
sanity-check references, mode 0 is the trivial off state, and mode 2
(exponential recovery) is documented as "weird" in the upstream and
kept for parity but flagged as such.

## Determinism

Modes 4 and 5 are stochastic. Callers must pass a
`numpy.random.Generator` to `FoulingModeStepper.step()` so test
fixtures can pin seeds and scenarios can replay runs:

```python
from bdsim import LiveSimulator
from bdsim.config import ProcessFaults, Settings
import numpy as np

pf = ProcessFaults(fouling_dynamic=True)
sim = LiveSimulator(settings=Settings(), pfaults=pf, seed=42)
rng = np.random.default_rng(seed=123)
sim.activate_fouling_mode_window(mode=4, duration_s=600, rng=rng)
while not sim.done:
    s = sim.step()
```

The `LiveSimulator.activate_fouling_mode_window(mode, duration_s, rng)`
API is the dashboard's entry point. While the window is active
(`sim.t < pfaults.fouling_mode_active_end_t`), the kernel applies the
ARMAX output on top of (or instead of, depending on flags) the
continuous-α path.

## ProcessFaults knobs

| Field | Default | Purpose |
|-------|---------|---------|
| `fouling_mode` | `0` | global mode selector (matches upstream) |
| `fouling_mode_xRG_weight` | `True` | mode 4 couples to glycerol mole fraction |
| `fouling_ar_eps_std` | `5e-4` | ARMAX innovation σ (modes 4/5) |
| `fouling_mode_default_window_s` | `3600.0` | default fault-window length |
| `fouling_mode_active_mode` | `0` | runtime overlay: 0 = off, 4 / 5 = ARMAX |
| `fouling_mode_active_end_t` | `-1.0` | sim time at which the active window expires |
| `fouling_mode_active_seed` | `None` | optional seed for ARMAX RNG (reproducibility) |

See [Config-surface](../30-engine/Config-surface.md) for the full table.

## Tests

- `tests/test_fouling_modes.py` — pure-Python stepper tests
  (modes 0–5, snapshot/restore, ARMAX determinism).
- `tests/test_fouling_modes_live.py` — LiveSimulator wiring
  (windowed-mode activation, priority over continuous α, end-of-window
  behaviour, byte-identical baseline when no window is active).

## References

- Upstream `fouling.m` — Fernandes 2019, Strelet Dec 2019
  (BDSIM_spectr reference distribution).
- [Config-surface](../30-engine/Config-surface.md) for fouling-mode windows context.
- [Config-surface](../30-engine/Config-surface.md) for the full `ProcessFaults` table.

Related: [Config-surface](../30-engine/Config-surface.md), [Config-surface](../30-engine/Config-surface.md), [Fingerprints-and-tests](../30-engine/Fingerprints-and-tests.md), [Home](../Home.md)
