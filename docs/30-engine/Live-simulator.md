---
tags: [engine]
aliases: [LiveSimulator, StepResult, Live API]
---

# Live-simulator

Module: `bdsim/live_simulator.py`. Class: `LiveSimulator`. Per-step payload: `StepResult`.

## Role

Stateful driver for **bdsim-dashboard**: one `step()` advances one `dt`, returns sensors/state for MQTT/HTTP. Same plant math as [Batch-driver](../30-engine/Batch-driver.md) when knobs match; live path has its own fingerprint pins.

## Lifecycle

```mermaid
sequenceDiagram
  participant D as Dashboard
  participant L as LiveSimulator
  participant ODE as ODEmodel
  D->>L: LiveSimulator(settings, seed, pfaults)
  loop until done
    D->>L: optional mutate faults / knobs
    D->>L: step()
    L->>ODE: integrate one dt
    L-->>D: StepResult
  end
```

## Pseudocode

```text
sim = LiveSimulator(settings, seed, pfaults, …)
while not sim.done:
    # mid-run: sim.sensor_faults.bias[i] = …
    #          sim.set_*_knob(...); fouling windows; CW trip helpers
    s = sim.step()   # StepResult: t, pv, uv, sv, sp, quality, …
```

## Live-only capabilities

- Mutate `sensor_faults` between steps (bias / stuck / dropouts)
- cooling-water pump trip CW pump trip / disturbance overlays
- Operator disturbance knobs (`live_*` fields on `ProcessFaults`)
- NIR/IR spectrum sensor spectra attached on fire times (`StepResult.spectra`)
- `activate_fouling_mode_window` for fouling-mode windows

> [!tip]
> Prefer first-class methods over private `_` attributes when extending the live API. See [`USER.md`](../../USER.md) for current patterns.

Related: [Batch-driver](../30-engine/Batch-driver.md), [Config-surface](../30-engine/Config-surface.md), [Sensors-and-control](../10-plant/Sensors-and-control.md), [Config-surface](../30-engine/Config-surface.md)
