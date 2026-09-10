---
tags: [orientation]
aliases: [bdsim overview, What is this repo]
---

# What-is-bdsim

**bdsim** is a Python port of Natércia C. P. Fernandes’ **BDSIM** (University of Coimbra, 2019): a closed-loop **biodiesel plant** simulator.

It models: **filter → reactor → heat exchanger → decanter → washer → dryer**, with sensors, [PID](../Acronyms.md#pid) loops, valve [stiction](../Glossary.md#valve-stiction), and a decanter [split neural network](../20-math/Decanter-split-NN.md).

## Plant flow

```mermaid
flowchart LR
    subgraph feed [Feed]
        O[Oil] --> R
        M[Methanol] --> R
    end
    F[Filter] --> R
    R[Reactor<br/>6-species<br/>kinetics] -->|TR| HX
    HX[Heat Exchanger<br/>+ fouling factor] -->|TR / TD| D
    D[Decanter<br/>+ split NN] --> L[Light phase<br/>→ washer → dryer]
    D --> HV[Heavy phase<br/>→ recycle]
    R -.PV TR.-> PID1[PID #1]
    R -.PV hH.-> PID3[PID #3]
    D -.PV TD.-> PID2[PID #2]
    R -.PV Foil.-> PID4[PID #4]
    PID1 -->|vinputo| R
    PID2 -->|Tmet| HX
    PID3 -->|vinputH| D
    PID4 -->|Qheat| HX
```

## Code flow

```mermaid
flowchart TD
    User[Caller] -->|run or run_with| Sim[bdsim.simulation]
    User -->|step| Live[bdsim.live_simulator]
    Sim --> Step[per-step helpers<br/>bdsim._step_helpers]
    Live --> Step
    Step --> ODE[Numba @njit RHS<br/>bdsim.ode]
    ODE --> Thermo[bdsim.thermo]
    ODE --> Kinetics[bdsim.kinetics]
    ODE --> Split[bdsim.split_nn<br/>pure-NumPy split]
    Sim --> Fouling[bdsim.fouling_modes<br/>5-mode stepper]
    Live --> Fouling
    Sim --> Spect[comp_spectrum<br/>bdsim.spectra]
    Live --> Spect
```

## What this repo is

- The **simulation engine** (math / plant dynamics)
- A **multi-source data generator**: process vectors, setpoints, optional quality latches, disturbances, NIR/IR spectra, wear/fouling

## What this repo is not

- Not the operator UI, MQTT bridge, or scenario catalog → those live in **bdsim-dashboard**
- Not where fusion / ML training pipelines are implemented → *simulate sources here; fuse elsewhere*
- Not a place to “improve” plant physics without an agreed change and [fingerprint](../Glossary.md#fingerprint) update

## Who uses it

1. **bdsim-dashboard** — `LiveSimulator` step-by-step
2. **Fault-detection / ML research** — batch `run` / `run_with` with fixed seeds
3. **Data-fusion consumers** — time-aligned streams from one plant run

Shared contract: same seed + same `ProcessFaults` → [byte-identical trajectory](../00-orientation/Byte-identical-contract.md).

> [!note]
> Root [`AGENTS.md`](../AGENTS.md) is the hard rule sheet for agents. This vault is the junior-friendly deep dive.

Related: [Home](../Home.md), [Repo-map](../00-orientation/Repo-map.md), [Process-flow](../10-plant/Process-flow.md), [Glossary](../Glossary.md)
