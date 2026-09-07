---
tags: [helper, reference]
aliases: [Abbreviations, Abbreviations list]
---

# Acronyms

Short forms used across the vault and codebase. See also [Glossary](./Glossary.md) for longer definitions.

| Acronym | Meaning | See |
|---------|---------|-----|
| **AE** | Algebraic equations (steady relations evaluated with the ODE) | [ODE-and-AE](./20-math/ODE-and-AE.md) |
| **ARMAX** | AutoRegressive–Moving-Average with eXogenous inputs (noise / fouling modes) | [Config-surface](./30-engine/Config-surface.md), [Config-surface](./30-engine/Config-surface.md) |
| **BDSIM** | Biodiesel Dynamic SIMulator (upstream MATLAB/Octave, Fernandes 2019) | [What-is-bdsim](./00-orientation/What-is-bdsim.md) |
| **CW** | Cooling water | [Units-and-streams](./10-plant/Units-and-streams.md), [Config-surface](./30-engine/Config-surface.md) |
| **DCS** | Distributed control system (industrial control analogy for live faults) | [Sensors-and-control](./10-plant/Sensors-and-control.md) |
| **DG** | Diglyceride | [Kinetics](./20-math/Kinetics.md), [Units-and-streams](./10-plant/Units-and-streams.md) |
| **DP** | Differential pressure (filter ΔP sensor, `pv[4]`) | [Channel-indices](./40-helpers/Channel-indices.md), [Sensors-and-control](./10-plant/Sensors-and-control.md) |
| **E** / **FAME** | Ester / Fatty Acid Methyl Ester (biodiesel product) | [Kinetics](./20-math/Kinetics.md), [Glossary](./Glossary.md) |
| **FFA** | Free fatty acid | [Config-surface](./30-engine/Config-surface.md), [ODE-and-AE](./20-math/ODE-and-AE.md) |
| **G** | Glycerol | [Kinetics](./20-math/Kinetics.md), [Decanter-split-NN](./20-math/Decanter-split-NN.md) |
| **HEX** | Heat exchanger | [Process-flow](./10-plant/Process-flow.md), [Config-surface](./30-engine/Config-surface.md) |
| **IR** | Infrared (spectrum bands with NIR) | [Config-surface](./30-engine/Config-surface.md) |
| **IV** | Iodine value (quality lab channel) | [Channel-indices](./40-helpers/Channel-indices.md), [Glossary](./Glossary.md) |
| **JIT** | Just-in-time compilation (see [Glossary](./Glossary.md#numba-kernel)) | [ODE-and-AE](./20-math/ODE-and-AE.md) |
| **M** | Methanol | [Kinetics](./20-math/Kinetics.md), [Units-and-streams](./10-plant/Units-and-streams.md) |
| **MG** | Monoglyceride | [Kinetics](./20-math/Kinetics.md) |
| **MLP** | Multi-layer perceptron (decanter split net) | [Decanter-split-NN](./20-math/Decanter-split-NN.md) |
| **MOC** | Map of content (this vault’s hub notes) | [Home](./Home.md) |
| **MQTT** | Message queue telemetry (dashboard side, not this repo) | [What-is-bdsim](./00-orientation/What-is-bdsim.md) |
| **NIR** | Near-infrared virtual spectrum sensor | [Config-surface](./30-engine/Config-surface.md) |
| **NN** | Neural network | [Decanter-split-NN](./20-math/Decanter-split-NN.md) |
| **ODE** | Ordinary differential equation (plant dynamics RHS) | [ODE-and-AE](./20-math/ODE-and-AE.md) |
| **OPC** | Open Platform Communications–style quality codes on live sensors | [Live-simulator](./30-engine/Live-simulator.md) |
| **PID** | Proportional–integral–derivative controller | [Sensors-and-control](./10-plant/Sensors-and-control.md) |
| **PV** | Process variable / measured value vector (`pv`) | [Channel-indices](./40-helpers/Channel-indices.md) |
| **QA** | Quality assurance lab (FAME%, water, IV) | [Config-surface](./30-engine/Config-surface.md), [Glossary](./Glossary.md) |
| **RHS** | Right-hand side of the ODE | [ODE-and-AE](./20-math/ODE-and-AE.md) |
| **RNG** | Random-number generator (seeded for fingerprints) | [Byte-identical-contract](./00-orientation/Byte-identical-contract.md) |
| **SP** | Setpoint vector (`sp`) | [Channel-indices](./40-helpers/Channel-indices.md), [Sensors-and-control](./10-plant/Sensors-and-control.md) |
| **SV** | State variable vector (`sv`) | [Channel-indices](./40-helpers/Channel-indices.md), [ODE-and-AE](./20-math/ODE-and-AE.md) |
| **TG** | Triglyceride | [Kinetics](./20-math/Kinetics.md) |
| **TRL** | Technology readiness level (this sim is TRL 5/6 plant stand-in) | [What-is-bdsim](./00-orientation/What-is-bdsim.md) |
| **UCO** | Used cooking oil (typical feedstock context) | [Units-and-streams](./10-plant/Units-and-streams.md) |
| **UV** | Input / manipulated variable vector (`uv` / `u`) | [Channel-indices](./40-helpers/Channel-indices.md) |

### Tag / tag-like names in code

| Name | Meaning |
|------|---------|
| **TR-101** | Reactor temperature loop / sensor (`pv[0]`) |
| **TD-201** | Decanter temperature (`pv[1]`) |
| **FICA-401** | Oil mass-flow indication/control (`pv[3]`) |
| **QA-101/102/103** | Latched FAME%, water ppm, IV when `quality_state=True` |

Related: [Glossary](./Glossary.md), [Home](./Home.md), [Channel-indices](./40-helpers/Channel-indices.md)
