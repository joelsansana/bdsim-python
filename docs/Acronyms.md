---
tags: [helper, reference]
aliases: [Abbreviations, Abbreviations list]
---

# Acronyms

Short forms used across the vault and codebase. See also [[Glossary]] for longer definitions.

| Acronym | Meaning | See |
|---------|---------|-----|
| **AE** | Algebraic equations (steady relations evaluated with the ODE) | [[ODE-and-AE]] |
| **ARMAX** | AutoRegressive–Moving-Average with eXogenous inputs (noise / fouling modes) | [[Layers-roadmap]], [[Config-surface]] |
| **BDSIM** | Biodiesel Dynamic SIMulator (upstream MATLAB/Octave, Fernandes 2019) | [[What-is-bdsim]] |
| **CW** | Cooling water | [[Units-and-streams]], [[Layers-roadmap]] |
| **DCS** | Distributed control system (industrial control analogy for live faults) | [[Sensors-and-control]] |
| **DG** | Diglyceride | [[Kinetics]], [[Units-and-streams]] |
| **DP** | Differential pressure (filter ΔP sensor, `pv[4]`) | [[Channel-indices]], [[Sensors-and-control]] |
| **E** / **FAME** | Ester / Fatty Acid Methyl Ester (biodiesel product) | [[Kinetics]], [[Glossary]] |
| **FFA** | Free fatty acid | [[Config-surface]], [[ODE-and-AE]] |
| **G** | Glycerol | [[Kinetics]], [[Decanter-split-NN]] |
| **HEX** | Heat exchanger | [[Process-flow]], [[Layers-roadmap]] |
| **IR** | Infrared (spectrum bands with NIR) | [[Layers-roadmap]] |
| **IV** | Iodine value (quality lab channel) | [[Channel-indices]], [[Glossary]] |
| **JIT** | Just-in-time compilation (see [[Glossary#Numba kernel]]) | [[ODE-and-AE]] |
| **M** | Methanol | [[Kinetics]], [[Units-and-streams]] |
| **MG** | Monoglyceride | [[Kinetics]] |
| **MLP** | Multi-layer perceptron (decanter split net) | [[Decanter-split-NN]] |
| **MOC** | Map of content (this vault’s hub notes) | [[Home]] |
| **MQTT** | Message queue telemetry (dashboard side, not this repo) | [[What-is-bdsim]] |
| **NIR** | Near-infrared virtual spectrum sensor | [[Layers-roadmap]] |
| **NN** | Neural network | [[Decanter-split-NN]] |
| **ODE** | Ordinary differential equation (plant dynamics RHS) | [[ODE-and-AE]] |
| **OPC** | Open Platform Communications–style quality codes on live sensors | [[Live-simulator]] |
| **PID** | Proportional–integral–derivative controller | [[Sensors-and-control]] |
| **PV** | Process variable / measured value vector (`pv`) | [[Channel-indices]] |
| **QA** | Quality assurance lab (FAME%, water, IV) | [[Layers-roadmap]], [[Glossary]] |
| **RHS** | Right-hand side of the ODE | [[ODE-and-AE]] |
| **RNG** | Random-number generator (seeded for fingerprints) | [[Byte-identical-contract]] |
| **SP** | Setpoint vector (`sp`) | [[Channel-indices]], [[Sensors-and-control]] |
| **SV** | State variable vector (`sv`) | [[Channel-indices]], [[ODE-and-AE]] |
| **TG** | Triglyceride | [[Kinetics]] |
| **TRL** | Technology readiness level (this sim is TRL 5/6 plant stand-in) | [[What-is-bdsim]] |
| **UCO** | Used cooking oil (typical feedstock context) | [[Units-and-streams]] |
| **UV** | Input / manipulated variable vector (`uv` / `u`) | [[Channel-indices]] |

### Tag / tag-like names in code

| Name | Meaning |
|------|---------|
| **TR-101** | Reactor temperature loop / sensor (`pv[0]`) |
| **TD-201** | Decanter temperature (`pv[1]`) |
| **FICA-401** | Oil mass-flow indication/control (`pv[3]`) |
| **QA-101/102/103** | Latched FAME%, water ppm, IV when `quality_state=True` |

Related: [[Glossary]], [[Home]], [[Channel-indices]]
