---
tags: [plant]
aliases: [Species, Streams]
---

# Units-and-streams

## Six species (reactor / phases)

Indexed consistently in kinetics and compositions:

| Index | Symbol | Name |
|-------|--------|------|
| 0 | TG | Triglyceride |
| 1 | DG | Diglyceride |
| 2 | MG | Monoglyceride |
| 3 | M | Methanol |
| 4 | E | Ester (FAME) |
| 5 | G | Glycerol |

Heavy-phase composition in the state vector stores only **M, E, G** (3 components) — see [Channel-indices](../40-helpers/Channel-indices.md).

## Main streams

- **Oil feed** — through filter; temperature `Toil` (`u[3]`), flow from filter + valve
- **Methanol feed** — `Tmet`, `Fmet` (`u[1]`, `u[2]`)
- **Cooling water / HEX** — duty `Qheat` (`u[4]`); external disturbances tracks ambient / CW T / CW pressure
- **Light vs heavy** — decanter split fractions η from [Decanter-split-NN](../20-math/Decanter-split-NN.md)

## Densities / cp

Pure-component and mixture properties live in `Parameters` and [Thermo](../20-math/Thermo.md) (`Mmx`, `cpmx`, `Vmolar`). Prefer SI (K, Pa, mol, m³, s); display helpers convert °C / kg/h for plots.

Related: [Process-flow](../10-plant/Process-flow.md), [Kinetics](../20-math/Kinetics.md), [Thermo](../20-math/Thermo.md), [Acronyms](../Acronyms.md)
