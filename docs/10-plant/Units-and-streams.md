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

Heavy-phase composition in the state vector stores only **M, E, G** (3 components) — see [[Channel-indices]].

## Main streams

- **Oil feed** — through filter; temperature `Toil` (`u[3]`), flow from filter + valve
- **Methanol feed** — `Tmet`, `Fmet` (`u[1]`, `u[2]`)
- **Cooling water / HEX** — duty `Qheat` (`u[4]`); Layer 2.6 tracks ambient / CW T / CW pressure
- **Light vs heavy** — decanter split fractions η from [[Decanter-split-NN]]

## Densities / cp

Pure-component and mixture properties live in `Parameters` and [[Thermo]] (`Mmx`, `cpmx`, `Vmolar`). Prefer SI (K, Pa, mol, m³, s); display helpers convert °C / kg/h for plots.

Related: [[Process-flow]], [[Kinetics]], [[Thermo]], [[Acronyms]]
