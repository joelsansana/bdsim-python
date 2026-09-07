---
tags: [moc, hub]
aliases: [MOC, Index, Start here]
---

# Home

Junior onboarding hub for the **bdsim** plant simulator. This vault explains *how the plant and engine work*; operator quick-starts live in root [`USER.md`](../USER.md), admin install in [`ADMIN.md`](../ADMIN.md), agent hard rules in [`AGENTS.md`](../AGENTS.md).

## Start here (suggested path)

1. [What-is-bdsim](./00-orientation/What-is-bdsim.md) — what this repo is and is not
2. [Process-flow](./10-plant/Process-flow.md) — filter → reactor → HEX → decanter → washer → dryer
3. [Byte-identical-contract](./00-orientation/Byte-identical-contract.md) — seed + faults → reproducible trajectories (runtime default vs legacy pin profiles)
4. [Config-surface](./30-engine/Config-surface.md) — `Settings`, `ProcessFaults`, faults
5. [Batch-driver](./30-engine/Batch-driver.md) and [Live-simulator](./30-engine/Live-simulator.md) — how time advances
6. [Channel-indices](./40-helpers/Channel-indices.md) — `sv` / `pv` / `uv` cheat sheet
7. [Practical dev workflow](./../AGENTS.md) — tests, fingerprints, what not to touch

## Maps of content

### Orientation
- [What-is-bdsim](./00-orientation/What-is-bdsim.md)
- [Repo-map](./00-orientation/Repo-map.md)
- [Byte-identical-contract](./00-orientation/Byte-identical-contract.md)
- [Practical dev workflow](./../AGENTS.md)

### Plant
- [Process-flow](./10-plant/Process-flow.md)
- [Units-and-streams](./10-plant/Units-and-streams.md)
- [Sensors-and-control](./10-plant/Sensors-and-control.md)

### Math
- [ODE-and-AE](./20-math/ODE-and-AE.md)
- [Kinetics](./20-math/Kinetics.md)
- [Thermo](./20-math/Thermo.md)
- [Decanter-split-NN](./20-math/Decanter-split-NN.md)

### Engine
- [Config-surface](./30-engine/Config-surface.md)
- [Batch-driver](./30-engine/Batch-driver.md)
- [Live-simulator](./30-engine/Live-simulator.md)
- [Fingerprints-and-tests](./30-engine/Fingerprints-and-tests.md)
- [Fouling-modes](./30-engine/Fouling-modes.md) (fouling-mode windows five-mode stepper walkthrough)

### Helpers
- [Acronyms](./Acronyms.md)
- [Glossary](./Glossary.md)
- [Channel-indices](./40-helpers/Channel-indices.md)
- [Pseudocode-conventions](./40-helpers/Pseudocode-conventions.md)

> [!tip]
> The vault is plain Markdown and renders cleanly on GitHub. The original Obsidian wikilinks (`[[Note]]`) have been converted to standard Markdown links; if you prefer to author in Obsidian, the file structure and naming will round-trip without surprises.

Related: [Acronyms](./Acronyms.md), [Glossary](./Glossary.md)
