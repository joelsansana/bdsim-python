---
tags: [moc, hub]
aliases: [MOC, Index, Start here]
---

# Home

Junior onboarding hub for the **bdsim** plant simulator. This vault explains *how the plant and engine work*; operator quick-starts live in root [`USER.md`](../USER.md), admin install in [`ADMIN.md`](../ADMIN.md), agent hard rules in [`AGENTS.md`](../AGENTS.md).

## Start here (suggested path)

1. [[What-is-bdsim]] — what this repo is and is not
2. [[Process-flow]] — filter → reactor → HEX → decanter → washer → dryer
3. [[Byte-identical-contract]] — seed + faults → reproducible trajectories (runtime default vs legacy pin profiles)
4. [[Config-surface]] — `Settings`, `ProcessFaults`, faults
5. [[Batch-driver]] and [[Live-simulator]] — how time advances
6. [[Channel-indices]] — `sv` / `pv` / `uv` cheat sheet
7. [[How-to-work-here]] — tests, fingerprints, what not to touch

## Maps of content

### Orientation
- [[What-is-bdsim]]
- [[Repo-map]]
- [[Byte-identical-contract]]
- [[How-to-work-here]]

### Plant
- [[Process-flow]]
- [[Units-and-streams]]
- [[Sensors-and-control]]

### Math
- [[ODE-and-AE]]
- [[Kinetics]]
- [[Thermo]]
- [[Decanter-split-NN]]

### Engine
- [[Config-surface]]
- [[Batch-driver]]
- [[Live-simulator]]
- [[Fingerprints-and-tests]]
- [[Layers-roadmap]]
- [[Fouling-modes]] (Layer 2.8b five-mode stepper walkthrough)

### Helpers
- [[Acronyms]]
- [[Glossary]]
- [[Channel-indices]]
- [[Pseudocode-conventions]]

> [!tip]
> Open this vault in Obsidian for graph view. Wikilinks (`[[Note]]`) are the graph edges.

Related: [[Acronyms]], [[Glossary]]
