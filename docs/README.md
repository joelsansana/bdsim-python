# docs/

Knowledge vault for the **bdsim** plant simulator.

The vault is plain Markdown — it renders cleanly on GitHub. It was originally
authored as an Obsidian vault (wikilinks have been converted to standard
Markdown links as of the 1.2.x docs cleanup); if you open the folder in
Obsidian, the file structure and naming still round-trip without surprises.

## Start here

- **[Home.md](Home.md)** — the vault entry point with a suggested reading path.
- **[../README.md](../README.md)** — project top-level README (install, run, programmatic use).
- **[../AGENTS.md](../AGENTS.md)** — agent hard rules and dev workflow.

## Layout

| Subdirectory | What lives here |
|--------------|------------------|
| `00-orientation/` | What this repo is, repo map, the byte-identical fingerprint contract |
| `10-plant/` | Process flow, units, sensor & control loops |
| `20-math/` | ODE / AE model, kinetics, thermo, decanter split neural network |
| `30-engine/` | Config surface, batch / live drivers, fouling modes, fingerprints |
| `40-helpers/` | Channel indices, pseudocode conventions |

Root vault files: [Acronyms.md](Acronyms.md), [Glossary.md](Glossary.md), [Home.md](Home.md).

## Upstream PDFs (optional)

Reference PDFs such as `Fernandes2019_BDSIM.pdf` and `manual.pdf` are
**gitignored** if placed under `docs/` (see [`../.gitignore`](../.gitignore)).
Drop local copies into `docs/` if you have them; they are **not** required to
install or run bdsim.
