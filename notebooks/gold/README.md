# Gold Octave vs Python compare notebooks

Visual parity for gold scenarios **S00–S07**: each notebook shows the original nine BDsim figures with **Octave (left)** and **Python (right)** side by side.

| Notebook | Scenario |
|----------|----------|
| `S00_compare.ipynb` | Healthy baseline |
| `S01_compare.ipynb` | Filter clogging + cleaning |
| `S02_compare.ipynb` | HEX linear fouling |
| `S03_compare.ipynb` | Side reactions / kinetic loss |
| `S04_compare.ipynb` | Sequential sensor fault calendar |
| `S05_compare.ipynb` | Oil-valve stiction + Foil SP steps |
| `S06_compare.ipynb` | Intermittent TD sensor fault |
| `S07_compare.ipynb` | Combined multi-fault |

**Data required** (gitignored):

- Octave ASCII: `data/results_octave/gold_dataset_1w/S##/`
- Python Parquet: `data/results_py/S##/`

Generate Python gold with `./scripts/run_gold_scenarios.sh` or the cluster runner; see [`docs/30-engine/Gold-parity-cluster.md`](../../docs/30-engine/Gold-parity-cluster.md).

**Open** in Cursor / VS Code with the project `.venv` kernel (plotly + pandas already installed). Helper: [`_compare.py`](_compare.py). To regenerate notebook shells after editing briefs: `python notebooks/gold/_gen_notebooks.py`.
