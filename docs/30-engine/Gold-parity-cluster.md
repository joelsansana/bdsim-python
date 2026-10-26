---
tags: [validation, cluster, gold]
aliases: [gold cluster, Parquet parity]
---

# Gold-parity-cluster

How to generate Python gold trajectories (S00–S07) on **anvil + forge** and compare them to Octave ASCII under `data/gold_dataset_1w/`.

SSH / topology background: BDSIM [`docs/cluster.md`](file:///home/eugeniu/repos/BDSIM/docs/cluster.md) (read-only). This repo owns the Python runners.

## Prerequisites

- Same absolute path to **bdsim-python** on anvil and forge
- `uv sync` (pandas/pyarrow included) on both nodes — creates `.venv/` used by the runners
- GNU **parallel**, **rsync**, SSH key login to forge
- PyYAML for `config/load_cluster_env.py` (`apt install python3-yaml`)
- Gold ASCII on the compare host: `data/gold_dataset_1w/S00`…`S07` (gitignored)

Cluster jobs prefer `${ROOT}/.venv/bin/python` (not `uv` on PATH). Non-interactive SSH often lacks `uv`; without a synced `.venv` on forge you get `ModuleNotFoundError: scipy`.

## Generate Parquet (cluster)

```bash
./scripts/run_gold_scenarios_cluster.sh SMOKE    # npoints=100 wiring check
./scripts/run_gold_scenarios_cluster.sh          # full week → results_py/S##/
./scripts/run_gold_scenarios_cluster.sh S00 S01
```

Local-only:

```bash
./scripts/run_gold_scenarios.sh SMOKE
JOBS=1 ./scripts/run_gold_scenarios.sh S00
# or:
uv run python -m bdsim.scenarios S00 --outdir results_py
```

Logs: `logs/gold_<timestamp>/`. Remotes sync **logs** and **`results_py/`** back to the submit host.

## Compare

```bash
uv run pytest tests/test_gold_parity.py -q          # SMOKE + loader
uv run pytest tests/test_gold_parity.py -m slow -q  # full-week vs gold
```

Slow tests prefer pre-built `results_py/S##`; without them only S00 auto-builds in-process.

## Design notes

- Solver remains **RK45** (not LSODA); RMSE bands allow Python↔Octave drift.
- S06 intermittence: NumPy RNG ≠ Octave — phenomenology / structure only.
- Export format: **Parquet only** (handbook §6 column names). See [[Python-vs-MATLAB-validation]].
