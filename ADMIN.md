# ADMIN.md — bdsim

System administration guide. Covers install, packaging, environment setup, dependency management, and CI-style health checks.

## Who you are

The person who keeps bdsim installable, runnable, and reproducible on this box (or any box). You care about: Python version, Numba cache, editable install, system vs. user site-packages, fingerprint regression, and runtime dependencies.

## Fingerprint-aligned install (required for pin tests)

Use this path whenever you run the fingerprint suite or need trajectories that match the pinned hashes in `tests/`.

**Reference stack for bdsim 1.1.1 pins:** Python **3.10** + committed [`uv.lock`](uv.lock) → `numpy==2.2.6`, `scipy==1.15.3`, `numba==0.66.0` (macOS arm64 reference host). Source of truth is the lockfile, not loose `pyproject.toml` ranges.

From this repo's root:

```bash
# 1. Python 3.10 (install if needed)
uv python install 3.10

# 2. Sync editable package + test extras FROM the committed lockfile
#    Do not delete or casually regenerate uv.lock — that can retarget pins.
uv sync --extra test

# 3. Verify stack + editable install path
uv run python -c "import sys,numpy,scipy,numba,bdsim; print(sys.version); print(numpy.__version__, scipy.__version__, numba.__version__); print(bdsim.__file__)"
```

Expect versions `2.2.6 1.15.3 0.66.0` on Python 3.10, and a path ending in this repo’s `bdsim/__init__.py`. Start bdsim-dashboard from the **same** `uv` environment.

Then run health checks with `uv run …` only (see below). First Numba compile of hot kernels can take ~30–60s; later runs use the `.nbi` cache.

> **Caveat:** Other OS/CPU/BLAS builds, or Python ≥3.11 (the lockfile selects different numpy/scipy wheels via markers), **may** still diverge from pins even after a correct `uv sync`. If fingerprint tests fail, paste the verify-stack output above before changing pins.

## Run-only install (fingerprints not guaranteed)

OK to **run** `python -m bdsim` or import the library. Bare `pip install` **ignores** `uv.lock` and resolves within `numpy>=1.24` / `scipy>=1.11` / … — fingerprint tests often fail.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[test]"
python -c "import bdsim; print(bdsim.__file__)"
```

### Fallback: system / user site (PEP 668)

Only if you cannot use `uv`/venv. On Debian/Ubuntu:

```bash
pip install --user --break-system-packages -e .
```

Same caveat: run-only; not for fingerprint parity.

## Dependencies

`pyproject.toml` lists:

- `numpy>=1.24`
- `numba>=0.58` — the JIT compiler. **First-run of a Numba kernel takes ~30-60s while it compiles; subsequent runs use the `.nbi` cache.**
- `scipy>=1.11` — `solve_ivp` (RK45) is the ODE integrator
- `torch>=2.0` — the decanter split MLP
- `plotly>=6.9.0` — figure HTML output

Optional (test extras):

- `pytest>=8.0`
- `pytest-asyncio>=0.23`

The decanter split neural net in `bdsim/split_nn.py` is a faithful port of the upstream 3-layer MLP. Weights and biases are **hardcoded** as NumPy constants in that module (and loaded into `DecanterSplitNet`); there is no `split_nn_weights.pt` file and no runtime weight download.

## Numba cache management

Numba JIT compiles each `@njit` function on first call and caches the compiled object in `__pycache__/*.nbi`. After a kernel change, the cache becomes stale and the next run is slow (30-60s recompile).

Force a clean rebuild:

```bash
find . -name "*.nbi" -delete
find . -name "__pycache__" -exec rm -rf {} +
```

Then re-run the smoke test to warm the cache:

```bash
uv run python -m pytest tests/test_smoke.py -q
```

## Fingerprint regression

`tests/test_smoke.py` and `tests/test_live_simulator.py` pin SHA-256 fingerprints over the full trajectory. A silent numerical drift in a kernel will fail loud.

Named profiles (see `docs/00-orientation/Byte-identical-contract.md` and `AGENTS.md`):

| Path | Profile | Meaning | Hash |
|---|---|---|---|
| batch | legacy fingerprint | Tests pass `fouling_dynamic=False` — **not** bare `ProcessFaults()` | `sv=c8807b23b14a9ad1` |
| batch | legacy fingerprint | | `pv=77def506dbfe25c9` |
| batch | legacy fingerprint | | `uv=17e620519474074a` |
| batch | Layer 2.5 fingerprint | Dynamic HEX fouling (`fouling_dynamic=True`) | `sv=696531c4...` |
| batch | Layer 2.6 (active disturbance) | Nonzero disturbance amplitudes | `sv=8865a8c3...` |
| live | legacy fingerprint | Matching legacy knobs | `sv=23c3c885694c3d24` |
| live | Layer 2.6 (active disturbance) | | `sv=bb763a9bde1d3fc9` |

Runtime default is bare `ProcessFaults()` (`fouling_dynamic=True`, `sv` width 22). When a fingerprint updates, that's a "we changed the math **or the numerical stack**" signal. Pins as of **1.1.1** match `uv.lock` (`numpy==2.2.6`, `scipy==1.15.3`, `numba==0.66.0`). Document the why in the commit body and update the pin in the test file. Don't suppress the test.

## Health checks

Use after a **fingerprint-aligned** install (`uv sync --extra test`):

```bash
# 1. Stack + install path (expect 2.2.6 1.15.3 0.66.0 on Python 3.10)
uv run python -c "import sys,numpy,scipy,numba,bdsim; print(sys.version); print(numpy.__version__, scipy.__version__, numba.__version__); print('OK:', bdsim.__file__)"

# 2. Smoke test (~3-4 min on first run with Numba compile)
uv run python -m pytest tests/test_smoke.py -q

# 3. Full test suite (~6:40 once Numba is warm) — includes fingerprint pins
uv run python -m pytest tests/ -q

# 4. Lint (E702/E401 in ode.py are expected — Numba multi-statement)
uv run ruff check bdsim/ tests/

# 5. CLI works
uv run python -m bdsim 42 --outdir /tmp/bdsim-check --no-plots
ls /tmp/bdsim-check/                     # expect 4 CSVs
```

## Output artifacts

`python3 -m bdsim <seed>` produces in `--outdir` (default `results/`):

- `inputs.csv`, `states.csv`, `measurements.csv`, `setpoints.csv` — 4 CSVs matching the upstream MATLAB column names byte-for-byte
- 9 Plotly HTML figures (when `--no-plots` is NOT set) — `fig01_tr_td_toil_qheat.html` through `fig09_setpoint_loop.html`
- `index.html` — combined view of all 9 figures

## Deployment context

bdsim is **not deployed** as a service. It's an installed library. The bdsim-dashboard project wraps it as a live process.

If you ever need bdsim to run as a long-lived process for some other consumer, the bdsim-dashboard `SimRunner` is the reference implementation (asyncio loop, per-step `LiveSimulator.step()` driven by a worker thread, state published to MQTT).

## Upstream reference

- **Original MATLAB:** Natércia C. P. Fernandes, 2019, University of Coimbra (`natercia@eq.uc.pt`). Upstream: `https://github.com/naterciafernandes/BDSIM`.
- **Docs vault:** [`docs/Home.md`](docs/Home.md) (see [`docs/README.md`](docs/README.md)).
- **Optional PDFs:** `Fernandes2019_BDSIM.pdf` and `manual.pdf` are gitignored at the repo root — drop local copies if you have them; not required to run.
- **Citation:** [`CITATION.cff`](CITATION.cff).
- **License:** GPLv3+ (matches upstream); see [`LICENSE`](LICENSE).
- **Maintainer scripts:** [`scripts/`](scripts/) — optional utilities only (see `scripts/README.md`).

## When to escalate to Joel

- A fingerprint drift in a non-touched file (regression in the legacy path)
- A test that previously passed and now fails
- A new dependency you need to add
- A change to the upstream-MATLAB-faithful behavior in the default path
- Any change to `pyproject.toml` version (also requires updating the
  `bdsim/__init__.py:__version__` and the bdsim-dashboard's minimum-version
  pin in lockstep — both fields are pinned to the same value today)

## Versioning

`__version__` lives in two places, kept in sync:

- `bdsim/__init__.py` — runtime `bdsim.__version__`
- `pyproject.toml` — `[project] version =`

A breaking change to the byte-identical fingerprint contract or to the
`ProcessFaults` schema triggers a minor bump. A patch bump only changes
internal kernels without touching fingerprints or the public schema.
The dashboard's `pyproject.toml` does **not** pin a minimum bdsim
version today — add it when the dashboard gains an installation
contract.
