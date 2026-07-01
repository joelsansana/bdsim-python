# ADMIN.md — bdsim

System administration guide. Covers install, packaging, environment setup, dependency management, and CI-style health checks.

## Who you are

The person who keeps bdsim installable, runnable, and reproducible on this box (or any box). You care about: Python version, Numba cache, editable install, system vs. user site-packages, fingerprint regression, and runtime dependencies.

## Standard install

```bash
cd ~/Documents/projects/bdsim
pip install --user --break-system-packages -e .
```

The `--user --break-system-packages` is required because Debian/Ubuntu's PEP 668 protection blocks system-wide pip installs. The editable install (`-e`) means changes to source are picked up on the next Python invocation without re-installing.

If you don't have the bdsim package in a system Python's path, the bdsim-dashboard cannot import it from a fresh shell. Verify after install:

```bash
python3 -c "import bdsim, bdsim.live_simulator, bdsim.simulation, bdsim.config; print(bdsim.__file__)"
```

Should print `~/Documents/projects/bdsim/bdsim/__init__.py`. If it prints `ModuleNotFoundError`, the editable install didn't land in this Python's site-packages — run the install command from a shell where this is the default `python3`.

## Alternative: venv

If the system install conflicts with something:

```bash
cd ~/Documents/projects/bdsim
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The dashboard is then started from the same venv.

## Dependencies

`pyproject.toml` lists:

- `numpy>=1.24`
- `numba>=0.58` — the JIT compiler. **First-run of a Numba kernel takes ~30-60s while it compiles; subsequent runs use the `.nbi` cache.**
- `scipy>=1.11` — `solve_ivp` (RK45) is the ODE integrator
- `torch>=2.0` — the decanter split MLP

Optional (test extras):

- `pytest>=8.0`
- `pytest-asyncio>=0.23`

The decanter split neural net in `bdsim/split_nn.py` was trained by upstream on a 3-layer MLP. The trained weights are in `bdsim/split_nn_weights.pt` (or generated on first run via `_ensure_weights()`).

## Numba cache management

Numba JIT compiles each `@njit` function on first call and caches the compiled object in `__pycache__/*.nbi`. After a kernel change, the cache becomes stale and the next run is slow (30-60s recompile).

Force a clean rebuild:

```bash
cd ~/Documents/projects/bdsim
find . -name "*.nbi" -delete
find . -name "__pycache__" -exec rm -rf {} +
```

Then re-run the smoke test to warm the cache:

```bash
python3 -m pytest tests/test_smoke.py -q
```

## Fingerprint regression

`tests/test_smoke.py` and `tests/test_live_simulator.py` pin SHA-256 fingerprints over the full trajectory. A silent numerical drift in a kernel will fail loud.

The pinned hashes (current):

| Path | Profile | Hash |
|---|---|---|
| batch | legacy default | `sv=6f61eb532b3284ee` |
| batch | legacy default | `pv=72a3d070452c8fb8` |
| batch | legacy default | `uv=53a404a4b3d7a63c` |
| batch | Layer 2.5 (HEX fouling dynamic) | `sv=1938fec8...` |
| batch | Layer 2.6 (active disturbance) | `sv=e2a29849...` |
| live | legacy default | `sv=f37fb5e0f70f5516` |
| live | Layer 2.6 (active disturbance) | `sv=13ea81f3af76ec5a` |

When a fingerprint updates, that's a "we changed the math" signal. Document the why in the commit body and update the pin in the test file. Don't suppress the test.

## Health checks

```bash
# 1. Install works
python3 -c "import bdsim; print('OK:', bdsim.__file__)"

# 2. Smoke test (12 tests, ~3-4 min on first run with Numba compile)
python3 -m pytest tests/test_smoke.py -q

# 3. Full test suite (~6:40 once Numba is warm)
python3 -m pytest tests/ -q

# 4. Lint (E702/E401 in ode.py are expected — Numba multi-statement)
ruff check bdsim/ tests/

# 5. CLI works
python3 -m bdsim 42 --outdir /tmp/bdsim-check --no-plots
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

- **Original MATLAB:** Natércia C. P. Fernandes, 2019, University of Coimbra (`natercia@eq.uc.pt`).
- **Reference PDFs in `docs/`:** `Fernandes2019_BDSIM.pdf`, `manual.pdf`.
- **License:** GPLv3+ (matches upstream).

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
