---
tags: [validation, documentation]
aliases: [MATLAB validation, Octave validation, upstream comparison]
---

# Python-vs-MATLAB validation (issue #20)

The [byte-identical fingerprint regression](../00-orientation/Byte-identical-contract.md)
tests prove the Python port is stable **against itself** across
runs and dependency versions. The "faithful port" claim, however,
rests on code-reading parity — there is no numerical diff against
a documented MATLAB/Octave reference run.

This page documents the **procedure for producing the reference
data** and the **tolerance envelope** the comparison uses.

## Reference trajectory (where it lives)

A single committed artefact, when produced:

- `tests/data/upstream_reference.npz`

The file carries the five canonical trajectory channels
(`t`, `sv`, `pv`, `uv`, `sp`) as NPY arrays. The shape and
schema are validated by `tests/test_matlab_validation.py`
(`test_reference_file_well_formed`); the test suite is
auto-skipped when the file is absent, so CI stays green on a
Python-only install.

## Producing the reference

1. **Install Octave or MATLAB.** Octave is the open-source path
   (`apt install octave` on Debian/Ubuntu, `brew install octave`
   on macOS). The upstream is Fernandes 2019 / Strelet Dec 2019
   `BDSIM_spectr`; the entry point is `BDSIM.m`.

2. **Run with the canonical configuration** used in the
   byte-identical fingerprint tests:

   - `seed = 42`
   - `fouling = 1` (legacy series)
   - `foulingpar = [3e-7]`
   - `tf = 260000` (≈ 72.2 h)
   - `dt = 5`
   - All quality / wear / spectrum masters off (so the
     trajectory is the upstream 21-component state)

3. **Capture the trajectory** as a `.npz`. The MATLAB/Octave
   snippet at the bottom of this page shows the save. The five
   arrays must have matching leading-axis lengths; the test
   scaffold checks this.

4. **Commit the file** to `tests/data/upstream_reference.npz`
   with a commit message that documents the Octave / MATLAB
   version and the seed. Anyone re-running the comparison can
   then reproduce the upstream run with the exact same tool
   chain.

## Tolerance envelope

The Python port uses `scipy.solve_ivp` with `method="RK45"`,
`rtol=1e-3`, `atol=1e-6`, `max_step=dt`. The upstream
`BDSIM_spectr` uses `ode45` (MATLAB) or `lsode` (Octave), which
share the RK45 basis but differ in step-size heuristics,
event detection, and Jacobian handling. The comparison
tolerances (set in `tests/test_matlab_validation.py`):

| Channel | Relative | Absolute | Why                                                                 |
| ------- | -------- | -------- | ------------------------------------------------------------------- |
| `t`     | 1e-9     | 1e-6     | `t` is a user input, not a solver output — should match exactly.    |
| `sv`    | 1e-3     | 1.0      | State vector — solver tolerance dominates.                          |
| `pv`    | 1e-1     | 5.0      | Measurements include a sensor-noise floor (default `noise_std`).     |
| `uv`    | 1e-3     | 1.0      | Input vector — solver-tolerance dominated.                          |
| `sp`    | 1e-9     | 1e-6     | Setpoint profile — no solver dependence.                            |

These are intentionally generous. Once a reference run is
committed, the tolerances can be tightened to the actual
solver diff observed on a particular run.

## Why the test is gated on a file

- The MATLAB/Octave tool chain is **not** a build dependency.
  Gating CI on the file's presence would force every
  contributor to install Octave just to run `pytest`.
- The reference is **deterministic by construction** — once
  the file is committed, the comparison is a one-liner that
  doesn't need to re-run the upstream solver on every CI
  cycle. The file is the contract; the test verifies the
  contract.

If the file is regenerated (e.g. after an upstream port
correction), the commit message must say so and the new hash
goes in the changelog alongside the regenerated fingerprint
pins.

## Octave / MATLAB save snippet

```matlab
% After the run() call in BDSIM.m:
t  = res(1,:)';
uv = res(2:7,:)';
sv = res(8:28,:)';
pv = res(29:33,:)';
sp = res(34:37,:)';
save('-v7', 'upstream_reference.npz', 't', 'uv', 'sv', 'pv', 'sp');
```

The Python side reads it back with `numpy.load` and asserts
`channel.shape[0] == t.shape[0]`.

## Status

- The test scaffold is in `tests/test_matlab_validation.py` and
  is auto-skipped today.
- The procedure is reproducible by anyone with Octave or
  MATLAB.
- **Action**: produce the reference trajectory (one-time,
  ~10 minutes of Octave work) and commit it under
  `tests/data/upstream_reference.npz`. After that, the test
  suite will gain a real Python-vs-MATLAB comparison.
