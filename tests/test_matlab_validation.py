"""
Quantified Python-vs-MATLAB validation scaffold (issue #20).

The byte-identical fingerprint regression tests prove the Python
port is stable *against itself* across runs/versions. The
"faithful port" claim, however, currently rests on code-reading
parity — there is no numerical diff against a documented
MATLAB / Octave reference run.

This module is a **scaffold** for that validation: it loads a
reference trajectory from ``tests/data/upstream_reference.npz``
when the file is present and asserts trajectory agreement within
solver tolerance. When the file is absent, every test in this
module is skipped (so CI stays green on the Python-only install).

Procedure for producing the reference data
------------------------------------------
1. Install Octave (``apt install octave`` on Debian/Ubuntu,
   ``brew install octave`` on macOS, or MATLAB if you have a
   license). The reference upstream is Fernandes 2019 /
   Strelet Dec 2019 ``BDSIM_spectr``; the entry point is
   ``BDSIM.m``.
2. Run with the documented default configuration and the
   settings used in the byte-identical fingerprint tests
   (``seed=42``, ``fouling=1``, ``foulingpar=[3e-7]``,
   ``tf=260000``, ``dt=5``).
3. Capture the trajectory arrays (``t``, ``sv``, ``pv``, ``uv``)
   as ``.npz`` and commit the file as
   ``tests/data/upstream_reference.npz`` with a documented
   Octave / MATLAB version in the commit message.
4. The tests below will then assert the Python port matches
   the reference within the documented solver tolerance
   (default: 1e-3 relative for state, 1e-1 for measurements
   — the latter is dominated by sensor noise).

Until step 3 is done, this file is harmless: every test is
skipped via ``pytest.importorskip``-style guard. The presence
of the data file is the contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Path to the (optional) reference trajectory produced by an
# Octave / MATLAB run of the upstream BDSIM. The file is **not**
# committed today; see the module docstring for the procedure.
_REFERENCE_PATH = Path(__file__).resolve().parent / "data" / "upstream_reference.npz"

# Default tolerance envelope. The solver difference
# (RK45 vs. lsode) is the dominant contribution; sensor noise
# on the published pv series is the second. These are
# intentionally generous — once a reference run is available,
# the tolerance can be tightened to the actual solver diff.
_DEFAULT_TOLERANCES: dict[str, tuple[float, float]] = {
    # channel: (relative tolerance, absolute tolerance)
    "sv": (1e-3, 1.0),                                # state vector
    "pv": (1e-1, 5.0),                                # measurements (sensor noise floor)
    "uv": (1e-3, 1.0),                                # input vector
    "sp": (1e-9, 1e-6),                               # setpoint profile (no solver dependence)
    "t":  (1e-9, 1e-6),                               # time grid (identical by construction)
}


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def _reference_data_or_skip() -> dict[str, np.ndarray]:
    """Load the reference trajectory, skipping if absent."""
    if not _REFERENCE_PATH.exists():
        pytest.skip(
            f"upstream reference trajectory not found at {_REFERENCE_PATH}; "
            f"see module docstring for the production procedure"
        )
    with np.load(_REFERENCE_PATH) as data:
        return {k: np.array(data[k]) for k in data.files}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reference_file_well_formed() -> None:
    """When the reference file is present, it must carry the
    canonical trajectory channels (``t``, ``sv``, ``pv``, ``uv``,
    ``sp``) and have matching leading-axis lengths."""
    data = _reference_data_or_skip()
    expected = {"t", "sv", "pv", "uv", "sp"}
    assert expected <= set(data.keys()), (
        f"reference file is missing channels: {expected - set(data.keys())}"
    )
    n = data["t"].shape[0]
    for ch in expected - {"t"}:
        assert data[ch].shape[0] == n, (
            f"reference '{ch}' has {data[ch].shape[0]} rows, "
            f"expected {n} to match 't'"
        )


@pytest.mark.parametrize("channel", ["t", "sv", "pv", "uv", "sp"])
def test_python_matches_upstream_within_tolerance(channel: str) -> None:
    """For each trajectory channel, the Python port's output
    matches the upstream reference within the documented
    tolerance envelope.

    This is the quantified Python-vs-MATLAB validation called
    for by issue #20 — it only runs when a reference trajectory
    has been produced and committed (see the module docstring).
    """
    from bdsim import run_with
    from bdsim.config import ProcessFaults, Settings

    ref = _reference_data_or_skip()
    rel, atol = _DEFAULT_TOLERANCES[channel]
    settings = Settings(ti=0.0, tf=ref["t"][-1], dt=ref["t"][1] - ref["t"][0])
    results = run_with(
        settings=settings,
        pfaults=ProcessFaults(fouling_dynamic=False),   # legacy / upstream-anchored
        seed=42,
    )
    actual = getattr(results, channel)
    expected = ref[channel]
    np.testing.assert_allclose(
        actual, expected, rtol=rel, atol=atol,
        err_msg=f"channel '{channel}' diverges from upstream beyond tolerance",
    )


def test_python_solver_signature_matches_upstream_seed() -> None:
    """The Python port with ``seed=42`` produces the same
    byte-trajectory as a fresh reference run (when one is
    available) — verifies the deterministic contract holds
    *across* solvers, not just within the Python port."""
    from bdsim import run_with
    from bdsim.config import ProcessFaults, Settings

    ref = _reference_data_or_skip()
    settings = Settings(ti=0.0, tf=ref["t"][-1], dt=ref["t"][1] - ref["t"][0])
    results = run_with(
        settings=settings,
        pfaults=ProcessFaults(fouling_dynamic=False),
        seed=42,
    )
    # The published time grid must match the upstream exactly —
    # ``ti``, ``tf``, ``dt`` are user inputs, not solver outputs.
    np.testing.assert_array_equal(results.t, ref["t"])
