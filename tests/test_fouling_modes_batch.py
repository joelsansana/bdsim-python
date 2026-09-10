"""
Tests for windowed fouling-mode path on the batch driver (issue #8).

Before #8, the batch ``run_with`` driver implemented only tiers 1
(dynamic α) and 3 (static legacy factor) of the fouling priority
system. The windowed mode-4/5 tier 2 was reachable only via
``LiveSimulator``. After #8, both paths route through
``FoulingModeStepper`` when ``pfaults.fouling_mode_active_mode`` is
4 or 5 and the current sim time is before
``pfaults.fouling_mode_active_end_t``.

Coverage:

- Default ``run_with()`` (no windowed fields set) produces identical
  ``Results.factor`` to the pre-#8 batch behaviour.
- ``run_with(fouling_mode_active_mode=4, ...)`` produces a factor
  in the window that differs from the static series.
- The same ``fouling_mode_active_seed`` reproduces the stochastic
  factor across runs.
- ``Results.factor`` records the per-step applied factor, so callers
  can inspect which path the kernel used.
- After the window expires, control returns to the static factor.
- Both the live and batch paths produce the same ``factor`` trajectory
  for the same fault schedule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bdsim import LiveSimulator, run_with
from bdsim.config import ProcessFaults, Settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _short_settings() -> Settings:
    return Settings(ti=0.0, tf=2000.0, dt=5.0)


def _default_pfaults() -> ProcessFaults:
    # fouling_mode_active_mode=0, end_t=-1.0, seed=None — no windowed
    # behaviour, identical to pre-#8 batch path.
    return ProcessFaults()


# ---------------------------------------------------------------------------
# Default run_with() is unchanged (fingerprint contract preserved)
# ---------------------------------------------------------------------------


def test_batch_default_factor_matches_legacy_series() -> None:
    """Without a windowed field and with dynamic α disabled, the
    batch path's factor must match the pre-baked ``_fouling`` series
    (fingerprint contract for the legacy profile)."""
    pf = ProcessFaults(fouling_dynamic=False)
    results = run_with(settings=_short_settings(), pfaults=pf, seed=42)
    # Without dynamic α, the factor at each step is 1 / (1 + k * t)
    # where k = pfaults.foulingpar[0] = 3e-7 and t = settings.ti + i*dt.
    k = pf.foulingpar[0]
    expected = 1.0 / (1.0 + k * results.t)
    np.testing.assert_allclose(results.factor, expected, rtol=1e-12)


def test_batch_default_results_factor_is_not_none() -> None:
    """``Results.factor`` is populated for the batch path too."""
    results = run_with(
        settings=_short_settings(),
        pfaults=ProcessFaults(fouling_dynamic=False),
        seed=42,
    )
    assert results.factor is not None
    assert results.factor.shape == (results.t.shape[0],)


# ---------------------------------------------------------------------------
# Windowed mode 4/5
# ---------------------------------------------------------------------------


def test_batch_windowed_mode_4_activates_fouling_stepper() -> None:
    """With ``fouling_mode_active_mode=4`` and a window that covers
    the whole run, the applied factor is the stepper's output (not
    the static ``1/(1 + k*t)`` series)."""
    pf = ProcessFaults(
        fouling_mode_active_mode=4,
        fouling_mode_active_end_t=1e9,             # effectively forever
        fouling_mode_active_seed=42,
    )
    results = run_with(settings=_short_settings(), pfaults=pf, seed=42)

    # Compute the static series for comparison.
    k = pf.foulingpar[0]
    expected_static = 1.0 / (1.0 + k * results.t)
    # The windowed factor differs (mode 4 is stochastic ARMAX).
    # Specifically, the variance in ``factor - expected_static`` is
    # non-trivial at later times.
    delta = results.factor - expected_static
    assert np.std(delta) > 1e-4, (
        f"windowed mode 4 factor should diverge from the static series; "
        f"got std={np.std(delta):.3e}"
    )


def test_batch_windowed_mode_5_activates_fouling_stepper() -> None:
    """Same check for mode 5 (ARMAX with φ=θ=0)."""
    pf = ProcessFaults(
        fouling_mode_active_mode=5,
        fouling_mode_active_end_t=1e9,
        fouling_mode_active_seed=7,
    )
    results = run_with(settings=_short_settings(), pfaults=pf, seed=42)
    k = pf.foulingpar[0]
    expected_static = 1.0 / (1.0 + k * results.t)
    delta = results.factor - expected_static
    assert np.std(delta) > 1e-4


def test_batch_windowed_mode_reproducible_with_seed() -> None:
    """Two runs with the same ``fouling_mode_active_seed`` produce
    identical ``Results.factor``."""
    pf = ProcessFaults(
        fouling_mode_active_mode=4,
        fouling_mode_active_end_t=1e9,
        fouling_mode_active_seed=2024,
    )
    r1 = run_with(settings=_short_settings(), pfaults=pf, seed=1)
    r2 = run_with(settings=_short_settings(), pfaults=pf, seed=1)
    np.testing.assert_array_equal(r1.factor, r2.factor)


def test_batch_window_expires_returns_to_static() -> None:
    """After ``fouling_mode_active_end_t``, the stepper is bypassed
    and the static series takes over."""
    # End the window very early (t=10s) so most of the run uses the
    # static series. Disable dynamic α so the kernel's tier-3 branch
    # is the static pre-baked factor (otherwise tier 1 — the dynamic
    # α slot — would mask the test).
    pf = ProcessFaults(
        fouling_dynamic=False,
        fouling_mode_active_mode=4,
        fouling_mode_active_end_t=10.0,
        fouling_mode_active_seed=99,
    )
    results = run_with(settings=_short_settings(), pfaults=pf, seed=42)

    # After end_t, the kernel falls through to factor[i] = 1/(1 + k*t).
    k = pf.foulingpar[0]
    expected_static = 1.0 / (1.0 + k * results.t)

    # Once we're well past end_t (say t > 1000 s), the factor should
    # match the static series.
    mask = results.t > 1000.0
    np.testing.assert_allclose(
        results.factor[mask], expected_static[mask], rtol=1e-12
    )


# ---------------------------------------------------------------------------
# Live / batch parity
# ---------------------------------------------------------------------------


def test_batch_and_live_produce_same_factor_for_same_window() -> None:
    """For the same ``ProcessFaults`` windowed config, batch
    ``run_with`` and ``LiveSimulator`` produce identical factor
    trajectories (issue #8)."""
    pf = ProcessFaults(
        fouling_mode_active_mode=4,
        fouling_mode_active_end_t=1e9,
        fouling_mode_active_seed=1234,
    )
    settings = _short_settings()

    batch_results = run_with(settings=settings, pfaults=pf, seed=42)
    live = LiveSimulator(settings=settings, pfaults=pf, seed=42)
    while not live.done:
        live.step()
    # ``_factor_history`` has length ``lt`` (one entry per step,
    # including t=0). The batch ``Results.factor`` drops the final
    # sample per the legacy convention, so compare against ``[:-1]``.
    n_steps = int(settings.tf / settings.dt) + 1
    live_factors = np.asarray(live._factor_history[:n_steps])

    np.testing.assert_allclose(
        batch_results.factor,
        live_factors[:-1],
        rtol=1e-12,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Sanity: invalid mode still rejected
# ---------------------------------------------------------------------------


def test_batch_fouling_mode_active_mode_rejects_modes_1_to_3() -> None:
    """``fouling_mode_active_mode`` only accepts 0, 4, or 5; the
    validation added in #9 still fires from the batch path."""
    with pytest.raises(ValueError, match="fouling_mode_active_mode"):
        ProcessFaults(fouling_mode_active_mode=1)
    with pytest.raises(ValueError, match="fouling_mode_active_mode"):
        ProcessFaults(fouling_mode_active_mode=3)
    with pytest.raises(ValueError, match="fouling_mode_active_mode"):
        ProcessFaults(fouling_mode_active_mode=6)
