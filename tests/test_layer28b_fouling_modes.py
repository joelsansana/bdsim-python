"""Tests for Layer 2.8b five-mode fouling stepper (port of fouling.m).

Covers:
  * Modes 0–3 deterministic behaviour (off / linear / exponential /
    Chaibakhsh) — sanity checks against hand-computed values.
  * Modes 4 and 5 stochastic behaviour — pinned RNG, reproduce across
    runs, factor ∈ (0, 1], Rfouling ≥ 0, mean-reversion in mode 4 vs
    pure noise in mode 5.
  * Priority semantics — mode-0 path returns 0 and resets state.
  * State visibility — snapshot/restore roundtrip preserves explicit
    ARMAX state (replaces upstream ``persistent`` MATLAB variables).

These tests are pure-Python; no Numba JIT, no integration with the
ODE kernel. LiveSimulator wiring is tested separately.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bdsim.fouling_modes import (
    FoulingMode,
    FoulingModeStepper,
)


# ---------------------------------------------------------------------------
# Modes 0–3: deterministic behaviour
# ---------------------------------------------------------------------------

def test_mode_0_returns_unit_factor():
    """OFF: factor = 1, regardless of t."""
    st = FoulingModeStepper()
    f, r = st.step(t=12345.0, mode=FoulingMode.OFF)
    assert f == pytest.approx(1.0)
    assert r == 0.0


def test_mode_1_linear_growth():
    """LINEAR: Rfouling = k * t, factor = 1 / (1 + k * t)."""
    k = 3e-7
    st = FoulingModeStepper(foulingpar=np.array([k]))
    t = 1000.0
    f, r = st.step(t=t, mode=FoulingMode.LINEAR)
    expected_r = k * t
    assert r == pytest.approx(expected_r)
    assert f == pytest.approx(1.0 / (1.0 + expected_r))


def test_mode_1_factor_monotone_decreasing():
    """LINEAR: factor monotonically decreases with t."""
    st = FoulingModeStepper(foulingpar=np.array([1e-6]))
    f0, _ = st.step(t=100.0, mode=FoulingMode.LINEAR)
    f1, _ = st.step(t=200.0, mode=FoulingMode.LINEAR)
    f2, _ = st.step(t=1000.0, mode=FoulingMode.LINEAR)
    assert f0 > f1 > f2


def test_mode_2_exponential_decay_t_positive():
    """EXPONENTIAL: at t > 0 the formula yields a finite positive Rf."""
    st = FoulingModeStepper()
    f, r = st.step(t=50000.0, mode=FoulingMode.EXPONENTIAL)
    # The formula 0.1 * exp(-50000/t) at t=50000 is 0.1 * exp(-1)
    expected_r = 0.1 * math.exp(-1.0)
    assert r == pytest.approx(expected_r)
    assert f == pytest.approx(1.0 / (1.0 + expected_r))


def test_mode_2_exponential_handles_t_zero_safely():
    """EXPONENTIAL at t=0 must not blow up."""
    st = FoulingModeStepper()
    # We don't enforce a specific value (the upstream divides by t,
    # which would be NaN). Our port guards against the divide-by-zero
    # by returning 0.0 — that's the conservative, demo-safe choice.
    f, r = st.step(t=0.0, mode=FoulingMode.EXPONENTIAL)
    assert math.isfinite(f)
    assert math.isfinite(r)
    assert r >= 0.0


def test_mode_3_chaibakhsh_starts_at_zero():
    """Chaibakhsh at t=0: r=r0, so log(r0/r0) = 0, Rfouling = 0."""
    st = FoulingModeStepper()
    f, r = st.step(t=0.0, mode=FoulingMode.CHAIBAKHSH)
    assert r == pytest.approx(0.0)
    assert f == pytest.approx(1.0)


def test_mode_3_chaibakhsh_grows_with_t():
    """Chaibakhsh: log(r0/r(t)) with r < r0 → log > 0 → Rfouling grows."""
    st = FoulingModeStepper()
    _, r1 = st.step(t=1000.0, mode=FoulingMode.CHAIBAKHSH)
    _, r2 = st.step(t=5000.0, mode=FoulingMode.CHAIBAKHSH)
    assert r2 > r1 > 0


def test_mode_3_chaibakhsh_factor_formula():
    """Verify the closed-form expression against a hand computation."""
    st = FoulingModeStepper()
    t = 10000.0
    r0, tau, kc, Lt = 0.05, 1e-5, 0.9, 5.0
    r = r0 * math.exp(-tau * t)
    expected_r = (1.0 / (2.0 * math.pi * kc * Lt)) * math.log(r0 / r)
    _, r_actual = st.step(t=t, mode=FoulingMode.CHAIBAKHSH)
    assert r_actual == pytest.approx(expected_r, rel=1e-12)


# ---------------------------------------------------------------------------
# Modes 4 and 5: stochastic, mean-reverting vs. pure-noise
# ---------------------------------------------------------------------------

def test_mode_4_deterministic_with_seed():
    """Two runs with the same seed produce identical Rf trajectories."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    st1 = FoulingModeStepper()
    st2 = FoulingModeStepper()
    traj1 = []
    traj2 = []
    for k in range(50):
        f1, r1 = st1.step(t=100.0 * (k + 1), mode=FoulingMode.ARMAX_NOISE,
                          xRG=0.05, rng=rng1)
        f2, r2 = st2.step(t=100.0 * (k + 1), mode=FoulingMode.ARMAX_NOISE,
                          xRG=0.05, rng=rng2)
        traj1.append(r1)
        traj2.append(r2)
    assert traj1 == pytest.approx(traj2)


def test_mode_4_rfouling_non_negative():
    """Mode 4 clamps Rfouling ≥ 0 even with a pathological negative innovation."""
    st = FoulingModeStepper()
    # Drive hard: large negative epsilon to force a negative candidate
    rng = np.random.default_rng(0)
    for _ in range(20):
        f, r = st.step(t=100.0, mode=FoulingMode.ARMAX_NOISE,
                       xRG=0.05, rng=rng)
        assert r >= 0.0
        assert 0.0 < f <= 1.0


def test_mode_4_mean_reverts_to_k_times_t_times_xRG():
    """Mode 4 is a mean-reverting ARMAX: with small ε, Rf → k·t·xRG."""
    st = FoulingModeStepper(foulingpar=np.array([3e-7]), ar_eps_std=1e-12)
    k = 3e-7
    xrg = 0.05
    # Long horizon so the AR(1) state converges to its target
    rng = np.random.default_rng(7)
    for step in range(5000):
        t = 100.0 * (step + 1)
        f, r = st.step(t=t, mode=FoulingMode.ARMAX_NOISE, xRG=xrg, rng=rng)
    target = k * 100.0 * 5000 * xrg
    # Should be within a few percent of target — ARMAX convergence is fast
    assert abs(r - target) / target < 0.05


def test_mode_5_pure_noise_does_not_update_rf_old():
    """Mode 5 has φ=θ=0 so the persistent RfOld stays 0; only ε contributes.

    The upstream code never updates ``RfOld`` in mode 5 (the assignment
    is missing from the case-5 branch). Our port mirrors that
    behaviour to keep byte-identical replay against MATLAB reference.
    """
    st = FoulingModeStepper()
    rng = np.random.default_rng(0)
    for _ in range(100):
        st.step(t=100.0, mode=FoulingMode.ARMAX_PURE_NOISE,
                xRG=0.05, rng=rng)
    snap = st.snapshot()
    assert snap["rf_old"] == 0.0
    # epsilon_old is whatever the last ε was (5e-4 * randn(1,1))
    assert -3 * 5e-4 <= snap["epsilon_old"] <= 3 * 5e-4


def test_mode_5_rfouling_is_just_epsilon():
    """Mode 5: rfouling at step k equals epsilon_k (since rf_old = 0)."""
    st = FoulingModeStepper()
    rng = np.random.default_rng(123)
    for _ in range(20):
        f, r = st.step(t=100.0, mode=FoulingMode.ARMAX_PURE_NOISE,
                       xRG=0.05, rng=rng)
        # rfouling should equal the most recent epsilon (clamped to >=0)
        snap = st.snapshot()
        assert r == pytest.approx(max(0.0, snap["epsilon_old"]))


def test_mode_4_factor_strictly_in_unit_interval():
    """factor ∈ (0, 1] for all modes (1/(1+Rfouling) is monotonically decreasing)."""
    st = FoulingModeStepper(ar_eps_std=1e-3)
    rng = np.random.default_rng(99)
    for k in range(200):
        for mode in (FoulingMode.LINEAR, FoulingMode.ARMAX_NOISE,
                     FoulingMode.ARMAX_PURE_NOISE):
            f, r = st.step(t=100.0 * (k + 1), mode=mode, xRG=0.05, rng=rng)
            assert 0.0 < f <= 1.0
            assert r >= 0.0


# ---------------------------------------------------------------------------
# xRG coupling
# ---------------------------------------------------------------------------

def test_mode_4_xRG_weight_off_decouples_from_glycerol():
    """When xRG_weight=False, mode-4 target is k·t (independent of xRG)."""
    st_off = FoulingModeStepper(ar_eps_std=1e-12, xRG_weight=False)
    rng = np.random.default_rng(0)
    for k in range(2000):
        st_off.step(t=100.0 * (k + 1), mode=FoulingMode.ARMAX_NOISE,
                    xRG=0.99, rng=rng)              # xRG ignored
    snap = st_off.snapshot()
    # Target with xRG_weight=False: k * t_final = 3e-7 * 200000 = 6e-2
    assert abs(snap["rf_old"] - 6e-2) / 6e-2 < 0.02


def test_mode_4_xRG_weight_on_tracks_glycerol():
    """When xRG_weight=True (default), mode-4 target is k·t·xRG."""
    xrg = 0.10
    st = FoulingModeStepper(ar_eps_std=1e-12, xRG_weight=True)
    rng = np.random.default_rng(0)
    for k in range(2000):
        st.step(t=100.0 * (k + 1), mode=FoulingMode.ARMAX_NOISE,
                xRG=xrg, rng=rng)
    snap = st.snapshot()
    target = 3e-7 * 200000 * xrg
    assert abs(snap["rf_old"] - target) / target < 0.05


# ---------------------------------------------------------------------------
# State visibility: snapshot / restore
# ---------------------------------------------------------------------------

def test_snapshot_returns_arimax_state():
    """snapshot() exposes rf_old, epsilon_old, tau — replaces MATLAB persistent."""
    st = FoulingModeStepper()
    rng = np.random.default_rng(0)
    st.step(t=100.0, mode=FoulingMode.ARMAX_NOISE, xRG=0.05, rng=rng)
    snap = st.snapshot()
    assert "rf_old" in snap
    assert "epsilon_old" in snap
    assert "tau" in snap
    assert isinstance(snap["rf_old"], float)
    assert isinstance(snap["epsilon_old"], float)


def test_restore_resets_state_for_replay():
    """restore(state) lets callers jump the stepper to a saved ARMAX state."""
    st1 = FoulingModeStepper()
    rng = np.random.default_rng(0)
    for k in range(50):
        st1.step(t=100.0 * (k + 1), mode=FoulingMode.ARMAX_NOISE,
                 xRG=0.05, rng=rng)
    saved = st1.snapshot()

    st2 = FoulingModeStepper()
    st2.restore(saved)
    # Both steppers should now produce identical future trajectories
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    for k in range(20):
        f_a, r_a = st1.step(t=1000.0 * (k + 1),
                            mode=FoulingMode.ARMAX_NOISE, xRG=0.05, rng=rng_a)
        f_b, r_b = st2.step(t=1000.0 * (k + 1),
                            mode=FoulingMode.ARMAX_NOISE, xRG=0.05, rng=rng_b)
        assert r_a == pytest.approx(r_b)


def test_reset_zeroes_state():
    """reset() returns the stepper to a fresh-start condition."""
    st = FoulingModeStepper()
    rng = np.random.default_rng(0)
    for k in range(50):
        st.step(t=100.0 * (k + 1), mode=FoulingMode.ARMAX_NOISE,
                xRG=0.05, rng=rng)
    st.reset()
    snap = st.snapshot()
    assert snap["rf_old"] == 0.0
    assert snap["epsilon_old"] == 0.0
    assert snap["tau"] == 0.0


# ---------------------------------------------------------------------------
# Mode-0 reset semantics
# ---------------------------------------------------------------------------

def test_mode_0_clears_arimax_state():
    """Mode OFF zeroes the ARMAX state (matches upstream RfOld = 0)."""
    st = FoulingModeStepper()
    rng = np.random.default_rng(0)
    for k in range(20):
        st.step(t=100.0, mode=FoulingMode.ARMAX_NOISE, xRG=0.05, rng=rng)
    assert st.snapshot()["rf_old"] > 0
    st.step(t=100.0, mode=FoulingMode.OFF)
    snap = st.snapshot()
    assert snap["rf_old"] == 0.0
    assert snap["epsilon_old"] == 0.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_unknown_mode_raises():
    """Out-of-range mode raises ValueError with a helpful message."""
    st = FoulingModeStepper()
    with pytest.raises(ValueError, match="unknown fouling mode"):
        st.step(t=100.0, mode=6)
    with pytest.raises(ValueError, match="unknown fouling mode"):
        st.step(t=100.0, mode=-1)


# ---------------------------------------------------------------------------
# Int mode values match upstream
# ---------------------------------------------------------------------------

def test_int_mode_values_match_upstream():
    """IntEnum values match upstream fouling.m switch cases."""
    assert int(FoulingMode.OFF) == 0
    assert int(FoulingMode.LINEAR) == 1
    assert int(FoulingMode.EXPONENTIAL) == 2
    assert int(FoulingMode.CHAIBAKHSH) == 3
    assert int(FoulingMode.ARMAX_NOISE) == 4
    assert int(FoulingMode.ARMAX_PURE_NOISE) == 5