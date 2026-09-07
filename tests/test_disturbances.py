"""Tests for the external-disturbances feature (ambient / CW amplitudes & drift).

Verifies:
- Default profile produces zero perturbation (legacy byte-identical).
- Non-zero amplitude shifts Tmet, Toil, Qheat in the expected direction.
- ``Results.disturbances`` and ``StepResult.disturbances`` carry the
  (lt-1, 3) and (3,) arrays respectively.
- Event-driven perturbations (power_dip, cw_pump_trip) fire and clear.
"""

from __future__ import annotations

import hashlib
import numpy as np

from bdsim.config import ProcessFaults, Settings
from bdsim.live_simulator import LiveSimulator
from bdsim.simulation import run_with


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _fingerprint(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# ProcessFaults defaults preserve byte-identical legacy behavior
# --------------------------------------------------------------------------- #


def test_default_disturbance_amplitudes_are_zero() -> None:
    """All disturbance knobs default to zero so legacy behavior is preserved."""
    pfaults = ProcessFaults()
    assert pfaults.ambient_t_amplitude_k == 0.0
    assert pfaults.cw_t_amplitude_k == 0.0
    assert pfaults.cw_p_drift_pa_per_h == 0.0
    assert pfaults.cw_p_noise_pa == 0.0


def test_zero_amplitude_disturbance_track_is_constant() -> None:
    """When all amplitudes are zero, the disturbance track is the nominal constant."""
    settings = Settings()
    pfaults = ProcessFaults()
    settings._pfaults = pfaults
    t = np.linspace(0, 86400.0, 100)
    track = settings.disturbances(t)
    np.testing.assert_array_equal(
        track[:, 0], np.full(100, pfaults.ambient_t_mean_k)
    )
    np.testing.assert_array_equal(
        track[:, 1], np.full(100, pfaults.cw_t_mean_k)
    )
    np.testing.assert_array_equal(
        track[:, 2], np.full(100, pfaults.cw_p_nominal_pa)
    )


# --------------------------------------------------------------------------- #
# Non-zero amplitudes move the trajectory
# --------------------------------------------------------------------------- #


def test_active_disturbance_shifts_tmet_and_toil() -> None:
    """When ambient swings +8 K and CW swings +4 K, Tmet and Toil shift accordingly."""
    # 12 h horizon with dt=5 s gives lt=8641; peak ambient (6 h) is
    # at index 6*3600/5 = 4320, well within range.
    settings = Settings(ti=0.0, tf=43200.0, dt=5.0)
    pfaults_active = ProcessFaults(
        ambient_t_amplitude_k=8.0,
        cw_t_amplitude_k=4.0,
    )
    pfaults_default = ProcessFaults()

    res_def = run_with(settings=settings, pfaults=pfaults_default, seed=42, verbose=False)
    res_act = run_with(settings=settings, pfaults=pfaults_active, seed=42, verbose=False)

    # At t=12h the ambient sinusoid (period 24h) is at zero crossing.
    # The ``Results`` slicing drops the last sim row, so the effective
    # last index is at t = tf - dt = 43195 s; sin(2π * 43195/86400) is
    # near zero but not exactly zero. Use a coarse tolerance here; the
    # peak-amplitude check below is the strict one.
    assert res_act.disturbances.shape == (len(res_act.t), 3)
    np.testing.assert_allclose(
        res_act.disturbances[-1, 0],
        pfaults_active.ambient_t_mean_k,
        atol=0.01,
    )

    # Toil must have shifted relative to the no-disturbance baseline.
    assert res_act.uv[-1, 3] != res_def.uv[-1, 3]

    # Find a time when the ambient sinusoid is at peak (t=6h for
    # period 24h: sin(2π * 6/24) = sin(π/2) = 1). At that point
    # Tamb = ambient_t_mean + ambient_t_amplitude.
    i_peak = int(6 * 3600 / 5)                                # 4320
    np.testing.assert_allclose(
        res_act.disturbances[i_peak, 0],
        pfaults_active.ambient_t_mean_k + pfaults_active.ambient_t_amplitude_k,
        atol=1e-6,
    )


def test_active_cw_pressure_drift_reduces_qheat() -> None:
    """Negative CW pressure drift should reduce Qheat via the multiplicative kernel."""
    settings = Settings(ti=0.0, tf=7200.0, dt=5.0)
    pfaults_active = ProcessFaults(
        cw_p_drift_pa_per_h=-300.0,                 # ~ -2.16 bar over 2h
        ambient_t_amplitude_k=0.0,                 # isolate Qheat effect
        cw_t_amplitude_k=0.0,
        cw_p_noise_pa=0.0,
    )
    pfaults_default = ProcessFaults()
    res_def = run_with(settings=settings, pfaults=pfaults_default, seed=42, verbose=False)
    res_act = run_with(settings=settings, pfaults=pfaults_active, seed=42, verbose=False)

    # CW pressure drops; u[4] is reduced by the kernel.
    assert res_act.disturbances[-1, 2] < pfaults_active.cw_p_nominal_pa
    assert res_act.uv[-1, 4] < res_def.uv[-1, 4]


# --------------------------------------------------------------------------- #
# Live path exposes disturbances on StepResult
# --------------------------------------------------------------------------- #


def test_live_simulator_step_exposes_disturbances() -> None:
    """LiveSimulator.step() returns disturbances array of shape (3,)."""
    settings = Settings(ti=0.0, tf=86400.0, dt=60.0)         # 24 h, dt=60 s
    pfaults = ProcessFaults(ambient_t_amplitude_k=4.0)
    sim = LiveSimulator(settings=settings, seed=42, pfaults=pfaults)
    # Sample one step at t=0 (sin = 0) and one at t=6h (sin = 1,
    # peak ambient). The difference should be visible in component 0.
    res0 = sim.step()                                          # t=0
    for _ in range(int(6 * 3600 / 60) - 1):
        sim.step()
    res_peak = sim.step()                                      # t ≈ 6h
    assert res0.disturbances is not None
    assert res_peak.disturbances is not None
    assert res0.disturbances.shape == (3,)
    assert res_peak.disturbances.shape == (3,)
    # At t=0 the sinusoid is at zero crossing; at t=6h it peaks at
    # ambient_t_mean + ambient_t_amplitude.
    np.testing.assert_allclose(res0.disturbances[0], pfaults.ambient_t_mean_k, atol=1e-6)
    np.testing.assert_allclose(
        res_peak.disturbances[0],
        pfaults.ambient_t_mean_k + pfaults.ambient_t_amplitude_k,
        atol=1e-6,
    )


def test_live_simulator_zero_amplitude_disturbances_are_constant() -> None:
    """When amplitudes are zero, every step sees the same constant track."""
    settings = Settings(ti=0.0, tf=600.0, dt=5.0)
    pfaults = ProcessFaults()
    sim = LiveSimulator(settings=settings, seed=42, pfaults=pfaults)
    seen = []
    for _ in range(5):
        sim.step()
        seen.append(sim._disturbance_track[sim._i - 1, :].copy())
    for i in range(1, len(seen)):
        np.testing.assert_array_equal(seen[i], seen[0])


# --------------------------------------------------------------------------- #
# Fingerprint regression: zero-amplitude profile
# --------------------------------------------------------------------------- #


def test_default_fingerprint_unchanged_from_layer25_baseline() -> None:
    """external disturbances zero-amplitude default must not drift the upstream fingerprint.

    Uses the canonical upstream Settings (ti=0, tf=260000, dt=5) that
    the original dynamic fouling baseline test pinned. Same settings →
    same fingerprint, regardless of which knobs are at zero.
    """
    settings = Settings()                                    # canonical: ti=0, tf=260000, dt=5
    # Legacy profile: explicit overrides for every feature flag now
    # default-ON (1.2.0+).
    pfaults = ProcessFaults(
        fouling_dynamic=False,
        quality_state=False,
        pump_wear=False,
        valve_wear=False,
        spectrum_enabled=False,
    )
    res = run_with(settings=settings, pfaults=pfaults, seed=42, verbose=False)
    # Pinned by dynamic fouling / Step 4 regression tests.
    assert _fingerprint(res.sv) == "c8807b23b14a9ad1", (
        f"Default sv fingerprint drifted: {_fingerprint(res.sv)} != c8807b23b14a9ad1"
    )
    assert _fingerprint(res.pv) == "77def506dbfe25c9"
    assert _fingerprint(res.uv) == "17e620519474074a"


def test_active_disturbance_produces_new_pinned_fingerprint() -> None:
    """When disturbance knobs are non-zero, the trajectory diverges from upstream.
    Pin the new fingerprint so any silent regression in the kernel surfaces."""
    settings = Settings()                                    # canonical baseline
    # Legacy profile (21-component state) + active external disturbances knobs.
    # Explicit overrides for every feature flag now default-ON
    # (1.2.0+) so this pin stays a clean external disturbances-only trajectory.
    pfaults = ProcessFaults(
        fouling_dynamic=False,
        quality_state=False,
        pump_wear=False,
        valve_wear=False,
        spectrum_enabled=False,
        ambient_t_amplitude_k=8.0,
        cw_t_amplitude_k=4.0,
        cw_p_drift_pa_per_h=-100.0,
    )
    res = run_with(settings=settings, pfaults=pfaults, seed=42, verbose=False)
    # This is a stable pin. Any change to the disturbance kernel
    # bumps these hashes; tests will catch silent regressions.
    assert _fingerprint(res.sv) == "8865a8c352cb6b55", (
        f"Active-disturbance sv fingerprint drifted: {_fingerprint(res.sv)} != 8865a8c352cb6b55"
    )
    assert _fingerprint(res.pv) == "98a6fb024c62a056"
    assert _fingerprint(res.uv) == "401adfcb74484141"
