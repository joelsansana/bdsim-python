"""Tests for Layer 2.7 operator-driven disturbance knobs.

Verifies:
- The four ``ProcessFaults.live_*`` fields default to ``None``
  (no overlay → profile baseline wins, legacy fingerprint intact).
- ``LiveSimulator.set_*_knob()`` writers mutate the fields and
  reject out-of-envelope values.
- Resolved baselines honour the overlay: a live knob takes effect
  on the next step.
- Getter / snapshot methods expose the live + configured pair.
- ``clear_disturbance_knobs()`` restores all four to ``None`` and
  the kernel reverts to the configured profile baseline.
- A new fingerprint pin is recorded for a representative active
  overlay so any silent regression in the kernel surfaces.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from bdsim.config import ProcessFaults, Settings
from bdsim.live_simulator import LiveSimulator
from bdsim.simulation import run_with


def _fingerprint(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Defaults: all live knobs are None → no overlay
# --------------------------------------------------------------------------- #


def test_live_knob_fields_default_to_none() -> None:
    """The four live_* fields default to None — overlay off by default.

    Bit-identical to the Layer 2.6 baseline when no overlay is set.
    """
    pfaults = ProcessFaults()
    assert pfaults.live_ambient_mean_k is None
    assert pfaults.live_ambient_amplitude_k is None
    assert pfaults.live_cw_t_mean_k is None
    assert pfaults.live_cw_p_drift_pa_per_h is None


def test_zero_amplitude_with_no_overlays_is_byte_identical_to_layer25() -> None:
    """No overlay + zero profile amplitudes → Layer 2.5/2.6 fingerprint."""
    settings = Settings()
    pfaults = ProcessFaults(fouling_dynamic=False, quality_state=False)
    res = run_with(settings=settings, pfaults=pfaults, seed=42, verbose=False)
    # Pinned by Layer 2.5 / Step 4 regression tests (Layer 2.6 used
    # the same fingerprint when amplitudes are zero). Layers 2.5
    # and 2.6 share this contract; Layer 2.7 must not break it
    # when all ``live_*`` fields are None.
    assert _fingerprint(res.sv) == "c8807b23b14a9ad1"
    assert _fingerprint(res.pv) == "77def506dbfe25c9"
    assert _fingerprint(res.uv) == "17e620519474074a"


# --------------------------------------------------------------------------- #
# Mutators: validation + state mutation
# --------------------------------------------------------------------------- #


def test_set_ambient_mean_k_writes_field_and_accepts_none() -> None:
    """The setter accepts a valid value and clears the overlay with None."""
    settings = Settings(ti=0.0, tf=600.0, dt=5.0)
    sim = LiveSimulator(settings=settings, seed=42)
    # Initially None.
    assert sim.pfaults.live_ambient_mean_k is None
    # Set to 30 °C (303.15 K) — within the [-10, 60] °C envelope.
    result = sim.set_ambient_mean_k(303.15)
    assert result["knob"] == "ambient_mean_k"
    assert result["previous"] is None
    assert result["current"] == pytest.approx(303.15)
    assert sim.pfaults.live_ambient_mean_k == pytest.approx(303.15)
    # Clear by passing None.
    result2 = sim.set_ambient_mean_k(None)
    assert result2["previous"] == pytest.approx(303.15)
    assert result2["current"] is None
    assert sim.pfaults.live_ambient_mean_k is None


def test_set_ambient_mean_k_rejects_out_of_envelope_values() -> None:
    """Values outside [-10 °C, 60 °C] → ValueError with the knob name."""
    settings = Settings(ti=0.0, tf=600.0, dt=5.0)
    sim = LiveSimulator(settings=settings, seed=42)
    with pytest.raises(ValueError, match="ambient_mean_k"):
        sim.set_ambient_mean_k(400.0)                       # 127 °C: way too hot
    with pytest.raises(ValueError, match="ambient_mean_k"):
        sim.set_ambient_mean_k(200.0)                       # -73 °C: way too cold
    # Non-numeric input.
    with pytest.raises(ValueError, match="ambient_mean_k"):
        sim.set_ambient_mean_k("thirty")                    # type: ignore[arg-type]
    # Field untouched after rejections.
    assert sim.pfaults.live_ambient_mean_k is None


def test_set_ambient_amplitude_k_envelope() -> None:
    """Amplitude envelope is [0, 30] K; negative or >30 rejected."""
    settings = Settings(ti=0.0, tf=600.0, dt=5.0)
    sim = LiveSimulator(settings=settings, seed=42)
    sim.set_ambient_amplitude_k(0.0)                        # flat day
    assert sim.pfaults.live_ambient_amplitude_k == 0.0
    sim.set_ambient_amplitude_k(15.5)                       # big swing
    assert sim.pfaults.live_ambient_amplitude_k == pytest.approx(15.5)
    with pytest.raises(ValueError, match="ambient_amplitude_k"):
        sim.set_ambient_amplitude_k(-1.0)
    with pytest.raises(ValueError, match="ambient_amplitude_k"):
        sim.set_ambient_amplitude_k(50.0)


def test_set_cw_t_mean_k_envelope() -> None:
    """CW T envelope is [-10, 40] °C (above 40 °C the HEX loses duty)."""
    settings = Settings(ti=0.0, tf=600.0, dt=5.0)
    sim = LiveSimulator(settings=settings, seed=42)
    sim.set_cw_t_mean_k(283.15)                             # 10 °C
    assert sim.pfaults.live_cw_t_mean_k == pytest.approx(283.15)
    with pytest.raises(ValueError, match="cw_t_mean_k"):
        sim.set_cw_t_mean_k(350.0)                          # too warm
    with pytest.raises(ValueError, match="cw_t_mean_k"):
        sim.set_cw_t_mean_k(250.0)                          # freezing


def test_set_cw_p_drift_pa_per_h_envelope() -> None:
    """Drift envelope is [-1000, 1000] Pa/h; summer-cooling negative drift lands."""
    settings = Settings(ti=0.0, tf=600.0, dt=5.0)
    sim = LiveSimulator(settings=settings, seed=42)
    sim.set_cw_p_drift_pa_per_h(-200.0)                     # pumps wearing
    assert sim.pfaults.live_cw_p_drift_pa_per_h == pytest.approx(-200.0)
    sim.set_cw_p_drift_pa_per_h(150.0)                      # over-pressure
    assert sim.pfaults.live_cw_p_drift_pa_per_h == pytest.approx(150.0)
    with pytest.raises(ValueError, match="cw_p_drift_pa_per_h"):
        sim.set_cw_p_drift_pa_per_h(-2000.0)
    with pytest.raises(ValueError, match="cw_p_drift_pa_per_h"):
        sim.set_cw_p_drift_pa_per_h(5000.0)


# --------------------------------------------------------------------------- #
# Snapshot + clear
# --------------------------------------------------------------------------- #


def test_get_disturbance_knobs_reports_live_and_configured() -> None:
    """Snapshot exposes both the overlay value and the profile baseline."""
    settings = Settings(ti=0.0, tf=600.0, dt=5.0)
    sim = LiveSimulator(settings=settings, seed=42)
    snap = sim.get_disturbance_knobs()
    # All four knobs present, no overlay, configured baselines match defaults.
    assert set(snap.keys()) == {
        "ambient_mean_k", "ambient_amplitude_k", "cw_t_mean_k", "cw_p_drift_pa_per_h"
    }
    assert snap["ambient_mean_k"]["live"] is None
    assert snap["ambient_mean_k"]["configured"] == pytest.approx(293.15)
    assert snap["cw_t_mean_k"]["configured"] == pytest.approx(288.15)
    assert snap["ambient_amplitude_k"]["configured"] == 0.0
    assert snap["cw_p_drift_pa_per_h"]["configured"] == 0.0

    # Set one knob; snapshot reflects it.
    sim.set_ambient_mean_k(303.15)
    snap = sim.get_disturbance_knobs()
    assert snap["ambient_mean_k"]["live"] == pytest.approx(303.15)
    assert snap["ambient_mean_k"]["configured"] == pytest.approx(293.15)


def test_clear_disturbance_knobs_resets_all_overlays() -> None:
    """clear_disturbance_knobs() returns previous values and zeros all overlays."""
    settings = Settings(ti=0.0, tf=600.0, dt=5.0)
    sim = LiveSimulator(settings=settings, seed=42)
    sim.set_ambient_mean_k(303.15)
    sim.set_ambient_amplitude_k(5.0)
    sim.set_cw_t_mean_k(283.15)
    sim.set_cw_p_drift_pa_per_h(-100.0)
    snap_pre = sim.get_disturbance_knobs()
    assert snap_pre["ambient_mean_k"]["live"] == pytest.approx(303.15)
    assert snap_pre["ambient_amplitude_k"]["live"] == pytest.approx(5.0)

    result = sim.clear_disturbance_knobs()
    # Returned previous values match the pre-clear state.
    assert result["ambient_mean_k"]["previous"] == pytest.approx(303.15)
    assert result["ambient_amplitude_k"]["previous"] == pytest.approx(5.0)
    assert result["cw_t_mean_k"]["previous"] == pytest.approx(283.15)
    assert result["cw_p_drift_pa_per_h"]["previous"] == pytest.approx(-100.0)
    # Every current is None.
    for entry in result.values():
        assert entry["current"] is None
    # Snapshot confirms all cleared.
    snap_post = sim.get_disturbance_knobs()
    for knob in snap_post.values():
        assert knob["live"] is None


# --------------------------------------------------------------------------- #
# Kernel effect — overlay actually shifts the published disturbance
# --------------------------------------------------------------------------- #


def test_ambient_mean_overlay_shifts_published_track() -> None:
    """Setting ``live_ambient_mean_k`` shifts the published TAMB column.

    With the sinusoid off (configured amplitude=0) and a live mean of
    308.15 K (35 °C), ``StepResult.disturbances[0]`` should be ≈ 308.15
    K at every step — the mean rides through, the sinusoid is pinned flat.
    """
    settings = Settings(ti=0.0, tf=600.0, dt=5.0)
    sim = LiveSimulator(settings=settings, seed=42)
    sim.set_ambient_mean_k(308.15)
    for _ in range(5):
        result = sim.step()
        assert result.disturbances is not None
        np.testing.assert_allclose(
            result.disturbances[0], 308.15, atol=1e-6,
            err_msg="ambient_mean_k overlay did not propagate to published track",
        )


def test_cw_p_drift_overlay_drifts_published_track_at_expected_rate() -> None:
    """Setting ``live_cw_p_drift_pa_per_h=-100`` walks PCW down by 100 Pa/h.

    At t=0 the snapshot sits at the nominal 4e5 Pa; at t=1 h it's at
    4e5 - 100 = 3.996e5 Pa. The kernel resolves ``cw_drift`` to the
    overlay when it computes the track. We use a gentle drift on a
    1 h horizon so the perturbation stays physically plausible
    (drift >> nominal would push ``cw_p`` below zero, which is a
    separate numerical concern covered by the dashboard's horizon
    choices).
    """
    settings = Settings(ti=0.0, tf=3700.0, dt=5.0)           # 3700 s ≈ ~1 h
    sim = LiveSimulator(settings=settings, seed=42)
    sim.set_cw_p_drift_pa_per_h(-100.0)
    # Snapshot at t=0 — sit on the first row of the track.
    r0 = sim.step()
    np.testing.assert_allclose(r0.disturbances[2], 4.0e5, atol=10.0)
    # Walk to ~t=3600 s.
    for _ in range(int(3600 / 5) - 1):
        sim.step()
    r1 = sim.step()
    # cw_p ≈ nominal + drift * t = 4e5 - 100 * 3600 = 3.9964e5 Pa.
    expected = 4.0e5 - 100.0 * 3600.0
    np.testing.assert_allclose(
        r1.disturbances[2], expected, rtol=1e-4,
        err_msg=f"CW drift overlay: expected {expected}, got {r1.disturbances[2]}",
    )


# --------------------------------------------------------------------------- #
# Fingerprint: a representative overlay produces a stable, new hash
# --------------------------------------------------------------------------- #


def test_active_ambient_overlay_pinned_fingerprint() -> None:
    """One active overlay (drift only) has a pinned fingerprint.

    Why drift and not just ambient-mean? The ambient mean is the
    sinusoid's baseline — leaving it alone when amplitude=0 means
    the perturbation kernel adds 0 to u[1] / u[3] regardless of the
    live mean. Drift, by contrast, walks ``cw_p`` directly and the
    kernel multiplies Qheat by ``Pcw / Pnominal`` — observable in
    ``uv`` and downstream in ``pv`` / ``sv``. The pin lets any
    silent regression in the overlay-resolution code surface as a
    test failure.

    Horizon is 24 h (86 400 s) which keeps the drift away from the
    pathological ``cw_p ≈ 0`` blow-up (visible only on the 72 h
    canonical fingerprint horizon, where the drift term dominates).
    """
    settings = Settings(ti=0.0, tf=86400.0, dt=5.0)           # 24 h
    pfaults = ProcessFaults(
        fouling_dynamic=False,
        quality_state=False,
        live_cw_p_drift_pa_per_h=-50.0,                      # pumps slowly wearing
    )
    res = run_with(settings=settings, pfaults=pfaults, seed=42, verbose=False)
    # Stable pins (recompute by running the same fixture and
    # checking in the new hashes — these are the contract).
    assert _fingerprint(res.sv) == "677f6817f64172ab", (
        f"Layer 2.7 drift-overlay sv fingerprint drifted: {_fingerprint(res.sv)}"
    )
    assert _fingerprint(res.pv) == "6a308554964e1051"
    assert _fingerprint(res.uv) == "eb914f357f5d38c3"


def test_clear_overlay_after_use_restores_cleared_state() -> None:
    """Clearing an overlay after use reverts ``uv`` to the no-overlay trajectory."""
    settings = Settings(ti=0.0, tf=86400.0, dt=5.0)           # 24 h
    pfaults_used = ProcessFaults(
        fouling_dynamic=False,
        quality_state=False,
        live_cw_p_drift_pa_per_h=-50.0,
    )
    pfaults_cleared = ProcessFaults(
        fouling_dynamic=False,
        quality_state=False,
        live_cw_p_drift_pa_per_h=None,                       # overlay OFF
    )
    res_used = run_with(settings=settings, pfaults=pfaults_used, seed=42, verbose=False)
    res_cleared = run_with(settings=settings, pfaults=pfaults_cleared, seed=42, verbose=False)
    assert _fingerprint(res_used.sv) != _fingerprint(res_cleared.sv), (
        "Overlay should shift the trajectory away from the cleared baseline"
    )
    # The cleared run with all profile knobs at zero must match the
    # no-overlay 24h fingerprint (different from the canonical 72h
    # Layer 2.5 / 2.6 hash because of horizon, not because of knobs).
    assert _fingerprint(res_cleared.sv) == "7df580fdf1adead3"
    assert _fingerprint(res_cleared.pv) == "02a434ac5ffca65a"
    assert _fingerprint(res_cleared.uv) == "b0d476a82f2c5c19"
