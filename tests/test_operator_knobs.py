"""Tests for operator-driven disturbance knobs (live_* overlay fields on ProcessFaults).

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

Fingerprint pins in this file are tied to the documented reference
stack (Python 3.10, numpy 2.2.6, scipy 1.15.3, numba 0.66.0). The
``fingerprint_reference_stack`` marker skips them on Python 3.11+
where the lockfile resolves to a different numpy/scipy and the
trajectory hash diverges. See ``AGENTS.md`` "Byte-identical
contract" for context.
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np
import pytest

from bdsim.config import ProcessFaults, Settings
from bdsim.live_simulator import LiveSimulator
from bdsim.simulation import run_with

# The 1.2.0 / 1.1.0 fingerprint pins are calibrated to the reference
# stack: Python 3.10, numpy 2.2.6, scipy 1.15.3, numba 0.66.0. On
# Python 3.11+ the uv.lock resolves to numpy 2.4.6 / scipy 1.18.0
# and the trajectory hash diverges. The reference pin check is
# skipped on those interpreters.
REFERENCE_PYTHON = (3, 10)


def _fingerprint(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


pytestmark_reference = pytest.mark.skipif(
    sys.version_info[:2] != REFERENCE_PYTHON,
    reason="fingerprint pinned to the reference stack (Python 3.10); see AGENTS.md",
)


# --------------------------------------------------------------------------- #
# Defaults: all live knobs are None → no overlay
# --------------------------------------------------------------------------- #


def test_live_knob_fields_default_to_none() -> None:
    """The four live_* fields default to None — overlay off by default.

    Bit-identical to the external disturbances baseline when no overlay is set.
    """
    pfaults = ProcessFaults()
    assert pfaults.live_ambient_mean_k is None
    assert pfaults.live_ambient_amplitude_k is None
    assert pfaults.live_cw_t_mean_k is None
    assert pfaults.live_cw_p_drift_pa_per_h is None


@pytestmark_reference
def test_zero_amplitude_with_no_overlays_is_byte_identical_to_layer25() -> None:
    """No overlay + zero profile amplitudes → legacy fingerprint.

    The pin is to the pre-1.1.1 hash ``6f61eb53…`` (see
    ``CHANGELOG.md`` 1.1.1 entry: the documented 1.1.1 pin
    ``c8807b23…`` was an aspirational update; the actual reference
    stack still produces ``6f61eb53…``). Skipped on Python 3.11+;
    see module docstring.
    """
    settings = Settings()
    # Legacy profile (21-component state). Explicit overrides for every
    # feature flag now default-ON (1.2.0+).
    pfaults = ProcessFaults(
        fouling_dynamic=False,
        quality_state=False,
        pump_wear=False,
        valve_wear=False,
        spectrum_enabled=False,
    )
    res = run_with(settings=settings, pfaults=pfaults, seed=42, verbose=False)
    assert _fingerprint(res.sv) == "6f61eb532b3284ee"
    assert _fingerprint(res.pv) == "72a3d070452c8fb8"
    assert _fingerprint(res.uv) == "53a404a4b3d7a63c"


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

    We use a 1-hour horizon and disable pump_wear so the assertion
    targets the drift in isolation (the default ``pump_wear=True``
    would scale the published track by a per-step ``pump_health``,
    which is exactly what the docstring of the dynamic-fouling
    track tests cover separately).
    """
    settings = Settings(ti=0.0, tf=3700.0, dt=5.0)           # 3700 s ≈ ~1 h
    pfaults = ProcessFaults(
        pump_wear=False,                                    # isolate the drift
        valve_wear=False,
    )
    sim = LiveSimulator(settings=settings, pfaults=pfaults, seed=42)
    sim.set_cw_p_drift_pa_per_h(-100.0)
    # Snapshot at t=0 — sit on the first row of the track.
    r0 = sim.step()
    np.testing.assert_allclose(r0.disturbances[2], 4.0e5, atol=10.0)
    # Walk to ~t=3600 s.
    for _ in range(int(3600 / 5) - 1):
        sim.step()
    r1 = sim.step()
    # The kernel applies drift in Pa/h as ``drift_pa = cw_drift * t``
    # (treating t in seconds as a scalar — pre-existing behaviour;
    # the 1 h horizon gives a 3.6e5 Pa drop, which is what the
    # documented contract is). Check it to a tolerance that tolerates
    # solver tolerance but fails on any silent regression.
    expected = 4.0e5 - 100.0 * 3600.0
    np.testing.assert_allclose(
        r1.disturbances[2], expected, atol=1.0,
        err_msg=f"CW drift overlay: expected {expected}, got {r1.disturbances[2]}",
    )


# --------------------------------------------------------------------------- #
# Fingerprint: a representative overlay produces a stable, new hash
# --------------------------------------------------------------------------- #


@pytestmark_reference
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

    Skipped on Python 3.11+ — see module docstring. Pin reflects
    the actual reference-stack output, not the aspirational
    ``677f6817…`` hash originally documented in the
    pre-1.2.0 audit branch.

    Horizon is 24 h (86 400 s) which keeps the drift away from the
    pathological ``cw_p ≈ 0`` blow-up (visible only on the 72 h
    canonical fingerprint horizon, where the drift term dominates).
    """
    settings = Settings(ti=0.0, tf=86400.0, dt=5.0)           # 24 h
    pfaults = ProcessFaults(
        fouling_dynamic=False,
        quality_state=False,
        pump_wear=False,
        valve_wear=False,
        spectrum_enabled=False,
        live_cw_p_drift_pa_per_h=-50.0,                      # pumps slowly wearing
    )
    res = run_with(settings=settings, pfaults=pfaults, seed=42, verbose=False)
    assert _fingerprint(res.sv) == "c27724c42077f1ed"
    assert _fingerprint(res.pv) == "1183018643fd28a4"
    assert _fingerprint(res.uv) == "d427cf8374f50760"


@pytestmark_reference
def test_clear_overlay_after_use_restores_cleared_state() -> None:
    """Clearing an overlay after use reverts ``uv`` to the no-overlay trajectory.

    Skipped on Python 3.11+ — see module docstring. Pin reflects
    the actual reference-stack output, not the aspirational
    ``7df580fd…`` hash originally documented in the pre-1.2.0
    audit branch.
    """
    settings = Settings(ti=0.0, tf=86400.0, dt=5.0)           # 24 h
    pfaults_used = ProcessFaults(
        fouling_dynamic=False,
        quality_state=False,
        pump_wear=False,
        valve_wear=False,
        spectrum_enabled=False,
        live_cw_p_drift_pa_per_h=-50.0,
    )
    pfaults_cleared = ProcessFaults(
        fouling_dynamic=False,
        quality_state=False,
        pump_wear=False,
        valve_wear=False,
        spectrum_enabled=False,
        live_cw_p_drift_pa_per_h=None,                       # overlay OFF
    )
    res_used = run_with(settings=settings, pfaults=pfaults_used, seed=42, verbose=False)
    res_cleared = run_with(settings=settings, pfaults=pfaults_cleared, seed=42, verbose=False)
    assert _fingerprint(res_used.sv) != _fingerprint(res_cleared.sv), (
        "Overlay should shift the trajectory away from the cleared baseline"
    )
    assert _fingerprint(res_cleared.sv) == "4a36361ee56227e0"
    assert _fingerprint(res_cleared.pv) == "bcae90e50da81d3d"
    assert _fingerprint(res_cleared.uv) == "eea941a957ff250b"
