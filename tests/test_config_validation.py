"""
Tests for ``bdsim.config`` constructor validation (issue #9).

Covers ``__post_init__`` validation on every config dataclass:
- ``Parameters``
- ``ProcessFaults``
- ``SensorFaults``
- ``ValveFaults``
- ``ARMAX``
- ``PIDController``
- ``Settings``

The rules are deliberately permissive: they catch clearly-broken input
(NaN, Inf, wrong shapes, negative physical quantities, out-of-enum
switches) but do not enforce tight operating-range bounds. These tests
confirm:

1. All current defaults pass validation (no fingerprint drift).
2. Clearly-broken inputs raise ``ValueError`` / ``TypeError`` with
   useful messages naming the offending field.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bdsim.config import (
    ARMAX,
    Parameters,
    PIDController,
    ProcessFaults,
    SensorFaults,
    Settings,
    ValveFaults,
)

# ---------------------------------------------------------------------------
# Defaults sanity: every dataclass accepts its documented defaults
# ---------------------------------------------------------------------------


def test_parameters_defaults_pass() -> None:
    p = Parameters()
    p.finalize()
    assert p.nc == 6


def test_processfaults_defaults_pass() -> None:
    pf = ProcessFaults()
    assert pf.fouling == 1
    assert pf.fouling_mode == 0


def test_sensorfaults_defaults_pass() -> None:
    sf = SensorFaults()
    assert sf.nsensors == 5


def test_valvefaults_defaults_pass() -> None:
    vf = ValveFaults()
    assert vf.S.shape == (2,)


def test_armax_defaults_pass() -> None:
    am = ARMAX()
    assert am.phi.shape == (6,)


def test_pidcontroller_defaults_pass() -> None:
    pid = PIDController()
    assert pid.kc.shape == (4,)


def test_settings_defaults_pass() -> None:
    s = Settings()
    # live_sp* are seeded from sp* in __post_init__
    assert s.live_sp1 == s.sp1


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def test_parameters_rejects_nc_lt_one() -> None:
    with pytest.raises(ValueError, match="nc: must be >= 1"):
        Parameters(nc=0)


def test_parameters_rejects_non_int_nc() -> None:
    with pytest.raises(TypeError, match="nc: expected int"):
        Parameters(nc=6.0)


def test_parameters_rejects_nan_rate_constant() -> None:
    bad = np.array([1.0, np.nan, 0.1, 0.1, 0.1, 0.0])
    with pytest.raises(ValueError, match="k0.*all entries must be finite"):
        Parameters(k0=bad)


def test_parameters_rejects_negative_rate_constant() -> None:
    bad = np.array([-0.1, 0.1, 0.1, 0.1, 0.1, 0.0])
    with pytest.raises(ValueError, match="k0.*non-negative"):
        Parameters(k0=bad)


def test_parameters_rejects_negative_density() -> None:
    bad = np.array([954.0, 983.0, -1030.0, 757.0, 844.0, 1340.0])
    with pytest.raises(ValueError, match="ro.*positive"):
        Parameters(ro=bad)


def test_parameters_rejects_zero_molar_mass() -> None:
    bad = np.array([0.854, 0.600, 0.346, 0.0, 0.286, 0.092])
    with pytest.raises(ValueError, match="M.*positive"):
        Parameters(M=bad)


def test_parameters_rejects_wrong_k0_shape() -> None:
    with pytest.raises(ValueError, match="k0.*must have shape"):
        Parameters(k0=np.array([1.0, 0.1, 0.1]))


def test_parameters_rejects_negative_R() -> None:
    with pytest.raises(ValueError, match="R.*positive"):
        Parameters(R=-1.0)


def test_parameters_rejects_negative_dHr_entries_are_allowed() -> None:
    """dHr can be negative (exothermic / endothermic); only NaN/Inf rejected."""
    # Default dHr already contains -58906.0 — confirm.
    assert Parameters().dHr.min() < 0
    # But NaN is rejected.
    bad = np.array([15699.0, 36899.0, np.nan])
    with pytest.raises(ValueError, match="dHr.*all entries must be finite"):
        Parameters(dHr=bad)


def test_parameters_rejects_inf_viscosity() -> None:
    with pytest.raises(ValueError, match="visco.*must be finite"):
        Parameters(visco=float("inf"))


def test_parameters_rejects_zero_VR() -> None:
    with pytest.raises(ValueError, match="VR.*positive"):
        Parameters(VR=0.0)


def test_parameters_rejects_floor_gt_trip() -> None:
    with pytest.raises(ValueError, match="pump_health_floor.*<= pump_health_trip_threshold"):
        Parameters(
            pump_health_floor=0.5,
            pump_health_trip_threshold=0.1,
        )


# ---------------------------------------------------------------------------
# ProcessFaults
# ---------------------------------------------------------------------------


def test_processfaults_rejects_nan_clog_fraction() -> None:
    with pytest.raises(ValueError, match="clog_fraction.*must be finite"):
        ProcessFaults(clog_fraction=float("nan"))


def test_processfaults_rejects_negative_clog_fraction() -> None:
    with pytest.raises(ValueError, match="clog_fraction.*non-negative"):
        ProcessFaults(clog_fraction=-0.1)


def test_processfaults_rejects_zero_DPclean() -> None:
    with pytest.raises(ValueError, match="DPclean.*positive"):
        ProcessFaults(DPclean=0.0)


def test_processfaults_rejects_bad_fouling() -> None:
    with pytest.raises(ValueError, match="fouling.*must be one of"):
        ProcessFaults(fouling=2)


def test_processfaults_rejects_bad_quality_lag_mode() -> None:
    with pytest.raises(ValueError, match="quality_lag_mode.*lab.*online"):
        ProcessFaults(quality_lag_mode="bogus")


def test_processfaults_rejects_zero_lab_cycle() -> None:
    with pytest.raises(ValueError, match="lab_cycle_s.*positive"):
        ProcessFaults(lab_cycle_s=0.0)


def test_processfaults_rejects_negative_amplitude() -> None:
    with pytest.raises(ValueError, match="ambient_t_amplitude_k.*non-negative"):
        ProcessFaults(ambient_t_amplitude_k=-1.0)


def test_processfaults_rejects_zero_ambient_period() -> None:
    with pytest.raises(ValueError, match="ambient_t_period_s.*positive"):
        ProcessFaults(ambient_t_period_s=0.0)


def test_processfaults_rejects_cw_pump_low_factor_zero() -> None:
    with pytest.raises(ValueError, match="cw_pump_low_factor"):
        ProcessFaults(cw_pump_low_factor=0.0)


def test_processfaults_rejects_cw_pump_low_factor_gt_one() -> None:
    with pytest.raises(ValueError, match="cw_pump_low_factor"):
        ProcessFaults(cw_pump_low_factor=1.5)


def test_processfaults_rejects_pump_floor_gt_trip() -> None:
    with pytest.raises(ValueError, match="pump_wear_floor.*<= pump_health_trip_threshold"):
        ProcessFaults(pump_wear_floor=0.5, pump_health_trip_threshold=0.1)


def test_processfaults_rejects_valve_floor_gt_ceiling() -> None:
    with pytest.raises(ValueError, match="valve_stiction_floor_pct.*<= valve_stiction_ceiling_pct"):
        ProcessFaults(valve_stiction_floor_pct=80.0, valve_stiction_ceiling_pct=50.0)


def test_processfaults_rejects_bad_spctr_cs() -> None:
    with pytest.raises(ValueError, match="spctr_cs.*must be one of"):
        ProcessFaults(spctr_cs=4)


def test_processfaults_rejects_bad_fouling_mode() -> None:
    with pytest.raises(ValueError, match="fouling_mode.*must be one of"):
        ProcessFaults(fouling_mode=6)


def test_processfaults_rejects_non_string_spectra_ref_path() -> None:
    with pytest.raises(TypeError, match="spectra_ref_path"):
        ProcessFaults(spectra_ref_path=123)             # type: ignore[arg-type]


def test_processfaults_rejects_nan_live_knob() -> None:
    with pytest.raises(ValueError, match="live_ambient_mean_k"):
        ProcessFaults(live_ambient_mean_k=float("nan"))


def test_processfaults_live_knob_none_is_ok() -> None:
    pf = ProcessFaults(live_ambient_mean_k=None)
    assert pf.live_ambient_mean_k is None


def test_processfaults_rejects_negative_noise_fame() -> None:
    with pytest.raises(ValueError, match="lab_noise_fame"):
        ProcessFaults(lab_noise_fame=-0.1)


# ---------------------------------------------------------------------------
# SensorFaults
# ---------------------------------------------------------------------------


def test_sensorfaults_rejects_zero_nsensors() -> None:
    with pytest.raises(ValueError, match="nsensors.*>= 1"):
        SensorFaults(nsensors=0)


def test_sensorfaults_rejects_non_int_nsensors() -> None:
    with pytest.raises(TypeError, match="nsensors.*expected int"):
        SensorFaults(nsensors=5.0)


def test_sensorfaults_rejects_wrong_signal_shape() -> None:
    with pytest.raises(ValueError, match="signal.*shape"):
        SensorFaults(signal=np.array([1, 0, 1]))


def test_sensorfaults_rejects_wrong_noise_std_shape() -> None:
    with pytest.raises(ValueError, match="noise_std.*shape"):
        SensorFaults(noise_std=np.array([0.1, 0.2, 0.3]))


def test_sensorfaults_rejects_negative_noise_std() -> None:
    bad = np.array([0.1, 0.1, -0.5, 5e-3, 300.0])
    with pytest.raises(ValueError, match="noise_std.*non-negative"):
        SensorFaults(noise_std=bad)


def test_sensorfaults_rejects_non_bool_signal() -> None:
    bad = np.array([0.5, 1.0, 0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="signal.*0 or 1"):
        SensorFaults(signal=bad)


def test_sensorfaults_rejects_out_of_range_bias_key() -> None:
    with pytest.raises(ValueError, match="bias.*outside the configured range"):
        SensorFaults(bias={5: 1.0})


def test_sensorfaults_rejects_negative_tmax_interm() -> None:
    bad = np.array([0.0, 3600.0, 0.0, -1.0, 0.0])
    with pytest.raises(ValueError, match="tmaxInterm.*non-negative"):
        SensorFaults(tmaxInterm=bad)


# ---------------------------------------------------------------------------
# ValveFaults
# ---------------------------------------------------------------------------


def test_valvefaults_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="J.*shape"):
        ValveFaults(S=np.array([0.0, 0.0]), J=np.array([0.0]))


def test_valvefaults_rejects_negative_S() -> None:
    bad = np.array([0.0, -1.0])
    with pytest.raises(ValueError, match="S.*non-negative"):
        ValveFaults(S=bad)


def test_valvefaults_rejects_negative_uindex() -> None:
    bad = np.array([6, -1])
    with pytest.raises(ValueError, match="uindex.*non-negative"):
        ValveFaults(uindex=bad)


# ---------------------------------------------------------------------------
# ARMAX
# ---------------------------------------------------------------------------


def test_armax_rejects_nan_phi() -> None:
    bad = np.array([np.nan, 0.05, 0.03, 0.0, 0.01, 0.0])
    with pytest.raises(ValueError, match="phi.*all entries must be finite"):
        ARMAX(phi=bad)


def test_armax_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="theta.*shape"):
        ARMAX(theta=np.array([0.0, 0.06]))


def test_armax_rejects_negative_unoise_std() -> None:
    bad = np.array([0.0, -0.1, 1.15, 0.0, 0.01, 0.0])
    with pytest.raises(ValueError, match="unoise_std.*non-negative"):
        ARMAX(unoise_std=bad)


# ---------------------------------------------------------------------------
# PIDController
# ---------------------------------------------------------------------------


def test_pidcontroller_rejects_zero_taui() -> None:
    bad = np.array([16960.0, 0.0, 3600.0, 15.0])
    with pytest.raises(ValueError, match="taui.*positive"):
        PIDController(taui=bad)


def test_pidcontroller_rejects_negative_taud() -> None:
    bad = np.array([0.0, 0.0, -1.0, 0.0])
    with pytest.raises(ValueError, match="taud.*non-negative"):
        PIDController(taud=bad)


def test_pidcontroller_rejects_lower_gt_upper() -> None:
    bad = np.array([400.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="lower_bound.*<= .*upper_bound"):
        PIDController(lower_bound=bad)


def test_pidcontroller_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="kc.*shape"):
        PIDController(kc=np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_rejects_tf_le_ti() -> None:
    with pytest.raises(ValueError, match="tf.*> ti"):
        Settings(ti=100.0, tf=50.0)


def test_settings_rejects_zero_dt() -> None:
    with pytest.raises(ValueError, match="dt.*positive"):
        Settings(dt=0.0)


def test_settings_rejects_nan_tf() -> None:
    with pytest.raises(ValueError, match="tf.*must be finite"):
        Settings(tf=float("nan"))


def test_settings_rejects_wrong_u0_shape() -> None:
    with pytest.raises(ValueError, match="u0.*shape"):
        Settings(u0=np.array([1.0, 2.0]))


def test_settings_rejects_nan_u0() -> None:
    bad = np.array([40.0, np.nan, 657.0, 60 + 273.15, 23615.0, 28.5])
    with pytest.raises(ValueError, match="u0.*all entries must be finite"):
        Settings(u0=bad)


def test_settings_rejects_zero_nic() -> None:
    with pytest.raises(ValueError, match="nic.*>= 1"):
        Settings(nic=0)


def test_settings_rejects_non_int_nic() -> None:
    with pytest.raises(TypeError, match="nic.*expected int"):
        Settings(nic=4.0)


def test_settings_rejects_bad_mode_1b() -> None:
    bad = np.array([1, 0, 1, -5])
    with pytest.raises(ValueError, match="mode_1b"):
        Settings(mode_1b=bad)


def test_settings_rejects_nan_sp() -> None:
    with pytest.raises(ValueError, match="sp1.*must be finite"):
        Settings(sp1=float("nan"))


def test_settings_rejects_wrong_u0_length() -> None:
    with pytest.raises(ValueError, match="u0.*shape"):
        Settings(u0=np.zeros(5))


# ---------------------------------------------------------------------------
# End-to-end: a construction-time validation failure doesn't bypass
# the byte-identical fingerprint contract (defaults still produce the
# canonical trajectory).
# ---------------------------------------------------------------------------


def test_validation_does_not_break_default_construction() -> None:
    """A ``LiveSimulator`` built from defaults still runs to completion."""
    from bdsim.live_simulator import LiveSimulator
    sim = LiveSimulator(settings=Settings(ti=0.0, tf=600.0, dt=5.0), seed=42)
    while not sim.done:
        sim.step()
    assert sim.done
    assert math.isfinite(float(sim._sv[-1, 6]))        # TR finite
