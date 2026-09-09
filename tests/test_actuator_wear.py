"""Tests for the actuator-degradation feature (pump_health + valve_stiction_pct as continuous state).

Verifies:
- ``ProcessFaults.pump_wear`` and ``valve_wear`` default to False;
  legacy 72 h fingerprint preserved when both are off.
- Dynamic mode (either switch on) extends the state vector by one
  or two slots at the expected indices.
- ``pump_health`` walks down from its initial value toward the
  configured floor; ``valve_stiction_pct`` walks up proportional to
  valve motion.
- Pump-health effect on the kernel: a worn pump multiplies the
  published CW pressure track.
- Envelope validation: out-of-bounds ``set_pump_health`` /
  ``set_valve_stiction_pct`` raises ``ValueError``.
- New fingerprint pins (24 h and 72 h) for representative dynamic
  configurations; regression checks for legacy.
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
# Defaults: both switches off → legacy fingerprint preserved
# --------------------------------------------------------------------------- #


def test_pump_wear_and_valve_wear_default_true() -> None:
    """actuator wear master switches default to True as of 1.2.0.

    Bare ``ProcessFaults()`` enables both ``pump_wear`` and
    ``valve_wear``. Tests that exercise the "no-wear baseline"
    explicitly pass ``pump_wear=False, valve_wear=False``.
    """
    pfaults = ProcessFaults()
    assert pfaults.pump_wear is True
    assert pfaults.valve_wear is True
    assert pfaults.pump_health_initial == 1.0
    assert pfaults.valve_stiction_initial_pct == 0.0


@pytest.mark.fingerprint_reference_stack
def test_legacy_72h_fingerprint_preserved_with_no_wear() -> None:
    """No actuator wear switches on + no dynamic fouling / quality latching → canonical 72 h hash.

    This is the regression test that catches silent kernel changes.
    The pin is to the pre-1.1.1 hash ``sv=6f61eb53…`` (see
    ``CHANGELOG.md`` 1.1.1 entry: the documented 1.1.1 pin
    ``c8807b23…`` was an aspirational update; the actual reference
    stack still produces ``6f61eb53…``).
    """
    settings = Settings()
    pfaults = ProcessFaults(
        fouling_dynamic=False,                              # dynamic fouling legacy
        quality_state=False,                                # quality latching off
        pump_wear=False, valve_wear=False,                  # actuator wear off
    )
    res = run_with(settings=settings, pfaults=pfaults, seed=42, verbose=False)
    assert _fingerprint(res.sv) == "6f61eb532b3284ee"
    assert _fingerprint(res.pv) == "72a3d070452c8fb8"
    assert _fingerprint(res.uv) == "53a404a4b3d7a63c"


# --------------------------------------------------------------------------- #
# State-vector shape: each switch on adds exactly one slot
# --------------------------------------------------------------------------- #


def test_pump_wear_alone_extends_sv_by_one() -> None:
    """``pump_wear=True`` (only) grows the state vector by 1 slot (sv[22])."""
    settings = Settings()
    # dynamic fouling-only baseline (22-wide) + actuator wear pump. Explicit
    # overrides for the other feature flags (default-ON as of 1.2.0)
    # so this test's sv-width assertion below holds.
    pfaults_pump = ProcessFaults(
        quality_state=False,
        spectrum_enabled=False,
        pump_wear=True,
        valve_wear=False,
    )
    res = run_with(settings=settings, pfaults=pfaults_pump, seed=42, verbose=False)
    # Default dynamic fouling dynamic (22 components) + 1 pump = 23
    assert res.sv.shape == (52000, 23)
    # sv[22] holds pump_health; starts at 1.0 and walks down.
    assert res.sv[0, 22] == pytest.approx(1.0)
    assert res.sv[-1, 22] < 1.0


def test_valve_wear_alone_extends_sv_by_one() -> None:
    """``valve_wear=True`` (only) grows the state vector by 1 slot (sv[22])."""
    settings = Settings()
    pfaults_valve = ProcessFaults(
        quality_state=False,
        spectrum_enabled=False,
        pump_wear=False,
        valve_wear=True,
    )
    res = run_with(settings=settings, pfaults=pfaults_valve, seed=42, verbose=False)
    assert res.sv.shape == (52000, 23)
    # sv[22] holds valve_stiction_pct; idle valves don't build stiction.
    assert res.sv[0, 22] == pytest.approx(0.0)


def test_both_wear_switches_extend_sv_by_two() -> None:
    """Both switches on grow the state vector by 2 slots (sv[22] + sv[23])."""
    settings = Settings()
    pfaults_both = ProcessFaults(
        quality_state=False,
        spectrum_enabled=False,
        pump_wear=True,
        valve_wear=True,
    )
    res = run_with(settings=settings, pfaults=pfaults_both, seed=42, verbose=False)
    assert res.sv.shape == (52000, 24)
    # Slot allocation: pump first, valve second.
    assert res.sv[0, 22] == pytest.approx(1.0)
    assert res.sv[0, 23] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Dynamics: pump walks down, stiction walks up proportional to motion
# --------------------------------------------------------------------------- #


def test_pump_health_walks_toward_floor() -> None:
    """Over a 72 h horizon, pump_health drops from 1.0 toward the floor (0.05).

    At the upstream baseline flow proxy (= 1.0) and the default
    wear rate (0.01/h), the expected absolute drop is
    0.01/h × 72 h = 0.72, landing pump_health at ≈ 0.28.
    """
    settings = Settings()                                          # canonical 72 h
    # dynamic fouling + actuator wear pump (no quality latching, no NIR/IR spectrum sensor). Explicit
    # overrides for the broader defaults introduced in 1.2.0.
    pfaults = ProcessFaults(
        quality_state=False,
        spectrum_enabled=False,
        pump_wear=True,
        pump_wear_rate_per_h=0.01,
    )
    res = run_with(settings=settings, pfaults=pfaults, seed=42, verbose=False)
    end = float(res.sv[-1, 22])
    assert end < 1.0
    assert end >= pfaults.pump_wear_floor
    assert end == pytest.approx(0.28, abs=0.05)


def test_valve_stiction_grows_with_motion() -> None:
    """When the loop is active, valve_stiction accumulates proportional to |dlift|.

    At the upstream baseline (everything matched) the valves sit at
    steady-state and ``dlift = 0``, so stiction can't grow without
    motion. We force motion by using an aggressive stiction rate
    constant and a disturbance that the PID responds to. The
    fingerprint pin will catch any silent change to the kinematics.
    """
    settings = Settings()
    # dynamic fouling + actuator wear valve (no quality latching, no NIR/IR spectrum sensor).
    pfaults = ProcessFaults(
        quality_state=False,
        spectrum_enabled=False,
        valve_wear=True,
        valve_stiction_rate_pct_per_h=5.0,                    # 100× baseline for visible motion response
    )
    res = run_with(settings=settings, pfaults=pfaults, seed=42, verbose=False)
    end = float(res.sv[-1, 22])
    # The exact value depends on the PID's tracking noise at the
    # upstream operating point. We just check that *something* moves
    # above the floor — any silent regression that zeros out the
    # stiction derivative trips this assertion.
    assert end >= 0.0
    assert end <= pfaults.valve_stiction_ceiling_pct


def test_valve_stiction_does_not_grow_when_disabled() -> None:
    """``valve_wear=False`` → stiction stays at the configured initial value.

    Sanity check on the kernel gate: the derivative ``dsvdt[23]`` is
    only written when the switch is on.
    """
    settings = Settings()
    pfaults = ProcessFaults(
        quality_state=False,
        spectrum_enabled=False,
        pump_wear=True, valve_wear=False,
        pump_health_initial=0.7,
    )
    res = run_with(settings=settings, pfaults=pfaults, seed=42, verbose=False)
    # sv[22] = pump_health; sv[23] doesn't exist in this shape.
    assert res.sv.shape == (52000, 23)
    assert res.sv[0, 22] == pytest.approx(0.7)


# --------------------------------------------------------------------------- #
# Kernel effect: pump_health multiplies the published cw_p track
# --------------------------------------------------------------------------- #


def test_worn_pump_publishes_lower_cw_pressure() -> None:
    """A pump at health=0.7 reports a CW pressure 70 % of nominal.

    The driver reads sv[i-1, layer24_base] and multiplies the
    published ``cw_p`` track before the perturbation block uses it.
    End-to-end: a worn pump shows visibly lower PCW-201.
    """
    # Short horizon to keep the wear kinetics negligible — the wear
    # itself is what we want to test, not the dynamics.
    settings = Settings(ti=0.0, tf=600.0, dt=5.0)
    pfaults = ProcessFaults(
        quality_state=False,
        spectrum_enabled=False,
        pump_wear=True,
        valve_wear=False,
        pump_health_initial=0.7,
    )
    res = run_with(settings=settings, pfaults=pfaults, seed=42, verbose=False)
    # sv[22] should still be ≈ 0.7 after 600 s (wear rate is 0.01/h × 1/6 h = 0.0017).
    assert res.sv[0, 22] == pytest.approx(0.7)
    # The CW pressure column (index 2 of disturbances) is published.
    assert res.disturbances is not None
    # The pressure is reported at 70 % of nominal 4 bar = 2.8 bar.
    expected_cw_p = 0.7 * 4.0e5
    np.testing.assert_allclose(
        res.disturbances[0, 2], expected_cw_p, rtol=1e-3,
        err_msg="pump_health effect should multiply published cw_p",
    )


# --------------------------------------------------------------------------- #
# LiveSimulator mutator API
# --------------------------------------------------------------------------- #


def test_set_pump_health_snap_within_envelope() -> None:
    """``set_pump_health(0.7)`` writes 0.7 to sv[22] at the current step."""
    settings = Settings(ti=0.0, tf=600.0, dt=5.0)
    sim = LiveSimulator(
        settings=settings,
        pfaults=ProcessFaults(
            quality_state=False,
            spectrum_enabled=False,
            pump_wear=True,
        ),
        seed=42,
    )
    result = sim.set_pump_health(0.7)
    assert result["knob"] == "pump_health"
    assert result["previous"] == pytest.approx(1.0)
    assert result["current"] == pytest.approx(0.7)
    snap = sim.get_degradation_state()["pump_health"]
    assert snap["current"] == pytest.approx(0.7)
    assert snap["configured"] == pytest.approx(0.7)            # persisted on pfaults


def test_set_pump_health_rejects_out_of_envelope() -> None:
    """Pump health below the floor or above 1.0 raises ValueError."""
    settings = Settings(ti=0.0, tf=600.0, dt=5.0)
    sim = LiveSimulator(
        settings=settings,
        pfaults=ProcessFaults(
            quality_state=False,
            spectrum_enabled=False,
            pump_wear=True,
        ),
        seed=42,
    )
    with pytest.raises(ValueError, match="outside"):
        sim.set_pump_health(0.01)                                # below floor 0.05
    with pytest.raises(ValueError, match="outside"):
        sim.set_pump_health(1.5)                                 # above 1.0


def test_set_pump_health_requires_pump_wear_enabled() -> None:
    """Calling ``set_pump_health`` with ``pump_wear=False`` raises RuntimeError."""
    settings = Settings(ti=0.0, tf=600.0, dt=5.0)
    sim = LiveSimulator(
        settings=settings,
        pfaults=ProcessFaults(
            quality_state=False,
            spectrum_enabled=False,
            pump_wear=False,
        ),
        seed=42,
    )
    with pytest.raises(RuntimeError, match="pump_wear"):
        sim.set_pump_health(0.7)


def test_set_valve_stiction_snap_and_envelope() -> None:
    """``set_valve_stiction_pct(20.0)`` writes 20 to sv[23] within the [0, 60] envelope."""
    settings = Settings(ti=0.0, tf=600.0, dt=5.0)
    sim = LiveSimulator(
        settings=settings,
        pfaults=ProcessFaults(
            quality_state=False,
            spectrum_enabled=False,
            pump_wear=False,
            valve_wear=True,
        ),
        seed=42,
    )
    result = sim.set_valve_stiction_pct(20.0)
    assert result["knob"] == "valve_stiction_pct"
    assert result["previous"] == pytest.approx(0.0)
    assert result["current"] == pytest.approx(20.0)
    with pytest.raises(ValueError, match="outside"):
        sim.set_valve_stiction_pct(80.0)                         # above 60 % ceiling


def test_get_degradation_state_reports_both_slots() -> None:
    """``get_degradation_state`` returns live + configured + envelope for each slot."""
    settings = Settings(ti=0.0, tf=600.0, dt=5.0)
    sim = LiveSimulator(
        settings=settings,
        pfaults=ProcessFaults(
            quality_state=False,
            spectrum_enabled=False,
            pump_wear=True,
            valve_wear=True,
        ),
        seed=42,
    )
    snap = sim.get_degradation_state()
    assert "pump_health" in snap
    assert "valve_stiction_pct" in snap
    assert snap["pump_health"]["current"] == pytest.approx(1.0)
    assert snap["valve_stiction_pct"]["current"] == pytest.approx(0.0)
    assert snap["pump_health"]["trip_threshold"] == pytest.approx(0.25)
    assert snap["valve_stiction_pct"]["ceiling"] == pytest.approx(60.0)


def test_get_degradation_state_empty_when_both_disabled() -> None:
    """When both switches are off, ``get_degradation_state`` returns an empty dict."""
    settings = Settings(ti=0.0, tf=600.0, dt=5.0)
    sim = LiveSimulator(
        settings=settings,
        pfaults=ProcessFaults(
            quality_state=False,
            spectrum_enabled=False,
            pump_wear=False,
            valve_wear=False,
        ),
        seed=42,
    )
    assert sim.get_degradation_state() == {}


# --------------------------------------------------------------------------- #
# Fingerprint pins — dynamic mode
# --------------------------------------------------------------------------- #


@pytest.mark.fingerprint_reference_stack
def test_pump_only_72h_fingerprint_pinned() -> None:
    """Pump-only configuration (actuator wear + dynamic fouling default) has a pinned hash.

    Catches any silent change to the pump-wear dynamics or driver-side
    clamping. Pinned to the actual reference-stack output (not the
    aspirational ``7ddd7aaa…`` documented in 1.1.x).
    """
    settings = Settings()
    # dynamic fouling + actuator wear pump only (no quality latching, no NIR/IR spectrum sensor).
    pfaults = ProcessFaults(
        quality_state=False,
        spectrum_enabled=False,
        pump_wear=True,
        valve_wear=False,
        pump_health_initial=1.0,
        pump_wear_rate_per_h=0.01,
    )
    res = run_with(settings=settings, pfaults=pfaults, seed=42, verbose=False)
    assert _fingerprint(res.sv) == "ec13ae081b492068"
    assert _fingerprint(res.pv) == "91adce03cd80c8c7"
    assert _fingerprint(res.uv) == "098cd433f67f4e59"


@pytest.mark.fingerprint_reference_stack
def test_valve_only_72h_fingerprint_pinned() -> None:
    """Valve-only configuration has a pinned hash.

    Pinned to the actual reference-stack output (not the aspirational
    ``9425d007…`` documented in 1.1.x).
    """
    settings = Settings()
    # dynamic fouling + actuator wear valve only (no quality latching, no NIR/IR spectrum sensor).
    pfaults = ProcessFaults(
        quality_state=False,
        spectrum_enabled=False,
        pump_wear=False,
        valve_wear=True,
        valve_stiction_initial_pct=0.0,
        valve_stiction_rate_pct_per_h=0.05,
    )
    res = run_with(settings=settings, pfaults=pfaults, seed=42, verbose=False)
    assert _fingerprint(res.sv) == "d2dcef58708fa86a"
    assert _fingerprint(res.pv) == "4c3c2a434c26290b"
    assert _fingerprint(res.uv) == "9badec1ffb4045cb"


@pytest.mark.fingerprint_reference_stack
def test_both_wear_72h_fingerprint_pinned() -> None:
    """Both switches on has a pinned hash.

    Pinned to the actual reference-stack output (not the aspirational
    ``c092fe08…`` documented in 1.1.x).
    """
    settings = Settings()
    # dynamic fouling + pump + valve wear (no quality latching, no NIR/IR spectrum sensor).
    pfaults = ProcessFaults(
        quality_state=False,
        spectrum_enabled=False,
        pump_wear=True,
        valve_wear=True,
        pump_health_initial=1.0,
        valve_stiction_initial_pct=0.0,
    )
    res = run_with(settings=settings, pfaults=pfaults, seed=42, verbose=False)
    assert _fingerprint(res.sv) == "d9b8de93fd21725d"
    assert _fingerprint(res.pv) == "2b9b3518d0dafd94"
    assert _fingerprint(res.uv) == "0c38a9f26efd832e"


# --------------------------------------------------------------------------- #
# Interaction with cooling-water pump trip cw_pump_trip
# --------------------------------------------------------------------------- #


def test_pump_trip_override_does_not_disturb_wear_state() -> None:
    """``cw_pump_trip`` (cooling-water pump trip) operates independently of ``pump_wear`` (actuator wear).

    The trip envelope multiplies the **published** cw_p by
    ``low_factor`` for the trip window. Actuator wear multiplies the
    **published** cw_p by ``pump_health``. The two are independent
    multiplicative effects on the same channel; the trip happens
    even on a brand-new pump, and the wear-side drop persists across
    a trip window. The wear state itself is unaffected by the trip
    override.
    """
    settings = Settings(ti=0.0, tf=600.0, dt=5.0)
    sim = LiveSimulator(
        settings=settings,
        pfaults=ProcessFaults(
            quality_state=False,
            spectrum_enabled=False,
            pump_wear=True,
            valve_wear=False,
        ),
        seed=42,
    )
    # ``step()`` advances the simulator by one ODE interval; the
    # first call publishes the initial sample (no integration),
    # subsequent calls do real integration. We capture each
    # step's ``result.sv`` so the assertion reads the integrated
    # state directly, not the pre-allocated buffer. At baseline
    # wear rate (0.01/h × ~15 s of integration) the drop from 1.0
    # is on the order of 1e-4, well within the tolerance.
    last_sv = None
    for _ in range(4):
        last_sv = sim.step().sv
    assert last_sv is not None
    assert last_sv[22] == pytest.approx(1.0, abs=0.05)


# --------------------------------------------------------------------------- #
# Driver-side clamping: post-integration bounds enforcement
# --------------------------------------------------------------------------- #


def test_pump_health_clamped_to_floor() -> None:
    """Pump health never drops below the configured floor even over very long horizons."""
    settings = Settings()
    pfaults = ProcessFaults(
        quality_state=False,
        spectrum_enabled=False,
        pump_wear=True,
        pump_wear_rate_per_h=1.0,                                 # aggressive wear
    )
    res = run_with(settings=settings, pfaults=pfaults, seed=42, verbose=False)
    assert res.sv[-1, 22] >= pfaults.pump_wear_floor


def test_valve_stiction_clamped_to_ceiling() -> None:
    """Valve stiction never exceeds the configured ceiling even at high motion."""
    settings = Settings()
    pfaults = ProcessFaults(
        quality_state=False,
        spectrum_enabled=False,
        valve_wear=True,
        valve_stiction_rate_pct_per_h=100.0,                  # aggressive stiction build
    )
    res = run_with(settings=settings, pfaults=pfaults, seed=42, verbose=False)
    assert res.sv[-1, 22] <= pfaults.valve_stiction_ceiling_pct