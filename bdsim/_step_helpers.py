"""Shared per-step helpers for the batch and live simulators (issue #10).

The batch driver :func:`bdsim.simulation.run_with` and the live driver
:class:`bdsim.live_simulator.LiveSimulator` were historically maintained
as two parallel implementations of the per-step ODE-integration-plus-
measurement loop. Issue #10 asked to extract the shared logic so a
change in one place is automatically reflected in the other.

The helpers in this module are **pure functions** (or minimal-mutation
functions over their explicit arguments) — no module-level mutable
state, no implicit coupling to the caller. They wrap the three blocks
the issue called out by name (``fouling update``, ``measurement``,
``controller step``) plus the related ``disturbances`` overlay (which
is structurally similar to the controller step and was duplicated in
the same two places).

What is *not* extracted
-----------------------
The remaining per-step pieces — ARMAX input update, valve stiction,
``solve_ivp`` integration, post-integration state clamping, filter
cleaning, quality state post-processing, and the measurement kernel
itself — stay inlined for now. The reasons are:

* The ARMAX update, valve stiction, filter cleaning, and
  ``solve_ivp`` integration all mutate per-step state in a
  loop-iteration-local way (e.g. ``valve_yOLD[k] = ...``). Extracting
  them means packing the per-step state into a small object that
  both drivers pass in and out — a larger refactor that the AGENTS.md
  "sacred hot path" rule discourages. These can be extracted in a
  follow-up.
* The measurement kernel :func:`bdsim.simulation._measurements` is
  already a single shared Numba-JIT function — the duplication is at
  the call site only, and the call site is one line.
* The live-only overlays (``_apply_disturbance_override`` for the
  cw_pump_trip handler and ``_apply_live_sensor_faults`` for the
  bias / dropouts / stuck knobs) are not in the batch path. The
  helpers below accept an optional callback for the disturbance
  override so the live driver can plug in its trip handler without
  the batch driver having to know about it.

Fingerprint contract
--------------------
The helpers in this module preserve the **exact statement order** of
the original inlined code. The byte-identical fingerprint contract
on ``res.sv`` / ``res.pv`` / ``res.uv`` is verified by
``tests/test_live_simulator.py::test_run_with_via_live_matches_batch``
and the pinned hashes in ``tests/test_actuator_wear.py``. Any change
that reorders statements (even a no-op move) can drift the last bits
or the RNG sequence and must be re-pinned.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from .fouling_modes import FoulingMode, FoulingModeStepper

if TYPE_CHECKING:
    from .config import ProcessFaults


def select_fouling_factor(
    t_now: float,
    sv: np.ndarray,
    pfaults: ProcessFaults,
    factor_series: np.ndarray,
    i: int,
    fouling_stepper: FoulingModeStepper,
    fouling_mode_rng: np.random.Generator,
    use_dynamic_alpha: bool,
) -> float:
    """Pick the per-step ``factor`` value used by the heat-exchanger RHS.

    Mirrors the priority system used in both
    :func:`bdsim.simulation.run_with` and
    :meth:`bdsim.live_simulator.LiveSimulator._advance_one_step` (issue #8):

      1) continuous α (dynamic fouling) — when ``fouling_dynamic=True``
      2) windowed mode 4/5 (fouling-mode windows) — when the active
         mode is 4 or 5 and ``t_now < fouling_mode_active_end_t``
      3) static legacy factor[i] — the pre-baked series

    Parameters
    ----------
    t_now
        Sim time at the start of the current step
        (``t[i - 1]`` in the caller's index convention).
    sv
        State-vector buffer, shape ``(lt, sv_width)``. ``sv[i - 1]``
        is the previous step's state.
    pfaults
        Process-faults bundle. Read-only.
    factor_series
        Pre-baked static ``factor`` series, shape ``(lt,)``.
    i
        Current step index (1-based).
    fouling_stepper
        :class:`FoulingModeStepper` for tier 2.
    fouling_mode_rng
        RNG consumed by modes 4/5.
    use_dynamic_alpha
        Tier 1 gate (``pfaults.fouling_dynamic``).

    Returns
    -------
    float
        The ``factor`` to be passed to ``rhs.set_factor(...)`` and
        recorded in the per-step ``factor_applied`` buffer.
    """
    mode_active = (
        pfaults.fouling_mode_active_mode in (FoulingMode.ARMAX_NOISE,
                                             FoulingMode.ARMAX_PURE_NOISE)
        and t_now < float(pfaults.fouling_mode_active_end_t)
    )
    if use_dynamic_alpha:
        return float(sv[i - 1, 21])
    if mode_active:
        # xRG is sv[5] (reactor glycerol mole fraction, 0-based).
        # Falls back to 0.0 if the state slot is unavailable (e.g.
        # legacy 21-component state with quality_state=False).
        xrg = float(sv[i - 1, 5]) if sv.shape[1] > 5 else 0.0
        factor_windowed, _ = fouling_stepper.step(
            t=t_now,
            mode=int(pfaults.fouling_mode_active_mode),
            xRG=xrg,
            rng=fouling_mode_rng,
        )
        return float(factor_windowed)
    return float(factor_series[i])


def apply_controller_step(
    i: int,
    u: np.ndarray,
    pvAUTO: np.ndarray,
    sp: np.ndarray,
    mode_1b: np.ndarray,
    uindexAUTO: np.ndarray,
    pid_states: list,
    nic: int,
) -> None:
    """Apply the four PID controller loops on the controller tick.

    Matches the original inlined block in
    :func:`bdsim.simulation.run_with` and
    :meth:`bdsim.live_simulator.LiveSimulator._advance_one_step`
    byte-for-byte. Mutates ``u`` in place (writes the new controller
    output into ``u[uindexAUTO[k]]`` for each active loop).

    Parameters
    ----------
    i
        Current step index (1-based).
    u
        Input-vector buffer, shape ``(ninputs,)``. Mutated in place.
    pvAUTO
        Per-loop controlled-variable measurement, shape
        ``(n_active_loops,)`` (``pv[pvindexAUTO]`` for the current
        step).
    sp
        Setpoint buffer, shape ``(lt, 4)``.
    mode_1b
        Loop-enable mask. The original inlined code calls
        ``.astype(bool)`` and uses ``np.where`` to map the active
        loops onto the pid_states and sp columns.
    uindexAUTO
        Per-loop manipulated-variable column index into ``u``.
    pid_states
        List of :class:`_PIDState` instances, one per loop.
    nic
        Controller update interval in steps. Controllers step when
        ``(i - 1) % nic == 0``.
    """
    if (i - 1) % nic == 0:
        mode = mode_1b.astype(bool)
        mode_idx = np.where(mode)[0]
        for k, ui in enumerate(uindexAUTO):
            meas = pvAUTO[k]
            sp_val = sp[i, mode_idx[k]]
            u[ui] = pid_states[mode_idx[k]].step(sp_val, meas)


def apply_disturbances(
    t_now: float,
    i: int,
    u: np.ndarray,
    sv: np.ndarray,
    pfaults: ProcessFaults,
    disturbance_track: np.ndarray | None,
    use_pump_wear: bool,
    pump_health_idx: int,
    apply_override: Callable[[float, float], float] | None = None,
) -> None:
    """Apply the external-disturbance overlay to ``u`` in place.

    Mirrors the perturbation block in
    :func:`bdsim.simulation.run_with` and
    :meth:`bdsim.live_simulator.LiveSimulator._advance_one_step`
    byte-for-byte. The block is gated on any of the disturbance
    amplitudes / live-knob overlays being non-``None`` / non-zero,
    or on ``use_pump_wear`` (so a worn pump still affects Qheat
    even when all the sinusoid amplitudes are zero).

    Parameters
    ----------
    t_now
        Sim time at the start of the current step (``t[i - 1]``).
    i
        Current step index (1-based).
    u
        Input-vector buffer, shape ``(ninputs,)``. Mutated in place.
    sv
        State-vector buffer, shape ``(lt, sv_width)``. Read-only
        here (``sv[i - 1, pump_health_idx]`` is the previous step's
        pump-health slot for the wear multiplier).
    pfaults
        Process-faults bundle. Read-only.
    disturbance_track
        Pre-built disturbance track, shape ``(lt, 3)`` with columns
        ``[Tamb, Tcw, Pcw]``. ``None`` skips the overlay (matches
        the batch path's ``disturbance_track is not None`` guard).
    use_pump_wear
        Whether the layer-2.4 pump-wear state slot is active.
    pump_health_idx
        Column index in ``sv`` of the pump-health slot. Only used
        when ``use_pump_wear`` is ``True``.
    apply_override
        Optional callback ``(t_now, cw_p) -> cw_p`` applied to the
        CW pressure before the Qheat scaling. The live path uses
        this for the :class:`cooling-water pump trip` override;
        the batch path passes ``None``.
    """
    if disturbance_track is None:
        return
    if not (
        pfaults.ambient_t_amplitude_k != 0.0
        or pfaults.cw_t_amplitude_k != 0.0
        or pfaults.cw_p_drift_pa_per_h != 0.0
        or pfaults.cw_p_noise_pa != 0.0
        or pfaults.live_ambient_mean_k is not None
        or pfaults.live_ambient_amplitude_k is not None
        or pfaults.live_cw_t_mean_k is not None
        or pfaults.live_cw_p_drift_pa_per_h is not None
        or use_pump_wear
    ):
        return
    amb, cw_t, cw_p = disturbance_track[i - 1, :]
    if apply_override is not None:
        cw_p = apply_override(t_now, float(cw_p))
    if use_pump_wear:
        cw_p = cw_p * float(sv[i - 1, pump_health_idx])
    # operator disturbance knobs: baseline references must match the resolved
    # means/amps used inside settings.disturbances(). Pull
    # them once so the deviation math is consistent.
    amb_mean_resolved = (
        pfaults.live_ambient_mean_k
        if pfaults.live_ambient_mean_k is not None
        else pfaults.ambient_t_mean_k
    )
    cw_mean_resolved = (
        pfaults.live_cw_t_mean_k
        if pfaults.live_cw_t_mean_k is not None
        else pfaults.cw_t_mean_k
    )
    # Tmet shifts with CW deviation from (resolved) baseline.
    u[1] = u[1] + (cw_t - cw_mean_resolved) * pfaults.met_cw_track
    # Toil shifts with ambient deviation from (resolved) baseline.
    u[3] = u[3] + (amb - amb_mean_resolved) * pfaults.oil_ambient_track
    # Qheat scales with CW pressure (lower pressure = less heat transfer).
    if pfaults.qheat_cw_scaling and pfaults.cw_p_nominal_pa > 0.0:
        u[4] = u[4] * (cw_p / pfaults.cw_p_nominal_pa)
