"""
Live, stateful per-step simulator.

This module exposes :class:`LiveSimulator` - a stateful object that
mirrors the batch driver in :mod:`bdsim.simulation` but advances one ODE
interval at a time. It exists for two reasons:

1. **Roadmap step 4 (fault injection).** External code can mutate
   ``sensor_faults.bias`` / ``.stuck`` / ``.dropouts``,
   ``valve_faults.S`` / ``.J``, or ``settings.u0`` (feedstock drift)
   between ``step()`` calls and the simulator will pick them up on the
   next iteration. The batch :func:`bdsim.simulation.run_with` cannot
   expose these knobs mid-run; the live driver can.

2. **Dashboard / live-consumer integration.** The bdsim-dashboard wants to drive a
   process simulator from an async event loop in the same style as a
   real plant DCS - request the next sample, publish it, service fault
   injections in between. Generators (``yield``) are awkward to drive
   from two threads; a stateful object with ``.step()`` is not.

Trajectory parity
-----------------
The inner loop of :meth:`LiveSimulator.step` is copy-lifted from
:func:`bdsim.simulation.run_with` with two minimal changes:

* the per-step arrays (``sv[i-1, :]``, ``factor[i]``, ``d[i-1, :]``,
  ``sfaults.signal[i, :]`` etc.) are scalars / sub-arrays held on
  ``self``;
* the live fault knobs (``sensor_faults.bias`` etc.) are applied to the
  measurement output after the batch path produces its row.

This means :meth:`bdsim.simulation.run_with` can be re-implemented as a
thin driver over :class:`LiveSimulator` and the smoke-test trajectory
must stay byte-identical. The regression test for that contract lives at
``tests/test_live_simulator.py::test_run_with_via_live_matches_batch``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from .config import (
    Parameters,
    ProcessFaults,
    SensorFaults,
    ValveFaults,
    ARMAX,
    PIDController,
    Settings,
    StepResult,
    Results,
)
from .ode import AEmodel, make_rhs
from .thermo import side_reactions
from .simulation import (
    _armax_update_jit,
    _clogging_kit,
    _fouling,
    _intermittence,
    _measurements,
    _PIDState,
    _stiction_step,
)
from .fouling_modes import FoulingMode, FoulingModeStepper


class LiveSimulator:
    """Stateful, per-step biodiesel process simulator.

    Construction mirrors :func:`bdsim.simulation.run_with` end-to-end so
    a subsequent ``step()`` produces the same trajectory row-for-row as
    the batch path. Use :meth:`reset` to rewind to t=0 without rebuilding
    the object.

    Live mutation surface
    ---------------------
    External code may modify between ``step()`` calls:

    * ``self.sensor_faults.bias`` - sensor index → additive offset (units)
    * ``self.sensor_faults.stuck`` - sensor index → sim time at which it
      became stuck. The held value is the *last published* sample (Q1
      decision from the step 4 design review). Sticky until cleared.
    * ``self.sensor_faults.dropouts`` - set of sensor indices currently
      outputting zero (OPC quality = "bad")
    * ``self.valve_faults.S`` / ``.J`` - stiction band / jump per valve
    * ``self.settings.u0`` - exogenous setpoint for feedstock drift
      (Fmet at index 2) / heat loss (Qheat at index 4)

    Everything else (``sensor_faults.a`` / ``.b`` / ``.signal``, the batch
    intermittency machinery) is set up at construction and then
    read-only for the lifetime of the run.

    Examples
    --------
    >>> from bdsim.live_simulator import LiveSimulator
    >>> sim = LiveSimulator(seed=42)
    >>> while not sim.done:
    ...     row = sim.step()
    ...     print(row.t, row.pv)
    >>> sim.reset()
    """

    def __init__(
        self,
        p: Parameters | None = None,
        pfaults: ProcessFaults | None = None,
        sfaults: SensorFaults | None = None,
        vfaults: ValveFaults | None = None,
        armax: ARMAX | None = None,
        pid: PIDController | None = None,
        settings: Settings | None = None,
        seed: int | None = None,
    ) -> None:
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.p = p or Parameters()
        self.p.finalize()
        self.pfaults = pfaults or ProcessFaults()
        self.sensor_faults = sfaults or SensorFaults()
        self.valve_faults = vfaults or ValveFaults()
        self.armax = armax or ARMAX()
        self.pid = pid or PIDController()
        self.settings = settings or Settings()

        # The setup is identical to ``run_with`` (lines 298-369 of the
        # original module) up to the point where the main loop starts.
        self._setup()
        self._last_published: dict[int, float] = {}    # sensor index → value (for stuck semantic)
        # cooling-water pump trip: single-slot mid-run override for the CW pressure
        # disturbance channel. None = no override (the kernel applies
        # the baseline sinusoidal profile only). Populated by a
        # ``cw_pump_trip`` handler (see faults.py) and cleared when
        # the trip expires. Single-slot matches ``power_dip``
        # semantics — a second trip replaces the first.
        self._disturbance_override: dict[str, Any] | None = None
        # NIR/IR spectrum sensor: NIR/IR spectrum generator. Built once at
        # construction so the reference spectra CSV is loaded lazily
        # only when the sensor is enabled. ``None`` when
        # ``pfaults.spectrum_enabled`` is False — saves the CSV read
        # on every LiveSimulator instantiation.
        self._spectrum_generator: SpectrumGenerator | None = None
        if pfaults is not None and pfaults.spectrum_enabled:
            from .spectra import SpectrumGenerator, SpectrumConfig
            self._spectrum_generator = SpectrumGenerator(
                SpectrumConfig(
                    enabled=True,
                    spctr_t=pfaults.spctr_t,
                    cs=pfaults.spctr_cs,
                    snr_db=pfaults.spctr_snr_db,
                    k=pfaults.spctr_k,
                    drift_a=pfaults.spctr_drift_a,
                    drift_b=pfaults.spctr_drift_b,
                    drift_c=pfaults.spctr_drift_c,
                    spectra_ref_path=pfaults.spectra_ref_path,
                    seed=seed,
                )
            )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Rewind to t=0 with the same configuration. RNG is *not* reseeded -
        the next run continues the random sequence. Use ``seed=...`` on a
        new :class:`LiveSimulator` if you want a deterministic re-run.
        """
        self._setup()
        self._last_published.clear()
        # cooling-water pump trip: clear any active cw_pump_trip override so the
        # next run starts from a clean disturbance state.
        self._disturbance_override = None

    # ------------------------------------------------------------------ #
    # Operator actions (Roadmap dynamic fouling)
    # ------------------------------------------------------------------ #

    def trigger_cleaning(self) -> dict[str, Any]:
        """Operator-initiated HEX/filter cleaning event.

        Snaps the current step's HEX fouling factor α (sv[21]) to
        ``alpha_clean`` (default 0.1), resets the filter pore radius
        (sv[18]) to ``rclean``, and records the cleaning time in
        ``self._tclean``. The next :meth:`step` propagates these
        values into the publish path.

        Returns a small dict with the change summary so callers can
        audit the action. In legacy mode (state vector has 21
        components) the α slot is silently skipped — the pore-radius
        reset still happens.

        Thread-safety: not thread-safe. Call from the same thread that
        owns the step loop (typically the FastAPI request handler
        invokes this via the runner's lock).
        """
        i = self._i
        changed: dict[str, Any] = {"step": i, "sim_time_s": float(self._t[i])}
        # Pore radius reset (always — this is the existing cleaning action).
        self._sv[i, 18] = self.p.rclean * 1e6
        changed["pore_radius_um"] = float(self._sv[i, 18])
        # HEX fouling reset (dynamic fouling — only when the state vector has the slot).
        if self._sv.shape[1] > 21:
            self._sv[i, 21] = self.p.alpha_clean
            changed["alpha"] = float(self._sv[i, 21])
        # Append the cleaning timestamp (matches the auto-trigger behaviour).
        self._tclean.append(float(self._t[i]))
        changed["cleaning_times_s"] = list(self._tclean)
        return changed

    # ------------------------------------------------------------------ #
    # Operator actions (Roadmap operator disturbance knobs — disturbance knobs)
    # ------------------------------------------------------------------ #
    #
    # These four setters let the dashboard's disturbance panel ride
    # the exogenous environment for the rest of the run. Each knob
    # is **persistent** (no time window, no expiry — they're the
    # "today's weather is hotter than usual" controls, not the
    # "brownout for 5 minutes" controls that :class:`FaultSpec`
    # already provides). Setting a knob to ``None`` clears the
    # overlay and reverts to the configured profile baseline.
    #
    # Validation is the minimal envelope: amplitudes drift between
    # physically plausible values, drift rate between ±1000 Pa/h,
    # mean temperatures between 0 °C and 60 °C (well outside the
    # biodiesel reactor's operating range). Out-of-envelope values
    # raise :class:`ValueError` so a bad UI input doesn't silently
    # poison the run.

    def set_ambient_mean_k(self, value: float | None) -> dict[str, float | None]:
        """Override ambient temperature setpoint (K).

        ``None`` clears the overlay and reverts to
        ``pfaults.ambient_t_mean_k``. Otherwise the value must lie in
        ``[263.15, 333.15]`` (-10 °C .. 60 °C). Takes effect on the
        next ``step()`` (the perturbation kernel recomputes the
        track from the live knob).
        """
        resolved = self._validate_knob("ambient_mean_k", value, lo=263.15, hi=333.15)
        prev = self.pfaults.live_ambient_mean_k
        self.pfaults.live_ambient_mean_k = resolved
        self._refresh_disturbance_track()
        return {"knob": "ambient_mean_k", "previous": prev, "current": resolved}

    def set_ambient_amplitude_k(self, value: float | None) -> dict[str, float | None]:
        """Override ambient daily-sinusoid amplitude (K).

        ``None`` clears the overlay; otherwise the value must lie in
        ``[0.0, 30.0]`` (the configured default is 8 K, the operator
        can pin it flat at 0 to mean "boring day, no swing").
        """
        resolved = self._validate_knob("ambient_amplitude_k", value, lo=0.0, hi=30.0)
        prev = self.pfaults.live_ambient_amplitude_k
        self.pfaults.live_ambient_amplitude_k = resolved
        self._refresh_disturbance_track()
        return {"knob": "ambient_amplitude_k", "previous": prev, "current": resolved}

    def set_cw_t_mean_k(self, value: float | None) -> dict[str, float | None]:
        """Override cooling-water inlet setpoint (K).

        ``None`` clears the overlay; otherwise the value must lie in
        ``[263.15, 313.15]`` (-10 °C .. 40 °C). Above ~40 °C the
        HEX starts losing duty, below 0 °C the cooling tower freezes.
        """
        resolved = self._validate_knob("cw_t_mean_k", value, lo=263.15, hi=313.15)
        prev = self.pfaults.live_cw_t_mean_k
        self.pfaults.live_cw_t_mean_k = resolved
        self._refresh_disturbance_track()
        return {"knob": "cw_t_mean_k", "previous": prev, "current": resolved}

    def set_cw_p_drift_pa_per_h(self, value: float | None) -> dict[str, float | None]:
        """Override cooling-water pressure drift rate (Pa/h).

        ``None`` clears the overlay; otherwise the value must lie
        in ``[-1000.0, 1000.0]``. Negative means the pumps are
        wearing (pressure drifts down over hours), positive means
        a fresh pump is over-pressurising. Daily-bin drift only —
        the long-term trend is the story, not noise.
        """
        resolved = self._validate_knob("cw_p_drift_pa_per_h", value, lo=-1000.0, hi=1000.0)
        prev = self.pfaults.live_cw_p_drift_pa_per_h
        self.pfaults.live_cw_p_drift_pa_per_h = resolved
        self._refresh_disturbance_track()
        return {"knob": "cw_p_drift_pa_per_h", "previous": prev, "current": resolved}

    def clear_disturbance_knobs(self) -> dict[str, dict[str, None]]:
        """Clear all operator-driven knob overlays in one call.

        Returns a small dict mapping knob name → previous (None is
        always returned as the ``current`` value). Mirrors the
        ``reset`` semantics of the dashboard "Restore defaults" button.
        """
        prevs: dict[str, dict[str, None]] = {}
        for knob, attr in (
            ("ambient_mean_k", "live_ambient_mean_k"),
            ("ambient_amplitude_k", "live_ambient_amplitude_k"),
            ("cw_t_mean_k", "live_cw_t_mean_k"),
            ("cw_p_drift_pa_per_h", "live_cw_p_drift_pa_per_h"),
        ):
            prev = getattr(self.pfaults, attr)
            setattr(self.pfaults, attr, None)
            prevs[knob] = {"previous": prev, "current": None}
        self._refresh_disturbance_track()
        return prevs

    def _refresh_disturbance_track(self) -> None:
        """Rebuild ``self._disturbance_track`` after a knob mutation.

        The track is computed once at :meth:`_setup` time and held
        steady through the run so the kernel's perturbation math
        (Tmet, Toil, Qheat shifts) stays stable. When an operator
        pushes a knob, only the **future** rows of the track reflect
        the new operating point — past rows stay put, so the
        trajectory doesn't time-warp. The rebuild is an O(lt - i)
        scan, cheap for typical demo horizons (a few thousand rows).

        Idempotent on the no-knob path: if no ``live_*`` overlay is
        set, the rebuilt track matches the kernel baseline exactly.
        """
        if not hasattr(self, "_t") or not hasattr(self, "_pfaults"):
            return                                          # setup hasn't run yet
        if self._pfaults is None:
            return
        # Past rows (indices < self._i) are published and immutable;
        # only the future part shifts under the new operating point.
        i_pivot = self._i
        t_full = self._t
        # Recompute the full track against the current pfaults.
        new_track = self.settings.disturbances(t_full)
        # For past rows (already-published), keep the prior track.
        if i_pivot > 0 and self._disturbance_track is not None:
            new_track[:i_pivot] = self._disturbance_track[:i_pivot]
        self._disturbance_track = new_track

    def get_disturbance_knobs(self) -> dict[str, dict[str, float | None]]:
        """Snapshot the active knob overlays + profile baselines.

        Returned shape::

            {
                "ambient_mean_k":      {"live": ..., "configured": ...},
                "ambient_amplitude_k": {"live": ..., "configured": ...},
                "cw_t_mean_k":         {"live": ..., "configured": ...},
                "cw_p_drift_pa_per_h": {"live": ..., "configured": ...},
            }

        ``live`` is the override value (``None`` means no overlay).
        ``configured`` is the un-overridden profile baseline the
        overlay replaces (or, if no overlay is set, the value the
        kernel is actually using).
        """
        return {
            "ambient_mean_k": {
                "live": self.pfaults.live_ambient_mean_k,
                "configured": self.pfaults.ambient_t_mean_k,
            },
            "ambient_amplitude_k": {
                "live": self.pfaults.live_ambient_amplitude_k,
                "configured": self.pfaults.ambient_t_amplitude_k,
            },
            "cw_t_mean_k": {
                "live": self.pfaults.live_cw_t_mean_k,
                "configured": self.pfaults.cw_t_mean_k,
            },
            "cw_p_drift_pa_per_h": {
                "live": self.pfaults.live_cw_p_drift_pa_per_h,
                "configured": self.pfaults.cw_p_drift_pa_per_h,
            },
        }

    @staticmethod
    def _validate_knob(
        name: str, value: float | None, *, lo: float, hi: float
    ) -> float | None:
        """Validate an operator-driven knob overlay value.

        ``None`` is always accepted (clear-overlay). Otherwise the
        value must be a real number in ``[lo, hi]`` (both inclusive
        at the extremes). Raises :class:`ValueError` with a hint
        about the envelope so the dashboard can show a useful error.
        """
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a real number or null, got {value!r}"
            ) from exc
        if not (lo <= v <= hi):
            raise ValueError(
                f"{name}={v!r} is outside the operator knob envelope [{lo}, {hi}]"
            )
        return v

    # ------------------------------------------------------------------ #
    # Operator actions (Roadmap actuator wear — actuator degradation)
    # ------------------------------------------------------------------ #
    #
    # These three mutators let the dashboard demonstrate sudden wear
    # events without waiting for the natural kinetics to play out. They
    # write directly into ``ProcessFaults`` so a subsequent rebuild
    # picks the value up, AND inject into the live sim immediately when
    # it's already running. Envelope validation is identical to the
    # dynamics constants on :class:`ProcessFaults`.

    def set_pump_health(self, value: float | None) -> dict[str, float | None]:
        """Snap the pump to a specific health state.

        ``None`` is treated as 1.0 (brand-new). The value must lie in
        ``[pump_wear_floor, 1.0]``; out-of-envelope raises ``ValueError``.
        The next integration step uses this as the starting state and
        wear continues from there.
        """
        pfaults = self.pfaults
        if not pfaults.pump_wear:
            raise RuntimeError(
                "set_pump_health requires pfaults.pump_wear=True; "
                "construct the simulator with ProcessFaults(pump_wear=True)"
            )
        if value is None:
            value = 1.0
        if not (pfaults.pump_wear_floor <= float(value) <= 1.0):
            raise ValueError(
                f"pump_health={value} is outside [{pfaults.pump_wear_floor}, 1.0]"
            )
        prev = float(self._sv[self._i, self._pump_health_idx])
        self._sv[self._i, self._pump_health_idx] = float(value)
        # Also persist so a future rebuild carries the new initial state.
        pfaults.pump_health_initial = float(value)
        return {"knob": "pump_health", "previous": prev, "current": float(value)}

    def set_valve_stiction_pct(self, value: float | None) -> dict[str, float | None]:
        """Snap the valve stiction to a specific percentage.

        ``None`` is treated as 0.0 (pristine). The value must lie in
        ``[valve_stiction_floor_pct, valve_stiction_ceiling_pct]``;
        out-of-envelope raises ``ValueError``.
        """
        pfaults = self.pfaults
        if not pfaults.valve_wear:
            raise RuntimeError(
                "set_valve_stiction_pct requires pfaults.valve_wear=True"
            )
        if value is None:
            value = 0.0
        lo, hi = pfaults.valve_stiction_floor_pct, pfaults.valve_stiction_ceiling_pct
        if not (lo <= float(value) <= hi):
            raise ValueError(
                f"valve_stiction_pct={value} is outside [{lo}, {hi}]"
            )
        prev = float(self._sv[self._i, self._valve_stiction_idx])
        self._sv[self._i, self._valve_stiction_idx] = float(value)
        pfaults.valve_stiction_initial_pct = float(value)
        return {"knob": "valve_stiction_pct", "previous": prev, "current": float(value)}

    def get_degradation_state(self) -> dict[str, dict[str, float | None]]:
        """Snapshot pump_health + valve_stiction_pct (current and configured).

        Each entry exposes ``current`` (the live state value) and
        ``configured`` (the un-overridden initial value the sim was
        built with). Returns an empty dict when both actuator wear switches
        are off.
        """
        out: dict[str, dict[str, float | None]] = {}
        pfaults = self.pfaults
        if pfaults.pump_wear:
            out["pump_health"] = {
                "current": float(self._sv[self._i, self._pump_health_idx]),
                "configured": float(pfaults.pump_health_initial),
                "trip_threshold": float(pfaults.pump_health_trip_threshold),
                "floor": float(pfaults.pump_wear_floor),
            }
        if pfaults.valve_wear:
            out["valve_stiction_pct"] = {
                "current": float(self._sv[self._i, self._valve_stiction_idx]),
                "configured": float(pfaults.valve_stiction_initial_pct),
                "ceiling": float(pfaults.valve_stiction_ceiling_pct),
            }
        return out

    # ------------------------------------------------------------------
    # fouling-mode windows: windowed fouling mode (modes 4 / 5) mutators
    # ------------------------------------------------------------------
    def activate_fouling_mode_window(
        self,
        mode: int | FoulingMode,
        duration_s: float,
        seed: int | None = None,
    ) -> dict:
        """Activate a windowed fouling-mode fault (fouling-mode windows).

        During the window the kernel uses the
        :class:`~bdsim.fouling_modes.FoulingModeStepper` to produce
        ``factor`` per step instead of the dynamic fouling α path or the
        legacy pre-baked series. When the window expires control
        returns to whichever path is higher priority (dynamic fouling
        continuous α if enabled, else the static legacy series).

        Args:
            mode: 4 or 5 (other modes are not windowed; they don't
                have stochastic state).
            duration_s: how long the window lasts, in sim seconds.
                Defaults to ``pfaults.fouling_mode_default_window_s``
                when ``None``.
            seed: optional RNG seed for the ARMAX stepper during
                this window. ``None`` → fresh seed derived from
                the current rng so consecutive windows differ.

        Returns:
            A dict with the active mode, end_t, and window length
            for diagnostic echo on the dashboard.
        """
        m = int(mode)
        if m not in (FoulingMode.ARMAX_NOISE, FoulingMode.ARMAX_PURE_NOISE):
            raise ValueError(
                f"fouling_mode_window only supports modes 4 or 5; got {m!r}"
            )
        if duration_s is None or duration_s <= 0.0:
            raise ValueError(
                f"duration_s must be positive; got {duration_s!r}"
            )
        pfaults = self.pfaults
        if seed is not None:
            self._fouling_mode_rng = np.random.default_rng(int(seed))
        pfaults.fouling_mode_active_mode = m
        pfaults.fouling_mode_active_end_t = float(self.t) + float(duration_s)
        # Reset the ARMAX state so each window starts fresh
        self._fouling_stepper.reset()
        return {
            "mode": m,
            "start_t": float(self.t),
            "end_t": float(pfaults.fouling_mode_active_end_t),
            "duration_s": float(duration_s),
        }

    def clear_fouling_mode_window(self) -> dict:
        """Deactivate the windowed fouling mode. Idempotent.

        Returns the cleared envelope (empty if no window was active).
        """
        pfaults = self.pfaults
        prev = {
            "mode": int(pfaults.fouling_mode_active_mode),
            "end_t": float(pfaults.fouling_mode_active_end_t),
        }
        pfaults.fouling_mode_active_mode = 0
        pfaults.fouling_mode_active_end_t = -1.0
        self._fouling_stepper.reset()
        return prev

    def get_fouling_mode_state(self) -> dict:
        """Snapshot the windowed fouling state for diagnostics / replay."""
        pfaults = self.pfaults
        return {
            "active_mode": int(pfaults.fouling_mode_active_mode),
            "active_end_t": float(pfaults.fouling_mode_active_end_t),
            "stepper_state": self._fouling_stepper.snapshot(),
        }

    @property
    def done(self) -> bool:
        """True once the configured end time has been reached."""
        return self._i >= self._lt

    @property
    def t(self) -> float:
        """Current sim time (seconds). After :meth:`step`, this is the
        end-of-step time. Initially ``self.settings.ti``. Once the sim
        has reached end-of-time (``done`` is ``True``), this returns
        ``settings.tf`` so callers reading ``sim.t`` after a run
        completes don't see an IndexError.
        """
        if self.done:
            return float(self.settings.tf)
        if self._i == 0:
            return self.settings.ti
        return float(self._t[self._i])

    @property
    def i(self) -> int:
        """Current step index in the pre-allocated time array."""
        return self._i

    # ------------------------------------------------------------------ #
    # Main per-step API
    # ------------------------------------------------------------------ #
    def step(self) -> StepResult:
        """Advance the simulator by one ODE interval.

        Returns
        -------
        StepResult
            A snapshot of the plant at the end of this interval. All
            arrays follow :class:`bdsim.config.StepResult` semantics.

        Raises
        ------
        RuntimeError
            If ``done`` is already True when ``step()`` is called.
        """
        if self.done:
            raise RuntimeError("LiveSimulator: step() called past the configured end time")

        i = self._i
        if i == 0:
            # First step is special: just publish the initial sample.
            result = self._initial_sample()
        else:
            result = self._advance_one_step(i)

        self._i += 1
        return result

    # ------------------------------------------------------------------ #
    # Collect a full trajectory (used by ``run_with``)
    # ------------------------------------------------------------------ #
    def run_to_completion(
        self,
        verbose: bool = False,
        collect_history: bool = False,
    ) -> Results:
        """Drive :meth:`step` until ``done`` and return a :class:`Results`
        object compatible with the legacy :func:`bdsim.simulation.run_with`.

        Trajectory parity with ``run_with`` is verified by
        ``tests/test_live_simulator.py::test_run_with_via_live_matches_batch``.
        Both paths call the same JIT kernels in the same order, so the
        byte-comparison should hold.

        ``collect_history`` defaults to ``False`` because the legacy path
        drops the last point and starts writing at i=1; if you want the
        *full* trajectory including the initial sample (i=0), set this to
        ``True``.
        """
        tic = time.perf_counter()
        if verbose:
            lt = self._lt
            print(f"BDSIM-Python — simulating {lt} time steps ({self.settings.tf / 3600:.1f} h)")

        # Pre-allocate the collection buffers to match the legacy shape:
        # ``lt - 1`` rows (we drop the last sample per the upstream
        # convention).
        lt_out = self._lt - 1
        sv_buf = np.empty((lt_out, self._sv.shape[1]))
        pv_buf = np.empty((lt_out, self.sensor_faults.nsensors))
        uv_buf = np.empty((lt_out, len(self._u0)))
        sp_buf = np.empty((lt_out, self._sp.shape[1]))
        xLend_buf = np.empty((lt_out, 6))
        yLend_buf = np.empty((lt_out, 6))

        # Drive step-by-step. The legacy Results object keeps the initial
        # sample (i=0) and drops the very last iteration (i=lt-1) to
        # match the upstream "no downstream measurement for the final
        # step" convention. Concretely: ``res.sv = sv[:-1, :]`` keeps
        # indices 0..lt-2 → ``lt-1`` rows. We mirror that by collecting
        # every step and slicing off the final row at the end.
        samples: list[StepResult] = []
        while not self.done:
            if verbose and self._i % max(1, (self._lt // 10)) == 0:
                print(f"  ... t = {self._t[self._i]:.0f} s "
                      f"({self._t[self._i] / 3600:.1f} h, "
                      f"{100 * self._i / self._lt:.0f}%)")
            samples.append(self.step())

        # Trim the very last sample to match the legacy shape.
        samples = samples[:-1]

        for j, result in enumerate(samples):
            sv_buf[j, :] = result.sv
            pv_buf[j, :] = result.pv
            uv_buf[j, :] = result.uv
            sp_buf[j, :] = result.sp
            if result.xLend is not None:
                xLend_buf[j, :] = result.xLend
            if result.yLend is not None:
                yLend_buf[j, :] = result.yLend

        t_out = self._t[:self._lt - 1]                     # indices 0..lt-2 (matches legacy sv[:-1])
        runtime = time.perf_counter() - tic
        if verbose:
            print(f"Runtime: {runtime:.2f} s")

        return Results(
            t=t_out,
            uv=uv_buf,
            sv=sv_buf,
            pv=pv_buf,
            sp=sp_buf,
            xLend=xLend_buf,
            yLend=yLend_buf,
            tclean=np.array(self._tclean, dtype=float),
            quality=self._quality_latched[:self._lt - 1, :].copy(),
            disturbances=self._disturbance_track[:self._lt - 1, :].copy(),
            factor=self._factor_history[:self._lt - 1].copy(),
        )

    # ------------------------------------------------------------------ #
    # Private: construction & per-step mechanics
    # ------------------------------------------------------------------ #
    def _setup(self) -> None:
        """Initialise all per-run state. Mirrors lines 298-369 of the
        legacy ``run_with`` exactly. After ``_setup``,
        ``self._i == 0`` and a subsequent ``step()`` publishes the
        initial-condition sample.
        """
        settings = self.settings
        p = self.p
        pfaults = self.pfaults
        sfaults = self.sensor_faults
        vfaults = self.valve_faults
        armax = self.armax
        pid = self.pid

        # --- time vector & derivatives
        self._t = np.arange(settings.ti, settings.tf + settings.dt / 2, settings.dt)
        self._lt = len(self._t)

        # --- filter constants
        K1F, K2F, K3F, K4F = _clogging_kit(p, pfaults)
        p.K1F, p.K2F, p.K3F, p.K4F = K1F, K2F, K3F, K4F
        p.factor = 1.0
        p.k0 = side_reactions(p.k0, pfaults.ratio_robs_r)

        # --- input vector normalisation (matches legacy)
        u0 = settings.u0.copy()
        u0[2] = u0[2] / p.Mm / 3600.0
        armax.unoise_std[2] = armax.unoise_std[2] / p.Mm / 3600.0
        self._u0 = u0
        self._u0_kg_h = settings.u0.copy()                 # cached for control writes

        # --- setpoint profile (matches legacy)
        sp = np.zeros((self._lt, 4))
        sp[:, 0] = settings.live_sp1
        sp[:, 1] = settings.live_sp2
        sp[:, 2] = settings.live_sp3
        sp[:, 3] = settings.live_sp4 + 100.0 * np.heaviside(self._t - 5000.0, 1.0)
        sp[:, 3] = sp[:, 3] / 3600.0
        self._sp = sp

        # --- PID normalisation
        pid.kc = pid.kc.copy()
        pid.kc[3] = pid.kc[3] * 3600.0
        pid.upper_bound = pid.upper_bound.copy()
        pid.lower_bound = pid.lower_bound.copy()

        # --- sensor fault pre-allocation (matches legacy)
        sfaults.b = np.zeros((self._lt, sfaults.nsensors))
        sfaults.b[:, 1] = 5e-6 * (self._t - 100000.0) * np.heaviside(self._t - 100000.0, 1.0)
        sfaults.b[:, 4] = 5e3 * (np.heaviside(self._t - 60000.0, 1.0)
                                - np.heaviside(self._t - 80000.0, 1.0))
        sfaults.a = np.ones((self._lt, sfaults.nsensors))
        sfaults.signal = np.ones((self._lt, sfaults.nsensors))
        sfaults.signal, sfaults.a, sfaults.b = _intermittence(
            self._t, sfaults.signal, sfaults.a, sfaults.b, sfaults
        )

        # --- ARMAX AUTO-mode forcing
        mode = settings.mode_1b.astype(bool)
        self._uindexAUTO = settings.uindex_1b[mode] - 1
        armax.phi[self._uindexAUTO] = 1.0
        armax.theta[self._uindexAUTO] = 0.0
        armax.eta[self._uindexAUTO] = 0.0
        armax.unoise_std[self._uindexAUTO] = 0.0

        # dynamic fouling: HEX fouling α is in sv[21] only when the dynamic
        # path is enabled. Legacy mode keeps the state vector at 21
        # components for byte-identical reproducibility.
        self._use_dynamic_alpha = pfaults.fouling_dynamic
        self._use_quality_state = pfaults.quality_state
        self._use_pump_wear = pfaults.pump_wear
        self._use_valve_wear = pfaults.valve_wear
        # Forward actuator wear kinetics overrides onto ``p`` so the
        # Numba kernel reads resolved values (per-hour → per-second,
        # etc). Pattern matches dynamic fouling / quality latching.
        p.apply_layer24_overrides(pfaults)

        # quality latching: quality state adds 6 components when enabled.
        # (set above alongside actuator wear flags)

        # external disturbances: external disturbance track. Built once at setup,
        # referenced per-step to perturb u[] (Tmet, Toil, Qheat).
        # When all amplitudes are zero, this reduces to identity.
        self._pfaults = pfaults
        settings._pfaults = pfaults
        self._disturbance_track: np.ndarray = settings.disturbances(self._t)
        self._quality_latched = np.zeros((self._lt, 3))
        self._last_lab_sample_t: float = -np.inf
        if pfaults.quality_lag_mode == "online":
            self._lab_period_s = pfaults.online_cycle_s
        else:
            self._lab_period_s = pfaults.lab_cycle_s
        self._pfaults = pfaults                                  # keep for noise levels

        # --- exogenous disturbance
        d = settings.exogenous(self._t)
        d[:, 2] = d[:, 2] / p.Mm / 3600.0
        for ui in self._uindexAUTO:
            d[:, ui] = 0.0
        self._d = d
        armax.unoise = armax.unoise_std * np.random.randn(len(u0))

        # --- PID instances
        pid_dt = settings.dt * settings.nic
        self._pid_states = [
            _PIDState(
                kc=pid.kc[k],
                taui=pid.taui[k],
                taud=pid.taud[k],
                lower=pid.lower_bound[k],
                upper=pid.upper_bound[k],
                dt=pid_dt,
                ioutput=u0[settings.uindex_1b[k] - 1],
            )
            for k in range(4)
        ]
        self._pvindexAUTO = settings.pvindex_1b[mode] - 1

        # --- pre-allocated history
        self._u = u0.copy()
        self._vpos = np.zeros((self._lt, len(vfaults.uindex)))
        self._vpos[0, :] = u0[vfaults.uindex - 1]

        # dynamic fouling: HEX fouling α is in sv[21] only when the dynamic
        # path is enabled. Legacy mode keeps the state vector at 21
        # components for byte-identical reproducibility. quality latching adds
        # 6 more components when enabled.
        sv_width = (
            len(settings.sv0)
            + (1 if self._use_dynamic_alpha else 0)
            + (6 if self._use_quality_state else 0)
            + (1 if self._use_pump_wear else 0)
            + (1 if self._use_valve_wear else 0)
        )
        self._sv = np.zeros((self._lt, sv_width))
        self._sv[0, :len(settings.sv0)] = settings.sv0
        if self._use_dynamic_alpha:
            self._sv[0, 21] = 0.05                              # HEX fouling α at start, lightly fouled
        if self._use_quality_state:
            self._sv[0, 22] = p.fame_eq
            self._sv[0, 23] = p.water_eq
            self._sv[0, 24] = p.iv_eq
            self._sv[0, 25] = p.ffa_ref
            self._sv[0, 26] = 0.01
            self._sv[0, 27] = p.iv_eq
        # actuator wear: continuous-state slots land *after* whatever the
        # legacy / dynamic fouling / quality latching stack produces. ``layer24_base``
        # gives the absolute index of the first actuator wear slot;
        # pump_health lives at base+0, valve_stiction at base+1 (when
        # pump_wear is also enabled) or base+0 (when only valve_wear).
        self._layer24_base = (
            len(settings.sv0)
            + (1 if self._use_dynamic_alpha else 0)
            + (6 if self._use_quality_state else 0)
        )
        if self._use_pump_wear:
            self._sv[0, self._layer24_base + 0] = float(pfaults.pump_health_initial)
        if self._use_valve_wear:
            self._sv[0, self._layer24_base + (1 if self._use_pump_wear else 0)] = float(
                pfaults.valve_stiction_initial_pct
            )
        # Helper getters so the perturbation block can read the right
        # row without recomputing the index every step.
        self._pump_health_idx = (
            self._layer24_base if self._use_pump_wear else None
        )
        self._valve_stiction_idx = (
            self._layer24_base + (1 if self._use_pump_wear else 0)
            if self._use_valve_wear
            else None
        )
        self._sv[0, 18] = settings.sv0[18] * 1e6

        self._xLend = np.zeros((self._lt, 6))
        self._yLend = np.zeros((self._lt, 6))
        self._xLend[0, :], self._yLend[0, :] = AEmodel(self._sv[0, :], p)

        self._uv = np.zeros((self._lt, len(u0)))
        self._uv[0, :] = u0

        self._pv = np.zeros((self._lt, sfaults.nsensors))
        self._pv[0, :] = _measurements(0, self._sv[0, :], self._uv[0, :], p, sfaults,
                                       sfaults.signal, sfaults.a, sfaults.b)
        self._pvAUTO = self._pv[0, self._pvindexAUTO]

        # t=0 quality sample (quality latching) so the dashboard has a value
        # before the first step completes.
        if self._use_quality_state:
            self._quality_latched[0, 0] = self._sv[0, 22] + self._pfaults.lab_noise_fame * np.random.randn()
            self._quality_latched[0, 1] = self._sv[0, 23] + self._pfaults.lab_noise_water * np.random.randn()
            self._quality_latched[0, 2] = self._sv[0, 24] + self._pfaults.lab_noise_iv * np.random.randn()
            self._quality_latched[0, 0] = float(np.clip(self._quality_latched[0, 0], 0.0, 100.0))
            self._quality_latched[0, 1] = float(np.clip(self._quality_latched[0, 1], 0.0, 5000.0))
            self._quality_latched[0, 2] = float(np.clip(self._quality_latched[0, 2], 0.0, 200.0))

        self._tclean: list[float] = []
        self._factor = _fouling(self._t, pfaults.fouling, pfaults.foulingpar)

        # fouling-mode windows: per-step factor recording buffer. Populated by
        # the priority-aware selection block in ``_advance_one_step``
        # so callers / tests can inspect which path was active at
        # each step. Sized ``lt`` and indexed ``[i]`` (the end-of-step
        # factor that the kernel actually applied).
        self._factor_history = np.empty(self._lt, dtype=float)
        # factor at t=0. With dynamic fouling dynamic α the initial α is
        # 0.05 → factor = 1/(1+0.05). Legacy mode: factor[0] = 1
        # (no fouling at t=0). Windowed mode is inactive at t=0 so
        # the legacy value applies.
        if self._use_dynamic_alpha:
            self._factor_history[0] = float(self._sv[0, 21])
        else:
            self._factor_history[0] = float(self._factor[0])

        # fouling-mode windows: windowed five-mode fouling stepper. Lives
        # alongside the pre-baked legacy factor series; the kernel
        # picks one of three paths per step:
        #   1) continuous α  (sv[21] when fouling_dynamic=True)
        #   2) windowed mode 4/5  (this stepper, when active)
        #   3) static legacy factor[i]
        # Priority is ``1 > 2 > 3`` per the fouling-mode windows plan.
        # The ARMAX RNG is seeded from ``pfaults.fouling_mode_active_seed``
        # (or from a fresh default_rng) so that scenario replays are
        # deterministic.
        self._fouling_stepper = FoulingModeStepper(
            foulingpar=pfaults.foulingpar,
            ar_eps_std=pfaults.fouling_ar_eps_std,
            xRG_weight=pfaults.fouling_mode_xRG_weight,
        )
        if pfaults.fouling_mode_active_seed is not None:
            self._fouling_mode_rng = np.random.default_rng(
                int(pfaults.fouling_mode_active_seed)
            )
        else:
            # Process-local default_rng — fresh each sim build, so
            # scenarios that don't pin a seed still get reproducible
            # behaviour for the lifetime of this LiveSimulator instance.
            self._fouling_mode_rng = np.random.default_rng()

        # --- valve stiction state
        self._nvalves = len(vfaults.uindex)
        self._valve_duOLD = np.zeros(self._nvalves)
        self._valve_stp = np.zeros(self._nvalves, dtype=int)
        self._valve_us = u0[vfaults.uindex - 1] + (vfaults.S - vfaults.J) / 2.0
        self._valve_d = -np.ones(self._nvalves, dtype=int)
        self._valve_yOLD = self._vpos[0, :].copy()

        self._unoiseOLD = armax.unoise.copy()
        self._rhs = make_rhs(
            p,
            use_dynamic_alpha=self._use_dynamic_alpha,
            use_quality_state=self._use_quality_state,
            use_pump_wear=self._use_pump_wear,
            use_valve_wear=self._use_valve_wear,
        )

        self._i = 0                                         # current step index

    def _initial_sample(self) -> StepResult:
        """Snapshot at i=0 (initial condition)."""
        # Build quality map from the live fault knobs (currently empty at start).
        quality = self._quality_map()
        return StepResult(
            t=float(self._t[0]),
            pv=self._pv[0, :].copy(),
            uv=self._uv[0, :].copy(),
            sv=self._sv[0, :].copy(),
            sp=self._sp[0, :].copy(),
            quality=quality,
            quality_latched=self._quality_latched[0, :].copy() if self._use_quality_state else None,
            disturbances=self._disturbance_track[0, :].copy(),
            xLend=self._xLend[0, :].copy(),
            yLend=self._yLend[0, :].copy(),
            spectra=self._maybe_sample_spectrum(0, 0),
        )

    def _advance_one_step(self, i: int) -> StepResult:
        """Inner body of the legacy `for i in range(1, lt)` loop.

        Reads from ``self._*`` arrays/objects, mutates them in place.
        Returns the snapshot for row ``i``.
        """
        vfaults = self.valve_faults
        p = self.p
        pfaults = self.pfaults
        sfaults = self.sensor_faults
        armax = self.armax
        settings = self.settings

        # ----------------- inputs update (ARMAX + PID)
        unoise = armax.unoise_std * np.random.randn(len(self._u0))
        u_new = np.empty(len(self._u0))
        _armax_update_jit(self._u, u_new, armax.phi, armax.theta, armax.eta,
                          unoise, self._unoiseOLD, self._d[i - 1, :])
        self._u = u_new
        self._unoiseOLD = unoise

        # external disturbances: external disturbance overlay. Same kernel as
        # the batch path in simulation.py: Tmet tracks CW deviation,
        # Toil tracks ambient deviation, Qheat scales with CW
        # pressure. Skipped entirely when all amplitudes are zero
        # to preserve the legacy byte-identical fingerprint.
        pfaults = self._pfaults
        if (
            pfaults.ambient_t_amplitude_k != 0.0
            or pfaults.cw_t_amplitude_k != 0.0
            or pfaults.cw_p_drift_pa_per_h != 0.0
            or pfaults.cw_p_noise_pa != 0.0
            or pfaults.live_ambient_mean_k is not None
            or pfaults.live_ambient_amplitude_k is not None
            or pfaults.live_cw_t_mean_k is not None
            or pfaults.live_cw_p_drift_pa_per_h is not None
            # actuator wear: pump_wear multiplies cw_p per-step so the
            # perturbation block must run even when the sinusoid /
            # drift / live knobs are all at zero. Symmetric with the
            # simulation.py driver.
            or pfaults.pump_wear
        ):
            amb, cw_t, cw_p = self._disturbance_track[i - 1, :]
            # cooling-water pump trip: apply cw_pump_trip override on top of the
            # baseline CW pressure. The override is single-slot — a
            # second trip replaces the first. No-op when no override
            # is active or when the trip has expired.
            cw_p = self._apply_disturbance_override(
                float(self._t[i - 1]), float(cw_p)
            )
            # actuator wear: multiply cw_p by current pump_health (read
            # from the previous step's state). pump_health ∈ [0, 1];
            # at 1.0 the multiplier is 1.0 and cw_p is unaffected.
            # This is the wear-side effect — the kernel evolves the
            # state slot, the driver applies the multiplier here so
            # the Qheat scaling block sees a worn-pump cw_p.
            if pfaults.pump_wear:
                cw_p = cw_p * float(self._sv[i - 1, self._pump_health_idx])
            # operator disturbance knobs: resolve operator-driven knob overlays so
            # the deviation math uses the same resolved baseline as
            # the track did. Pull once, use locally.
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
            self._u[1] = self._u[1] + (cw_t - cw_mean_resolved) * pfaults.met_cw_track
            self._u[3] = self._u[3] + (amb - amb_mean_resolved) * pfaults.oil_ambient_track
            if pfaults.qheat_cw_scaling and pfaults.cw_p_nominal_pa > 0.0:
                self._u[4] = self._u[4] * (cw_p / pfaults.cw_p_nominal_pa)

        if (i - 1) % settings.nic == 0:
            mode = settings.mode_1b.astype(bool)
            for k, ui in enumerate(self._uindexAUTO):
                meas = self._pvAUTO[k]
                sp_val = self._sp[i, np.where(mode)[0][k]]
                # Use the live setpoint if it differs from the pre-built
                # profile at the current index - this lets ``POST /control``
                # propagate without rebuilding the SP array.
                self._u[ui] = self._pid_states[np.where(mode)[0][k]].step(sp_val, meas)
        self._uv[i, :] = self._u

        # ----------------- valve stiction
        for k in range(self._nvalves):
            ui = vfaults.uindex[k] - 1
            y, du, us, stp, ddir = _stiction_step(
                self._u[ui], self._uv[i - 1, ui], self._valve_yOLD[k], self._valve_duOLD[k],
                self._valve_us[k], self._valve_stp[k], self._valve_d[k],
                vfaults.S[k], vfaults.J[k],
            )
            self._vpos[i, k] = y
            self._valve_duOLD[k] = du
            self._valve_us[k] = us
            self._valve_stp[k] = stp
            self._valve_d[k] = ddir
            self._valve_yOLD[k] = y

        # ----------------- ODE integration
        uu = self._u.copy()
        uu[vfaults.uindex - 1] = self._vpos[i, :]
        self._rhs.set_u(uu)
        # dynamic fouling / fouling-mode windows: factor selection with explicit priority:
        #   1) continuous α (dynamic fouling) — when fouling_dynamic=True
        #   2) windowed mode 4/5 (fouling-mode windows) — when fouling_mode_active
        #   3) static legacy factor[i] — pre-baked series
        # Selection mirrors the simulation.py priority, so the live and
        # batch paths produce identical factor trajectories for any
        # given fault schedule.
        t_now = float(self._t[i - 1])
        mode_active = (
            pfaults.fouling_mode_active_mode in (FoulingMode.ARMAX_NOISE,
                                                 FoulingMode.ARMAX_PURE_NOISE)
            and t_now < float(pfaults.fouling_mode_active_end_t)
        )
        if self._use_dynamic_alpha:
            applied_factor = float(self._sv[i - 1, 21])
            self._rhs.set_factor(applied_factor)
        elif mode_active:
            # xRG is sv[5] (reactor glycerol mole fraction, 0-based).
            # Falls back to 0.0 if the state slot is unavailable
            # (e.g. legacy 21-component state with quality_state=False).
            xrg = float(self._sv[i - 1, 5]) if self._sv.shape[1] > 5 else 0.0
            factor_windowed, _ = self._fouling_stepper.step(
                t=t_now,
                mode=int(pfaults.fouling_mode_active_mode),
                xRG=xrg,
                rng=self._fouling_mode_rng,
            )
            applied_factor = factor_windowed
            self._rhs.set_factor(applied_factor)
        else:
            applied_factor = float(self._factor[i])
            self._rhs.set_factor(applied_factor)
        # fouling-mode windows: record the applied factor so callers can inspect
        # which path (continuous α / windowed / static) the kernel
        # used for this step. Cheap (1 float per step).
        self._factor_history[i] = applied_factor
        sv_init = self._sv[i - 1, :].copy()
        sol = solve_ivp(self._rhs, (self._t[i - 1], self._t[i]), sv_init,
                        method="RK45", rtol=1e-3, atol=1e-6,
                        max_step=settings.dt)
        if sol.success is False:
            self._sv[i, :] = self._sv[i - 1, :]
        else:
            self._sv[i, :] = sol.y[:, -1]

        # dynamic fouling post-integration handling of sv[21] (α) — only in
        # dynamic mode. Legacy mode keeps the 21-component state.
        if self._use_dynamic_alpha:
            self._sv[i, 21] = float(np.clip(self._sv[i, 21], 0.0, 1.0))

        # actuator wear: post-integration clamping on pump_health and
        # valve_stiction_pct. Driver-side enforcement so we don't
        # accumulate numerical drift outside the operating envelope.
        if self._use_pump_wear:
            self._sv[i, self._pump_health_idx] = float(np.clip(
                self._sv[i, self._pump_health_idx],
                pfaults.pump_wear_floor,
                1.0,
            ))
        if self._use_valve_wear:
            self._sv[i, self._valve_stiction_idx] = float(np.clip(
                self._sv[i, self._valve_stiction_idx],
                pfaults.valve_stiction_floor_pct,
                pfaults.valve_stiction_ceiling_pct,
            ))

        # ----------------- filter cleaning
        if self._pv[i - 1, 4] >= pfaults.DPclean:
            self._sv[i, 18] = p.rclean * 1e6
            if self._use_dynamic_alpha:
                self._sv[i, 21] = p.alpha_clean                   # snap α to 0.1
            var = pfaults.filter_std * np.random.randn()
            p.K2F = p.K2F + (4 * p.visco / np.pi) * p.cv**2 * var
            p.K4F = p.K4F + (8 * p.visco / np.pi) * var
            self._tclean.append(self._t[i])

        # ----------------- quality state post-processing (quality latching)
        if self._use_quality_state:
            self._sv[i, 22] = float(np.clip(self._sv[i, 22], 0.0, 100.0))
            self._sv[i, 23] = float(np.clip(self._sv[i, 23], 0.0, 5000.0))
            self._sv[i, 24] = float(np.clip(self._sv[i, 24], 0.0, 200.0))
            self._sv[i, 25] = float(np.clip(self._sv[i, 25], 0.0, 0.5))
            self._sv[i, 26] = float(np.clip(self._sv[i, 26], 0.0, 0.5))
            self._sv[i, 27] = float(np.clip(self._sv[i, 27], 0.0, 200.0))
            if (self._t[i] - self._last_lab_sample_t) >= self._lab_period_s:
                self._quality_latched[i, 0] = self._sv[i, 22] + self._pfaults.lab_noise_fame * np.random.randn()
                self._quality_latched[i, 1] = self._sv[i, 23] + self._pfaults.lab_noise_water * np.random.randn()
                self._quality_latched[i, 2] = self._sv[i, 24] + self._pfaults.lab_noise_iv * np.random.randn()
                self._quality_latched[i, 0] = float(np.clip(self._quality_latched[i, 0], 0.0, 100.0))
                self._quality_latched[i, 1] = float(np.clip(self._quality_latched[i, 1], 0.0, 5000.0))
                self._quality_latched[i, 2] = float(np.clip(self._quality_latched[i, 2], 0.0, 200.0))
                self._last_lab_sample_t = self._t[i]
            else:
                self._quality_latched[i, :] = self._quality_latched[i - 1, :]
        else:
            self._quality_latched[i, :] = np.nan

        # ----------------- measurements (batch path)
        self._xLend[i, :], self._yLend[i, :] = AEmodel(self._sv[i, :], p)
        pv_row = _measurements(i, self._sv[i, :], uu, p, sfaults,
                               sfaults.signal, sfaults.a, sfaults.b)
        self._pv[i, :] = pv_row
        self._pvAUTO = self._pv[i, self._pvindexAUTO]

        # ----------------- LIVE FAULT OVERLAY (Roadmap step 4)
        # Apply bias / dropouts / stuck on top of the batch result. These
        # modifications only flow into the StepResult; the underlying
        # pv array still reflects the batch truth so that
        # ``run_to_completion`` + ``Results`` stays identical to legacy.
        pv_out, quality = self._apply_live_sensor_faults(pv_row, i)
        self._update_last_published(pv_out)

        return StepResult(
            t=float(self._t[i]),
            pv=pv_out,
            uv=self._uv[i, :].copy(),
            sv=self._sv[i, :].copy(),
            sp=self._sp[i, :].copy(),
            quality=quality,
            quality_latched=self._quality_latched[i, :].copy() if self._use_quality_state else None,
            # cooling-water pump trip: reflect any active cw_pump_trip override in
            # the published disturbance column. The kernel above
            # already applied the override to ``self._u[4]`` for the
            # ODE step; here we update the published snapshot so the
            # dashboard / live consumer sees the actual CW pressure
            # rather than the unperturbed baseline track. The
            # ``self._disturbance_track`` is left untouched so a
            # future run (or a reset) starts from a clean baseline.
            disturbances=self._published_disturbances(i),
            xLend=self._xLend[i, :].copy(),
            yLend=self._yLend[i, :].copy(),
            spectra=self._maybe_sample_spectrum(i, self._t[i]),
        )

    # ------------------------------------------------------------------ #
    # Live fault overlay
    # ------------------------------------------------------------------ #
    def _published_disturbances(self, i: int) -> np.ndarray:
        """Return the disturbance snapshot for row ``i`` with any
        active cw_pump_trip override applied to channel 2 (PCW).

        Baseline channels (Tamb, Tcw, PCW) come straight from
        ``self._disturbance_track[i]``. The PCW channel is overridden
        when the trip is active so the dashboard / live consumer
        sees the actual pressure the kernel used, not the unperturbed
        baseline.

        Returns a fresh copy so callers can't mutate the track.
        """
        row = self._disturbance_track[i, :].copy()
        if self._disturbance_override is not None and self._disturbance_override.get("channel") == "pcw":
            # The kernel already applied this same factor on
            # ``self._u[4]``; here we mirror it on the published
            # snapshot so the surface matches the underlying state.
            t_here = float(self._t[i])
            row[2] = float(self._apply_disturbance_override(t_here, float(row[2])))
        # actuator wear: pump_health multiplies the published PCW so
        # the dashboard surface reflects what the kernel saw. The
        # kernel applies the same factor on ``u[4]`` in the
        # perturbation block; we mirror it here so the snapshot
        # matches. Without this mirror, the dashboard would show
        # the unperturbed baseline.
        if self.pfaults.pump_wear and self._pump_health_idx is not None:
            row[2] = float(row[2]) * float(self._sv[i, self._pump_health_idx])
        return row

    def _apply_disturbance_override(self, t: float, cw_p_baseline: float) -> float:
        """Apply a cw_pump_trip envelope on top of the baseline CW pressure.

        Returns the override-applied CW pressure (Pa). When no override
        is active (or the trip has expired) the baseline is returned
        unchanged. The envelope is ramp-down / hold / ramp-up:

            t < start_t              → 1.00
            start_t <= t < end_t     → linear in/out, held at low_factor
            t >= end_t               → 1.00

        ``low_factor`` is the floor (0.3 = 70% drop). ``ramp_s`` is the
        ramp duration at both edges. The function is pure: same input
        always returns the same output, no side effects.

        cooling-water pump trip. See bdsim.live_simulator.LiveSimulator._apply_disturbance_override.
        """
        ovr = self._disturbance_override
        if ovr is None or "channel" not in ovr or ovr["channel"] != "pcw":
            return cw_p_baseline
        start_t = float(ovr["start_t"])
        end_t = float(ovr["end_t"])
        low_factor = float(ovr.get("low_factor", 0.3))
        ramp_s = float(ovr.get("ramp_s", 30.0))
        if t < start_t or t >= end_t:
            return cw_p_baseline
        # Ramp-down: start_t → start_t + ramp_s
        if t < start_t + ramp_s:
            r = (t - start_t) / ramp_s
            factor = 1.0 - (1.0 - low_factor) * r
        # Hold: start_t + ramp_s → end_t - ramp_s
        elif t < end_t - ramp_s:
            factor = low_factor
        # Ramp-up: end_t - ramp_s → end_t
        else:
            r = (t - (end_t - ramp_s)) / ramp_s
            factor = low_factor + (1.0 - low_factor) * r
        return cw_p_baseline * factor

    def _apply_live_sensor_faults(
        self, pv_row: np.ndarray, i: int,
    ) -> tuple[np.ndarray, dict[int, str]]:
        """Apply bias / dropout / stuck on top of the batched pv row.

        Returns (pv_with_faults, quality_map).
        """
        sfaults = self.sensor_faults
        out = pv_row.copy()
        quality: dict[int, str] = {}

        # Initialise all sensors as "good" by default.
        for k in range(sfaults.nsensors):
            quality[k] = "good"

        # Dropout: zero out the sensor and flag as "bad".
        for k in sfaults.dropouts:
            if 0 <= k < sfaults.nsensors:
                out[k] = np.nan          # NaN is more honest than 0 (operators expect this)
                quality[k] = "bad"

        # Bias: additive offset on the published value. Quality stays "good"
        # (a biased sensor still publishes reasonable numbers - operators
        # detect bias via correlation, not the value itself).
        for k, b in sfaults.bias.items():
            if 0 <= k < sfaults.nsensors:
                out[k] = out[k] + b
                quality[k] = "uncertain"

        # Stuck: hold the *last published* value (Q1 decision). The PID
        # therefore reacts to a frozen measurement and the closed-loop
        # response diverges from the un-faulted trajectory.
        for k in sfaults.stuck:
            if 0 <= k < sfaults.nsensors and k in self._last_published:
                out[k] = self._last_published[k]
                quality[k] = "uncertain"

        return out, quality

    def _update_last_published(self, pv_row: np.ndarray) -> None:
        """Cache the (post-fault) pv row so a future ``stuck`` event can
        freeze at this exact value.
        """
        for k in range(len(pv_row)):
            v = pv_row[k]
            if np.isfinite(v):                       # NaN sensors don't update the cache
                self._last_published[k] = float(v)

    def _quality_map(self) -> dict[int, str]:
        """Build a 'good' quality map (used for the initial-condition sample
        before any fault is applied)."""
        return {k: "good" for k in range(self.sensor_faults.nsensors)}

    # ------------------------------------------------------------------ #
    # NIR/IR spectrum sensor: spectrum sampler (post-process, fired per spctr_t)
    # ------------------------------------------------------------------ #
    def _maybe_sample_spectrum(self, i: int, t: float):
        """Fire the spectrum sensor at step ``i`` if cadence has elapsed.

        Returns ``None`` when:
        - the spectrum sensor is disabled (``pfaults.spectrum_enabled=False``)
        - or ``spctr_t`` has not elapsed since the previous fire.

        Caller must attach the result to ``StepResult.spectra``
        verbatim — it is already None-when-disabled and None-when-
        not-firing.
        """
        gen = self._spectrum_generator
        if gen is None:
            return None
        # Reactor composition is sv[0:6]; light phase is sv[7:13];
        # heavy phase is sv[13:16] (3-species).
        sv_row = self._sv[i, :]
        x_reactor = sv_row[0:6]
        x_light = sv_row[7:13]
        x_heavy = sv_row[13:16]
        return gen.maybe_sample(t=float(t), step_idx=i,
                                x_reactor=x_reactor, x_light=x_light,
                                x_heavy=x_heavy)

    # ------------------------------------------------------------------ #
    # Context manager (Q2 approved: yes)
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "LiveSimulator":
        return self

    def __exit__(self, *exc: Any) -> None:
        # No resources to release; reset to a clean state for next use.
        self.reset()
