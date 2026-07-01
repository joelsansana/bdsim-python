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

2. **Lepanto FDE integration.** The bdsim-dashboard wants to drive a
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

import logging
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

logger = logging.getLogger(__name__)


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

    @property
    def done(self) -> bool:
        """True once the configured end time has been reached."""
        return self._i >= self._lt

    @property
    def t(self) -> float:
        """Current sim time (seconds). After :meth:`step`, this is the
        end-of-step time. Initially ``self.settings.ti``.
        """
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

        # Layer 2.5: HEX fouling α is in sv[21] only when the dynamic
        # path is enabled. Legacy mode keeps the state vector at 21
        # components for byte-identical reproducibility.
        self._use_dynamic_alpha = pfaults.fouling_dynamic

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

        # Layer 2.5: HEX fouling α is in sv[21] only when the dynamic
        # path is enabled. Legacy mode keeps the state vector at 21
        # components for byte-identical reproducibility.
        self._sv = np.zeros((self._lt, len(settings.sv0) + (1 if self._use_dynamic_alpha else 0)))
        self._sv[0, :len(settings.sv0)] = settings.sv0
        if self._use_dynamic_alpha:
            self._sv[0, 21] = 0.05                              # HEX fouling α at start, lightly fouled
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

        self._tclean: list[float] = []
        self._factor = _fouling(self._t, pfaults.fouling, pfaults.foulingpar)

        # --- valve stiction state
        self._nvalves = len(vfaults.uindex)
        self._valve_duOLD = np.zeros(self._nvalves)
        self._valve_stp = np.zeros(self._nvalves, dtype=int)
        self._valve_us = u0[vfaults.uindex - 1] + (vfaults.S - vfaults.J) / 2.0
        self._valve_d = -np.ones(self._nvalves, dtype=int)
        self._valve_yOLD = self._vpos[0, :].copy()

        self._unoiseOLD = armax.unoise.copy()
        self._rhs = make_rhs(p, use_dynamic_alpha=self._use_dynamic_alpha)

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
            xLend=self._xLend[0, :].copy(),
            yLend=self._yLend[0, :].copy(),
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
        # Layer 2.5: factor selection — same convention as simulation.py.
        if self._use_dynamic_alpha:
            self._rhs.set_factor(self._sv[i - 1, 21])
        else:
            self._rhs.set_factor(self._factor[i])
        sv_init = self._sv[i - 1, :].copy()
        sol = solve_ivp(self._rhs, (self._t[i - 1], self._t[i]), sv_init,
                        method="RK45", rtol=1e-3, atol=1e-6,
                        max_step=settings.dt)
        if sol.success is False:
            self._sv[i, :] = self._sv[i - 1, :]
        else:
            self._sv[i, :] = sol.y[:, -1]

        # Layer 2.5 post-integration handling of sv[21] (α) — only in
        # dynamic mode. Legacy mode keeps the 21-component state.
        if self._use_dynamic_alpha:
            self._sv[i, 21] = float(np.clip(self._sv[i, 21], 0.0, 1.0))

        # ----------------- filter cleaning
        if self._pv[i - 1, 4] >= pfaults.DPclean:
            self._sv[i, 18] = p.rclean * 1e6
            if self._use_dynamic_alpha:
                self._sv[i, 21] = p.alpha_clean                   # snap α to 0.1
            var = pfaults.filter_std * np.random.randn()
            p.K2F = p.K2F + (4 * p.visco / np.pi) * p.cv**2 * var
            p.K4F = p.K4F + (8 * p.visco / np.pi) * var
            self._tclean.append(self._t[i])

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
            xLend=self._xLend[i, :].copy(),
            yLend=self._yLend[i, :].copy(),
        )

    # ------------------------------------------------------------------ #
    # Live fault overlay
    # ------------------------------------------------------------------ #
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
    # Context manager (Q2 approved: yes)
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "LiveSimulator":
        return self

    def __exit__(self, *exc: Any) -> None:
        # No resources to release; reset to a clean state for next use.
        self.reset()
