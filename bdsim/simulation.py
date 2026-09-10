"""
Main simulation driver (replaces ``BDsim.m``).

Loop structure mirrors the MATLAB upstream:
1. Build filter constants from process faults
2. Initialise ARMAX noise, controller state, sensor-fault arrays
3. For each time step:
   a. Update inputs (ARMAX disturbance + PID if AUTO mode)
   b. Apply valve stiction to manipulated variables
   c. Integrate ODE on ``[t(i-1), t(i)]`` with ``solve_ivp``
   d. Clean filter if ``DPfilter >= DPclean``
   e. Compute measurements with sensor faults

Differences from the MATLAB upstream:
- Single integration call per interval (``solve_ivp`` with ``RK45``,
  matching ``ode45`` defaults). Upstream used ``lsode`` in Octave (LSODA)
  and ``ode45`` in MATLAB; results agree to within solver tolerance.
- Sensor-fault arrays are pre-allocated on startup instead of growing in
  the loop.
- All persistent-state MATLAB functions (``pid_controller``, ``valves``,
  ``inputs``) are replaced by explicit state objects on the driver.
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

from ._step_helpers import (
    apply_controller_step,
    apply_disturbances,
    select_fouling_factor,
)
from .config import (
    ARMAX,
    Parameters,
    PIDController,
    ProcessFaults,
    Results,
    SensorFaults,
    Settings,
    ValveFaults,
)
from .fouling_modes import FoulingModeStepper
from .ode import AEmodel, _qoil_jit, make_rhs
from .thermo import side_reactions

# -----------------------------------------------------------------------------
# Filter constants (clogging_kit.m)
# -----------------------------------------------------------------------------

def _clogging_kit(p: Parameters, pfaults: ProcessFaults) -> tuple[float, float, float, float]:
    """Compute the four filter constants from geometry and fault severity."""
    K1F = 1e12 * pfaults.clog_fraction / (2 * np.pi * p.zF * p.np_)
    K2F = 1e24 * 4 * p.visco * p.zF / (p.np_ * np.pi) * p.cv**2
    K3F = p.Ppump * p.cv**2
    K4F = 1e24 * 8 * p.visco * p.zF / (p.np_ * np.pi)
    return K1F, K2F, K3F, K4F


# -----------------------------------------------------------------------------
# Fouling (fouling.m)
# -----------------------------------------------------------------------------

def _fouling(t: np.ndarray, fouling: int, foulingpar: np.ndarray) -> np.ndarray:
    Rfouling = (fouling != 0) * foulingpar[0] * t
    return 1.0 / (1.0 + Rfouling)


# -----------------------------------------------------------------------------
# Sensor intermittence (intermittence.m)
# -----------------------------------------------------------------------------

def _intermittence(
    t: np.ndarray,
    signal: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    sfaults: SensorFaults,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand ``sfaults`` to per-timestep arrays (matches upstream behaviour)."""
    ti = t[0]
    tf = t[-1]
    dt = t[1] - t[0]

    for k in range(sfaults.nsensors):
        if not sfaults.isIntermit[k]:
            continue
        tt = ti
        while tt < tf:
            tnew = tt + sfaults.tmaxInterm[k] * np.random.rand()
            tnew = min(tnew - (tnew % dt), tf)
            if np.random.rand() < 0.5:
                ind1 = round((tt - ti) / dt + 1)
                ind2 = round((tnew - ti) / dt + 1)
                signal[ind1:ind2 + 1, k] = 1.0
                a[ind1:ind2 + 1, k] = 1.0
                b[ind1:ind2 + 1, k] = 0.0
            tt = tnew
    return signal, a, b


# -----------------------------------------------------------------------------
# PID controller (pid_controller.m) — JIT-compiled inner step
# -----------------------------------------------------------------------------

@njit(cache=True)
def _pid_step_jit(setpoint: float, measure: float,
                  kc: float, taui: float, taud: float,
                  lower: float, upper: float, dt: float,
                  error1: float, error2: float, output: float,
                  out_state: np.ndarray) -> float:
    """One PID step. Mutates the persistent state in ``out_state``:
    ``[error1, error2, output]``. Returns the new output value.
    """
    erro = setpoint - measure
    out = output + kc * (
        erro - error1
        + dt / taui * erro
        + (erro - 2.0 * error1 + error2) * taud / dt
    )
    if out < lower:
        out = lower
    elif out > upper:
        out = upper
    out_state[0] = erro
    out_state[1] = error1
    out_state[2] = out
    return out


class _PIDState:
    """Per-loop PID controller. Uses the JIT kernel for the inner step."""

    def __init__(self, kc: float, taui: float, taud: float,
                 lower: float, upper: float, dt: float,
                 ioutput: float, ierror1: float = 0.0, ierror2: float = 0.0):
        self.kc = kc
        self.taui = taui
        self.taud = taud
        self.lower = lower
        self.upper = upper
        self.dt = dt
        self._state = np.array([ierror1, ierror2, ioutput], dtype=np.float64)

    def step(self, setpoint: float, measure: float) -> float:
        return _pid_step_jit(
            setpoint, measure, self.kc, self.taui, self.taud,
            self.lower, self.upper, self.dt,
            self._state[0], self._state[1], self._state[2],
            self._state,
        )


# -----------------------------------------------------------------------------
# Valve stiction (stiction.m + valves.m) — JIT kernel
# -----------------------------------------------------------------------------

@njit(cache=True)
def _stiction_step_jit(u: float, u_old: float, du_old: float, y_old: float,
                       us: float, stp: int, d_dir: int,
                       S: float, J: float,
                       out_state: np.ndarray) -> None:
    """One evaluation of the Kano stiction model.

    Mutates ``out_state[:] = [y, du, us, stp, d_dir]``.
    """
    if u < 0.0:
        u = 0.0
    elif u > 100.0:
        u = 100.0

    du = u - u_old

    # The upstream "valve just stopped" branch: when direction flips AND
    # valve was moving (stp==0), snapshot us = u_old and set stp = 1.
    if du * du_old <= 0.0 and stp == 0:
        us = u_old
        stp = 1

    y = y_old
    if stp == 0:
        y = u - d_dir / 2.0 * (S - J)
    elif -d_dir * (u - us) > S:
        d_dir = -d_dir
        y = u - d_dir / 2.0 * (S - J)
        stp = 0
    elif d_dir * (u - us) > J:
        y = u - d_dir / 2.0 * (S - J)
        stp = 0
    else:
        y = y_old

    out_state[0] = y
    out_state[1] = du
    out_state[2] = us
    out_state[3] = stp
    out_state[4] = d_dir


def _stiction_step(u: float, u_old: float, y_old: float, du_old: float,
                   us: float, stp: int, d_dir: int,
                   S: float, J: float) -> tuple[float, float, float, int, int]:
    """One evaluation of the Kano stiction model (thin Python wrapper)."""
    out = np.zeros(5)
    _stiction_step_jit(u, u_old, du_old, y_old, us, stp, d_dir, S, J, out)
    return out[0], out[1], out[2], int(out[3]), int(out[4])


# -----------------------------------------------------------------------------
# Measurements with sensor faults (measurements.m)
# -----------------------------------------------------------------------------

@njit(cache=True)
def _measurements_jit(sv: np.ndarray, uu: np.ndarray,
                      K2F: float, K3F: float, K4F: float, roo: float,
                      signal_row: np.ndarray, a_row: np.ndarray,
                      b_row: np.ndarray, noise_std: np.ndarray,
                      out: np.ndarray, randn_buf: np.ndarray) -> None:
    """JIT-compiled measurement computation.

    ``out`` and ``randn_buf`` must be pre-allocated by the caller (length
    ``nsensors``). Random numbers are pulled from ``randn_buf`` to keep the
    kernel deterministic given a seeded RNG on the Python side.
    """
    Qo = _qoil_jit(sv[18], uu[0] / 100.0, K2F, K3F)
    pv = out
    pv[0] = sv[6]
    pv[1] = sv[17]
    pv[2] = sv[16]
    pv[3] = Qo * roo
    r4 = sv[18] * sv[18] * sv[18] * sv[18]
    pv[4] = K4F * Qo / r4

    for k in range(pv.shape[0]):
        pv[k] = signal_row[k] * (
            a_row[k] * pv[k] + b_row[k] + noise_std[k] * randn_buf[k]
        )


def _measurements(i: int, sv: np.ndarray, uu: np.ndarray,
                  p: Parameters, sfaults: SensorFaults,
                  signal: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the 5 sensor measurements with fault injection (JIT)."""
    pv = np.empty(sfaults.nsensors)
    randn_buf = np.random.randn(sfaults.nsensors)
    _measurements_jit(
        sv, uu, p.K2F, p.K3F, p.K4F, p.roo,
        signal[i, :], a[i, :], b[i, :],
        sfaults.noise_std,
        pv, randn_buf,
    )
    return pv


# -----------------------------------------------------------------------------
# ARMAX disturbance model (JIT kernel for the per-step update)
# -----------------------------------------------------------------------------

@njit(cache=True)
def _armax_update_jit(u: np.ndarray, u_out: np.ndarray,
                      phi: np.ndarray, theta: np.ndarray,
                      eta: np.ndarray,
                      unoise: np.ndarray, unoiseOLD: np.ndarray,
                      d_prev: np.ndarray) -> None:
    """One ARMAX(1,1,1) update step: ``u = phi*u + theta*noiseOLD + noise + eta*d``."""
    for k in range(u.shape[0]):
        u_out[k] = phi[k] * u[k] + theta[k] * unoiseOLD[k] + unoise[k] + eta[k] * d_prev[k]

def run(seed: int | None = None, verbose: bool = True) -> Results:
    """Run the closed-loop simulation with the upstream default settings.

    Parameters
    ----------
    seed : int, optional
        RNG seed for reproducibility. If None, no seed is set.
    verbose : bool, default True
        Print progress messages (matches upstream ``display()`` calls).

    Returns
    -------
    Results
        Trajectories ready for plotting or CSV export.
    """
    return run_with(seed=seed, verbose=verbose)


def run_with(
    p: Parameters | None = None,
    pfaults: ProcessFaults | None = None,
    sfaults: SensorFaults | None = None,
    vfaults: ValveFaults | None = None,
    armax: ARMAX | None = None,
    pid: PIDController | None = None,
    settings: Settings | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> Results:
    """Run the simulation with explicit overrides for every config block."""
    if seed is not None:
        np.random.seed(seed)

    p = p or Parameters()
    p.finalize()
    pfaults = pfaults or ProcessFaults()
    sfaults = sfaults or SensorFaults()
    vfaults = vfaults or ValveFaults()
    armax = armax or ARMAX()
    pid = pid or PIDController()
    settings = settings or Settings()

    tic = time.perf_counter()

    # ---------------------------------------------------------------- time
    t = np.arange(settings.ti, settings.tf + settings.dt / 2, settings.dt)
    lt = len(t)

    # ----- external disturbances: external disturbance track (lt x 3)
    # Bind pfaults into settings so Settings.disturbances(t) can read
    # the knobs without callers having to thread pfaults through.
    settings._pfaults = pfaults
    disturbance_track = settings.disturbances(t)

    # ---------------------------------------------------- filter constants
    K1F, K2F, K3F, K4F = _clogging_kit(p, pfaults)
    p.K1F, p.K2F, p.K3F, p.K4F = K1F, K2F, K3F, K4F
    p.factor = 1.0                                            # updated below per-step

    # Side reactions
    p.k0 = side_reactions(p.k0, pfaults.ratio_robs_r)

    # --------------------------------------------------- input vector prep
    u0 = settings.u0.copy()
    u0[2] = u0[2] / p.Mm / 3600.0                             # kg/h → mol/s
    armax.unoise_std[2] = armax.unoise_std[2] / p.Mm / 3600.0

    sp = np.zeros((lt, 4))
    sp[:, 0] = settings.sp1
    sp[:, 1] = settings.sp2
    sp[:, 2] = settings.sp3
    sp[:, 3] = settings.sp4 + 100.0 * np.heaviside(t - 5000.0, 1.0)

    sp[:, 3] = sp[:, 3] / 3600.0                               # kg/h → kg/s
    pid.kc = pid.kc.copy()
    pid.kc[3] = pid.kc[3] * 3600.0                            # %/(kg/h) → %/(kg/s)
    pid.upper_bound = pid.upper_bound.copy()
    pid.upper_bound[1] = pid.upper_bound[1] / 1.0              # (no-op, matches upstream)
    pid.lower_bound = pid.lower_bound.copy()

    sfaults.b = np.zeros((lt, sfaults.nsensors))
    sfaults.b[:, 1] = 5e-6 * (t - 100000.0) * np.heaviside(t - 100000.0, 1.0)
    sfaults.b[:, 4] = 5e3 * (np.heaviside(t - 60000.0, 1.0)
                             - np.heaviside(t - 80000.0, 1.0))
    sfaults.a = np.ones((lt, sfaults.nsensors))
    sfaults.signal = np.ones((lt, sfaults.nsensors))

    sfaults.signal, sfaults.a, sfaults.b = _intermittence(
        t, sfaults.signal, sfaults.a, sfaults.b, sfaults
    )

    # Force ARMAX coefficients for manipulated variables to zero in AUTO mode
    mode = settings.mode_1b.astype(bool)
    uindexAUTO = settings.uindex_1b[mode] - 1                  # convert to 0-based
    armax.phi[uindexAUTO] = 1.0
    armax.theta[uindexAUTO] = 0.0
    armax.eta[uindexAUTO] = 0.0
    armax.unoise_std[uindexAUTO] = 0.0

    # Exogenous disturbance profile
    d = settings.exogenous(t)
    d[:, 2] = d[:, 2] / p.Mm / 3600.0                         # kg/h → mol/s
    # Upstream zeros the disturbance rows that are manipulated in AUTO
    for ui in uindexAUTO:
        d[:, ui] = 0.0

    armax.unoise = armax.unoise_std * np.random.randn(len(u0))

    # -------------------------------------------------------- PID instances
    pid_dt = settings.dt * settings.nic
    pid_states = [
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
    pvindexAUTO = settings.pvindex_1b[mode] - 1

    # ----------------------------------------------------- pre-allocate
    u = u0.copy()
    vpos = np.zeros((lt, len(vfaults.uindex)))
    vpos[0, :] = u0[vfaults.uindex - 1]

    # dynamic fouling: HEX fouling α is in sv[21] only when the dynamic path
    # is enabled. Legacy mode keeps the state vector at 21 components
    # for byte-identical reproducibility vs. the upstream baseline.
    use_dynamic_alpha = pfaults.fouling_dynamic

    # quality latching: quality state adds 6 components when enabled
    # (sv[22:28]). When False, state vector stops at 22 (dynamic fouling only).
    use_quality_state = pfaults.quality_state

    # actuator wear: pump and valve degradation each add one continuous
    # state slot. Independent of dynamic fouling / quality latching — demos can
    # enable either, both, or neither. The state-vector width grows
    # by 1 or 2 as appropriate.
    use_pump_wear = pfaults.pump_wear
    use_valve_wear = pfaults.valve_wear

    # actuator wear: forward the kinetics overrides onto ``p`` so the
    # Numba kernel reads resolved values. Pattern matches dynamic fouling.
    p.apply_layer24_overrides(pfaults)

    # State vector width: 21 (legacy), 22 (+ dynamic fouling), 28 (+ quality latching),
    # +1 if pump_wear (sv[22] in legacy mode, sv[28] in quality mode),
    # +1 if valve_wear (sv[23] in legacy mode, sv[29] in quality mode).
    sv_width = (
        len(settings.sv0)
        + (1 if use_dynamic_alpha else 0)
        + (6 if use_quality_state else 0)
        + (1 if use_pump_wear else 0)
        + (1 if use_valve_wear else 0)
    )

    sv = np.zeros((lt, sv_width))
    sv[0, :len(settings.sv0)] = settings.sv0
    if use_dynamic_alpha:
        sv[0, 21] = 0.05                                       # HEX fouling α at start, lightly fouled
    if use_quality_state:
        # Initialise true instantaneous quality at the equilibrium
        # targets. This is the "perfect online analyser" reading;
        # the lab-cycle latched values are computed separately.
        sv[0, 22] = p.fame_eq                                  # FAME%
        sv[0, 23] = p.water_eq                                 # water, ppm
        sv[0, 24] = p.iv_eq                                    # IV
        sv[0, 25] = p.ffa_ref                                  # FFA in feed (mass fraction)
        sv[0, 26] = 0.01                                       # water in feed (1% by mass, typical UCO)
        sv[0, 27] = p.iv_eq                                    # IV in feed
    # actuator wear: continuous-state slots live *after* whatever the
    # legacy / dynamic fouling / quality latching stack produces. We compute the
    # base index so the initial values land on the right row whether
    # quality mode is on or off.
    layer24_base = (
        len(settings.sv0)
        + (1 if use_dynamic_alpha else 0)
        + (6 if use_quality_state else 0)
    )
    if use_pump_wear:
        sv[0, layer24_base + 0] = float(pfaults.pump_health_initial)
    if use_valve_wear:
        sv[0, layer24_base + (1 if use_pump_wear else 0)] = float(
            pfaults.valve_stiction_initial_pct
        )
    sv[0, 18] = settings.sv0[18] * 1e6                         # μm for numerical stability

    xLend = np.zeros((lt, 6))
    yLend = np.zeros((lt, 6))
    xLend[0, :], yLend[0, :] = AEmodel(sv[0, :], p)

    uv = np.zeros((lt, len(u0)))
    uv[0, :] = u0
    pv = np.zeros((lt, sfaults.nsensors))
    pv[0, :] = _measurements(0, sv[0, :], uv[0, :], p, sfaults,
                              sfaults.signal, sfaults.a, sfaults.b)
    pvAUTO = pv[0, pvindexAUTO]

    tclean = []

    # ---- quality latching: lab-cycle latching ------------------------------
    # quality_latched[i, :] holds the most-recent lab sample for
    # (FAME, water, IV) at sim step i. The latched value stays
    # constant between lab cycles — this is the time-lag structure
    # between OT (instantaneous) and lab (sampled) measurements.
    # Initial sample at t=0 so the dashboard
    # has something to show on startup.
    quality_latched = np.zeros((lt, 3))
    last_lab_sample_t: float = -np.inf                       # force a sample at t=0
    # Take the t=0 sample synchronously so the dashboard has a value
    # before the first integration step completes.
    if use_quality_state:
        quality_latched[0, 0] = sv[0, 22] + pfaults.lab_noise_fame * np.random.randn()
        quality_latched[0, 1] = sv[0, 23] + pfaults.lab_noise_water * np.random.randn()
        quality_latched[0, 2] = sv[0, 24] + pfaults.lab_noise_iv * np.random.randn()
        quality_latched[0, 0] = float(np.clip(quality_latched[0, 0], 0.0, 100.0))
        quality_latched[0, 1] = float(np.clip(quality_latched[0, 1], 0.0, 5000.0))
        quality_latched[0, 2] = float(np.clip(quality_latched[0, 2], 0.0, 200.0))
    # Lab cycle period: lab mode uses lab_cycle_s (default 15 min),
    # online mode uses online_cycle_s (default 60 s).
    if pfaults.quality_lag_mode == "online":
        lab_period_s = pfaults.online_cycle_s
    else:
        lab_period_s = pfaults.lab_cycle_s
    factor = _fouling(t, pfaults.fouling, pfaults.foulingpar)

    # fouling-mode windows: windowed five-mode fouling stepper. The
    # batch driver now mirrors the LiveSimulator priority system
    # (issue #8): when ``pfaults.fouling_mode_active_mode`` is set to
    # 4 or 5 and the current sim time is before
    # ``fouling_mode_active_end_t``, the stepper's ``factor`` overrides
    # the static series for that step. The stepper is allocated
    # eagerly so the kernel can take the same code path as the live
    # driver; when the windowed condition is never satisfied (the
    # default), the stepper is unused and the trajectory is byte-
    # identical to the pre-#8 batch behaviour.
    fouling_stepper = FoulingModeStepper(
        foulingpar=pfaults.foulingpar,
        ar_eps_std=pfaults.fouling_ar_eps_std,
        xRG_weight=pfaults.fouling_mode_xRG_weight,
    )
    if pfaults.fouling_mode_active_seed is not None:
        fouling_mode_rng = np.random.default_rng(
            int(pfaults.fouling_mode_active_seed)
        )
    else:
        fouling_mode_rng = np.random.default_rng()
    # factor_applied: per-step recording of which path the kernel
    # actually used (continuous α / windowed mode 4-5 / static
    # legacy). Mirrors LiveSimulator._factor_history and is what
    # callers see as ``Results.factor``.
    factor_applied = np.empty(lt, dtype=float)
    # factor at t=0. With dynamic fouling the initial α is 0.05 →
    # factor = 1/(1+0.05). Legacy mode: factor[0] = 1
    # (no fouling at t=0). Windowed mode is inactive at t=0 so the
    # priority system collapses to (dynamic α) OR (static legacy).
    if use_dynamic_alpha:
        factor_applied[0] = float(sv[0, 21])
    else:
        factor_applied[0] = float(factor[0])

    # Valve stiction state
    nvalves = len(vfaults.uindex)
    valve_duOLD = np.zeros(nvalves)
    valve_stp = np.zeros(nvalves, dtype=int)
    valve_us = u0[vfaults.uindex - 1] + (vfaults.S - vfaults.J) / 2.0
    valve_d = -np.ones(nvalves, dtype=int)
    valve_yOLD = vpos[0, :].copy()

    unoiseOLD = armax.unoise.copy()

    rhs = make_rhs(
        p,
        use_dynamic_alpha=use_dynamic_alpha,
        use_quality_state=use_quality_state,
        use_pump_wear=use_pump_wear,
        use_valve_wear=use_valve_wear,
    )

    # ============================================================ main loop
    if verbose:
        print(f"BDSIM-Python — simulating {lt} time steps ({settings.tf / 3600:.1f} h)")
        progress_every = max(1, lt // 10)

    for i in range(1, lt):
        if verbose and i % progress_every == 0:
            print(f"  ... t = {t[i]:.0f} s ({t[i] / 3600:.1f} h, {100 * i / lt:.0f}%)")

        # ----------------- inputs update (ARMAX + PID)
        unoise = armax.unoise_std * np.random.randn(len(u0))
        u_new = np.empty(len(u0))
        _armax_update_jit(u, u_new, armax.phi, armax.theta, armax.eta,
                          unoise, unoiseOLD, d[i - 1, :])
        u = u_new
        unoiseOLD = unoise

        # ----------------- external disturbances + operator knobs: external disturbance overlay
        # Apply the perturbation kernel: shifts to u[1] (Tmet), u[3]
        # (Toil), and a multiplicative scale on u[4] (Qheat). When
        # all amplitudes are zero (default) we skip the kernel
        # entirely to preserve the legacy byte-identical fingerprint
        # (FP operation ordering matters: even an identity u *= 1.0
        # introduces last-bit drift after Numba-JIT).
        #
        # operator disturbance knobs: operator knob overlays (live_* fields) take
        # effect here too — the deviation-from-baseline terms must
        # use the same baseline as the track used to generate
        # ``amb / cw_t / cw_p``, so we resolve the knobs once at the
        # top of the conditional. When any knob is non-zero (either
        # baseline profile or live overlay) the kernel runs.
        apply_disturbances(
            t_now=float(t[i - 1]),
            i=i,
            u=u,
            sv=sv,
            pfaults=pfaults,
            disturbance_track=disturbance_track,
            use_pump_wear=use_pump_wear,
            pump_health_idx=layer24_base + 0,
            apply_override=None,
        )

        apply_controller_step(
            i=i,
            u=u,
            pvAUTO=pvAUTO,
            sp=sp,
            mode_1b=settings.mode_1b,
            uindexAUTO=uindexAUTO,
            pid_states=pid_states,
            nic=settings.nic,
        )
        uv[i, :] = u

        # ----------------- valve stiction
        for k in range(nvalves):
            ui = vfaults.uindex[k] - 1
            y, du, us, stp, ddir = _stiction_step(
                u[ui], uv[i - 1, ui], valve_yOLD[k], valve_duOLD[k],
                valve_us[k], valve_stp[k], valve_d[k],
                vfaults.S[k], vfaults.J[k],
            )
            vpos[i, k] = y
            valve_duOLD[k] = du
            valve_us[k] = us
            valve_stp[k] = stp
            valve_d[k] = ddir
            valve_yOLD[k] = y

        # ----------------- ODE integration
        uu = u.copy()
        uu[vfaults.uindex - 1] = vpos[i, :]
        rhs.set_u(uu)
        # dynamic fouling / fouling-mode windows: factor selection with explicit
        # priority. Mirrors LiveSimulator (issue #8):
        #   1) continuous α (dynamic fouling) — when fouling_dynamic=True
        #   2) windowed mode 4/5 (fouling-mode windows) — when fouling_mode_active
        #   3) static legacy factor[i] — pre-baked series
        # Both paths now produce the same factor trajectory for any
        # given fault schedule; the fingerprint contract is preserved
        # because the new tier 2 is bypassed whenever
        # ``fouling_mode_active_mode`` is 0 (the default).
        applied_factor = select_fouling_factor(
            t_now=float(t[i - 1]),
            sv=sv,
            pfaults=pfaults,
            factor_series=factor,
            i=i,
            fouling_stepper=fouling_stepper,
            fouling_mode_rng=fouling_mode_rng,
            use_dynamic_alpha=use_dynamic_alpha,
        )
        rhs.set_factor(applied_factor)
        factor_applied[i] = applied_factor
        # NB: the `factor` argument is *also* ignored by the JIT when
        # use_dynamic_alpha=True (the RHS uses sv[21] directly). The
        # value passed here is just a placeholder for the legacy branch.

        # Note: state variable sv(19) is stored in micrometres upstream
        # (numerical stability). Convert from μm for the RHS, then back.
        sv_init = sv[i - 1, :].copy()
        # The MATLAB upstream uses ode45 (RK45) by default and lsode (LSODA)
        # in Octave. We use RK45 here with `max_step=dt` to match the
        # upstream per-interval behaviour while keeping per-step cost
        # predictable. Numba-JIT-compiled RHS keeps the inner loop fast.
        sol = solve_ivp(rhs, (t[i - 1], t[i]), sv_init,
                        method="RK45", rtol=1e-3, atol=1e-6,
                        max_step=settings.dt)
        if sol.success is False:
            if verbose:
                print(f"  ⚠️  solver failed at t={t[i]:.0f} s: {sol.message}")
            sv[i, :] = sv[i - 1, :]                                # freeze; continue
        else:
            sv[i, :] = sol.y[:, -1]

        # dynamic fouling post-integration handling of sv[21] (α) — only when
        # the state vector is wide enough (dynamic mode). Legacy mode
        # leaves sv at its 21-component upstream shape for byte-identical
        # reproducibility.
        if use_dynamic_alpha:
            sv[i, 21] = float(np.clip(sv[i, 21], 0.0, 1.0))

        # actuator wear: post-integration clamping on pump_health and
        # valve_stiction_pct. The kernel writes raw derivatives; the
        # driver enforces physical bounds so we don't accumulate
        # numerical drift outside the operating envelope.
        if use_pump_wear:
            sv[i, layer24_base + 0] = float(np.clip(
                sv[i, layer24_base + 0],
                pfaults.pump_wear_floor,
                1.0,
            ))
        if use_valve_wear:
            sv[i, layer24_base + (1 if use_pump_wear else 0)] = float(np.clip(
                sv[i, layer24_base + (1 if use_pump_wear else 0)],
                pfaults.valve_stiction_floor_pct,
                pfaults.valve_stiction_ceiling_pct,
            ))

        # ----------------- filter cleaning
        if pv[i - 1, 4] >= pfaults.DPclean:
            sv[i, 18] = p.rclean * 1e6
            if use_dynamic_alpha:
                sv[i, 21] = p.alpha_clean                          # snap α to 0.1
            var = pfaults.filter_std * np.random.randn()
            p.K2F = p.K2F + (4 * p.visco / np.pi) * p.cv**2 * var
            p.K4F = p.K4F + (8 * p.visco / np.pi) * var
            tclean.append(t[i])
            if verbose:
                print(f"  Filter cleaning performed at t = {t[i]:.1f} s")

        # ----------------- quality state post-processing (quality latching)
        # Clamp the true instantaneous quality to physical bounds.
        # FAME% ∈ [0, 100], water ∈ [0, 5000] ppm, IV ∈ [0, 200].
        # Feedstock state also clamped to physical bounds.
        if use_quality_state:
            sv[i, 22] = float(np.clip(sv[i, 22], 0.0, 100.0))           # FAME%
            sv[i, 23] = float(np.clip(sv[i, 23], 0.0, 5000.0))         # water, ppm
            sv[i, 24] = float(np.clip(sv[i, 24], 0.0, 200.0))          # IV
            sv[i, 25] = float(np.clip(sv[i, 25], 0.0, 0.5))            # FFA feed
            sv[i, 26] = float(np.clip(sv[i, 26], 0.0, 0.5))            # water feed
            sv[i, 27] = float(np.clip(sv[i, 27], 0.0, 200.0))          # IV feed

            # Lab-cycle latching: at each lab boundary, sample the true
            # quality with analytical noise and hold the value until the
            # next cycle. This is the time-lag structure.
            if (t[i] - last_lab_sample_t) >= lab_period_s:
                quality_latched[i, 0] = sv[i, 22] + pfaults.lab_noise_fame * np.random.randn()
                quality_latched[i, 1] = sv[i, 23] + pfaults.lab_noise_water * np.random.randn()
                quality_latched[i, 2] = sv[i, 24] + pfaults.lab_noise_iv * np.random.randn()
                # Re-clamp latched values to physical bounds.
                quality_latched[i, 0] = float(np.clip(quality_latched[i, 0], 0.0, 100.0))
                quality_latched[i, 1] = float(np.clip(quality_latched[i, 1], 0.0, 5000.0))
                quality_latched[i, 2] = float(np.clip(quality_latched[i, 2], 0.0, 200.0))
                last_lab_sample_t = t[i]
            else:
                # Hold previous latched value.
                quality_latched[i, :] = quality_latched[i - 1, :]
        else:
            # Quality state disabled — published latched values are
            # NaN-equivalent so the dashboard can detect "not active".
            quality_latched[i, :] = np.nan

        xLend[i, :], yLend[i, :] = AEmodel(sv[i, :], p)
        pv[i, :] = _measurements(i, sv[i, :], uu, p, sfaults,
                                 sfaults.signal, sfaults.a, sfaults.b)
        pvAUTO = pv[i, pvindexAUTO]

    # ------------------------------------------------ trim last point (upstream)
    uv = uv[:-1, :]
    sv = sv[:-1, :]
    pv = pv[:-1, :]
    sp = sp[:-1, :]
    xLend = xLend[:-1, :]
    yLend = yLend[:-1, :]
    t_out = t[:-1]

    # Note: uv[:, 2] is left in mol/s here; :meth:`Results.in_display_units`
    # converts to kg/h for plotting/CSV export. Doing the conversion in
    # only one place avoids the previous double-conversion bug where the
    # kg/h values ended up 115× too large.

    # actuator wear: apply pump_health multiplier to the published PCW
    # channel (index 2) when pump_wear is on. The kernel applies the
    # same factor on u[4] in the perturbation block; we mirror it on
    # the published snapshot here so dashboards / live consumers
    # see what the kernel actually used. We mutate a copy because the
    # caller may want to inspect the unperturbed baseline elsewhere.
    # The pump_health slot index depends on which other layers are
    # enabled — same convention as the kernel's layer24_base.
    disturbances_out = disturbance_track[:-1, :]
    if pfaults.pump_wear:
        pump_health_slot = (
            21
            + (1 if pfaults.fouling_dynamic else 0)
            + (6 if pfaults.quality_state else 0)
        )
        pump_health_track = sv[: disturbances_out.shape[0], pump_health_slot]
        disturbances_out = disturbances_out.copy()
        disturbances_out[:, 2] = (
            disturbances_out[:, 2] * pump_health_track
        )

    runtime = time.perf_counter() - tic
    if verbose:
        print(f"Runtime: {runtime:.2f} s")

    return Results(
        t=t_out, uv=uv, sv=sv, pv=pv, sp=sp,
        xLend=xLend, yLend=yLend,
        tclean=np.array(tclean),
        quality=quality_latched[:-1, :],
        disturbances=disturbances_out,
        factor=factor_applied[:-1],
    )