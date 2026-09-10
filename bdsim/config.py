"""
Typed configuration objects (replacing the script-style ``system_parameters.m``
and ``user_settings.m`` of the MATLAB upstream).

The defaults reproduce exactly the values from the upstream ``user_settings.m``
so that ``run()`` reproduces the upstream trajectory to within solver tolerance.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .spectra import SpectrumSample


# -----------------------------------------------------------------------------
# Validation helpers (issue #9)
# -----------------------------------------------------------------------------
#
# These helpers back ``__post_init__`` validation on every config dataclass.
# The rules are deliberately permissive — they catch clearly-broken input
# (NaN, Inf, wrong shapes, negative physical quantities, out-of-enum
# switches) but do not enforce tight operating-range bounds. All current
# defaults pass; the byte-identical fingerprint contract is preserved.
#
# Error class policy: ``ValueError`` for value / range / shape problems,
# ``TypeError`` for wrong-type inputs (per TRY004).

def _err(name: str, msg: str) -> ValueError:
    return ValueError(f"{name}: {msg}")


def _check_finite(name: str, value: float) -> None:
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"{name}: expected a real number, got {type(value).__name__}"
        )
    if not math.isfinite(float(value)):
        raise _err(name, f"must be finite; got {value!r}")


def _check_finite_array(name: str, value: np.ndarray) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"{name}: expected np.ndarray, got {type(value).__name__}"
        )
    if not np.all(np.isfinite(value)):
        raise _err(name, "all entries must be finite")


def _check_nonneg(name: str, value: float) -> None:
    _check_finite(name, value)
    if float(value) < 0.0:
        raise _err(name, f"must be non-negative; got {value!r}")


def _check_positive(name: str, value: float) -> None:
    _check_finite(name, value)
    if float(value) <= 0.0:
        raise _err(name, f"must be positive; got {value!r}")


def _check_in_set(name: str, value: int, allowed: tuple[int, ...]) -> None:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise TypeError(
            f"{name}: expected an integer in {allowed}, got {type(value).__name__}"
        )
    if int(value) not in allowed:
        raise _err(
            name, f"must be one of {allowed}; got {value!r}"
        )


def _check_shape(
    name: str, value: np.ndarray, expected: tuple[int, ...]
) -> None:
    if value.shape != expected:
        raise _err(
            name, f"must have shape {expected}; got {tuple(value.shape)}"
        )


# -----------------------------------------------------------------------------
# Process parameters (formerly system_parameters.m)
# -----------------------------------------------------------------------------

@dataclass
class Parameters:
    """Process constants — kinetic, geometric, physical.

    All quantities are in SI unless explicitly noted in the field comment.
    The defaults match ``system_parameters.m`` upstream verbatim.
    """

    # Kinetic
    nc: int = 6
    k0: np.ndarray = field(default_factory=lambda: np.array(
        [1.3613e-01, 4.6645e+01, 3.8708e-01, 1.4710e+07, 4.5477e+04, 0.0]
    ))                                                          # m³ mol⁻¹ s⁻¹
    Ea: np.ndarray = field(default_factory=lambda: np.array(
        [3.0225e+04, 4.4646e+04, 2.8941e+04, 7.2779e+04, 5.6093e+04, 0.0]
    ))                                                          # J mol⁻¹
    dHr: np.ndarray = field(default_factory=lambda: np.array(
        [15699.0, 36899.0, -58906.0]
    ))                                                          # J mol⁻¹

    # Physical — densities, molar masses, heat capacities
    ro: np.ndarray = field(default_factory=lambda: np.array(
        [954.0, 983.0, 1030.0, 757.0, 844.0, 1340.0]
    ))                                                          # kg m⁻³ (at 60 °C)
    M: np.ndarray = field(default_factory=lambda: np.array(
        [0.854, 0.600, 0.346, 0.032, 0.286, 0.092]
    ))                                                          # kg mol⁻¹
    cpmol: np.ndarray = field(default_factory=lambda: np.array(
        [2110.0, 2188.0, 2381.0, 2785.0, 2146.0, 2556.0]
    ) * np.array([0.854, 0.600, 0.346, 0.032, 0.286, 0.092]))  # J mol⁻¹ K⁻¹
    R: float = 8.314472                                         # J mol⁻¹ K⁻¹

    # Oil stream
    xo: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0, 0, 0], dtype=float))
    Mo: float = 0.854                                          # kg mol⁻¹
    roo: float = 954.0                                         # kg m⁻³
    vmolo: float = 0.854 / 954.0                               # m³ mol⁻¹
    cpmolo: float = 2110.0 * 0.854                             # J mol⁻¹ K⁻¹
    visco: float = 20.8e-3                                     # Pa s

    # Methanol stream
    xm: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1, 0, 0], dtype=float))
    Mm: float = 0.032                                          # kg mol⁻¹
    rom: float = 757.0                                         # kg m⁻³
    cmolm: float = 2785.0 * 0.032                              # J mol⁻¹ K⁻¹

    # Geometric
    VR: float = 20.0                                           # m³
    aD: float = 6.0                                            # m²
    hD: float = 1.5                                            # m

    # Filter / pump geometry
    zF: float = 1e-3                                           # m
    cv: float = 4.86e-6                                        # m³ s⁻¹ Pa⁻¹ᐟ²
    Ppump: float = 2.07e5                                      # Pa
    rclean: float = 15e-6                                      # m
    np_: float = 1.7e8                                         # number of pores (avoid name clash)

    # Valves
    kvH: float = 1.0                                           # %
    tauvH: float = 15.0                                        # s
    NHmax: float = 10.0                                        # mol s⁻¹
    kvo: float = 1.0                                           # %
    tauvo: float = 15.0                                        # s

    # Derived: filled by :meth:`finalize`
    vmol: np.ndarray | None = None
    cpmolm: float = 0.0                                        # populated by finalize

    # Filter constants — populated from process faults at startup
    K1F: float = 0.0
    K2F: float = 0.0
    K3F: float = 0.0
    K4F: float = 0.0

    # ---- HEX fouling dynamics (Roadmap dynamic fouling) -------------------------
    # α is a continuous state in sv[21]. dα/dt has two terms:
    #   accumulation: k_f0 * FFA_factor(T) * exp(-E_a_f / (R * TR))
    #   decay:        k_decay * α
    # Sane defaults give α ∈ [~0.05, ~0.7] over a 72 h horizon at the
    # default operating point. See bdsim/ode.py:_ode_rhs_jit for application.
    k_f0: float = 2.5e-4                                      # base deposition rate, 1/s
    E_a_f: float = 1.8e4                                      # activation energy, J/mol
    k_decay: float = 5.0e-7                                   # self-cleaning decay, 1/s
    ffa_ref: float = 0.05                                     # reference FFA fraction (dimensionless)
    alpha_clean: float = 0.1                                  # snap value on cleaning event

    # ---- quality latching: quality dynamics (feature) -------------------
    # First-order relaxation of true quality state toward equilibrium.
    # Time constants chosen so FAME ~ 30–60 min, water ~ 15–30 min,
    # IV ~ hours at default operating point. k_q = 1/tau (1/s).
    k_fame: float = 5.0e-4                                    # 1/s → tau ~ 33 min
    k_water: float = 8.0e-4                                   # 1/s → tau ~ 21 min
    k_iv: float = 2.0e-4                                      # 1/s → tau ~ 83 min
    # Equilibrium targets at default operating conditions.
    fame_eq: float = 97.0                                     # EN 14214 spec ≥ 96.5%
    water_eq: float = 200.0                                   # EN 14214 limit ≤ 500 ppm
    iv_eq: float = 60.0                                       # typical UCO
    # Feedstock random-walk noise (small per-step deviations).
    ffa_feed_noise: float = 1.0e-6                            # 1/sqrt(s) σ
    water_feed_noise: float = 1.0e-7
    iv_feed_noise: float = 1.0e-3

    # ---- actuator wear: actuator degradation kinetics constants ----
    # Pump: dh/dt = -k_pump_wear * (Q/Qnom)^p. The driver converts
    # the per-hour ``ProcessFaults.pump_wear_rate_per_h`` to a
    # per-second rate constant here so the Numba kernel sees the
    # canonical form.
    k_pump_wear: float = 0.01 / 3600.0                         # per-second baseline wear rate at Q=Qnom
    p_pump_wear: float = 1.5                                  # flow exponent (matches real pump curves)
    pump_health_floor: float = 0.05                           # can't go to absolute zero head
    pump_health_trip_threshold: float = 0.25                  # scenario-side trigger threshold
    # Valve stiction: dstiction/dt = k_stiction * |dlift/dt|.
    # k_stiction is in (%/stroke) per unit valve motion. The driver
    # passes the per-second rate so the kernel can apply it directly.
    k_valve_stiction: float = 0.05 / 3600.0                   # %/s per (lift%/s)
    valve_stiction_ceiling: float = 60.0                      # % — at this point the loop is unstable

    def __post_init__(self) -> None:
        """Validate physically-required fields at construction time (issue #9).

        Rules are deliberately permissive — they catch NaN/Inf, wrong
        shapes, and clearly-impossible physical values (negative
        densities, zero rate constants, etc.) without enforcing tight
        operating-range bounds. All current defaults pass.
        """
        # nc and the per-species array length
        if not isinstance(self.nc, (int, np.integer)) or isinstance(self.nc, bool):
            raise TypeError(f"nc: expected int, got {type(self.nc).__name__}")
        if self.nc < 1:
            raise _err("nc", f"must be >= 1; got {self.nc}")
        n = int(self.nc)

        # R is a positive constant
        _check_positive("R", self.R)

        # Per-species arrays
        _check_shape("k0", self.k0, (n,))
        _check_finite_array("k0", self.k0)
        if np.any(self.k0 < 0.0):
            raise _err("k0", "all rate constants must be non-negative")

        _check_shape("Ea", self.Ea, (n,))
        _check_finite_array("Ea", self.Ea)
        if np.any(self.Ea < 0.0):
            raise _err("Ea", "all activation energies must be non-negative")

        _check_shape("ro", self.ro, (n,))
        _check_finite_array("ro", self.ro)
        if np.any(self.ro <= 0.0):
            raise _err("ro", "all densities must be positive")

        _check_shape("M", self.M, (n,))
        _check_finite_array("M", self.M)
        if np.any(self.M <= 0.0):
            raise _err("M", "all molar masses must be positive")

        if self.cpmol is not None:
            _check_shape("cpmol", self.cpmol, (n,))
            _check_finite_array("cpmol", self.cpmol)
            if np.any(self.cpmol < 0.0):
                raise _err("cpmol", "all heat capacities must be non-negative")

        # Reaction enthalpy vector (can be negative for endothermic / exothermic)
        _check_shape("dHr", self.dHr, (3,))
        _check_finite_array("dHr", self.dHr)

        # Composition vectors
        _check_shape("xo", self.xo, (n,))
        _check_finite_array("xo", self.xo)
        if np.any(self.xo < 0.0):
            raise _err("xo", "all oil-stream fractions must be non-negative")
        _check_shape("xm", self.xm, (n,))
        _check_finite_array("xm", self.xm)
        if np.any(self.xm < 0.0):
            raise _err("xm", "all methanol-stream fractions must be non-negative")

        # Stream scalars (cpmolm is populated by finalize() and stays at
        # 0.0 until then — skip its check here).
        for name in ("Mo", "Mm", "roo", "rom", "vmolo", "visco"):
            _check_positive(name, getattr(self, name))

        # Geometry / hardware
        for name in ("VR", "aD", "hD", "zF", "cv", "Ppump", "rclean", "np_",
                     "kvH", "tauvH", "NHmax", "kvo", "tauvo"):
            _check_positive(name, getattr(self, name))

        # Fouling kinetics
        for name in ("k_f0", "E_a_f", "k_decay", "ffa_ref", "alpha_clean"):
            _check_nonneg(name, getattr(self, name))

        # Quality dynamics
        for name in ("k_fame", "k_water", "k_iv"):
            _check_nonneg(name, getattr(self, name))
        for name in ("fame_eq", "water_eq", "iv_eq"):
            _check_nonneg(name, getattr(self, name))
        for name in ("ffa_feed_noise", "water_feed_noise", "iv_feed_noise"):
            _check_nonneg(name, getattr(self, name))

        # Actuator wear kinetics
        _check_nonneg("k_pump_wear", self.k_pump_wear)
        _check_nonneg("p_pump_wear", self.p_pump_wear)
        _check_nonneg("pump_health_floor", self.pump_health_floor)
        _check_nonneg("pump_health_trip_threshold", self.pump_health_trip_threshold)
        if self.pump_health_floor > self.pump_health_trip_threshold:
            raise _err(
                "pump_health_floor",
                f"must be <= pump_health_trip_threshold "
                f"({self.pump_health_floor} > {self.pump_health_trip_threshold})",
            )
        _check_nonneg("k_valve_stiction", self.k_valve_stiction)
        _check_nonneg("valve_stiction_ceiling", self.valve_stiction_ceiling)

    def finalize(self) -> None:
        """Recompute derived quantities that depend on M and ro."""
        self.vmol = self.M / self.ro

    def apply_layer24_overrides(self, pfaults: ProcessFaults) -> None:
        """Apply actuator wear kinetics overrides from ``pfaults``.

        Called by the driver after ``Parameters()`` is constructed so
        any caller-set ``ProcessFaults.pump_wear_rate_per_h`` etc.
        flow into the Numba-visible kinetics constants. Same pattern
        as dynamic fouling / quality latching's override paths.
        """
        # Pump wear rate: convert per-hour → per-second so the kernel
        # multiplies by dt directly.
        self.k_pump_wear = float(pfaults.pump_wear_rate_per_h) / 3600.0
        self.p_pump_wear = float(pfaults.pump_wear_flow_exponent)
        self.pump_health_floor = float(pfaults.pump_wear_floor)
        self.pump_health_trip_threshold = float(pfaults.pump_health_trip_threshold)
        # Valve stiction rate: same per-hour → per-second conversion.
        # The kernel multiplies by |dlift/dt| (lift%/s) to give %/s.
        self.k_valve_stiction = float(pfaults.valve_stiction_rate_pct_per_h) / 3600.0
        self.valve_stiction_ceiling = float(pfaults.valve_stiction_ceiling_pct)


# -----------------------------------------------------------------------------
# Process faults (clogging, fouling, side reactions)
# -----------------------------------------------------------------------------

@dataclass
class ProcessFaults:
    """Process-level fault activation.

    Mirrors the ``pfaults.*`` block of ``user_settings.m``.
    """

    clog_fraction: float = 5.95e-7
    DPclean: float = 1e5                                       # Pa
    filter_std: float = 5e-15
    ratio_robs_r: float = 0.9
    fouling: int = 1                                           # 0 off, 1 on
    foulingpar: np.ndarray = field(default_factory=lambda: np.array([3e-7]))
    fouling_dynamic: bool = True                              # dynamic fouling: α evolves as a state when True.
                                                              # When False, behaviour matches the legacy
                                                              # pre-baked series (factor = 1/(1 + Rf)).
                                                              # Default ON since 1.0; bare ProcessFaults() is
                                                              # NOT the legacy fingerprint profile.

    # ---- quality latching: quality state + feedstock quality -------------------
    # When quality_state=True, sv0 grows by 6 components:
    #   sv[22] = FAME%   (instantaneous true value, 0..100)
    #   sv[23] = water   (instantaneous true value, ppm)
    #   sv[24] = IV      (instantaneous true value, g I2/100g)
    #   sv[25] = FFA_feed   (mass fraction, 0..1)
    #   sv[26] = water_feed (mass fraction, 0..1)
    #   sv[27] = IV_feed    (g I2/100g, typical 50..80)
    # Published channels QA-101/102/103 carry the *latched* lab samples
    # (sampled once per lab_cycle_s), not the instantaneous truth — this
    # is the time-lag structure between OT (instantaneous) and lab
    # (sampled) measurements that downstream fault-detection models can use.
    # quality_lag_mode = "lab" → 15-min default lab cycle.
    # quality_lag_mode = "online" → 60-s NIR cycle (online analyser).
    # quality_state=False preserves the upstream 21-component state.
    quality_state: bool = True                               # quality latching master switch. Default ON as of 1.2.0:
                                                              # demos that want the legacy pre-2.1 fingerprint must
                                                              # pass quality_state=False explicitly.
    quality_lag_mode: str = "lab"
    lab_cycle_s: float = 15.0 * 60.0                           # 15 minutes default
    online_cycle_s: float = 60.0                                # 1 minute for NIR
    lab_noise_fame: float = 0.3                                # % absolute
    lab_noise_water: float = 20.0                              # ppm absolute
    lab_noise_iv: float = 1.0                                  # g I2/100g absolute

    # ---- external disturbances: external disturbances ----
    # All default to zero amplitude / zero drift so the legacy
    # fingerprint is preserved when disturbance_profile is left at
    # defaults (the perturbation kernel adds 0 to every channel).
    ambient_t_mean_k: float = 293.15                           # 20 °C baseline
    ambient_t_amplitude_k: float = 0.0                         # daily sinusoid amplitude, K
    ambient_t_period_s: float = 24.0 * 3600.0                  # 24 h period
    cw_t_mean_k: float = 288.15                                # 15 °C cooling-water inlet baseline
    cw_t_amplitude_k: float = 0.0                              # seasonal sinusoid amplitude, K
    cw_t_period_s: float = 7.0 * 24.0 * 3600.0                 # 7-day period (slow)
    cw_p_nominal_pa: float = 4.0e5                             # 4 bar nominal CW pressure
    cw_p_drift_pa_per_h: float = 0.0                           # slow drift, Pa/h
    cw_p_noise_pa: float = 0.0                                 # small jitter (1σ), Pa
    # Mapping coefficients (linear first-order model). Defaults to 1.0
    # so 0-amplitude disturbance profile produces byte-identical
    # legacy behavior.
    met_cw_track: float = 0.3                                  # Tmet shift per K of CW deviation
    oil_ambient_track: float = 0.7                             # Toil shift per K of ambient deviation
    qheat_cw_scaling: bool = True                              # Qheat *= Pwater_cw / cw_p_nominal_pa

    # ------------------------------------------------------------------
    # cooling-water pump trip: cw_pump_trip mid-run override knobs. All default
    # values are conservative and aligned with the dashboard
    # FaultSpec defaults so a no-fault sim is byte-identical to the
    # external-disturbances fingerprint. Override is single-slot (a second trip
    # replaces the first); the kernel applies the envelope on top of
    # the baseline sinusoidal profile.
    # ------------------------------------------------------------------
    cw_pump_low_factor: float = 0.3                            # pressure floor during trip (1.0 = no drop, 0.0 = zero)
    cw_pump_ramp_s: float = 30.0                               # ramp-down + ramp-up duration, seconds
    cw_pump_default_duration_s: float = 600.0                 # default trip duration when the FaultSpec doesn't set one

    # ------------------------------------------------------------------
    # operator disturbance knobs: operator-driven disturbance knob overlays.
    #
    # ``None`` (default) means "use the configured profile value"
    # (ambient_t_mean_k / ambient_t_amplitude_k / cw_t_mean_k /
    # cw_p_drift_pa_per_h above). When an operator pushes a new
    # value through ``LiveSimulator.set_*_knob()``, the field is
    # written to a float and the kernel reads from there instead.
    #
    # All four default to ``None`` so the legacy external disturbances
    # fingerprint is preserved (the kernel reads ``None`` → falls
    # through to the configured profile value, which is identical
    # to today's behaviour). Knobs persist for the rest of the run
    # unless cleared back to ``None``.
    #
    # Period fields are NOT knob-overridable here on purpose: the
    # demo story is "operator tweaks the operating point", not
    # "operator changes the physics of the weather sinusoid".
    # ------------------------------------------------------------------
    live_ambient_mean_k: float | None = None                   # override ambient_t_mean_k
    live_ambient_amplitude_k: float | None = None             # override ambient_t_amplitude_k
    live_cw_t_mean_k: float | None = None                     # override cw_t_mean_k
    live_cw_p_drift_pa_per_h: float | None = None            # override cw_p_drift_pa_per_h

    # ------------------------------------------------------------------
    # actuator wear: actuator degradation as continuous state.
    #
    # Two master switches, both default ``False``. When both are
    # False the state vector is unchanged from operator disturbance knobs (21
    # components legacy, 22 + α in dynamic fouling mode, 28 in quality latching
    # mode). When ``pump_wear=True`` the state vector grows by one
    # slot (sv[22] = pump_health ∈ [0, 1]). When ``valve_wear=True``
    # it grows by another slot (sv[23] = valve_stiction_pct ∈
    # [0, 100]). The two switches are independent — demos can enable
    # one or both.
    #
    # ``pump_health`` multiplies the CW pressure nominal, so a worn
    # pump at ``pump_health = 0.7`` delivers only 70 % of the
    # nominal head. The trip probability is NOT kernel-side — the
    # scenario runner can read ``pump_health`` from a derived tag and
    # schedule a ``cw_pump_trip`` when it crosses the trip
    # threshold. Trip scheduling stays where it is (cooling-water pump trip);
    # actuator wear only adds the underlying wear curve.
    #
    # ``valve_stiction_pct`` reduces the effective valve gain via
    # ``kv_eff = kv * (1 - stiction / 100)``. Closed-loop
    # oscillation in TR-101 / TD-201 becomes visible as stiction
    # grows. The existing ``valve_stiction`` fault event still
    # works as an instantaneous deadband injection; actuator wear
    # models the slow build-up of that stiction.
    # ------------------------------------------------------------------
    pump_wear: bool = True                                     # actuator wear: pump degradation state (sv[22]).
                                                               # Default ON as of 1.2.0 — pass pump_wear=False
                                                               # explicitly for the legacy fingerprint profile.
    valve_wear: bool = True                                    # actuator wear: valve stiction state (sv[23]).
                                                               # Default ON as of 1.2.0 — pass valve_wear=False
                                                               # explicitly for the legacy fingerprint profile.

    # Initial values for the continuous-state slots. The driver
    # writes these into sv[22] / sv[23] at construction time.
    pump_health_initial: float = 1.0                           # 1.0 = brand-new impeller, 0.0 = end-of-life
    valve_stiction_initial_pct: float = 0.0                    # 0 % = pristine, 100 % = full stroke stuck

    # Kinetics constants. Defaults chosen so the demo ``pump_wear``
    # scenario lands the pump in the trip zone within ~30 min of
    # accelerated wear, and valve_stiction grows to a visibly
    # oscillating regime in the same window. All values are tunable
    # via ``ProcessFaults(...)`` overrides.
    pump_wear_rate_per_h: float = 0.01                         # dh/dt baseline (per-second × 3600), at nominal flow
    pump_wear_flow_exponent: float = 1.5                       # dh/dt ∝ (Q/Qnom)^p — higher flow → faster wear
    pump_wear_floor: float = 0.05                              # lower bound — pump can't go to absolute zero head
    pump_health_trip_threshold: float = 0.25                   # below this → trip becomes likely (scenario-side)
    valve_stiction_rate_pct_per_h: float = 0.05                # grows proportional to |dlift/dt|
    valve_stiction_floor_pct: float = 0.0                      # lower bound; can be negative in theory but never here
    valve_stiction_ceiling_pct: float = 60.0                   # upper bound — at 60 % the loop is already unstable

    # ------------------------------------------------------------------
    # NIR/IR spectrum sensor: NIR/IR virtual spectrum sensor (port of upstream
    # ``comp_spectrum.m``). Master switch is ON by default as of
    # 1.2.0 — the spectrum is post-process only (does NOT perturb
    # the ODE state vector), so the trajectory fingerprint is
    # unaffected; only ``StepResult.spectra`` is populated at fire
    # times. Pass ``spectrum_enabled=False`` to skip it entirely.
    # The generator fires every ``spctr_t`` seconds and attaches a
    # ``SpectrumSample`` to ``StepResult.spectra`` at those times
    # (None between fires).
    # ------------------------------------------------------------------
    spectrum_enabled: bool = True                              # Default ON as of 1.2.0 — set False to skip the
                                                               # NIR/IR virtual spectrum sensor entirely.
    spctr_t: float = 3600.0                                     # spectrum sampling period (s), default 1 h
    spctr_cs: int = 2                                           # Skoog photometric noise: 0..3
    spctr_snr_db: float = 30.0                                  # additive white Gaussian noise SNR
    spctr_k: float = 0.03                                       # photometric noise scale (Skoog: 0.3%T)
    spctr_drift_a: float = 0.01                                 # scatter baseline (a + b*wn + c*abs)
    spctr_drift_b: float = 0.0001
    spctr_drift_c: float = 1.05
    spectra_ref_path: str | None = None                         # None → bundled bdsim/data/spectra_ref.csv

    # ------------------------------------------------------------------
    # fouling-mode windows: windowed five-mode fouling stepper (port of
    # upstream ``fouling.m``). Modes 4 and 5 are stochastic ARMAX
    # fault-injection paths; the LiveSimulator kernel applies them
    # during an active fault window, then control returns to
    # dynamic fouling (continuous α) or to the static legacy series
    # (factor = 1/(1 + Rf)).
    #
    # Priority when multiple paths are configured:
    #   continuous α  >  windowed mode 4/5  >  static legacy series.
    #
    # The ``fouling_mode`` integer matches the upstream
    # ``pfaults.fouling`` switch case so a ProcessFaults(...)
    # override is a drop-in for legacy code. The
    # ``fouling_mode_active_*`` triple is the runtime overlay
    # written by the dashboard fault handler (FOULING_MODE_4 /
    # FOULING_MODE_5 events); defaults to "off" so a no-fault
    # sim is byte-identical to the operator-knobs fingerprint.
    # ------------------------------------------------------------------
    fouling_mode: int = 0                                      # 0..5 — global mode selector (matches upstream)
    fouling_mode_xRG_weight: bool = True                       # if True, mode-4 target includes xRG (glycerol coupling)
    fouling_ar_eps_std: float = 5e-4                           # ARMAX innovation σ (modes 4/5) — matches upstream
    fouling_mode_default_window_s: float = 3600.0              # default window if the FaultSpec doesn't set one
    fouling_mode_active_mode: int = 0                          # 0 = no active window; 4 or 5 = ARMAX mode active
    fouling_mode_active_end_t: float = -1.0                    # sim time at which the active window expires
    fouling_mode_active_seed: int | None = None                 # optional seed for the ARMAX RNG (reproducibility)

    def __post_init__(self) -> None:
        """Validate ``ProcessFaults`` at construction time (issue #9).

        Permissive: catches NaN/Inf, negative physical quantities, wrong
        shapes, and out-of-range enum-like fields. All current defaults
        pass — the byte-identical fingerprint contract is preserved.
        """
        # Filter physics
        _check_nonneg("clog_fraction", self.clog_fraction)
        _check_positive("DPclean", self.DPclean)
        _check_nonneg("filter_std", self.filter_std)
        _check_positive("ratio_robs_r", self.ratio_robs_r)
        _check_in_set("fouling", self.fouling, (0, 1))
        _check_shape("foulingpar", self.foulingpar, (1,))
        _check_finite_array("foulingpar", self.foulingpar)
        if float(self.foulingpar[0]) < 0.0:
            raise _err("foulingpar", "must be non-negative")

        # Quality latching
        if self.quality_lag_mode not in ("lab", "online"):
            raise _err(
                "quality_lag_mode",
                f"must be 'lab' or 'online'; got {self.quality_lag_mode!r}",
            )
        _check_positive("lab_cycle_s", self.lab_cycle_s)
        _check_positive("online_cycle_s", self.online_cycle_s)
        _check_nonneg("lab_noise_fame", self.lab_noise_fame)
        _check_nonneg("lab_noise_water", self.lab_noise_water)
        _check_nonneg("lab_noise_iv", self.lab_noise_iv)

        # Disturbance profile
        _check_finite("ambient_t_mean_k", self.ambient_t_mean_k)
        _check_nonneg("ambient_t_amplitude_k", self.ambient_t_amplitude_k)
        _check_positive("ambient_t_period_s", self.ambient_t_period_s)
        _check_finite("cw_t_mean_k", self.cw_t_mean_k)
        _check_nonneg("cw_t_amplitude_k", self.cw_t_amplitude_k)
        _check_positive("cw_t_period_s", self.cw_t_period_s)
        _check_positive("cw_p_nominal_pa", self.cw_p_nominal_pa)
        _check_finite("cw_p_drift_pa_per_h", self.cw_p_drift_pa_per_h)
        _check_nonneg("cw_p_noise_pa", self.cw_p_noise_pa)
        _check_finite("met_cw_track", self.met_cw_track)
        _check_finite("oil_ambient_track", self.oil_ambient_track)

        # CW pump trip envelope
        if not (0.0 < self.cw_pump_low_factor <= 1.0):
            raise _err(
                "cw_pump_low_factor",
                f"must be in (0.0, 1.0]; got {self.cw_pump_low_factor!r}",
            )
        _check_nonneg("cw_pump_ramp_s", self.cw_pump_ramp_s)
        _check_nonneg("cw_pump_default_duration_s", self.cw_pump_default_duration_s)

        # Operator knob overlays — None or finite float
        for name in (
            "live_ambient_mean_k",
            "live_ambient_amplitude_k",
            "live_cw_t_mean_k",
            "live_cw_p_drift_pa_per_h",
        ):
            v = getattr(self, name)
            if v is not None:
                _check_finite(name, v)

        # Actuator wear
        _check_nonneg("pump_wear_rate_per_h", self.pump_wear_rate_per_h)
        _check_nonneg("pump_wear_flow_exponent", self.pump_wear_flow_exponent)
        _check_nonneg("pump_wear_floor", self.pump_wear_floor)
        _check_nonneg("pump_health_trip_threshold", self.pump_health_trip_threshold)
        if self.pump_wear_floor > self.pump_health_trip_threshold:
            raise _err(
                "pump_wear_floor",
                f"must be <= pump_health_trip_threshold "
                f"({self.pump_wear_floor} > {self.pump_health_trip_threshold})",
            )
        _check_nonneg("valve_stiction_rate_pct_per_h", self.valve_stiction_rate_pct_per_h)
        _check_nonneg("valve_stiction_floor_pct", self.valve_stiction_floor_pct)
        _check_nonneg("valve_stiction_ceiling_pct", self.valve_stiction_ceiling_pct)
        if self.valve_stiction_floor_pct > self.valve_stiction_ceiling_pct:
            raise _err(
                "valve_stiction_floor_pct",
                f"must be <= valve_stiction_ceiling_pct "
                f"({self.valve_stiction_floor_pct} > {self.valve_stiction_ceiling_pct})",
            )

        # NIR/IR spectrum sensor
        _check_positive("spctr_t", self.spctr_t)
        _check_in_set("spctr_cs", self.spctr_cs, (0, 1, 2, 3))
        _check_finite("spctr_snr_db", self.spctr_snr_db)
        _check_nonneg("spctr_k", self.spctr_k)
        _check_nonneg("spctr_drift_a", self.spctr_drift_a)
        _check_nonneg("spctr_drift_b", self.spctr_drift_b)
        _check_nonneg("spctr_drift_c", self.spctr_drift_c)
        if self.spectra_ref_path is not None and not isinstance(
            self.spectra_ref_path, (str, os.PathLike)
        ):
            raise TypeError(
                "spectra_ref_path: expected str or os.PathLike or None, "
                f"got {type(self.spectra_ref_path).__name__}"
            )

        # Fouling-mode windows
        _check_in_set("fouling_mode", self.fouling_mode, (0, 1, 2, 3, 4, 5))
        _check_in_set(
            "fouling_mode_active_mode", self.fouling_mode_active_mode, (0, 4, 5)
        )
        _check_nonneg("fouling_ar_eps_std", self.fouling_ar_eps_std)
        _check_positive("fouling_mode_default_window_s", self.fouling_mode_default_window_s)
        _check_finite("fouling_mode_active_end_t", self.fouling_mode_active_end_t)


# -----------------------------------------------------------------------------
# Sensor faults
# -----------------------------------------------------------------------------

@dataclass
class SensorFaults:
    """Sensor-level fault activation.

    The MATLAB upstream builds ``sfaults.signal``, ``sfaults.a``, ``sfaults.b``
    as ``lt x nsensors`` arrays sized to the simulation horizon. To keep the
    Python port memory-friendly we keep only the *templates* here and let
    :class:`Simulation` expand them to full-length arrays on startup.

    See ``measurements`` in ``measurements.m`` for the fault application
    formula: ``pv = signal * (a * v + b + noise_std * randn)``.
    """

    nsensors: int = 5

    # Defaults: no fault
    signal: np.ndarray | None = None                           # bool, shape (nsensors,)
    a: np.ndarray | None = None                                # shape (nsensors,)
    b: np.ndarray | None = None                                # shape (nsensors,)

    isIntermit: np.ndarray = field(
        default_factory=lambda: np.array([0, 1, 0, 0, 0])
    )
    tmaxInterm: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 3600.0, 0.0, 0.0, 0.0])
    )
    # Sensor noise standard deviations (in measurement units).
    # Upstream MATLAB defaults are: [0.1, 0.1, 5e-3, 5.0, 300].
    # We override Foil (index 3) to 5e-3 kg/s — the upstream value of 5.0
    # corresponds to 18,000 kg/h σ on a 3,050 kg/h signal (590% relative
    # noise), which produces visibly broken-looking plots. 5e-3 kg/s ≈ 18
    # kg/h σ ≈ 0.6% relative noise — realistic for a Coriolis flowmeter.
    noise_std: np.ndarray = field(
        default_factory=lambda: np.array([0.1, 0.1, 5e-3, 5e-3, 300.0])
    )

    # User-overridable fault templates (matched by sensor index)
    drift_b: dict[int, callable] = field(default_factory=dict)
    bias_b: dict[int, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Live fault knobs (mid-run mutable for fault-injection scenarios).
    # These fields let external code inject faults between ODE steps
    # without rebuilding the simulator. The dashboard's fault handler
    # registry writes into these; ``LiveSimulator.step()`` reads them.
    #
    # -- ``bias`` : sensor index → additive offset (in measurement units)
    # -- ``stuck`` : sensor index → sim time at which it became stuck
    #    (sticky until cleared; held value is the *last published* sample,
    #    not the last good one — matching what most DCS systems do)
    # -- ``dropouts`` : set of sensor indices currently outputting NaN/0
    # All three coexist with the legacy ``a``/``b``/``signal`` path used
    # by the batch ``run_with()`` driver. The batch path ignores these
    # fields; the live path reads them.
    # ------------------------------------------------------------------ #
    bias: dict[int, float] = field(default_factory=dict)
    stuck: dict[int, float] = field(default_factory=dict)
    dropouts: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Validate ``SensorFaults`` at construction time (issue #9).

        Permissive: catches NaN/Inf, wrong-shape arrays, negative
        noise std, and non-bool signal arrays. All current defaults
        pass.
        """
        if not isinstance(self.nsensors, (int, np.integer)) or isinstance(self.nsensors, bool):
            raise TypeError(
                f"nsensors: expected int, got {type(self.nsensors).__name__}"
            )
        if self.nsensors < 1:
            raise _err("nsensors", f"must be >= 1; got {self.nsensors}")
        n = int(self.nsensors)

        # signal / a / b: shape-checked only when provided.
        for name in ("signal", "a", "b"):
            v = getattr(self, name)
            if v is not None:
                if not isinstance(v, np.ndarray):
                    raise TypeError(
                        f"{name}: expected np.ndarray or None, "
                        f"got {type(v).__name__}"
                    )
                _check_shape(name, v, (n,))
                _check_finite_array(name, v)

        # bool signal — must be all 0/1
        if self.signal is not None and not np.all((self.signal == 0) | (self.signal == 1)):
            raise _err("signal", "must be all 0 or 1 (bool-like)")

        # Per-sensor flags / times
        _check_shape("isIntermit", self.isIntermit, (n,))
        _check_finite_array("isIntermit", self.isIntermit)
        _check_shape("tmaxInterm", self.tmaxInterm, (n,))
        _check_finite_array("tmaxInterm", self.tmaxInterm)
        if np.any(self.tmaxInterm < 0.0):
            raise _err("tmaxInterm", "all entries must be non-negative")

        # Noise std must be non-negative
        _check_shape("noise_std", self.noise_std, (n,))
        _check_finite_array("noise_std", self.noise_std)
        if np.any(self.noise_std < 0.0):
            raise _err("noise_std", "all entries must be non-negative")

        # Live-knob dicts / set: keys must be valid sensor indices.
        for name in ("drift_b", "bias_b", "bias", "stuck"):
            d = getattr(self, name)
            for k in d:
                if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
                    raise TypeError(
                        f"{name}: key must be int sensor index, "
                        f"got {type(k).__name__}"
                    )
                if not (0 <= int(k) < n):
                    raise _err(
                        name,
                        f"key {k!r} is outside the configured range [0, {n})",
                    )
        for idx in self.dropouts:
            if not isinstance(idx, (int, np.integer)) or isinstance(idx, bool):
                raise TypeError(
                    f"dropouts: element must be int sensor index, "
                    f"got {type(idx).__name__}"
                )
            if not (0 <= int(idx) < n):
                raise _err(
                    "dropouts",
                    f"element {idx!r} is outside the configured range [0, {n})",
                )


# -----------------------------------------------------------------------------
# Valve faults (stiction)
# -----------------------------------------------------------------------------

@dataclass
class ValveFaults:
    """Stiction parameters per valve.

    The stiction model follows Kano et al. (IFAC DYCOPS). With ``S=0`` and
    ``J=0`` the valve passes the controller order through unchanged.
    """

    S: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    J: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    uindex: np.ndarray = field(default_factory=lambda: np.array([6, 1]))

    def __post_init__(self) -> None:
        """Validate ``ValveFaults`` at construction time (issue #9).

        All three arrays must share the same length (one entry per
        valve). Stiction parameters must be non-negative.
        """
        for name in ("S", "J", "uindex"):
            v = getattr(self, name)
            if not isinstance(v, np.ndarray):
                raise TypeError(
                    f"{name}: expected np.ndarray, got {type(v).__name__}"
                )
        n = int(self.S.shape[0])
        _check_shape("J", self.J, (n,))
        _check_shape("uindex", self.uindex, (n,))
        _check_finite_array("S", self.S)
        _check_finite_array("J", self.J)
        if np.any(self.S < 0.0):
            raise _err("S", "all entries must be non-negative")
        if np.any(self.J < 0.0):
            raise _err("J", "all entries must be non-negative")
        # uindex are valve → input channel indices; integers in [0, ninputs).
        # We don't pin ninputs here (it lives on Parameters.nc=6 upstream),
        # so just check integer + non-negative.
        if not np.all((self.uindex >= 0) & (self.uindex < 1000)):
            raise _err("uindex", "all entries must be non-negative integers")


# -----------------------------------------------------------------------------
# ARMAX disturbance model
# -----------------------------------------------------------------------------

@dataclass
class ARMAX:
    """ARMAX(1,1,1) coefficients per input variable.

    ``phi * u_prev + theta * noise_prev + noise + eta * d_prev``
    """

    phi: np.ndarray = field(default_factory=lambda: np.array(
        [0.0, 0.05, 0.03, 0.0, 0.01, 0.0]
    ))
    theta: np.ndarray = field(default_factory=lambda: np.array(
        [0.0, 0.06, 0.07, 0.0, 0.0, 0.0]
    ))
    eta: np.ndarray = field(default_factory=lambda: np.array(
        [0.0, 0.95, 0.97, 0.0, 0.98, 0.0]
    ))
    unoise_std: np.ndarray = field(default_factory=lambda: np.array(
        [0.0, 0.10, 1.15, 0.0, 0.01, 0.0]
    ))
    unoise: np.ndarray | None = None                          # populated by Simulation

    def __post_init__(self) -> None:
        """Validate ``ARMAX`` at construction time (issue #9)."""
        for name in ("phi", "theta", "eta", "unoise_std"):
            v = getattr(self, name)
            if not isinstance(v, np.ndarray):
                raise TypeError(
                    f"{name}: expected np.ndarray, got {type(v).__name__}"
                )
        n = int(self.phi.shape[0])
        for name in ("phi", "theta", "eta"):
            _check_shape(name, getattr(self, name), (n,))
            _check_finite_array(name, getattr(self, name))
        _check_shape("unoise_std", self.unoise_std, (n,))
        _check_finite_array("unoise_std", self.unoise_std)
        if np.any(self.unoise_std < 0.0):
            raise _err("unoise_std", "all entries must be non-negative")


# -----------------------------------------------------------------------------
# PID controller parameters
# -----------------------------------------------------------------------------

@dataclass
class PIDController:
    """PID gains, time constants, and saturation bounds for one loop.

    All indices are 0-based (Python convention) — the MATLAB upstream uses
    1-based; ``Simulation._build_pid`` does the conversion.
    """

    kc: np.ndarray = field(default_factory=lambda: np.array(
        [12.0, -10542.0, -1315.0, 3.4e-3]
    ))
    taui: np.ndarray = field(default_factory=lambda: np.array(
        [16960.0, 8350.0, 3600.0, 15.0]
    ))
    taud: np.ndarray = field(default_factory=lambda: np.array(
        [0.0, 0.0, 0.0, 0.0]
    ))
    lower_bound: np.ndarray = field(default_factory=lambda: np.array(
        [30 + 273.15, 0.0, 0.0, 0.0]
    ))
    upper_bound: np.ndarray = field(default_factory=lambda: np.array(
        [65 + 273.15, 40000.0, 100.0, 100.0]
    ))

    def __post_init__(self) -> None:
        """Validate ``PIDController`` at construction time (issue #9).

        All five arrays must have the canonical length 4 (one entry per
        loop). ``taui`` must be positive (a zero or negative integral
        time constant makes the loop diverge). ``taud`` must be
        non-negative. ``lower_bound <= upper_bound`` per loop.
        """
        for name in ("kc", "taui", "taud", "lower_bound", "upper_bound"):
            v = getattr(self, name)
            if not isinstance(v, np.ndarray):
                raise TypeError(
                    f"{name}: expected np.ndarray, got {type(v).__name__}"
                )
            _check_shape(name, v, (4,))
            _check_finite_array(name, v)
        if np.any(self.taui <= 0.0):
            raise _err("taui", "all entries must be positive")
        if np.any(self.taud < 0.0):
            raise _err("taud", "all entries must be non-negative")
        if np.any(self.lower_bound > self.upper_bound):
            raise _err(
                "lower_bound",
                "all entries must be <= the corresponding upper_bound entry",
            )


# -----------------------------------------------------------------------------
# Top-level settings bundle
# -----------------------------------------------------------------------------

@dataclass
class Settings:
    """Time vector, initial conditions, control loop wiring.

    The defaults reproduce the upstream ``user_settings.m`` trajectory.
    """

    # Time
    ti: float = 0.0
    tf: float = 260000.0
    dt: float = 5.0

    # Initial state vector (21 components by default; 22 when HEX fouling is dynamic)
    #
    # Legacy mode (pfaults.fouling_dynamic=False): exactly 21 components,
    # byte-identical to the upstream baseline. The driver ignores sv[21].
    # Dynamic mode (pfaults.fouling_dynamic=True): sv0 grows to 22
    # components, with α at [21] initialised to 0.05 (lightly fouled).
    # The driver chooses the right shape at construction time.
    sv0: np.ndarray = field(default_factory=lambda: np.array([
        # Reactor composition + temperature
        0.002455, 0.000553, 4.67e-05, 0.42353, 0.43022, 0.14319, 60.4 + 273.15,
        # Decanter light phase
        4.2864e-03, 9.6570e-04, 8.1554e-05, 2.4287e-01, 7.5098e-01, 8.1550e-04,
        # Decanter heavy phase + interface level + temperature
        6.6574e-01, 1.5852e-04, 3.3410e-01, 0.5, 50.0 + 273.15,
        # Filter pore radius
        15e-6,
        # Valve lifts
        40.0, 28.5,
    ]))

    # Initial input vector: vinputo %, Tmet K, Fmet kg/h, Toil K, Qheat W, vinputH %
    u0: np.ndarray = field(default_factory=lambda: np.array(
        [40.0, 50 + 273.15, 657.0, 60 + 273.15, 23615.0, 28.5]
    ))

    # Disturbance profile (lt x 6). Defaults: Tmet daily sinusoid + Qheat step.
    def exogenous(self, t: np.ndarray) -> np.ndarray:
        u0 = self.u0
        d = np.tile(u0, (len(t), 1))
        d[:, 1] = d[:, 1] + 3.0 * np.sin(np.pi / (12 * 3600) * t)
        d[:, 4] = d[:, 4] + 1000.0 * np.heaviside(t - 100000.0, 1.0)
        return d

    # external disturbances + operator knobs: external disturbance channel (lt x 3).
    # Returns [Tambient, Twater_cw, Pwater_cw] for every step in the
    # sim horizon. Perturbations from the daily sinusoids + slow
    # drift; the scenario runner can add event-grade perturbations
    # on top of this baseline (power_dip, cw_pump_trip).
    #
    # All components default to constant values when amplitude/drift
    # knobs are zero — preserves byte-identical legacy behaviour.
    #
    # operator disturbance knobs: when an operator pushes a knob override through
    # ``LiveSimulator.set_*_knob()``, the corresponding
    # ``pfaults.live_*_mean_k`` / ``live_*_amplitude_k`` /
    # ``live_cw_p_drift_pa_per_h`` field becomes a float and is used
    # in place of the underlying profile knob. ``None`` falls through
    # to the configured profile value (external disturbances default).
    def disturbances(self, t: np.ndarray) -> np.ndarray:
        pfaults = self._pfaults                  # injected by Simulation during build
        if pfaults is None:
            # Fallback: zero-amplitude profile. Callers without
            # pfaults binding get a constant nominal disturbance
            # profile (Tambient = 293.15 K, etc).
            return np.column_stack([
                np.full(len(t), 293.15),
                np.full(len(t), 288.15),
                np.full(len(t), 4.0e5),
            ])
        # operator disturbance knobs: resolve operator-driven knob overlays onto the
        # baseline profile knobs. Read-once here so the kernel stays
        # a single read per attribute per step.
        amb_mean = (
            pfaults.live_ambient_mean_k
            if pfaults.live_ambient_mean_k is not None
            else pfaults.ambient_t_mean_k
        )
        amb_amp = (
            pfaults.live_ambient_amplitude_k
            if pfaults.live_ambient_amplitude_k is not None
            else pfaults.ambient_t_amplitude_k
        )
        cw_mean = (
            pfaults.live_cw_t_mean_k
            if pfaults.live_cw_t_mean_k is not None
            else pfaults.cw_t_mean_k
        )
        cw_drift = (
            pfaults.live_cw_p_drift_pa_per_h
            if pfaults.live_cw_p_drift_pa_per_h is not None
            else pfaults.cw_p_drift_pa_per_h
        )
        amb = (
            amb_mean
            + amb_amp
            * np.sin(2.0 * np.pi * t / pfaults.ambient_t_period_s)
        )
        cw_t = (
            cw_mean
            + pfaults.cw_t_amplitude_k
            * np.sin(2.0 * np.pi * t / pfaults.cw_t_period_s)
        )
        # CW pressure: nominal + signed drift over time + jitter.
        # Drift knob is the only override-prone parameter; the
        # override uses ``cw_drift`` resolved above.
        drift_pa = cw_drift * t
        if pfaults.cw_p_noise_pa > 0.0:
            jitter = pfaults.cw_p_noise_pa * np.random.randn(len(t))
        else:
            jitter = 0.0
        cw_p = pfaults.cw_p_nominal_pa + drift_pa + jitter
        return np.column_stack([amb, cw_t, cw_p])

    # Internal binding: ``Simulation._setup`` patches this with the
    # active ``ProcessFaults`` so ``disturbances(t)`` can read the
    # knobs. Public API is still ``Settings.disturbances(t)`` so
    # callers don't need to thread pfaults through every call.
    _pfaults = None

    # Loop wiring (1-based in upstream; converted to 0-based at Simulation build time)
    mode_1b: np.ndarray = field(default_factory=lambda: np.array([1, 0, 1, 1]))
    pvindex_1b: np.ndarray = field(default_factory=lambda: np.array([1, 2, 3, 4]))
    uindex_1b: np.ndarray = field(default_factory=lambda: np.array([4, 5, 6, 1]))

    sp1: float = 60.4 + 273.15                                # TR setpoint, K
    sp2: float = 50.0 + 273.15                                # TD setpoint, K
    sp3: float = 0.5                                          # hH setpoint, m
    sp4: float = 3000.0                                       # Foil setpoint, kg/h

    nic: int = 4                                              # controller update every nic steps

    # ------------------------------------------------------------------ #
    # Live-mutable setpoints (Roadmap step 4). The ``LiveSimulator`` reads
    # these on every PID tick; ``POST /control`` writes into them so the
    # PID picks up the change on the next ``nic`` boundary.
    #
    # Important: ``simulation.run_with`` (the batch driver) does NOT honor
    # ``live_sp*`` — it reads the static ``sp1..sp4`` once at setup. Mutating
    # ``live_sp*`` only takes effect via ``LiveSimulator``. If you need a
    # setpoint sweep in a batch run, edit ``sp1..sp4`` directly before calling
    # ``run_with`` (or use the pre-baked ``sp[:, 3] += 100.0 * heaviside(...)``
    # style that simulation.py uses for ``sp4``).
    #
    # The ``__post_init__`` mirror keeps ``live_sp*`` seeded from ``sp*`` so
    # a freshly-built ``LiveSimulator`` starts at the documented setpoints.
    # Do not write to ``sp*`` and ``live_sp*`` separately — pick one.
    # ------------------------------------------------------------------ #
    live_sp1: float = 0.0
    live_sp2: float = 0.0
    live_sp3: float = 0.0
    live_sp4: float = 0.0

    def __post_init__(self) -> None:
        """Validate ``Settings`` at construction time (issue #9), then
        seed the live setpoints from the configured sp* values.

        Time axis must satisfy ``tf > ti`` and ``dt > 0``. ``u0``
        must be a length-6 finite vector. The loop-wiring index arrays
        (``mode_1b``, ``pvindex_1b``, ``uindex_1b``) must each be length 4
        with integer entries in ``[0, 6)``.
        """
        _check_finite("ti", self.ti)
        _check_finite("tf", self.tf)
        if not (self.tf > self.ti):
            raise _err("tf", f"must be > ti ({self.ti}); got {self.tf}")
        _check_positive("dt", self.dt)

        if self.u0 is None:
            raise _err("u0", "must be a numpy array of length 6")
        if not isinstance(self.u0, np.ndarray):
            raise TypeError(
                f"u0: expected np.ndarray, got {type(self.u0).__name__}"
            )
        _check_shape("u0", self.u0, (6,))
        _check_finite_array("u0", self.u0)

        if self.sv0 is not None:
            if not isinstance(self.sv0, np.ndarray):
                raise TypeError(
                    f"sv0: expected np.ndarray or None, got {type(self.sv0).__name__}"
                )
            _check_finite_array("sv0", self.sv0)

        for name in ("mode_1b", "pvindex_1b", "uindex_1b"):
            v = getattr(self, name)
            if not isinstance(v, np.ndarray):
                raise TypeError(
                    f"{name}: expected np.ndarray, got {type(v).__name__}"
                )
            _check_shape(name, v, (4,))
            # The arrays carry 1-based indices from the upstream MATLAB;
            # require integer-typed or castable, value range loose.
            if not np.all((v >= 0) & (v < 1000)):
                raise _err(
                    name,
                    "all entries must be non-negative integers < 1000",
                )

        if not isinstance(self.nic, (int, np.integer)) or isinstance(self.nic, bool):
            raise TypeError(
                f"nic: expected int, got {type(self.nic).__name__}"
            )
        if self.nic < 1:
            raise _err("nic", f"must be >= 1; got {self.nic}")

        for name in ("sp1", "sp2", "sp3", "sp4"):
            _check_finite(name, getattr(self, name))

        # Seed live setpoints from configured sp* values (existing
        # behavior — preserved verbatim).
        self.live_sp1 = self.sp1
        self.live_sp2 = self.sp2
        self.live_sp3 = self.sp3
        self.live_sp4 = self.sp4


# -----------------------------------------------------------------------------
# Per-step result (live simulator)
# -----------------------------------------------------------------------------


@dataclass
class StepResult:
    """Per-step payload returned by :class:`bdsim.simulation.LiveSimulator.step`.

    Mirrors the per-row structure of :class:`Results` but for a single time
    step. All arrays have the same column shapes as the batched version
    (``pv`` is ``(nsensors,)``, ``uv`` is ``(6,)``, ``sv`` is ``(21,)``,
    ``sp`` is ``(4,)``).

    ``quality`` maps sensor-index → OPC-style quality code (``"good"``,
    ``"bad"``, ``"uncertain"``). It is the public signal for sensor
    faults (dropout / stuck / bias) so downstream consumers do not need
    to inspect the sim state directly.

    ``quality_latched`` is the lab-cycle latched measurement payload
    (quality latching). Shape ``(3,)`` with ``[FAME%, water_ppm, IV]``. Empty
    array when ``ProcessFaults.quality_state`` is False.

    See :class:`bdsim.simulation.LiveSimulator` for the canonical usage.
    """

    t: float                                                # sim time at end of step, seconds
    pv: np.ndarray                                          # measurements, (nsensors,)
    uv: np.ndarray                                          # input vars,  (6,)
    sv: np.ndarray                                          # state vars,  (22,) — includes HEX fouling α at [21]
    sp: np.ndarray                                          # setpoints,   (4,)
    quality: dict[int, str] = field(default_factory=dict)   # sensor idx → quality
    quality_latched: np.ndarray | None = None               # lab-cycle latched values, (3,) when quality_state=True
    disturbances: np.ndarray | None = None                  # external disturbances: (3,) [Tamb, Tcw, Pcw]; None when off
    xLend: np.ndarray | None = None                         # washer/dryer output, (6,)
    yLend: np.ndarray | None = None                         # dryer mass fractions, (6,)
    spectra: SpectrumSample | None = None                    # NIR/IR spectrum sensor: SpectrumSample at fire times, else None


# -----------------------------------------------------------------------------
# Results container
# -----------------------------------------------------------------------------

@dataclass
class Results:
    """Trajectories returned by :func:`run`.

    All time series are length ``lt-1`` (the upstream drops the final point
    because the last ODE integration step has no downstream measurement;
    we follow the same convention).
    """

    t: np.ndarray                                            # seconds
    uv: np.ndarray                                            # input vars,  (lt-1, 6)
    sv: np.ndarray                                            # state vars,  (lt-1, 22) — includes HEX fouling α at [21]
    pv: np.ndarray                                            # measurements, (lt-1, 5)
    sp: np.ndarray                                            # setpoints,   (lt-1, 4)
    xLend: np.ndarray                                         # light-phase end comp, (lt-1, 6)
    yLend: np.ndarray                                         # light-phase mass frac, (lt-1, 6)
    tclean: np.ndarray                                        # filter cleaning times, s
    quality: np.ndarray | None = None                         # latched lab samples, (lt-1, 3): [FAME%, water ppm, IV]
                                                              # quality latching — present when quality_state=True; NaN rows otherwise
    disturbances: np.ndarray | None = None                    # external disturbances: external disturbance track,
                                                              # (lt-1, 3): [Tamb_K, Tcw_K, Pcw_Pa].
    factor: np.ndarray | None = None                          # fouling-mode windows: HEX fouling factor applied at each
                                                              # step (length lt-1). Populated by both the
                                                              # batch ``run_with`` path and the
                                                              # ``LiveSimulator`` path so downstream
                                                              # consumers (tests, dashboard, external
                                                              # fault-detection) can inspect which path
                                                              # (continuous α / windowed mode 4-5 /
                                                              # static legacy) the kernel used.

    # Display-unit conversions (matching upstream's final plotting block)
    def in_display_units(self) -> Results:
        """Convert to °C / kg/h as the upstream plotting block does.

        Returns a *new* Results object; original data is unchanged.
        """
        r = copy.deepcopy(self)

        r.uv[:, 1] -= 273.15
        r.uv[:, 3] -= 273.15
        r.sv[:, 6] -= 273.15
        r.sv[:, 17] -= 273.15
        r.pv[:, 0] -= 273.15
        r.pv[:, 1] -= 273.15
        r.sp[:, 0] -= 273.15
        r.sp[:, 1] -= 273.15
        r.pv[:, 3] *= 3600.0
        r.sp[:, 3] *= 3600.0
        r.uv[:, 2] = r.uv[:, 2] * 0.032 * 3600.0   # kg/h for methanol, Mm = 0.032 kg/mol
        return r
