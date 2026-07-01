"""
Typed configuration objects (replacing the script-style ``system_parameters.m``
and ``user_settings.m`` of the MATLAB upstream).

The defaults reproduce exactly the values from the upstream ``user_settings.m``
so that ``run()`` reproduces the upstream trajectory to within solver tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


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
    vmolo_local: float = 0.0
    cpmolo_local: float = 0.0

    # Filter constants — populated from process faults at startup
    K1F: float = 0.0
    K2F: float = 0.0
    K3F: float = 0.0
    K4F: float = 0.0

    # ---- HEX fouling dynamics (Roadmap Layer 2.5) -------------------------
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

    def finalize(self) -> None:
        """Recompute derived quantities that depend on M and ro."""
        self.vmol = self.M / self.ro
        self.vmolo_local = self.vmolo
        self.cpmolo_local = self.cpmolo
        self.cpmolm = self.cpmolm


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
    fouling_dynamic: bool = True                              # Layer 2.5: α evolves as a state when True
                                                              # When False, behaviour matches the legacy
                                                              # pre-baked series (factor = 1/(1 + Rf)).


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
    # Live fault knobs (mid-run mutable for the Lepanto FDE integration).
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
    # Live-mutable setpoints (Roadmap step 4). The ``LiveSimulator`` mirrors
    # the scalar ``sp1..sp4`` into these on construction; ``POST /control``
    # writes into them so the PID picks up the change on the next ``nic``
    # boundary. The mirror is kept in sync — callers should not write to
    # both.
    # ------------------------------------------------------------------ #
    live_sp1: float = 0.0
    live_sp2: float = 0.0
    live_sp3: float = 0.0
    live_sp4: float = 0.0

    def __post_init__(self) -> None:
        # Always re-sync from the scalar defaults after dataclass init.
        # The ``live_sp*`` fields exist so external code can mutate them
        # at runtime; the baseline values come from ``sp1..sp4``.
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

    See :class:`bdsim.simulation.LiveSimulator` for the canonical usage.
    """

    t: float                                                # sim time at end of step, seconds
    pv: np.ndarray                                          # measurements, (nsensors,)
    uv: np.ndarray                                          # input vars,  (6,)
    sv: np.ndarray                                          # state vars,  (22,) — includes HEX fouling α at [21]
    sp: np.ndarray                                          # setpoints,   (4,)
    quality: dict[int, str] = field(default_factory=dict)   # sensor idx → quality
    xLend: np.ndarray | None = None                         # washer/dryer output, (6,)
    yLend: np.ndarray | None = None                         # dryer mass fractions, (6,)


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

    # Display-unit conversions (matching upstream's final plotting block)
    def in_display_units(self) -> "Results":
        """Convert to °C / kg/h as the upstream plotting block does.

        Returns a *new* Results object; original data is unchanged.
        """
        import copy
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
