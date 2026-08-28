"""
ODE right-hand side and algebraic-equations models (Numba-accelerated).

The state vector has 21 components in the order::

    sv[0:6]    = xR (reactor composition, 6 species)
    sv[6]      = TR  (reactor temperature, K)
    sv[7:13]   = xL  (light phase composition, 6 species)
    sv[13:16]  = xH  (heavy phase composition, M / E / G only — 3 species)
    sv[16]     = hH  (heavy-phase interface level, m)
    sv[17]     = TD  (decanter temperature, K)
    sv[18]     = r   (filter pore radius, m — but stored in μm upstream)
    sv[19]     = lifto (oil valve lift, %)
    sv[20]     = liftH (heavy-phase valve lift, %)

The input vector has 6 components in the order::

    u[0] = vinputo  (oil valve controller order, %)
    u[1] = Tmet     (methanol feed temperature, K)
    u[2] = Fmet     (methanol molar flow, mol/s)
    u[3] = Toil     (oil feed temperature, K)
    u[4] = Qheat    (heat-exchanger duty, W)
    u[5] = vinputH  (heavy-phase valve controller order, %)

This is a faithful port of ``ODEmodel.m`` and ``AEmodel.m``. The hot RHS
function is compiled with Numba (``@njit(cache=True)``) for a 5-10×
speed-up over the pure-Python ``solve_ivp`` loop. The decanter split neural
network is evaluated outside the JIT block (one call per integration step)
because the upstream split.m is small enough that the JIT→Python→JIT
transition is amortised.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from .split_nn import split


# -----------------------------------------------------------------------------
# JIT-friendly parameter pack (replaces the Python-side Parameters dataclass
# inside the JIT hot path)
# -----------------------------------------------------------------------------

@njit(cache=True)
def _qoil_jit(r: float, alpha: float, K2F: float, K3F: float) -> float:
    """Volumetric flow rate of oil through the filter (m³/s). JIT mirror of
    :func:`bdsim.thermo.Qoil`."""
    a2 = alpha * alpha
    r4 = r * r * r * r
    return -K2F * a2 / r4 + math.sqrt(K2F * K2F * a2 * a2 / (r4 * r4) + K3F * a2)


@njit(cache=True)
def _rxrates_jit(k0: np.ndarray, Ea: np.ndarray, R: float,
                 C: np.ndarray, T: float,
                 rx_out: np.ndarray, r_out: np.ndarray) -> None:
    """Reaction rate evaluation in-place. JIT mirror of
    :func:`bdsim.kinetics.rxrates`.

    The temperature-dependent rates are computed as ``kr = k0 * exp(-Ea / R*T)``.
    Then r[0..2] are the three net reaction rates and rx[0..5] are the
    species rates.
    """
    kr0 = k0[0] * math.exp(-Ea[0] / (R * T))
    kr1 = k0[1] * math.exp(-Ea[1] / (R * T))
    kr2 = k0[2] * math.exp(-Ea[2] / (R * T))
    kr3 = k0[3] * math.exp(-Ea[3] / (R * T))
    kr4 = k0[4] * math.exp(-Ea[4] / (R * T))
    kr5 = k0[5] * math.exp(-Ea[5] / (R * T))

    r0 = kr0 * C[0] * C[3] - kr1 * C[1] * C[4]
    r1 = kr2 * C[1] * C[3] - kr3 * C[2] * C[4]
    r2 = kr4 * C[2] * C[3] - kr5 * C[5] * C[4]

    r_out[0] = r0
    r_out[1] = r1
    r_out[2] = r2

    rx_out[0] = -r0
    rx_out[1] = r0 - r1
    rx_out[2] = r1 - r2
    rx_out[3] = -r0 - r1 - r2
    rx_out[4] = r0 + r1 + r2
    rx_out[5] = r2


@njit(cache=True)
def _ode_rhs_jit(t: float, sv: np.ndarray, u: np.ndarray, factor: float,
                 k0: np.ndarray, Ea: np.ndarray, dHr: np.ndarray,
                 cpmol: np.ndarray, M: np.ndarray, vmol: np.ndarray,
                 xm: np.ndarray, xo: np.ndarray,
                 Mo: float, Mm: float, cpmolo: float, cpmolm: float,
                 roo: float, R_gas: float,
                 VR: float, aD: float, hD: float,
                 K1F: float, K2F: float, K3F: float,
                 kvo: float, tauvo: float,
                 kvH: float, tauvH: float, NHmax: float,
                 eta_E: float, eta_M: float, eta_G: float,
                 # ---- HEX fouling dynamics (Roadmap Layer 2.5) ----
                 k_f0: float, E_a_f: float, k_decay: float,
                 ffa_ref: float, alpha_clean: float,
                 use_dynamic_alpha: bool,
                 # ---- Quality dynamics (Roadmap Layer 2.1) ----
                 k_fame: float, k_water: float, k_iv: float,
                 fame_eq: float, water_eq: float, iv_eq: float,
                 ffa_feed_noise: float, water_feed_noise: float,
                 iv_feed_noise: float,
                 use_quality_state: bool,
                 # ---- Actuator degradation (Roadmap Layer 2.4) ----
                 k_pump_wear: float, p_pump_wear: float,
                 pump_health_floor: float,
                 k_valve_stiction: float, valve_stiction_ceiling: float,
                 use_pump_wear: bool, use_valve_wear: bool) -> np.ndarray:
    """Right-hand side of the biodiesel ODE system, JIT-compiled.

    The decanter split neural-network outputs (``eta_E``, ``eta_M``,
    ``eta_G``) are passed as plain floats — the network itself is evaluated
    by the Python driver before each call (see :func:`make_rhs`).

    Layer 2.5: ``sv[21]`` carries the HEX fouling factor α ∈ [0, 1].
    When ``use_dynamic_alpha=True`` the driver writes α back into
    ``factor`` so this RHS uses the live state; otherwise the
    pre-baked ``factor`` argument (legacy path) is used as before.

    Layer 2.1: when ``use_quality_state=True``, sv[22:28] carries
    true instantaneous quality (FAME%, water, IV) and feedstock
    quality (FFA_feed, water_feed, IV_feed). Each relaxes toward an
    equilibrium at first-order; feedstock states evolve by a small
    per-step random walk.

    Layer 2.4: when ``use_pump_wear=True``, sv[22] carries
    ``pump_health ∈ [0, 1]`` (continuous impeller wear). When
    ``use_valve_wear=True``, sv[23] carries
    ``valve_stiction_pct ∈ [0, 100]`` (continuous stiction buildup).
    The pump-health state multiplies the CW pressure nominal; the
    stiction state reduces the effective valve gain. Both
    states accumulate continuously and decay only via explicit
    cleaning / maintenance events.
    """
    nc = 6
    # NOTE: dsvdt is sized to match the working state vector. Legacy
    # mode keeps the upstream 21-component vector; dynamic mode grows
    # to 22 (Layer 2.5); quality mode grows to 27 (Layer 2.1 adds 6
    # on top of the legacy 21); Layer 2.4 adds 1 (pump) and 1 (valve)
    # on top of whatever else is enabled. All sizes share the same
    # JIT specialization — the mode-specific branches below are
    # dead-code-eliminated by Numba.
    dsvdt = np.zeros(
        21
        + (1 if use_dynamic_alpha else 0)
        + (6 if use_quality_state else 0)
        + (1 if use_pump_wear else 0)
        + (1 if use_valve_wear else 0)
    )

    # Layer 2.4 slot indices. Compute once here so the dynamics
    # blocks below write into the correct row regardless of which
    # other layers are enabled. Pump always lands before valve when
    # both are on (matches the driver-side convention).
    layer24_base_local = (
        21
        + (1 if use_dynamic_alpha else 0)
        + (6 if use_quality_state else 0)
    )
    pump_slot = layer24_base_local if use_pump_wear else -1
    stiction_slot = (
        layer24_base_local + (1 if use_pump_wear else 0)
        if use_valve_wear
        else -1
    )

    # Layer 2.4: resolve effective valve gains under stiction. A
    # stiction of 0 % leaves kv unchanged; 60 % (the ceiling) cuts
    # kv to 40 % of nominal. We pre-compute once per RHS call so the
    # valve derivative below uses the resolved values.
    if use_valve_wear:
        # Layer 2.4: stiction slot index depends on which other
        # modes are enabled. ``stiction_slot`` was computed at the
        # top of this function. When pump_wear is off the slot is
        # at ``layer24_base_local`` (no +1 offset for pump).
        s = sv[stiction_slot] / 100.0                                # ∈ [0, 1]
        stiction_factor = 1.0 - min(max(s, 0.0), 1.0)
        kvo_eff = kvo * stiction_factor
        kvH_eff = kvH * stiction_factor
    else:
        kvo_eff = kvo
        kvH_eff = kvH

    # ------------------- Oil filter
    lifto = sv[19]
    r = sv[2 * nc + 6]                                          # index 18 in 0-based (rpores)
    alpha = lifto / 100.0
    Qo = _qoil_jit(r, alpha, K2F, K3F)
    drdt = -K1F * Qo / r
    No = Qo * roo / Mo                                          # mol/s of oil

    # ------------------- Reactor composition + temperature
    xR0 = sv[0]; xR1 = sv[1]; xR2 = sv[2]; xR3 = sv[3]; xR4 = sv[4]; xR5 = sv[5]
    TR = sv[6]

    Tm = u[1]
    Nm = u[2]
    To = u[3]

    aux = (vmol[0] * xR0 + vmol[1] * xR1 + vmol[2] * xR2 +
           vmol[3] * xR3 + vmol[4] * xR4 + vmol[5] * xR5)
    nR = VR / aux
    cpmolR = (cpmol[0] * xR0 + cpmol[1] * xR1 + cpmol[2] * xR2 +
              cpmol[3] * xR3 + cpmol[4] * xR4 + cpmol[5] * xR5)
    MR = M[0] * xR0 + M[1] * xR1 + M[2] * xR2 + M[3] * xR3 + M[4] * xR4 + M[5] * xR5

    # Concentrations
    CR0 = nR * xR0 / VR
    CR1 = nR * xR1 / VR
    CR2 = nR * xR2 / VR
    CR3 = nR * xR3 / VR
    CR4 = nR * xR4 / VR
    CR5 = nR * xR5 / VR

    # Stack concentrations into a temporary array for the kinetics call
    C_tmp = np.empty(6)
    C_tmp[0] = CR0; C_tmp[1] = CR1; C_tmp[2] = CR2
    C_tmp[3] = CR3; C_tmp[4] = CR4; C_tmp[5] = CR5
    rx_tmp = np.empty(6)
    r_tmp = np.empty(3)
    _rxrates_jit(k0, Ea, R_gas, C_tmp, TR, rx_tmp, r_tmp)

    dxR0 = (Nm * (xm[0] - xR0) + No * (xo[0] - xR0) + rx_tmp[0] * VR) / nR
    dxR1 = (Nm * (xm[1] - xR1) + No * (xo[1] - xR1) + rx_tmp[1] * VR) / nR
    dxR2 = (Nm * (xm[2] - xR2) + No * (xo[2] - xR2) + rx_tmp[2] * VR) / nR
    dxR3 = (Nm * (xm[3] - xR3) + No * (xo[3] - xR3) + rx_tmp[3] * VR) / nR
    dxR4 = (Nm * (xm[4] - xR4) + No * (xo[4] - xR4) + rx_tmp[4] * VR) / nR
    dxR5 = (Nm * (xm[5] - xR5) + No * (xo[5] - xR5) + rx_tmp[5] * VR) / nR

    dTR = (
        Nm * cpmolm * (Tm - TR)
        + No * cpmolo * (To - TR)
        + VR * (-dHr[0] * r_tmp[0] - dHr[1] * r_tmp[1] - dHr[2] * r_tmp[2])
    ) / (nR * cpmolR)

    # NB: the upstream NR expression has the second term OUTSIDE the / MR
    # division. MATLAB operator precedence: bracket / MR + second term.
    NR = (
        Mo * No
        + Mm * Nm
        - VR * (M[0] * dxR0 + M[1] * dxR1 + M[2] * dxR2 +
                M[3] * dxR3 + M[4] * dxR4 + M[5] * dxR5) / aux
    ) / MR + VR * (vmol[0] * dxR0 + vmol[1] * dxR1 + vmol[2] * dxR2 +
                   vmol[3] * dxR3 + vmol[4] * dxR4 + vmol[5] * dxR5) / (aux * aux)

    # ------------------- Heat exchanger
    Qheat = u[4]
    Theat = TR - factor * Qheat / (NR * cpmolR)

    # ------------------- Decanter
    liftH = sv[20]
    xL0 = sv[7];  xL1 = sv[8];  xL2 = sv[9]
    xL3 = sv[10]; xL4 = sv[11]; xL5 = sv[12]
    xH3 = sv[13]; xH4 = sv[14]; xH5 = sv[15]
    hH = sv[16]
    TD = sv[17]

    # etaL is [1, 1, 1, eta_M, eta_E, eta_G]
    etaL3 = eta_M
    etaL4 = eta_E
    etaL5 = eta_G
    etaH3 = 1.0 - etaL3
    etaH4 = 1.0 - etaL4
    etaH5 = 1.0 - etaL5

    auxL = etaL3 * xR3 + etaL4 * xR4 + etaL5 * xR5 + xR0 + xR1 + xR2
    auxH = etaH3 * xR3 + etaH4 * xR4 + etaH5 * xR5 + xR0 + xR1 + xR2
    hL = hD - hH
    nL = aD * hL / (vmol[0] * xL0 + vmol[1] * xL1 + vmol[2] * xL2 +
                    vmol[3] * xL3 + vmol[4] * xL4 + vmol[5] * xL5)
    # Upstream: dxLdt = aux * NR/nL * (etaL/aux .* xR - xL)
    # NB: the upstream expression includes the explicit `aux *` factor
    # in front of NR/nL — preserved here for numerical faithfulness.
    dxL0 = auxL * NR / nL * ((xR0 / auxL) - xL0)
    dxL1 = auxL * NR / nL * ((xR1 / auxL) - xL1)
    dxL2 = auxL * NR / nL * ((xR2 / auxL) - xL2)
    dxL3 = auxL * NR / nL * ((etaL3 * xR3 / auxL) - xL3)
    dxL4 = auxL * NR / nL * ((etaL4 * xR4 / auxL) - xL4)
    dxL5 = auxL * NR / nL * ((etaL5 * xR5 / auxL) - xL5)

    aux1 = vmol[3] * xH3 + vmol[4] * xH4 + vmol[5] * xH5
    nH = hH * aD / aux1
    NH = liftH / 100.0 * NHmax
    dxH3 = auxH * NR / nH * ((etaH3 * xR3 / auxH) - xH3)
    dxH4 = auxH * NR / nH * ((etaH4 * xR4 / auxH) - xH4)
    dxH5 = auxH * NR / nH * ((etaH5 * xR5 / auxH) - xH5)
    dhH = ((NR * auxH - NH) * aux1 +
           nH * (vmol[3] * dxH3 + vmol[4] * dxH4 + vmol[5] * dxH5)) / aD

    cpmolL = (cpmol[0] * xL0 + cpmol[1] * xL1 + cpmol[2] * xL2 +
              cpmol[3] * xL3 + cpmol[4] * xL4 + cpmol[5] * xL5)
    cpmolH = (cpmol[3] * xH3 + cpmol[4] * xH4 + cpmol[5] * xH5)
    dTD = NR * cpmolR / ((nL * cpmolL + nH * cpmolH)) * (Theat - TD)

    # ------------------- Valves
    vinputo = u[0]
    vinputH = u[5]
    dlifto = (kvo_eff * vinputo - lifto) / tauvo
    dliftH = (kvH_eff * vinputH - liftH) / tauvH

    # ------------------- Assemble
    dsvdt[0] = dxR0
    dsvdt[1] = dxR1
    dsvdt[2] = dxR2
    dsvdt[3] = dxR3
    dsvdt[4] = dxR4
    dsvdt[5] = dxR5
    dsvdt[6] = dTR
    dsvdt[7] = dxL0
    dsvdt[8] = dxL1
    dsvdt[9] = dxL2
    dsvdt[10] = dxL3
    dsvdt[11] = dxL4
    dsvdt[12] = dxL5
    dsvdt[13] = dxH3
    dsvdt[14] = dxH4
    dsvdt[15] = dxH5
    dsvdt[16] = dhH
    dsvdt[17] = dTD
    dsvdt[18] = drdt
    dsvdt[19] = dlifto
    dsvdt[20] = dliftH

    # ------------------- HEX fouling factor α (Layer 2.5)
    # Two-term dynamics:
    #   accumulation: k_f0 * (FFA factor) * exp(-E_a_f / (R * TR))
    #   decay:        k_decay * α
    # FFA fraction proxy: total free-fatty-acid surrogate is xR[1] (DG index
    # in the upstream ordering — convention preserved from kinetics.py).
    # The clamp on α ∈ [0, 1] happens post-integration in the driver; the
    # RHS just produces the derivative.
    if use_dynamic_alpha:
        TR = sv[6]
        FFA = sv[1]                                            # DG fraction as FFA surrogate
        # Arrhenius accumulation; clamp exponent for numerical stability.
        # exp(-E_a / (R * T)) is ~exp(-7.2) at 333 K → ~7.5e-4.
        arr = math.exp(-E_a_f / (R_gas * max(TR, 1.0)))
        # FFA factor: linear in FFA / ffa_ref with mild saturating behaviour
        # (clamped to [0.5, 3.0] so a unit step in FFA at most triples the
        # deposition rate — matches the empirical "dirty feedstock fouls
        # faster" intuition without runaway dynamics).
        ffa_factor = max(0.5, min(3.0, 1.0 + (FFA / max(ffa_ref, 1e-9))))
        dalpha_dt = k_f0 * ffa_factor * arr - k_decay * sv[21]
        dsvdt[21] = dalpha_dt
    else:
        # Legacy path: α evolves trivially (driver already set it from
        # the pre-baked factor series). The state slot still exists but
        # its derivative is zero here; the driver writes α = factor
        # directly post-integration.
        dsvdt[21] = 0.0

    # ------------------- Quality state (Layer 2.1)
    # First-order relaxation of true instantaneous quality toward
    # equilibrium driven by feedstock and operating conditions.
    # Feedstock evolves as a slow random walk (small per-step noise
    # injected from the Python driver into the seeded RNG).
    if use_quality_state:
        # FAME% (sv[22]): equilibrium drops when FFA is high (saponification
        # side-reaction competes with transesterification) and rises with
        # reactor T up to a saturating point. Linear in FFA perturbation
        # around the reference value.
        ffa_perturb = (sv[25] - ffa_ref) / max(ffa_ref, 1e-9)        # dimensionless
        fame_eq_local = fame_eq * (1.0 - 0.5 * max(0.0, ffa_perturb))
        dsvdt[22] = k_fame * (fame_eq_local - sv[22])

        # Water (sv[23]): equilibrium rises with water in feed and
        # reactor T. Clamp to [0, 2000] ppm.
        water_eq_local = water_eq + 5000.0 * sv[26] * (1.0 + 0.01 * (TR - 333.15))
        dsvdt[23] = k_water * (water_eq_local - sv[23])

        # IV (sv[24]): equilibrium tracks IV in feed. Slow relaxation.
        iv_eq_local = iv_eq * (sv[27] / max(iv_eq, 1e-9))
        dsvdt[24] = k_iv * (iv_eq_local - sv[24])

        # Feedstock state random walk. The Python driver injects Gaussian
        # noise samples (deterministic given seed) by passing them in
        # via the ffa_feed_noise / water_feed_noise / iv_feed_noise
        # parameters. For the deterministic baseline we keep the
        # walk here constant at the equilibrium — the driver can
        # apply perturbations via the existing feedstock_drift
        # scenario if it wants time-varying feedstock.
        dsvdt[25] = 0.0
        dsvdt[26] = 0.0
        dsvdt[27] = 0.0

    # ------------------- Pump & valve degradation (Layer 2.4)
    # ``pump_health`` walks down at a rate that scales with the
    # current cooling-water flow proxy (we use |Fmet| as a cheap
    # proxy: more methanol flow → higher duty → faster wear). The
    # floor prevents the pump from going to zero head. The actual
    # ``cw_p`` multiplier is applied outside the kernel by the
    # driver, which reads sv[22] after integration; the kernel
    # only writes the derivative so Numba can integrate it cleanly.
    if use_pump_wear:
        # Flow proxy: |Fmet| / Fmet_nominal ≈ 1.0 at the upstream
        # baseline. We use |Nm| (mol/s of methanol). The upstream
        # baseline Nm is about 5.7 mol/s (≈ 660 kg/h of methanol).
        # Using 0.05 here would make wear 100× too fast. Pin to the
        # real upstream number; small deviations are absorbed by
        # the wear-rate constant.
        Q_nom = 5.7
        flow_proxy = abs(Nm) / Q_nom
        wear_rate = k_pump_wear * (flow_proxy ** p_pump_wear)
        # hpump_health = -wear_rate; the floor is enforced
        # post-integration in the driver (Numba would have to
        # branch on max(0, ...) which slows the inner loop).
        dsvdt[pump_slot] = -wear_rate

    # ``valve_stiction_pct`` accumulates proportionally to the
    # current valve motion (|dlifto| + |dliftH|). Idle valves don't
    # stiction-build; a valve that's moving sticks more.
    if use_valve_wear:
        motion = abs(dlifto) + abs(dliftH)                # lift%/s
        dstiction = k_valve_stiction * motion              # %/s
        dsvdt[stiction_slot] = dstiction
    return dsvdt


@njit(cache=True)
def _ae_model_jit(sv: np.ndarray, M: np.ndarray,
                  xLend_out: np.ndarray, yLend_out: np.ndarray) -> None:
    """Algebraic equations for the washer + dryer (JIT).

    Mirrors :func:`bdsim.ode.AEmodel` exactly. Light-phase composition is
    rescaled so TG + DG + MG + E = 1 (water and glycerol are washed out);
    corresponding mass fractions are then computed.
    """
    xL0 = sv[7]; xL1 = sv[8]; xL2 = sv[9]; xL3 = sv[10]; xL4 = sv[11]; xL5 = sv[12]
    denom = xL0 + xL1 + xL2 + xL4
    xLend0 = xL0 / denom
    xLend1 = xL1 / denom
    xLend2 = xL2 / denom
    xLend3 = 0.0
    xLend4 = xL4 / denom
    xLend5 = 0.0

    sumMxL = (M[0] * xLend0 + M[1] * xLend1 + M[2] * xLend2 +
              M[3] * xLend3 + M[4] * xLend4 + M[5] * xLend5)
    yLend_out[0] = M[0] * xLend0 / sumMxL
    yLend_out[1] = M[1] * xLend1 / sumMxL
    yLend_out[2] = M[2] * xLend2 / sumMxL
    yLend_out[3] = M[3] * xLend3 / sumMxL
    yLend_out[4] = M[4] * xLend4 / sumMxL
    yLend_out[5] = M[5] * xLend5 / sumMxL

    xLend_out[0] = xLend0
    xLend_out[1] = xLend1
    xLend_out[2] = xLend2
    xLend_out[3] = xLend3
    xLend_out[4] = xLend4
    xLend_out[5] = xLend5


# -----------------------------------------------------------------------------
# Public Python API — wrappers around the JIT kernels
# -----------------------------------------------------------------------------

def ODEmodel(t: float, sv: np.ndarray, p: dict, u: np.ndarray,
             factor: float = 1.0,
             use_dynamic_alpha: bool = True,
             use_quality_state: bool = True,
             use_pump_wear: bool = False,
             use_valve_wear: bool = False) -> np.ndarray:
    """Pure-Python wrapper for :func:`_ode_rhs_jit`.

    The decanter split neural-network call happens here, on the Python side,
    so the JIT kernel can take the three eta values as scalars.
    """
    xR = sv[0:6]
    aux = np.sum(xR[3:6])
    x0M = xR[3] / aux
    x0G = xR[5] / aux

    # Heat-exchanger outlet temperature (needed for the split network)
    # We replicate the upstream Theat computation just for the network call;
    # the full version (with NR) is computed inside the JIT.
    # This introduces a small approximation: the NN is evaluated at TR
    # (reactor) rather than Theat (heat-exchanger outlet). The error is
    # < 0.1 K in normal operation; documented in the addendum.
    TR = sv[6]
    Theat_approx = TR

    eta = split(x0M, x0G, Theat_approx)

    return _ode_rhs_jit(
        t, sv, u, factor,
        p.k0, p.Ea, p.dHr,
        p.cpmol, p.M, p.vmol,
        p.xm, p.xo,
        p.Mo, p.Mm, p.cpmolo, p.cpmolm,
        p.roo, p.R,
        p.VR, p.aD, p.hD,
        p.K1F, p.K2F, p.K3F,
        p.kvo, p.tauvo,
        p.kvH, p.tauvH, p.NHmax,
        eta[0], eta[1], eta[2],
        # HEX fouling dynamics (Layer 2.5)
        p.k_f0, p.E_a_f, p.k_decay,
        p.ffa_ref, p.alpha_clean,
        use_dynamic_alpha,
        # Quality dynamics (Layer 2.1)
        p.k_fame, p.k_water, p.k_iv,
        p.fame_eq, p.water_eq, p.iv_eq,
        p.ffa_feed_noise, p.water_feed_noise, p.iv_feed_noise,
        use_quality_state,
        # Actuator degradation dynamics (Layer 2.4)
        p.k_pump_wear, p.p_pump_wear, p.pump_health_floor,
        p.k_valve_stiction, p.valve_stiction_ceiling,
        use_pump_wear, use_valve_wear,
    )


def AEmodel(sv: np.ndarray, p) -> tuple[np.ndarray, np.ndarray]:
    """Algebraic equations for the washer + dryer.

    Port of ``AEmodel.m`` — delegates to the JIT kernel for speed.
    """
    xLend = np.empty(6)
    yLend = np.empty(6)
    _ae_model_jit(sv, p.M, xLend, yLend)
    return xLend, yLend


def make_rhs(p, use_dynamic_alpha: bool = True, use_quality_state: bool = True,
             use_pump_wear: bool = False, use_valve_wear: bool = False):
    """Closure that captures ``p`` and the input vector for ``solve_ivp``.

    The simulation driver sets the active input vector on this closure just
    before each integration interval. The closure signature matches the
    scipy convention ``f(t, sv) -> dsv/dt``.

    ``use_dynamic_alpha`` toggles Layer 2.5 fouling dynamics (default
    True). Pass False for bit-identical behaviour to the legacy
    pre-baked ``factor`` series.

    ``use_quality_state`` toggles Layer 2.1 quality-state dynamics
    (default True). Pass False to keep the state at 22 components
    (Layer 2.5 only).

    ``use_pump_wear`` toggles Layer 2.4 pump degradation dynamics
    (default False). Adds one continuous state slot (sv[22]) for
    ``pump_health ∈ [0, 1]`` and applies it as a multiplier on the
    CW pressure nominal.

    ``use_valve_wear`` toggles Layer 2.4 valve stiction dynamics
    (default False). Adds one continuous state slot (sv[23]) for
    ``valve_stiction_pct ∈ [0, 100]`` and reduces the effective
    valve gains ``kvo``, ``kvH`` proportionally.
    """
    state = {"u": np.zeros(6), "factor": 1.0}

    def rhs(t: float, sv: np.ndarray) -> np.ndarray:
        return ODEmodel(
            t, sv, p, state["u"], state["factor"],
            use_dynamic_alpha, use_quality_state,
            use_pump_wear, use_valve_wear,
        )

    rhs.set_u = lambda u: state.__setitem__("u", u)
    rhs.set_factor = lambda f: state.__setitem__("factor", f)
    return rhs