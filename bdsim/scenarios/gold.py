"""Gold validation scenarios S00–S15 (mirror BDSIM ``scenarios/*.m``).

S00–S07: golden suite — 1 week, ``dt=30``, ``nic=4``, noise off, all AUTO.
S08–S15: extended set (2026-11-16 scenario-review chain, coverage gaps G1–G8 of
``docs/scenario_coverage.md`` in the Octave repo). They deviate deliberately from
the golden ``nic`` (see ``SCENARIO_TIMEBASE``): at ``nic=4`` the Foil loop is
discretely unstable at the golden time base (Octave doc §4bis), so the extended
scenarios run ``nic=1``; S15 takes the Foil loop to MANUAL. Per-scenario horizons:
1 week (S08–S10, S12, S13), 1 month (S11, S14), 6 months (S15).

Shared defaults: noise off (unless stated), all AUTO (unless stated), legacy
21-wide ``ProcessFaults`` (Layer masters off). Demo schedules disabled so
calendars match Octave gold rather than ``user_settings.m``.

RNG caveat: S08 (sensor noise), S13 (intermittence masks) and S14 (new-filter
draws) consume random streams. Octave legacy RNG ≠ NumPy at the same seed, so
those compare at fingerprint level only (Port_execution.md §5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import (
    ARMAX,
    PIDController,
    ProcessFaults,
    Results,
    SensorFaults,
    Settings,
    ValveFaults,
)
from ..io_parquet import save_parquet
from ..simulation import run_with

T_DAY = 86400.0
T_WEEK = 7.0 * T_DAY
T_MONTH = 30.0 * T_DAY
T_HALFYEAR = 180.0 * T_DAY

SCENARIO_IDS = (
    "S00", "S01", "S02", "S03", "S04", "S05", "S06", "S07",
    "S08", "S09", "S10", "S11", "S12", "S13", "S14", "S15",
)

# Per-scenario time base (tf [s], dt [s], nic). S00–S07 mirror the golden suite;
# S08–S15 mirror the extended-scenario headers in the Octave repo
# (scenarios/GOLD_SCENARIOS.md, "Extended scenarios S08–S15").
SCENARIO_TIMEBASE: dict[str, tuple[float, float, int]] = {
    "S00": (T_WEEK, 30.0, 4),
    "S01": (T_WEEK, 30.0, 4),
    "S02": (T_WEEK, 30.0, 4),
    "S03": (T_WEEK, 30.0, 4),
    "S04": (T_WEEK, 30.0, 4),
    "S05": (T_WEEK, 30.0, 4),
    "S06": (T_WEEK, 30.0, 4),
    "S07": (T_WEEK, 30.0, 4),
    "S08": (T_WEEK, 30.0, 1),
    "S09": (T_WEEK, 30.0, 1),
    "S10": (T_WEEK, 30.0, 1),
    "S11": (T_MONTH, 60.0, 1),
    "S12": (T_WEEK, 30.0, 1),
    "S13": (T_WEEK, 30.0, 1),
    "S14": (T_MONTH, 60.0, 1),
    "S15": (T_HALFYEAR, 300.0, 1),
}

_GOLD_SV0 = np.array([
    0.002455, 0.000553, 0.0000467, 0.42353, 0.43022, 0.14319, 60.4 + 273.15,
    4.2864e-03, 9.6570e-04, 8.1554e-05, 2.4287e-01, 7.5098e-01, 8.1550e-04,
    6.6574e-01, 1.5852e-04, 3.3410e-01, 0.5, 50.0 + 273.15,
    15e-6,
    40.0, 28.5,
])

_GOLD_U0 = np.array([40.0, 50 + 273.15, 657.0, 60 + 273.15, 23615.0, 28.5])


def _legacy_pfaults(**overrides: Any) -> ProcessFaults:
    """Healthy process defaults + Layer masters off (upstream 21-wide)."""
    kw: dict[str, Any] = dict(
        clog_fraction=0.0,
        DPclean=float("inf"),
        filter_std=0.0,
        ratio_robs_r=1.0,
        fouling=0,
        foulingpar=np.array([1.0]),
        fouling_dynamic=False,
        quality_state=False,
        pump_wear=False,
        valve_wear=False,
        spectrum_enabled=False,
    )
    kw.update(overrides)
    return ProcessFaults(**kw)


def _quiet_sfaults(
    lt: int,
    *,
    signal: np.ndarray | None = None,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    is_intermit: np.ndarray | None = None,
    tmax_interm: np.ndarray | None = None,
    noise_std: np.ndarray | None = None,
) -> SensorFaults:
    n = 5
    return SensorFaults(
        nsensors=n,
        signal=np.ones((lt, n)) if signal is None else signal,
        a=np.ones((lt, n)) if a is None else a,
        b=np.zeros((lt, n)) if b is None else b,
        isIntermit=np.zeros(n) if is_intermit is None else is_intermit,
        tmaxInterm=np.zeros(n) if tmax_interm is None else tmax_interm,
        noise_std=np.zeros(n) if noise_std is None else noise_std,
    )


def _gold_settings(
    t: np.ndarray,
    *,
    sp_profile: np.ndarray | None = None,
    nic: int = 4,
    mode_1b: np.ndarray | None = None,
    exogenous=None,
) -> Settings:
    lt = len(t)
    if sp_profile is None:
        sp_profile = np.column_stack([
            np.full(lt, 60.4 + 273.15),
            np.full(lt, 50.0 + 273.15),
            np.full(lt, 0.5),
            np.full(lt, 3000.0),
        ])
    settings = Settings(
        ti=float(t[0]),
        tf=float(t[-1]),
        dt=float(t[1] - t[0]) if lt > 1 else 30.0,
        sv0=_GOLD_SV0.copy(),
        u0=_GOLD_U0.copy(),
        mode_1b=np.array([1, 1, 1, 1]) if mode_1b is None else mode_1b,
        pvindex_1b=np.array([1, 2, 3, 4]),
        uindex_1b=np.array([4, 5, 6, 1]),
        sp1=60.4 + 273.15,
        sp2=50.0 + 273.15,
        sp3=0.5,
        sp4=3000.0,
        nic=nic,
        use_demo_schedules=False,
        sp_profile=sp_profile,
    )
    if exogenous is not None:
        # S09 (and similar): custom load-disturbance profile. Instance
        # attribute shadows the dataclass method; simulation calls
        # ``settings.exogenous(t)``.
        settings.exogenous = exogenous
    return settings


def _gold_armax() -> ARMAX:
    z = np.zeros(6)
    return ARMAX(phi=z.copy(), theta=z.copy(), eta=z.copy(), unoise_std=z.copy())


def _gold_pid() -> PIDController:
    return PIDController()


def _gold_vfaults(
    S: np.ndarray | None = None,
    J: np.ndarray | None = None,
) -> ValveFaults:
    return ValveFaults(
        S=np.array([0.0, 0.0]) if S is None else S,
        J=np.array([0.0, 0.0]) if J is None else J,
        uindex=np.array([6, 1]),
    )


def _time_grid(
    *,
    ti: float = 0.0,
    tf: float = T_WEEK,
    dt: float = 30.0,
    npoints: int | None = None,
) -> np.ndarray:
    """Build ``t``; optional Octave-style ``BD_NPOINTS`` truncate (+1 before end-trim)."""
    t = np.arange(ti, tf + dt / 2, dt)
    if npoints is not None and npoints > 1 and npoints < len(t):
        # Octave keeps n_out+1 samples before end-trim → n_out after.
        n = min(len(t), int(npoints) + 1)
        t = t[:n]
    return t


@dataclass
class GoldBundle:
    settings: Settings
    pfaults: ProcessFaults
    sfaults: SensorFaults
    vfaults: ValveFaults
    armax: ARMAX
    pid: PIDController


def build_gold_bundle(scenario_id: str, t: np.ndarray) -> GoldBundle:
    """Build config blocks for one gold scenario on time grid ``t``.

    S00–S07: golden suite overrides (see Octave ``scenarios/GOLD_SCENARIOS.md``).
    S08–S15: extended set — mirrors the override blocks in the Octave scenario
    files ``S08_*.m`` … ``S15_*.m`` one-for-one (same profiles, same time base,
    ``nic=1``; S09/S15 take loop 4 to MANUAL with the oil order driven by the
    ARMAX exogenous channel).
    """
    sid = scenario_id.upper()
    if sid not in SCENARIO_IDS:
        raise ValueError(f"unknown gold scenario {scenario_id!r}; expected one of {SCENARIO_IDS}")

    lt = len(t)
    # Octave ``heaviside(0)`` is 0.5 (half-step on the boundary sample).
    # Ignore a second arg so call sites that used ``np.heaviside(..., 1.0)``
    # still get Octave semantics.
    def H(x: np.ndarray, _h0: float = 0.5) -> np.ndarray:
        return np.heaviside(x, 0.5)

    tf, dt, nic = SCENARIO_TIMEBASE[sid]

    pfaults = _legacy_pfaults()
    sfaults = _quiet_sfaults(lt)
    vfaults = _gold_vfaults()
    sp = np.column_stack([
        np.full(lt, 60.4 + 273.15),
        np.full(lt, 50.0 + 273.15),
        np.full(lt, 0.5),
        np.full(lt, 3000.0),
    ])
    mode_1b = np.array([1, 1, 1, 1])
    armax = _gold_armax()
    exogenous: Any = None

    if sid == "S01":
        pfaults = _legacy_pfaults(
            clog_fraction=5.95e-7,
            DPclean=1e5,
            filter_std=0.0,
        )
    elif sid == "S02":
        pfaults = _legacy_pfaults(
            fouling=1,
            foulingpar=np.array([3e-7]),
        )
    elif sid == "S03":
        pfaults = _legacy_pfaults(ratio_robs_r=0.9)
    elif sid == "S04":
        sig = np.ones((lt, 5))
        a = np.ones((lt, 5))
        b = np.zeros((lt, 5))
        # day 1–2: DP bias
        b[:, 4] = 5e3 * (H(t - 1 * T_DAY) - H(t - 2 * T_DAY))
        # day 2–3.5: TD drift
        b[:, 1] = (
            5e-6 * (t - 2 * T_DAY)
            * (H(t - 2 * T_DAY) - H(t - 3.5 * T_DAY))
        )
        # day 3.5–4.5: TR scaling
        a[:, 0] = 1.0 + 0.05 * (
            H(t - 3.5 * T_DAY) - H(t - 4.5 * T_DAY)
        )
        # day 4.5–5.5: Foil stuck at 3000 kg/h
        a[:, 3] = 1.0 - H(t - 4.5 * T_DAY) + H(t - 5.5 * T_DAY)
        b[:, 3] = 3000.0 * (
            H(t - 4.5 * T_DAY) - H(t - 5.5 * T_DAY)
        )
        # day 5.5–6.5: hH loss of signal
        # Octave heaviside(0)=0.5 would put 0.5 on the edge samples;
        # SensorFaults.signal must be bool-like {0,1}, so snap the half-step
        # (one-sample edge only; interval of loss matches Octave otherwise).
        sig[:, 2] = 1.0 - H(t - 5.5 * T_DAY) + H(t - 6.5 * T_DAY)
        sig = np.where(sig >= 0.5, 1.0, 0.0)
        sfaults = _quiet_sfaults(lt, signal=sig, a=a, b=b)
    elif sid == "S05":
        sp[:, 3] = (
            3000.0
            + 100.0 * H(t - 1 * T_DAY)
            - 100.0 * H(t - 4 * T_DAY)
        )
        vfaults = _gold_vfaults(
            S=np.array([0.0, 3.0]),
            J=np.array([0.0, 5.0]),
        )
    elif sid == "S06":
        b = np.zeros((lt, 5))
        b[:, 1] = 0.5 * H(t - 1 * T_DAY)
        sfaults = _quiet_sfaults(
            lt,
            b=b,
            is_intermit=np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
            tmax_interm=np.array([0.0, 3600.0, 0.0, 0.0, 0.0]),
        )
    elif sid == "S07":
        pfaults = _legacy_pfaults(
            DPclean=1e5,
            clog_fraction=5.95e-7,
            filter_std=0.0,
            ratio_robs_r=0.95,
            fouling=1,
            foulingpar=np.array([3e-7]),
        )
        b = np.zeros((lt, 5))
        b[:, 1] = 0.5 * (
            H(t - 3 * T_DAY) - H(t - 4 * T_DAY)
        )
        sfaults = _quiet_sfaults(lt, b=b)
    elif sid == "S08":
        # G1: measurement noise on all five sensors (user_settings magnitudes).
        sfaults = _quiet_sfaults(
            lt,
            noise_std=np.array([0.1, 0.1, 5e-3, 5.0, 300.0]),
        )
    elif sid == "S09":
        # G2: ARMAX load disturbances + Foil loop in MANUAL. Deterministic
        # (unoise_std = 0): oil order 40 -> 45 % at day 2 -> 40 % at day 4,
        # Tmet daily sine +-3 K.
        mode_1b = np.array([1, 1, 1, 0])
        armax = _gold_armax()
        armax.eta = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        def _exogenous_s09(t_: np.ndarray) -> np.ndarray:
            d = np.tile(_GOLD_U0, (len(t_), 1))
            d[:, 0] = 40.0 + 5.0 * H(t_ - 2 * T_DAY) - 5.0 * H(t_ - 4 * T_DAY)
            d[:, 1] = d[:, 1] + 3.0 * np.sin(2.0 * np.pi / T_DAY * t_)
            return d

        exogenous = _exogenous_s09
    elif sid == "S10":
        # G3: setpoint steps on the TR, TD and hH loops (servo regime).
        sp = np.column_stack([
            (60.4 + 273.15) + 2.0 * (H(t - 1 * T_DAY) - H(t - 3 * T_DAY)),
            (50.0 + 273.15) + 2.0 * (H(t - 2 * T_DAY) - H(t - 5 * T_DAY)),
            0.5 + 0.1 * (H(t - 4 * T_DAY) - H(t - 6 * T_DAY)),
            np.full(lt, 3000.0),
        ])
    elif sid == "S11":
        # G4: fouling sized so the required Qheat (23615*(1+Rf) W) crosses the
        # 40 kW PID upper bound at Rf > 0.693, i.e. t > 1.155e6 s (~day 13.4);
        # from then on Qheat is pinned at the bound and TD drifts above SP.
        pfaults = _legacy_pfaults(
            fouling=1,
            foulingpar=np.array([6e-7]),
        )
    elif sid == "S12":
        # G5: stiction variants — valve 1 (H) no-offset slip (S=J=2) and
        # valve 2 (oil) pure deadband (S=3, J=0), forced by hH/Foil SP steps.
        sp[:, 2] = 0.5 + 0.1 * (H(t - 1 * T_DAY) - H(t - 4 * T_DAY))
        sp[:, 3] = (
            3000.0
            + 100.0 * H(t - 1 * T_DAY)
            - 100.0 * H(t - 4 * T_DAY)
        )
        vfaults = _gold_vfaults(
            S=np.array([2.0, 3.0]),
            J=np.array([2.0, 0.0]),
        )
    elif sid == "S13":
        # G7: intermittence masks on three (sensor, fault) pairs: TR scaling
        # (day 1-4), TD drift (from day 2), Foil stuck (from day 5). Masks are
        # RNG-dependent: fingerprint-level parity only.
        a = np.ones((lt, 5))
        b = np.zeros((lt, 5))
        sig = np.ones((lt, 5))
        a[:, 0] = 1.0 + 0.05 * (H(t - 1 * T_DAY) - H(t - 4 * T_DAY))
        b[:, 1] = 5e-6 * (t - 2 * T_DAY) * H(t - 2 * T_DAY)
        a[:, 3] = 1.0 - H(t - 5 * T_DAY)
        b[:, 3] = 3000.0 * H(t - 5 * T_DAY)
        sfaults = _quiet_sfaults(
            lt,
            signal=sig,
            a=a,
            b=b,
            is_intermit=np.array([1.0, 1.0, 0.0, 1.0, 0.0]),
            tmax_interm=np.array([3600.0, 3600.0, 0.0, 3600.0, 0.0]),
        )
    elif sid == "S14":
        # G6: one month of clogging with ~14 cleanings and new-filter
        # variability. filter_std = 6e10 ~ 1 % relative jitter of the
        # clean-filter constants (the shipped user_settings 5e-15 is
        # numerically nil — see Octave S14 header for the scale analysis).
        pfaults = _legacy_pfaults(
            clog_fraction=5.95e-7,
            DPclean=1e5,
            filter_std=6e10,
        )
    elif sid == "S15":
        # G8: six-month degradation accumulation — clogging (~84 cleanings),
        # fouling e=3e-7 (Qheat pins at 40 kW from ~day 27), catalyst loss
        # 0.9. Foil loop in MANUAL with a constant 40 % oil order (ARMAX
        # pass-through): removes the discretely unstable loop at PID period
        # 300 s and covers the MANUAL/ARMAX regime at long horizon.
        mode_1b = np.array([1, 1, 1, 0])
        armax = _gold_armax()
        armax.eta = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pfaults = _legacy_pfaults(
            clog_fraction=5.95e-7,
            DPclean=1e5,
            filter_std=0.0,
            ratio_robs_r=0.9,
            fouling=1,
            foulingpar=np.array([3e-7]),
        )

    return GoldBundle(
        settings=_gold_settings(
            t,
            sp_profile=sp,
            nic=nic,
            mode_1b=mode_1b,
            exogenous=exogenous,
        ),
        pfaults=pfaults,
        sfaults=sfaults,
        vfaults=vfaults,
        armax=armax,
        pid=_gold_pid(),
    )


def run_gold_scenario(
    scenario_id: str,
    *,
    seed: int = 42,
    npoints: int | None = None,
    tf: float | None = None,
    dt: float | None = None,
    outdir: str | Path | None = None,
    verbose: bool = True,
) -> Results:
    """Run one gold scenario; optionally write Parquet under ``outdir``.

    ``tf``/``dt`` default to the per-scenario time base (``SCENARIO_TIMEBASE``);
    explicit values override it (e.g. SMOKE truncation via ``npoints``).
    """
    tf_def, dt_def, _nic = SCENARIO_TIMEBASE[scenario_id.upper()]
    if tf is None:
        tf = tf_def
    if dt is None:
        dt = dt_def
    t = _time_grid(tf=tf, dt=dt, npoints=npoints)
    bundle = build_gold_bundle(scenario_id, t)
    results = run_with(
        pfaults=bundle.pfaults,
        sfaults=bundle.sfaults,
        vfaults=bundle.vfaults,
        armax=bundle.armax,
        pid=bundle.pid,
        settings=bundle.settings,
        seed=seed,
        verbose=verbose,
    )
    if outdir is not None:
        save_parquet(results, outdir=outdir)
    return results
