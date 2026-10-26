"""Gold validation scenarios S00–S07 (mirror BDSIM ``scenarios/*.m``).

Shared defaults: 1 week, ``dt=30``, noise off, all AUTO, legacy 21-wide
``ProcessFaults`` (Layer masters off). Demo schedules disabled so calendars
match Octave gold rather than ``user_settings.m``.
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

SCENARIO_IDS = ("S00", "S01", "S02", "S03", "S04", "S05", "S06", "S07")

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
) -> SensorFaults:
    n = 5
    return SensorFaults(
        nsensors=n,
        signal=np.ones((lt, n)) if signal is None else signal,
        a=np.ones((lt, n)) if a is None else a,
        b=np.zeros((lt, n)) if b is None else b,
        isIntermit=np.zeros(n) if is_intermit is None else is_intermit,
        tmaxInterm=np.zeros(n) if tmax_interm is None else tmax_interm,
        noise_std=np.zeros(n),
    )


def _gold_settings(
    t: np.ndarray,
    *,
    sp_profile: np.ndarray | None = None,
) -> Settings:
    lt = len(t)
    if sp_profile is None:
        sp_profile = np.column_stack([
            np.full(lt, 60.4 + 273.15),
            np.full(lt, 50.0 + 273.15),
            np.full(lt, 0.5),
            np.full(lt, 3000.0),
        ])
    return Settings(
        ti=float(t[0]),
        tf=float(t[-1]),
        dt=float(t[1] - t[0]) if lt > 1 else 30.0,
        sv0=_GOLD_SV0.copy(),
        u0=_GOLD_U0.copy(),
        mode_1b=np.array([1, 1, 1, 1]),
        pvindex_1b=np.array([1, 2, 3, 4]),
        uindex_1b=np.array([4, 5, 6, 1]),
        sp1=60.4 + 273.15,
        sp2=50.0 + 273.15,
        sp3=0.5,
        sp4=3000.0,
        nic=4,
        use_demo_schedules=False,
        sp_profile=sp_profile,
    )


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
    """Build config blocks for one gold scenario on time grid ``t``."""
    sid = scenario_id.upper()
    if sid not in SCENARIO_IDS:
        raise ValueError(f"unknown gold scenario {scenario_id!r}; expected one of {SCENARIO_IDS}")

    lt = len(t)
    # Octave ``heaviside(0)`` is 0.5 (half-step on the boundary sample).
    # Ignore a second arg so call sites that used ``np.heaviside(..., 1.0)``
    # still get Octave semantics.
    def H(x: np.ndarray, _h0: float = 0.5) -> np.ndarray:
        return np.heaviside(x, 0.5)

    pfaults = _legacy_pfaults()
    sfaults = _quiet_sfaults(lt)
    vfaults = _gold_vfaults()
    sp = np.column_stack([
        np.full(lt, 60.4 + 273.15),
        np.full(lt, 50.0 + 273.15),
        np.full(lt, 0.5),
        np.full(lt, 3000.0),
    ])

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

    return GoldBundle(
        settings=_gold_settings(t, sp_profile=sp),
        pfaults=pfaults,
        sfaults=sfaults,
        vfaults=vfaults,
        armax=_gold_armax(),
        pid=_gold_pid(),
    )


def run_gold_scenario(
    scenario_id: str,
    *,
    seed: int = 42,
    npoints: int | None = None,
    tf: float = T_WEEK,
    dt: float = 30.0,
    outdir: str | Path | None = None,
    verbose: bool = True,
) -> Results:
    """Run one gold scenario; optionally write Parquet under ``outdir``."""
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
