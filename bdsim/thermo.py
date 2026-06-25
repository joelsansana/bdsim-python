"""
Thermodynamic and mixture-property leaf functions.

These are pure NumPy functions ported 1:1 from the MATLAB originals. They
have no side effects, no persistent state, and no fault handling — that
makes them the easiest layer to unit-test and to call from external code.

Units:
- :func:`Qoil` — m³/s
- :func:`Vmolar` — m³/mol
- :func:`Mmx` — kg/mol (molar mass of a mixture)
- :func:`cpmx` — J mol⁻¹ K⁻¹ (molar cp) or J kg⁻¹ K⁻¹ (massic cp)
- :func:`side_reactions` — applies a multiplicative correction to pre-exponential
  factors (dimensionless)
"""

from __future__ import annotations

import numpy as np


def Qoil(r: np.ndarray | float, alpha: np.ndarray | float, p: dict) -> np.ndarray:
    """Volumetric flow rate of oil through the filter (m³/s).

    Port of ``Qoil.m``. Computes the flow as a function of the pore radius
    ``r`` (m), the valve lift fraction ``alpha`` (-), and the filter
    constants packed into ``p``.

    Parameters
    ----------
    r : array_like
        Pore radius in metres.
    alpha : array_like
        Valve opening fraction (0–1).
    p : dict
        Filter constants. Must expose ``K2F`` and ``K3F`` attributes.

    Returns
    -------
    np.ndarray
        Volumetric flow rate, m³/s.
    """
    r = np.asarray(r, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    return -p.K2F * alpha**2 / r**4 + np.sqrt(
        p.K2F**2 * alpha**4 / r**8 + p.K3F * alpha**2
    )


def Vmolar(M: np.ndarray, ro: np.ndarray) -> np.ndarray:
    """Molar volume (m³/mol) from molar mass and mass density.

    Port of ``Vmolar.m``.
    """
    return np.asarray(M) / np.asarray(ro)


def Mmx(M: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Mean molar mass (kg/mol) of a mixture.

    Port of ``Mmx.m``.
    """
    M = np.asarray(M, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.sum(x * M)


def cpmx(cp: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Mixture heat capacity (molar or massic).

    Port of ``cpmx.m``. Same numerical operation regardless of whether
    inputs are molar or massic — callers must be consistent.
    """
    cp = np.asarray(cp, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.sum(x * cp)


def side_reactions(A: np.ndarray | float, ratio_robs_r: float) -> np.ndarray | float:
    """Effective pre-exponential factor after side-reaction deactivation.

    Port of ``side_reactions.m``. Simply multiplies ``A`` by ``ratio_robs_r``.
    """
    return np.asarray(A) * ratio_robs_r