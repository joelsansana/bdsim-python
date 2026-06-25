"""
Transesterification reaction kinetics.

Three-step reversible reaction network (TG ↔ DG ↔ MG → glycerol + 3 E),
modelled with Arrhenius temperature dependence.

Port of ``rxrates.m``.
"""

from __future__ import annotations

import numpy as np


def rxrates(
    k0: np.ndarray,
    Ea: np.ndarray,
    R: float,
    C: np.ndarray,
    T: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward/reverse rates and net species rates.

    Parameters
    ----------
    k0 : array_like, shape (6,)
        Pre-exponential factors for the 6 elementary reactions (forward and
        reverse of the three steps).
    Ea : array_like, shape (6,)
        Activation energies, J/mol.
    R : float
        Universal gas constant, J mol⁻¹ K⁻¹.
    C : array_like, shape (6,)
        Molar concentrations of the 6 species (TG, DG, MG, M, E, G), mol m⁻³.
    T : float
        Temperature, K.

    Returns
    -------
    rx : np.ndarray, shape (6,)
        Net species rates, mol m⁻³ s⁻¹. Positive = production.
    r : np.ndarray, shape (3,)
        Net rates of the three reactions, mol m⁻³ s⁻¹.
    """
    k0 = np.asarray(k0, dtype=float)
    Ea = np.asarray(Ea, dtype=float)
    C = np.asarray(C, dtype=float)

    kr = k0 * np.exp(-Ea / (R * T))                       # m³ mol⁻¹ s⁻¹
    r = np.array([
        kr[0] * C[0] * C[3] - kr[1] * C[1] * C[4],
        kr[2] * C[1] * C[3] - kr[3] * C[2] * C[4],
        kr[4] * C[2] * C[3] - kr[5] * C[5] * C[4],
    ])

    rx = np.array([
        -r[0],                  # TG
         r[0] - r[1],           # DG
               r[1] - r[2],     # MG
        -r[0] - r[1] - r[2],    # M
         r[0] + r[1] + r[2],    # E
                          r[2], # G
    ])
    return rx, r