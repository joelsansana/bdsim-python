"""
bdsim — Biodiesel Process Simulator (Python port)

A Python port of the BDSIM MATLAB/Octave simulator by Natércia C. P. Fernandes
(University of Coimbra, 2019), used as a TRL 5/6 simulation environment for
fault-detection research.

Original repository: https://github.com/naterciafernandes/BDSIM
Licence: GPLv3+ (same as upstream)

Public API
----------
- :func:`run` — execute the closed-loop simulation with default settings
- :func:`run_with` — execute with custom parameter / fault overrides
- :class:`Parameters`, :class:`Settings`, :class:`ProcessFaults`,
  :class:`SensorFaults`, :class:`ValveFaults`, :class:`ARMAX`,
  :class:`PIDController`, :class:`Results` — typed configuration objects
- :class:`DecanterSplitNet` — trainable PyTorch port of the decanter split
  neural network (Brásio et al.)

The port is faithful to the MATLAB semantics but uses modern Python idioms
(type hints, dataclasses, NumPy vectorisation, scipy.integrate.solve_ivp).
"""

from .config import (
    Parameters,
    ProcessFaults,
    SensorFaults,
    ValveFaults,
    ARMAX,
    PIDController,
    Settings,
    Results,
)
from .thermo import Qoil, Vmolar, Mmx, cpmx, side_reactions
from .kinetics import rxrates
from .split_nn import DecanterSplitNet, split
from .ode import ODEmodel, AEmodel
from .simulation import run, run_with

__version__ = "1.0.0"
__all__ = [
    "Parameters",
    "ProcessFaults",
    "SensorFaults",
    "ValveFaults",
    "ARMAX",
    "PIDController",
    "Settings",
    "Results",
    "Qoil", "Vmolar", "Mmx", "cpmx", "side_reactions",
    "rxrates",
    "DecanterSplitNet", "split",
    "ODEmodel", "AEmodel",
    "run", "run_with",
]