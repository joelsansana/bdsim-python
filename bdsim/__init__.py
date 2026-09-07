"""
bdsim - Biodiesel Process Simulator (Python port)

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
  :class:`PIDController`, :class:`Results`, :class:`StepResult` — typed configuration objects
- :class:`DecanterSplitNet` — trainable PyTorch port of the decanter split
  neural network (Brásio et al.)
- :class:`SpectrumGenerator`, :class:`SpectrumConfig`, :class:`SpectrumSample`,
  :func:`comp_spectrum` — NIR/IR virtual spectrum sensor
  (port of upstream ``comp_spectrum.m``). Sample at reactor / light-phase /
  heavy-phase decanter, Beer-Lambert + photometric noise + AWGN + drift.
  See :class:`ProcessFaults.spectrum_enabled` to enable.
- :class:`bdsim.live_simulator.LiveSimulator` — stateful per-step driver.
  Mutate ``sensor_faults.bias`` / ``.stuck`` / ``.dropouts`` between
  :meth:`~bdsim.live_simulator.LiveSimulator.step` calls to inject
  faults mid-run (live fault injection).

The port is faithful to the MATLAB semantics but uses modern Python idioms
(type hints, dataclasses, NumPy vectorisation, scipy.integrate.solve_ivp).
"""

from .config import (
    ARMAX,
    Parameters,
    PIDController,
    ProcessFaults,
    Results,
    SensorFaults,
    Settings,
    StepResult,
    ValveFaults,
)
from .kinetics import rxrates
from .live_simulator import LiveSimulator
from .ode import AEmodel, ODEmodel
from .simulation import run, run_with
from .spectra import (
    SpectrumConfig,
    SpectrumGenerator,
    SpectrumSample,
    comp_spectrum,
)
from .split_nn import DecanterSplitNet, split
from .thermo import Mmx, Qoil, Vmolar, cpmx, side_reactions

__version__ = "1.2.0"
__all__ = [
    "ARMAX",
    "AEmodel",
    "DecanterSplitNet",
    "LiveSimulator",
    "Mmx",
    "ODEmodel",
    "PIDController",
    "Parameters",
    "ProcessFaults",
    "Qoil",
    "Results",
    "SensorFaults",
    "Settings",
    "SpectrumConfig",
    "SpectrumGenerator",
    "SpectrumSample",
    "StepResult",
    "ValveFaults",
    "Vmolar",
    "comp_spectrum",
    "cpmx",
    "run",
    "run_with",
    "rxrates",
    "side_reactions",
    "split",
] 