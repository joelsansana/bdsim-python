"""fouling-mode windows: five-mode fouling stepper (port of upstream ``fouling.m``).

This module is the Python port of Fernandes 2019 / Strelet Dec 2019
``fouling.m`` from the BDSIM_spectr reference distribution. It exposes a
:class:`FoulingModeStepper` that maintains the ARMAX state explicitly
(``RfOld``, ``epsilon_OLD``) — the upstream uses ``persistent`` MATLAB
variables, which is unsafe for multi-instance or restart scenarios. Our
port makes this state explicit and bounded.

Three paths cooperate to produce the heat-exchanger efficiency factor
used in the energy balance ``Theat = TR - factor * Qheat / (NR * cpmolR)``:

  1. **dynamic fouling** (continuous α) — when ``pfaults.fouling_dynamic=True``
     the state vector carries ``sv[21]`` (α ∈ [0, 1]) and ``factor``
     follows the dynamics. This is the "physics-based slow fouling"
     story.
  2. **fouling-mode windows (this module, windowed modes 4/5)** — during an active
     fault event the stepper overrides ``factor`` with the ARMAX-mode
     output. This is the "fast, stochastic, fault-injection" story:
     intermittent feedstock-impurity spikes that look like ARMAX noise
     to a downstream correlation engine.
  3. **dynamic fouling fallback (static)** — when neither above applies,
     ``factor = 1 / (1 + foulingpar * t)`` (the legacy pre-baked
     series) is used.

Priority when all three are configured:
``continuous α  >  windowed mode 4/5  >  static``.

Modes 0–3 of the upstream are ported for completeness but the bdsim-dashboard
scenario catalog only exercises modes 4 and 5; modes 1 and 3 are wired into the
``FoulingModeStepper`` because they're well-defined and serve as
sanity-check references, mode 0 is the trivial off state, and mode 2
(exponential recovery) is documented as "weird" in the upstream and
kept for parity but flagged as such.

Determinism: modes 4 and 5 are stochastic. Callers must pass a
``numpy.random.Generator`` to :meth:`step` so test fixtures can pin
seeds and scenarios can replay runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


class FoulingMode(IntEnum):
    """Five-mode fouling model from upstream ``fouling.m``.

    The integer values match the upstream ``switch fouling`` cases so
    a ``ProcessFaults.fouling`` integer from the upstream user
    settings file maps directly.
    """

    OFF = 0
    LINEAR = 1
    EXPONENTIAL = 2
    CHAIBAKHSH = 3
    ARMAX_NOISE = 4
    ARMAX_PURE_NOISE = 5


@dataclass
class FoulingModeStepper:
    """Step-by-step driver for the five-mode fouling model.

    The stepper replaces the upstream's ``persistent`` MATLAB variables
    with explicit dataclass state. This makes the state visible
    (callers can ``snapshot()`` for diagnostics or replay), bounded
    (no hidden cross-call coupling), and unit-testable (no need to
    inspect module-level globals).

    Args:
        foulingpar: ``[k]`` array; ``k`` is the linear fouling rate
            (mode 1) and the ARMAX mean-reversion rate (modes 4/5).
            Default matches upstream ``user_settings.m`` (3e-7).
        ar_eps_std: 1σ noise on the ARMAX innovation ε (modes 4/5
            only). Default 5e-4 matches upstream ``fouling.m:42``.
        xRG_weight: when ``True``, modes 4/5 multiply ``k * t * xRG``
            (glycerol-coupling — faster fouling on dirty feedstock).
            Default ``True`` matches upstream behaviour.
    """

    foulingpar: np.ndarray = field(
        default_factory=lambda: np.array([3e-7])
    )
    ar_eps_std: float = 5e-4
    xRG_weight: bool = True

    # --- Explicit state (replaces MATLAB ``persistent`` variables)
    _rf_old: float = field(default=0.0, init=False, repr=False)
    _epsilon_old: float = field(default=0.0, init=False, repr=False)
    # The exponential mode (2) uses a ``tau`` parameter that the
    # upstream also stashes in a ``persistent`` variable. Default
    # ``None`` forces the mode-2 branch to write its tau=50000 on
    # every step (matches upstream behaviour; mode 2 is rarely used).
    _tau: float = field(default=0.0, init=False, repr=False)

    # --- Mode 3 (Chaibakhsh) parameters, fixed by upstream
    _chaibakhsh_r0: float = 0.05
    _chaibakhsh_tau: float = 1e-5
    _chaibakhsh_kc: float = 0.9
    _chaibakhsh_Lt: float = 5.0

    # --- Mode 4 ARMAX coefficients, fixed by upstream
    _armax_phi: float = 0.90
    _armax_theta: float = 0.75

    # ----------------------------------------------------------------
    # Per-step driver
    # ----------------------------------------------------------------
    def step(
        self,
        t: float,
        mode: int | FoulingMode,
        xRG: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[float, float]:
        """Advance the stepper one tick.

        Args:
            t: sim time in seconds (used by modes 1, 2, 3, 4).
            mode: one of :class:`FoulingMode` (int or enum).
            xRG: reactor glycerol mole fraction (sv[5] in 0-based
                Python indexing; sv(6) in 1-based MATLAB indexing).
                Used by modes 4 and 5 when ``xRG_weight=True``.
            rng: ``numpy.random.Generator`` for ARMAX noise. If
                ``None`` and the mode is stochastic (4 or 5), a fresh
                default generator is created — but this is **not
                reproducible** and tests must pass an explicit ``rng``.

        Returns:
            ``(factor, Rfouling)`` — ``factor`` is the heat-exchanger
            efficiency multiplier applied to ``Qheat`` in the energy
            balance; ``Rfouling`` is the accumulated resistance
            (m⁻¹·K/W or whatever units the upstream uses, raw).
        """
        m = int(mode)
        k = float(self.foulingpar[0])

        if m == 0:                                              # OFF
            rfouling = 0.0 * t                                  # 0 (literal upstream form)
            self._rf_old = 0.0
            self._epsilon_old = 0.0
        elif m == 1:                                            # LINEAR
            rfouling = k * t
        elif m == 2:                                            # EXPONENTIAL
            self._tau = 50000.0
            rfouling = 1e-1 * np.exp(-self._tau / t) if t > 0 else 0.0
        elif m == 3:                                            # CHAIBAKHSH
            r = self._chaibakhsh_r0 * np.exp(-self._chaibakhsh_tau * t)
            rfouling = (1.0 / (2.0 * np.pi * self._chaibakhsh_kc * self._chaibakhsh_Lt)) * np.log(
                self._chaibakhsh_r0 / r
            )
        elif m == 4:                                            # ARMAX(1,1,1)-like
            if rng is None:
                rng = np.random.default_rng()
            epsilon = self.ar_eps_std * rng.standard_normal()
            xrg_term = xRG if self.xRG_weight else 1.0
            rfouling = (
                self._rf_old
                + self._armax_phi * (k * t * xrg_term - self._rf_old)
                + epsilon
                - self._armax_theta * self._epsilon_old
            )
            if rfouling < 0:
                rfouling = 0.0
            self._epsilon_old = epsilon
            self._rf_old = rfouling
        elif m == 5:                                            # ARMAX with φ=θ=0
            if rng is None:
                rng = np.random.default_rng()
            epsilon = self.ar_eps_std * rng.standard_normal()
            xrg_term = xRG if self.xRG_weight else 1.0
            rfouling = (
                self._rf_old
                + 0.0 * (k * t * xrg_term - self._rf_old)
                + epsilon
                - 0.0 * self._epsilon_old
            )
            if rfouling < 0:
                rfouling = 0.0
            self._epsilon_old = epsilon
            # NOTE: upstream does NOT update _rf_old for mode 5; matches.
        else:
            raise ValueError(
                f"unknown fouling mode {m!r}; expected 0..5"
            )

        factor = 1.0 / (1.0 + rfouling)
        return float(factor), float(rfouling)

    # ----------------------------------------------------------------
    # Diagnostics / replay
    # ----------------------------------------------------------------
    def snapshot(self) -> dict:
        """Return the explicit ARMAX state for diagnostics or replay."""
        return {
            "rf_old": float(self._rf_old),
            "epsilon_old": float(self._epsilon_old),
            "tau": float(self._tau),
        }

    def restore(self, state: dict) -> None:
        """Restore from a prior :meth:`snapshot` for replay scenarios."""
        self._rf_old = float(state.get("rf_old", 0.0))
        self._epsilon_old = float(state.get("epsilon_old", 0.0))
        self._tau = float(state.get("tau", 0.0))

    def reset(self) -> None:
        """Zero the ARMAX state. Equivalent to a fresh sim start."""
        self._rf_old = 0.0
        self._epsilon_old = 0.0
        self._tau = 0.0
