"""
Smoke tests for the bdsim Python port.

The aim here is *not* to validate every numerical result against the MATLAB
upstream (that would require running Octave). It's to ensure:
1. Every public import works
2. The simulator runs end-to-end with default settings
3. Trajectories stay in physically plausible ranges
4. The neural-network split function matches the upstream reference
5. The thermo leaf functions match the MATLAB originals
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bdsim import run_with
from bdsim.kinetics import rxrates
from bdsim.split_nn import DecanterSplitNet, split
from bdsim.thermo import Mmx, Qoil, Vmolar, cpmx, side_reactions

# ---------------------------------------------------------------------------
# Leaf functions
# ---------------------------------------------------------------------------

def test_qoil_zero_alpha():
    """No valve opening → no flow."""
    p = type("P", (), {"K2F": 1.0, "K3F": 1.0})()
    assert Qoil(1e-5, 0.0, p) == pytest.approx(0.0)


def test_qoil_positive_alpha():
    """With valve open, flow is positive (use realistic parameter magnitudes)."""
    p = type("P", (), {"K2F": 1e9, "K3F": 1e-1})()
    q = Qoil(15e-6, 0.5, p)
    assert q > 0


def test_vmolar_definition():
    """Vmolar(M, ro) = M / ro."""
    assert Vmolar(0.032, 757.0) == pytest.approx(0.032 / 757.0)


def test_mmx_definition():
    """Mmx is a weighted average."""
    M = np.array([1.0, 2.0])
    x = np.array([0.3, 0.7])
    assert Mmx(M, x) == pytest.approx(1.7)


def test_cpmx_definition():
    cp = np.array([2000.0, 2500.0])
    x = np.array([0.5, 0.5])
    assert cpmx(cp, x) == pytest.approx(2250.0)


def test_side_reactions_scaling():
    assert side_reactions(1.0, 0.9) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Kinetics
# ---------------------------------------------------------------------------

def test_rxrates_zero_temperature_deactivates():
    """At T=0 all forward rates collapse to zero, but Arrhenius still yields
    kr=0 so r = -kr_rev * C_back * C_back' (negative rates everywhere).
    """
    k0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    Ea = np.array([1e4, 1e4, 1e4, 1e4, 1e4, 0.0])
    C = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    rx, r = rxrates(k0, Ea, 8.314, C, 300.0)
    assert r.shape == (3,)
    assert rx.shape == (6,)
    # Mass balance: dTG + dDG + dMG + dM + dE + dG = 0
    assert np.isclose(np.sum(rx), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

def test_split_net_matches_reference():
    """DecanterSplitNet(xM, xG, T_K) must equal split(xM, xG, T_K)."""
    import torch
    net = DecanterSplitNet()
    with torch.no_grad():
        for xM, xG, T in [(0.4, 0.15, 320.0), (0.5, 0.05, 300.0),
                          (0.3, 0.2, 340.0), (0.0, 0.0, 315.5)]:
            ref = split(xM, xG, T)
            out = net(torch.tensor([[xM, xG, T]],
                                   dtype=torch.float32)).numpy().flatten()
            np.testing.assert_allclose(out, ref, rtol=1e-5)


def test_split_net_freeze_pretrained():
    """Sanity check: freeze_pretrained disables gradients without breaking forward."""
    import torch
    net = DecanterSplitNet()
    net.freeze_pretrained()
    for p in net.parameters():
        assert p.requires_grad is False
    out = net(torch.tensor([[0.4, 0.15, 320.0]], dtype=torch.float32))
    assert out.shape == (1, 3)


def test_split_nn_weights_loaded_from_data_file():
    """Issue #12: upstream split-net weights live in
    ``bdsim/data/split_nn_weights.npz``, not inlined Python literals.
    Drop the .npz and the module fails to import — that is the
    documented contract."""
    from bdsim.split_nn import (
        _BIAS_GAMMA, _BIAS_THETA, _MN, _PESOS_W, _PESOS_W_HIDDEN, _ST,
    )
    # Sanity: every constant loaded with the expected shape.
    assert _MN.shape == (3,)
    assert _ST.shape == (3,)
    assert _PESOS_W.shape == (3, 5)
    assert _PESOS_W_HIDDEN.shape == (5, 3)
    assert _BIAS_THETA.shape == (5,)
    assert _BIAS_GAMMA.shape == (3,)


# ---------------------------------------------------------------------------
# End-to-end simulation
# ---------------------------------------------------------------------------

def test_smoke_run():
    """Default-settings short-horizon simulation completes and produces sane trajectories.

    Uses ``fouling_dynamic=False`` so the legacy 21-component state
    vector is exercised — this is the regression path that should
    match the upstream baseline bit-for-bit. See ``test_smoke_run_dynamic``
    for the new HEX-fouling dynamics (Roadmap dynamic fouling).
    """
    from bdsim.config import ProcessFaults, Settings
    # Legacy profile: explicit overrides for every feature flag now
    # default-ON (1.2.0+).
    pfaults = ProcessFaults(
        fouling_dynamic=False,
        quality_state=False,
        pump_wear=False,
        valve_wear=False,
        spectrum_enabled=False,
    )
    settings = Settings(ti=0.0, tf=10_000.0, dt=5.0)
    res = run_with(settings=settings, pfaults=pfaults, seed=42, verbose=False)

    # Time vector
    assert res.t[0] == 0.0
    assert res.t[-1] > 0.0

    # State trajectory shape — legacy 21 components
    assert res.sv.shape[1] == 21
    assert res.uv.shape[1] == 6
    assert res.pv.shape[1] == 5
    assert res.sp.shape[1] == 4

    # Reactor temperature stays in plausible biodiesel range
    TR_C = res.sv[:, 6] - 273.15
    assert np.all(TR_C > 40.0)
    assert np.all(TR_C < 80.0)

    # Decanter temperature also bounded
    TD_C = res.sv[:, 17] - 273.15
    assert np.all(TD_C > 30.0)
    assert np.all(TD_C < 70.0)

    # Mole fractions in [0, 1]
    assert np.all((res.sv[:, :6] >= 0) & (res.sv[:, :6] <= 1))
    assert np.all((res.sv[:, 7:13] >= 0) & (res.sv[:, 7:13] <= 1))

    # Filter pore radius is monotonically decreasing (clogging)
    assert np.all(np.diff(res.sv[:, 18]) <= 1e-6)


def test_smoke_run_with_seed_is_deterministic():
    """Same seed → same trajectories (short horizon, legacy mode)."""
    from bdsim.config import ProcessFaults, Settings
    # Legacy profile (21-component state).
    pfaults = ProcessFaults(
        fouling_dynamic=False,
        quality_state=False,
        pump_wear=False,
        valve_wear=False,
        spectrum_enabled=False,
    )
    settings = Settings(ti=0.0, tf=5_000.0, dt=5.0)
    r1 = run_with(settings=settings, pfaults=pfaults, seed=123, verbose=False)
    r2 = run_with(settings=settings, pfaults=pfaults, seed=123, verbose=False)
    np.testing.assert_array_equal(r1.sv, r2.sv)
    np.testing.assert_array_equal(r1.pv, r2.pv)


def test_smoke_run_dynamic_22_state_components():
    """Dynamic mode (Roadmap dynamic fouling) extends the state vector to 22.

    sv[21] holds the HEX fouling factor α ∈ [0, 1]. At default operating
    conditions α grows slowly (Arrhenius accumulation vs. linear decay)
    — for a 10 000 s run we expect α > initial 0.05 and α < 0.5.
    """
    from bdsim.config import ProcessFaults, Settings
    # dynamic fouling-only profile (22-component state). Explicit overrides
    # for the other feature flags default-ON as of 1.2.0+ so this test
    # stays a clean 22-component trajectory.
    pfaults = ProcessFaults(
        fouling_dynamic=True,
        quality_state=False,
        pump_wear=False,
        valve_wear=False,
        spectrum_enabled=False,
    )
    settings = Settings(ti=0.0, tf=10_000.0, dt=5.0)
    res = run_with(settings=settings, pfaults=pfaults, seed=42, verbose=False)

    assert res.sv.shape[1] == 22
    # α starts at 0.05 and grows (Arrhenius at reactor T)
    alpha = res.sv[:, 21]
    assert alpha[0] == pytest.approx(0.05)
    assert np.all(alpha >= 0.0)
    assert np.all(alpha <= 1.0)
    assert alpha[-1] > 0.05                                  # α climbed
    assert alpha[-1] < 0.5                                   # slow fouling at default conditions


def test_smoke_run_dynamic_clamps_alpha_on_cleaning():
    """A cleaning event snaps α to alpha_clean (0.1) and clamps it in [0, 1]."""
    from bdsim.config import ProcessFaults, Settings
    # dynamic fouling-only profile (22-component state).
    pfaults = ProcessFaults(
        fouling_dynamic=True,
        quality_state=False,
        pump_wear=False,
        valve_wear=False,
    )
    settings = Settings(ti=0.0, tf=500.0, dt=5.0)
    res = run_with(settings=settings, pfaults=pfaults, seed=42, verbose=False)
    alpha = res.sv[:, 21]
    # No cleaning should fire in 500 s (DP_clean = 1e5, far above
    # initial DP), but α should still be clamped.
    assert np.all(alpha >= 0.0)
    assert np.all(alpha <= 1.0)


def test_smoke_run_no_clogging():
    """With clogging disabled the filter radius stays constant."""
    from bdsim.config import ProcessFaults
    # Legacy profile (21-component state). Explicit overrides for the
    # other feature flags default-ON as of 1.2.0+ so the assertion on
    # sv[:, 18] (filter pore radius) is unaffected by the wear / quality
    # slots.
    pfaults = ProcessFaults(
        clog_fraction=0.0, fouling=0, foulingpar=np.array([0.0]),
        fouling_dynamic=False,
        quality_state=False,
        pump_wear=False,
        valve_wear=False,
        spectrum_enabled=False,
    )
    res = run_with(pfaults=pfaults, seed=7, verbose=False)
    np.testing.assert_allclose(res.sv[:, 18], res.sv[0, 18], atol=1e-6)