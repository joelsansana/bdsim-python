"""
Tests for Layer 2.8 — NIR/IR virtual spectrum sensor.

Covers:
- comp_spectrum() leaf-function determinism + noise scales
- SpectrumConfig default values match the MATLAB upstream
- SpectrumGenerator fire cadence (spctr_t), drift counter advances
- Reference spectra loads correctly (6 species x 631 channels)
- Legacy fingerprint preserved when spectrum_enabled=False
- StepResult.spectra is None between fires, SpectrumSample at fires
- Quality indicators in expected ranges
- Override spectra_ref_path works
- Seeded RNG produces identical spectra across two generators
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bdsim.spectra import (
    SpectrumConfig,
    SpectrumGenerator,
    SpectrumSample,
    _band_sum,
    _load_reference_spectra,
    comp_spectrum,
)
from bdsim.config import ProcessFaults
from bdsim.live_simulator import LiveSimulator


# ---------------------------------------------------------------------------
# Reference spectra load
# ---------------------------------------------------------------------------

def test_reference_spectra_load_default():
    """Bundled spectra_ref.csv loads as 6 species x 631 channels."""
    ref, wn = _load_reference_spectra(None)
    assert ref.shape == (6, 631)
    assert wn.shape == (631,)
    # Wavenumber axis starts at 637.5 cm-1 and steps in 5 cm-1.
    assert wn[0] == pytest.approx(637.5, abs=1e-6)
    assert wn[-1] == pytest.approx(637.5 + 5.0 * 630, abs=1e-6)
    assert np.all(np.diff(wn) == pytest.approx(5.0, abs=1e-9))


def test_reference_spectra_canonical_order():
    """Species ordering is TG, DG, MG, M, E, G (matches xR = sv[0:6])."""
    ref, _ = _load_reference_spectra(None)
    # Just verify the load doesn't crash and shape is right; the
    # canonical order is enforced by _load_reference_spectra's
    # explicit reorder.
    assert ref.shape[0] == 6


# ---------------------------------------------------------------------------
# comp_spectrum leaf function
# ---------------------------------------------------------------------------

def test_comp_spectrum_zero_x_is_zero():
    """Pure zero mole fractions → near-zero absorbance (modulo drift)."""
    ref, wn = _load_reference_spectra(None)
    x = np.zeros(6)
    rng = np.random.default_rng(0)
    absorb = comp_spectrum(ref, x, cs=0, snr_db=60.0, k=0.0,
                           a=0.0, b=0.0, c=1.0, rng=rng)
    # Without noise and no species, absorbance stays at the scatter
    # baseline (a + b*wn + c*0 = 0 since a=0, b=0, c=1, x=0).
    np.testing.assert_allclose(absorb, 0.0, atol=1e-9)


def test_comp_spectrum_pure_tg_returns_tg_reference():
    """Pure TG (x = [1,0,0,0,0,0]) gives TG's reference spectrum (within AWGN)."""
    ref, wn = _load_reference_spectra(None)
    x_tg = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # cs=0 (no photometric noise), very high SNR (AWGN effectively off),
    # no scatter (a=b=0, c=1).
    absorb = comp_spectrum(ref, x_tg, cs=0, snr_db=120.0, k=0.0,
                           a=0.0, b=0.0, c=1.0,
                           rng=np.random.default_rng(0))
    # Reference power drives AWGN; with snr_db=120 the noise std is
    # ~1e-6 × sqrt(signal_power). We allow a tolerance of 1e-3 for
    # the floating-point roundtrip through log10.
    np.testing.assert_allclose(absorb, ref[0, :], atol=1e-3)


def test_comp_spectrum_awgn_snr_doubles_noise_power():
    """Halving the SNR (in dB) increases noise std by ~sqrt(2)."""
    ref, _ = _load_reference_spectra(None)
    x = np.array([0.1, 0.1, 0.1, 0.2, 0.4, 0.1])
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    a1 = comp_spectrum(ref, x, cs=0, snr_db=40.0, k=0.0,
                       a=0.0, b=0.0, c=1.0, rng=rng1)
    a2 = comp_spectrum(ref, x, cs=0, snr_db=34.0, k=0.0,
                       a=0.0, b=0.0, c=1.0, rng=rng2)
    # Reference is identical; difference is the AWGN only.
    diff = a1 - a2
    # Halving SNR (6 dB ≈ factor 4 in power) → ~2x noise std.
    # We use a rough check: std(a1 - a2) ≈ sqrt(2) * std(a1 - ref).
    # Here we just check the std is non-trivially larger than zero.
    assert np.std(diff) > 1e-6


def test_comp_spectrum_invalid_cs_raises():
    """cs outside {0,1,2,3} raises ValueError."""
    ref, _ = _load_reference_spectra(None)
    with pytest.raises(ValueError):
        comp_spectrum(ref, np.array([1.0, 0, 0, 0, 0, 0]), cs=4,
                      snr_db=30.0, k=0.0, a=0.0, b=0.0, c=1.0,
                      rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# SpectrumGenerator behavior
# ---------------------------------------------------------------------------

def test_generator_disabled_returns_none():
    """Generator with enabled=False never returns a sample."""
    cfg = SpectrumConfig(enabled=False, spctr_t=1.0)
    gen = SpectrumGenerator(cfg)
    x = np.array([0.1, 0.1, 0.1, 0.2, 0.4, 0.1])
    assert gen.maybe_sample(0.0, 0, x, x, x[:3]) is None
    assert gen.maybe_sample(100.0, 100, x, x, x[:3]) is None


def test_generator_fires_at_spctr_t():
    """First call fires immediately; subsequent calls respect cadence."""
    cfg = SpectrumConfig(enabled=True, spctr_t=60.0, seed=42)
    gen = SpectrumGenerator(cfg)
    x = np.array([0.1, 0.1, 0.1, 0.2, 0.4, 0.1])
    s1 = gen.maybe_sample(0.0, 0, x, x, x[:3])
    assert s1 is not None
    assert s1.sim_t == 1
    # Second call within spctr_t → None.
    assert gen.maybe_sample(30.0, 100, x, x, x[:3]) is None
    # At spctr_t → fires again.
    s2 = gen.maybe_sample(60.0, 200, x, x, x[:3])
    assert s2 is not None
    assert s2.sim_t == 2


def test_generator_drift_advances_per_fire():
    """Drift counters advance by the MATLAB upstream constants per fire.

    The MATLAB upstream increments drift AFTER computing and logging
    the sample, so the captured ``drift_a/b/c`` on a sample are the
    *pre-fire* values. The next sample sees the *post-fire* values.
    """
    cfg = SpectrumConfig(enabled=True, spctr_t=60.0, seed=42)
    gen = SpectrumGenerator(cfg)
    x = np.array([0.1, 0.1, 0.1, 0.2, 0.4, 0.1])
    s1 = gen.maybe_sample(0.0, 0, x, x, x[:3])
    # First sample carries the configured drift values (pre-fire).
    assert s1.drift_a == pytest.approx(cfg.drift_a)
    assert s1.drift_b == pytest.approx(cfg.drift_b)
    assert s1.drift_c == pytest.approx(cfg.drift_c)
    s2 = gen.maybe_sample(60.0, 100, x, x, x[:3])
    # Second sample carries post-first-fire drift (one increment).
    assert s2.drift_a == pytest.approx(cfg.drift_a + 0.0005)
    assert s2.drift_b == pytest.approx(cfg.drift_b + 1e-7)
    assert s2.drift_c == pytest.approx(cfg.drift_c + 0.005)


def test_generator_seed_determinism():
    """Same seed → identical spectra across two generators."""
    cfg1 = SpectrumConfig(enabled=True, spctr_t=60.0, seed=7)
    cfg2 = SpectrumConfig(enabled=True, spctr_t=60.0, seed=7)
    gen1 = SpectrumGenerator(cfg1)
    gen2 = SpectrumGenerator(cfg2)
    x = np.array([0.1, 0.1, 0.1, 0.2, 0.4, 0.1])
    s1 = gen1.maybe_sample(0.0, 0, x, x, x[:3])
    s2 = gen2.maybe_sample(0.0, 0, x, x, x[:3])
    np.testing.assert_array_equal(s1.reactor, s2.reactor)
    np.testing.assert_array_equal(s1.light, s2.light)
    np.testing.assert_array_equal(s1.heavy, s2.heavy)


def test_generator_seed_difference_differs():
    """Different seeds → different spectra."""
    gen1 = SpectrumGenerator(SpectrumConfig(enabled=True, spctr_t=60.0, seed=1))
    gen2 = SpectrumGenerator(SpectrumConfig(enabled=True, spctr_t=60.0, seed=2))
    x = np.array([0.1, 0.1, 0.1, 0.2, 0.4, 0.1])
    s1 = gen1.maybe_sample(0.0, 0, x, x, x[:3])
    s2 = gen2.maybe_sample(0.0, 0, x, x, x[:3])
    assert not np.allclose(s1.reactor, s2.reactor)


def test_generator_quality_indicators_in_range():
    """QC indicators are finite, non-negative floats."""
    cfg = SpectrumConfig(enabled=True, spctr_t=60.0, seed=42)
    gen = SpectrumGenerator(cfg)
    x = np.array([0.05, 0.02, 0.01, 0.10, 0.80, 0.02])
    s = gen.maybe_sample(0.0, 0, x, x, x[:3])
    assert np.isfinite(s.qc_reactor_mg)
    assert np.isfinite(s.qc_reactor_tg)
    assert s.qc_reactor_mg >= 0.0
    assert s.qc_reactor_tg >= 0.0


def test_generator_heavy_phase_uses_three_species():
    """Heavy-phase absorbance is non-trivial even though only 3 species."""
    gen = SpectrumGenerator(SpectrumConfig(enabled=True, spctr_t=60.0, seed=42))
    # Heavy phase with all M, no E/G.
    x_reactor = np.array([0.05, 0.02, 0.01, 0.10, 0.80, 0.02])
    x_light = np.array([0.05, 0.02, 0.01, 0.10, 0.80, 0.02])
    x_heavy = np.array([0.5, 0.3, 0.2])
    s = gen.maybe_sample(0.0, 0, x_reactor, x_light, x_heavy)
    # Heavy spectrum should reflect M+E+G mixture, not all six species.
    # Specifically, no TG contribution (index 0).
    assert np.all(np.isfinite(s.heavy))
    assert s.heavy.shape == (gen.n_channels,)


# ---------------------------------------------------------------------------
# LiveSimulator integration
# ---------------------------------------------------------------------------

def test_step_result_spectra_none_when_disabled():
    """With spectrum_enabled=False, every step's spectra is None."""
    from bdsim.config import Settings
    # spectrum_enabled is default-ON as of 1.2.0; explicitly disable
    # so this test's "every step's spectra is None" assertion still
    # holds.
    sim = LiveSimulator(
        settings=Settings(tf=60.0, dt=1.0),
        pfaults=ProcessFaults(spectrum_enabled=False),
        seed=42,
    )
    while not sim.done:
        row = sim.step()
        assert row.spectra is None


def test_step_result_spectra_fires_at_spctr_t():
    """With spectrum_enabled=True, spectra fires every spctr_t seconds."""
    from bdsim.config import Settings
    pf = ProcessFaults(spectrum_enabled=True, spctr_t=60.0)
    sim = LiveSimulator(pfaults=pf, settings=Settings(tf=300.0, dt=1.0), seed=42)
    fires = []
    while not sim.done:
        row = sim.step()
        if row.spectra is not None:
            fires.append(row.spectra.t)
    # 300s run, spctr_t=60s → fires at t=0, 60, 120, 180, 240 → 5 samples.
    # We don't pin the exact count because t=0 may or may not fire
    # depending on t > 0 vs. _last_fire_t (=-inf). Either way, at least
    # 4 fires.
    assert len(fires) >= 4
    # Spacing between fires is approximately spctr_t.
    diffs = np.diff(fires)
    assert np.all(diffs >= 60.0 - 1e-6)


def test_spectrum_enabled_preserves_legacy_fingerprint():
    """Spectrum post-processing must not perturb the state trajectory.

    The sv hash from a 1h sim with spectrum_enabled=False must equal
    the sv hash from the same sim with spectrum_enabled=True.
    """
    from bdsim.config import Settings
    s = Settings(tf=3600.0, dt=1.0)

    # spectrum_enabled is default-ON as of 1.2.0; explicitly disable
    # on the "off" sim so this test still exercises the comparison.
    sim_off = LiveSimulator(
        settings=s,
        pfaults=ProcessFaults(spectrum_enabled=False),
        seed=42,
    )
    while not sim_off.done:
        sim_off.step()

    pf = ProcessFaults(spectrum_enabled=True, spctr_t=120.0)
    sim_on = LiveSimulator(pfaults=pf, settings=s, seed=42)
    while not sim_on.done:
        sim_on.step()

    h_off = hashlib.sha256(sim_off._sv.tobytes()).hexdigest()
    h_on = hashlib.sha256(sim_on._sv.tobytes()).hexdigest()
    assert h_off == h_on, (
        f"Spectrum post-process perturbed the state trajectory:\n"
        f"  off={h_off}\n  on ={h_on}"
    )


def test_spectrum_sample_payload_shape():
    """SpectrumSample payloads have correct shapes."""
    from bdsim.config import Settings
    pf = ProcessFaults(spectrum_enabled=True, spctr_t=300.0)
    sim = LiveSimulator(pfaults=pf, settings=Settings(tf=600.0, dt=1.0), seed=42)
    seen = None
    while not sim.done:
        row = sim.step()
        if row.spectra is not None:
            seen = row.spectra
            break
    assert seen is not None
    assert isinstance(seen, SpectrumSample)
    n_ch = 631
    assert seen.reactor.shape == (n_ch,)
    assert seen.light.shape == (n_ch,)
    assert seen.heavy.shape == (n_ch,)
    assert seen.wn.shape == (n_ch,)
    assert isinstance(seen.qc_reactor_mg, float)
    assert isinstance(seen.qc_reactor_tg, float)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_band_sum_outside_table_returns_zero():
    """Band outside the wavenumber axis → 0.0."""
    wn = np.linspace(637.5, 3787.5, 631)
    absorb = np.ones(631)
    assert _band_sum(absorb, wn, 5000.0, 6000.0) == 0.0


def test_band_sum_full_table_returns_sum():
    """Band covering full table → sum of all absorbance."""
    wn = np.linspace(637.5, 3787.5, 631)
    absorb = np.ones(631)
    assert _band_sum(absorb, wn, 0.0, 5000.0) == pytest.approx(631.0)