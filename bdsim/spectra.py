"""
NIR/IR absorbance spectrum model — NIR/IR spectrum sensor (Roadmap).

Port of ``BDSIM_spectr/comp_spectrum.m`` (Eugeniu Strelet, Dec 2019).

The model is a Beer-Lambert absorbance computation: given the molar
fractions of a mixture and a table of pure-species reference spectra,
the mixture's absorbance at each wavelength is the mole-fraction-
weighted sum of the species absorbances. Photometric noise, Gaussian
white noise, and instrument drift are then layered on top.

Three process locations are supported (the MATLAB upstream supports
five but our state vector only models up to the decanter; we don't
have the washer/dryer downstream state slots ``sv(22:33)``):

* ``"reactor"`` — 6 species (TG/DG/MG/M/E/G)
* ``"light"``   — 6 species (light phase decanter)
* ``"heavy"``   — 3 species (M/E/G only — TG/DG/MG don't go to heavy phase)

Per the upstream convention, spectra are *not* generated every step —
they fire every ``spctr_t`` seconds (default 3600 s = 1 hour). Between
fires, ``result.spectra`` is ``None``.

Reference spectra source
------------------------
The default reference spectra live in ``bdsim/data/spectra_ref.csv``
(GPL-3, Fernandes/Strelet, 2019). 630 wavenumber channels covering
the NIR range 637.5–3310 cm⁻¹. Override with a user-supplied path
via ``ProcessFaults.spectra_ref_path``.

.. note::
    These are **demo-grade** synthetic spectra — Beer-Lambert
    absorbance sums of mole fractions, not real NIR peaks from a
    laboratory assay. The correlation engine downstream treats them
    as a sensor-class input stream, not as a real QC measurement.

Public surface
--------------
- :class:`SpectrumConfig` — dataclass with all knobs.
- :class:`SpectrumSample` — dataclass for a single fired sample.
- :func:`comp_spectrum` — single-sample computation (Beer-Lambert +
  noise + scatter).
- :class:`SpectrumGenerator` — stateful per-location sampler that
  caches the reference table and handles ``spctr_t`` cadence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np

# Maximum size of a user-supplied reference spectra CSV, in bytes.
# The bundled ``bdsim/data/spectra_ref.csv`` is ~36 KB; a real NIR
# table for 1000 channels x 100 species in float64 is ~8 MB. 50 MB
# is generous defense-in-depth against a malicious or runaway override.
_REFERENCE_SPECTRA_MAX_BYTES: int = 50 * 1024 * 1024

# Whitelist of accepted file extensions. Restricting to common tabular
# text formats prevents the override path from being pointed at
# arbitrary binary files (devices, FIFOs, /proc/* on Linux).
_REFERENCE_SPECTRA_EXTENSIONS: frozenset[str] = frozenset({".csv", ".txt", ".tsv"})


def _validate_reference_path(path: str | os.PathLike[str]) -> Path:
    """Defense-in-depth validation for a user-supplied reference-spectra path.

    Resolves symlinks and ``..`` segments, rejects directories, special
    files, and unsupported extensions, and caps the file size. The
    bundled default path (``None``) is handled in the caller and is not
    validated here.

    Parameters
    ----------
    path
        User-supplied path string or :class:`os.PathLike`.

    Returns
    -------
    Path
        Resolved absolute :class:`Path` to a regular file.

    Raises
    ------
    TypeError
        If ``path`` is not a string or :class:`os.PathLike`.
    FileNotFoundError
        If the resolved path does not exist or is not a regular file.
    ValueError
        If the file extension is not in the whitelist or the file is
        larger than :data:`_REFERENCE_SPECTRA_MAX_BYTES`.
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(
            f"reference spectra path must be str or os.PathLike, got {type(path).__name__}"
        )

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        # ``is_file()`` is False for directories, symlink loops, devices,
        # and missing paths; collapse them into a single error class so
        # callers don't have to disambiguate.
        raise FileNotFoundError(
            f"reference spectra path is not a regular file: {resolved}"
        )

    if resolved.suffix.lower() not in _REFERENCE_SPECTRA_EXTENSIONS:
        raise ValueError(
            f"reference spectra path has unsupported extension "
            f"{resolved.suffix!r}; expected one of "
            f"{sorted(_REFERENCE_SPECTRA_EXTENSIONS)}: {resolved}"
        )

    size = resolved.stat().st_size
    if size > _REFERENCE_SPECTRA_MAX_BYTES:
        raise ValueError(
            f"reference spectra file is too large ({size} bytes > "
            f"{_REFERENCE_SPECTRA_MAX_BYTES} byte cap): {resolved}"
        )

    return resolved

# Number of species per location. The MATLAB upstream supports the
# dryer locations too (sv(22:27), sv(28:33)) but our state vector
# doesn't model washer/dryer downstream — we only have the reactor
# and decanter.
LOCATION_SPECIES: dict[str, int] = {
    "reactor": 6,    # TG, DG, MG, M, E, G
    "light": 6,      # full 6-species light phase
    "heavy": 3,      # M, E, G only (TG/DG/MG excluded by physics)
}


@dataclass
class SpectrumConfig:
    """Configuration for the virtual NIR/IR sensor.

    All fields map directly onto ``user_settings.m`` from the MATLAB
    upstream.

    Attributes
    ----------
    enabled: bool
        Master switch. When ``False`` no spectra are generated and
        ``result.spectra`` is always ``None``.
    spctr_t: float
        Sampling period in seconds. Spectra fire every ``spctr_t``
        seconds. Default 3600 (1 hour) — matches the MATLAB upstream.
    cs: int
        Photometric noise model (Skoog table 13-3):
        0 = no noise, 1 = constant (k), 2 = Poisson-like (√(T+T²)),
        3 = proportional (T).
    snr_db: float
        Additive Gaussian white noise SNR in dB.
    k: float
        Photometric noise scale, typically 0.3%T (Skoog).
    drift_a, drift_b, drift_c: float
        Instrument drift scatter coefficients. The MATLAB upstream
        advances them every fire (``a += 0.0005``, ``b += 1e-7``,
        ``c += 0.005``) to simulate slow instrument drift.
    spectra_ref_path: str | None
        Override path to the reference spectra CSV. When ``None``,
        the bundled ``bdsim/data/spectra_ref.csv`` is used.
    seed: int | None
        RNG seed for deterministic noise.
    """

    enabled: bool = False
    spctr_t: float = 3600.0
    cs: int = 2
    snr_db: float = 30.0
    k: float = 0.03
    drift_a: float = 0.01
    drift_b: float = 0.0001
    drift_c: float = 1.05
    spectra_ref_path: str | None = None
    seed: int | None = None


@dataclass
class SpectrumSample:
    """A single fired spectrum sample.

    Attributes
    ----------
    t: float
        Sim time at which the sample fired (seconds).
    sim_t: int
        Step index (mirrors the MATLAB ``i_spctr`` counter).
    reactor: np.ndarray
        Absorbance vector at the reactor outlet, shape
        ``(n_channels,)``. ``None`` if the location was not
        generated this tick.
    light: np.ndarray | None
        Light phase decanter absorbance, ``(n_channels,)``.
    heavy: np.ndarray | None
        Heavy phase decanter absorbance, ``(n_channels,)``.
    wn: np.ndarray
        Wavenumber axis, shape ``(n_channels,)`` — same for all
        locations in a sample.
    qc_reactor_mg: float
        Quality indicator (proxy): glycerin contamination at reactor.
    qc_reactor_tg: float
        Quality indicator (proxy): unreacted triglyceride at reactor.
    drift_a, drift_b, drift_c: float
        Drift coefficients at the time of this sample.
    """

    t: float
    sim_t: int
    reactor: np.ndarray
    light: np.ndarray
    heavy: np.ndarray
    wn: np.ndarray
    qc_reactor_mg: float
    qc_reactor_tg: float
    drift_a: float
    drift_b: float
    drift_c: float


def _load_reference_spectra(path: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Load reference spectra from CSV.

    When ``path`` is ``None``, the bundled ``bdsim/data/spectra_ref.csv``
    is loaded from package resources. When ``path`` is provided, it is
    validated by :func:`_validate_reference_path` (resolved, extension
    and size checked, regular file required) before being opened.

    Returns
    -------
    (ref_spctrs, wn)
        ``ref_spctrs`` shape ``(6, n_channels)`` — one row per species
        in canonical order TG, DG, MG, M, E, G. ``wn`` is the
        wavenumber axis, shape ``(n_channels,)``.

    Raises
    ------
    FileNotFoundError
        If ``path`` is provided but does not resolve to a regular file.
    ValueError
        If ``path`` has an unsupported extension or exceeds the size cap.
    TypeError
        If ``path`` is not a string or :class:`os.PathLike`.

    The MATLAB upstream reads ``spectra_data = csvread('spectra_ref.csv')``
    and uses ``[DG;E;G;M;MG;TG]`` ordering, then indexes into it
    depending on location. We re-order to canonical TG-first so the
    ``x_frac @ ref_spctrs`` product is unambiguous.
    """
    if path is None:
        # Use the bundled resource.
        ref_text = resources.files("bdsim.data").joinpath("spectra_ref.csv").read_text()
        from io import StringIO
        # Header row is the wavenumber axis; species absorbance rows
        # follow. We use ndmin=2 to coerce a 1D shape into a row, and
        # then split header from body manually.
        lines = ref_text.strip().splitlines()
        wn = np.array([float(x) for x in lines[0].split(",")])
        data = np.loadtxt(StringIO("\n".join(lines[1:])), delimiter=",")
    else:
        # Validate before opening: defense-in-depth against a
        # path-traversal / arbitrary-file-read primitive if
        # ``spectra_ref_path`` is ever exposed through a network-facing
        # surface (e.g. a dashboard endpoint). See issue #22.
        resolved = _validate_reference_path(path)
        with open(resolved, encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
        wn = np.array([float(x) for x in lines[0].split(",")])
        data = np.loadtxt(lines[1:], delimiter=",")

    # ``data`` shape: (n_species, n_channels). The MATLAB upstream
    # uses ordering [DG, E, G, M, MG, TG] (see ``system_parameters.m``
    # comment "spectra_data = csvread('spectra_ref.csv'); % [DG;E;G;M;MG;TG]").
    # We reorder to canonical TG, DG, MG, M, E, G — matches the
    # mole-fraction order used by ``xR = sv[0:6]``.
    # MATLAB 0-indexed upstream: 0=DG, 1=E, 2=G, 3=M, 4=MG, 5=TG
    # Canonical (TG, DG, MG, M, E, G) → reorder as [5, 0, 4, 3, 1, 2]
    ref_spctrs = data[[5, 0, 4, 3, 1, 2], :]  # shape (6, n_channels)

    return ref_spctrs, wn


def comp_spectrum(
    ref_spctrs: np.ndarray,
    x_frac: np.ndarray,
    cs: int,
    snr_db: float,
    k: float,
    a: float,
    b: float,
    c: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Compute the composite absorbance spectrum of a mixture.

    Direct port of ``comp_spectrum.m``.

    Parameters
    ----------
    ref_spctrs: np.ndarray
        Reference spectra, shape ``(n_species, n_channels)``.
    x_frac: np.ndarray
        Mole fractions of the species present in this mixture, shape
        ``(n_species_present,)``. ``n_species_present`` may be less
        than ``n_species`` of the reference table (e.g. heavy phase
        has only 3 species); pad with zeros or pass a subset of
        ``ref_spctrs`` matching the species that are actually
        present.
    cs: int
        Photometric noise model: 0, 1, 2, or 3.
    snr_db: float
        Additive white Gaussian noise SNR in dB.
    k: float
        Photometric noise scale.
    a, b, c: float
        Instrument drift scatter coefficients.
    rng: np.random.Generator | None
        Optional RNG instance for deterministic noise.

    Returns
    -------
    np.ndarray
        Absorbance vector, shape ``(n_channels,)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Beer-Lambert: absorbance = sum of (x_i * ref_i).
    spctr_abs = x_frac @ ref_spctrs  # shape (n_channels,)
    # Invert to transmittance. Saturated by 1e-12 to keep the log finite.
    spctr_trn = np.power(10.0, -spctr_abs)
    spctr_trn = np.clip(spctr_trn, 1e-12, None)

    # Photometric noise (Skoog table 13-3).
    if cs == 0:
        s_t = np.zeros_like(spctr_trn)
    elif cs == 1:
        s_t = k * np.ones_like(spctr_trn)
    elif cs == 2:
        s_t = k * np.sqrt(spctr_trn ** 2 + spctr_trn)
    elif cs == 3:
        s_t = k * spctr_trn
    else:
        raise ValueError(f"comp_spectrum: invalid photometric noise model cs={cs}")
    spctr_trn = spctr_trn + s_t * rng.standard_normal(spctr_trn.shape)
    spctr_trn = np.clip(spctr_trn, 1e-12, None)

    spctr_abs = -np.log10(spctr_trn)

    # Additive Gaussian white noise (matches MATLAB ``awgn`` semantics).
    # MATLAB's awgn(x, snr, 'measured') measures the signal power first,
    # then adds white noise to give the requested SNR.
    signal_power = np.mean(spctr_abs ** 2)
    if signal_power > 0:
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise_std = np.sqrt(noise_power)
        spctr_abs = spctr_abs + noise_std * rng.standard_normal(spctr_abs.shape)

    # Instrument drift scatter: absorbance = a + b*wn + c*absorbance.
    # Caller passes the wavenumber axis as a separate vector (see
    # ``_drift_array`` below) by multiplying b by wn inside the loop
    # call. We expose the scalar form here for unit-testability.
    return spctr_abs


def _scatter(spctr_abs: np.ndarray, wn: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Apply instrument drift scatter: ``a + b*wn + c*absorbance``.

    Returns a new array; does not mutate the input.
    """
    return a + b * wn + c * spctr_abs


@dataclass
class SpectrumGenerator:
    """Stateful spectrum sampler.

    Owns the cached reference spectra, the drift counters (which
    advance every fire to simulate slow instrument drift), the RNG,
    and the per-location species selectors. One instance is built
    per :class:`~bdsim.live_simulator.LiveSimulator` and queried via
    :meth:`maybe_sample` once per step.

    The generator is **pure signal processing** — it has no awareness
    of the ODE state other than the mole-fraction vector the caller
    passes in. This keeps it fast to test in isolation.
    """

    config: SpectrumConfig
    ref_spctrs: np.ndarray = field(init=False)
    wn: np.ndarray = field(init=False)
    _rng: np.random.Generator = field(init=False)
    _last_fire_t: float = field(init=False, default=-np.inf)
    _drift_a: float = field(init=False)
    _drift_b: float = field(init=False)
    _drift_c: float = field(init=False)
    _sample_counter: int = field(init=False, default=0)
    # species indices into ref_spctrs for each location
    _species_idx: dict[str, np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self.ref_spctrs, self.wn = _load_reference_spectra(self.config.spectra_ref_path)
        self._rng = np.random.default_rng(self.config.seed)
        self._drift_a = self.config.drift_a
        self._drift_b = self.config.drift_b
        self._drift_c = self.config.drift_c
        # For each location, the indices into the 6-species canonical
        # table. Heavy phase skips TG/DG/MG (they don't separate).
        self._species_idx = {
            "reactor": np.arange(6),
            "light": np.arange(6),
            "heavy": np.array([3, 4, 5]),  # M, E, G
        }

    @property
    def n_channels(self) -> int:
        return self.wn.shape[0]

    def maybe_sample(
        self,
        t: float,
        step_idx: int,
        x_reactor: np.ndarray,
        x_light: np.ndarray,
        x_heavy: np.ndarray,
    ) -> SpectrumSample | None:
        """Sample the spectrum at ``t`` if the cadence has elapsed.

        Returns ``None`` between fires. Caller must check the return
        and only attach to ``StepResult`` when not ``None``.

        The mole fractions are padded/truncated to match each location's
        species count. The heavy phase passes only M/E/G; we use the
        3-species subset of the reference table.
        """
        if not self.config.enabled:
            return None
        if t - self._last_fire_t + 1e-9 < self.config.spctr_t:
            return None

        self._last_fire_t = t
        self._sample_counter += 1

        # Sample each location.
        reactor = self._sample_one(x_reactor, np.arange(6))
        light = self._sample_one(x_light, np.arange(6))
        # Heavy phase: only 3 species. Use the species subset and
        # renormalize the mole fractions to those 3 species so the
        # Beer-Lambert sum is meaningful (consistent with the
        # MATLAB upstream which pads TG/DG/MG with zeros).
        heavy_full = np.zeros(6)
        heavy_full[3:6] = x_heavy[:3]
        heavy = self._sample_one(heavy_full, np.arange(6))

        # Quality indicators: proxy = peak absorbance in a target band.
        # We use a simple band sum rather than a peak search because
        # the demo spectra are smooth synthetic curves.
        # Glycerin (G) and methanol (M) absorb strongly in the 3300-3500
        # cm⁻¹ region but our reference table only goes to ~3310;
        # use the high-wavenumber channels as a proxy for "free
        # glycerol" and "free methanol" indicators.
        # TG peak is in the mid-NIR around 1750 cm⁻¹.
        qc_mg = _band_sum(reactor, self.wn, 3200.0, 3310.0)
        qc_tg = _band_sum(reactor, self.wn, 1700.0, 1800.0)

        sample = SpectrumSample(
            t=float(t),
            sim_t=self._sample_counter,
            reactor=reactor,
            light=light,
            heavy=heavy,
            wn=self.wn.copy(),
            qc_reactor_mg=float(qc_mg),
            qc_reactor_tg=float(qc_tg),
            drift_a=self._drift_a,
            drift_b=self._drift_b,
            drift_c=self._drift_c,
        )

        # Advance drift counters (matches MATLAB upstream convention).
        self._drift_a += 0.0005
        self._drift_b += 1e-7
        self._drift_c += 0.005

        return sample

    def _sample_one(self, x_full: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Sample one location.

        ``x_full`` is the full 6-species vector (zero-padded for the
        heavy phase). ``idx`` says which reference rows to use.
        """
        ref_sub = self.ref_spctrs[idx, :]
        x_sub = x_full[idx]
        absorb = comp_spectrum(
            ref_sub,
            x_sub,
            cs=self.config.cs,
            snr_db=self.config.snr_db,
            k=self.config.k,
            a=self._drift_a,
            b=self._drift_b,
            c=self._drift_c,
            rng=self._rng,
        )
        return _scatter(absorb, self.wn, self._drift_a, self._drift_b, self._drift_c)


def _band_sum(absorb: np.ndarray, wn: np.ndarray, lo: float, hi: float) -> float:
    """Sum of absorbance values in a wavenumber band.

    Cheap proxy used as a "quality indicator" — peaks in this band
    indicate presence of the corresponding species. Returns 0.0 when
    the band is outside the reference table range.
    """
    mask = (wn >= lo) & (wn <= hi)
    if not np.any(mask):
        return 0.0
    return float(np.sum(absorb[mask]))