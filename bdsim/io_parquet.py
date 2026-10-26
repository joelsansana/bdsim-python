"""Parquet I/O for bdsim trajectories (Port_execution handbook §6).

Final port exports use float64 Parquet with snappy compression and stable
snake_case column names. ASCII gold under ``data/gold_dataset_*`` is loaded
via :func:`load_ascii_gold` into the same schema for parity tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Results

# ---------------------------------------------------------------------------
# Column schemas (handbook §6)
# ---------------------------------------------------------------------------

MEASUREMENT_COLS = [
    "t_s", "TR_C", "TD_C", "hH_m", "Foil_kg_h", "DPfilter_Pa",
]
INPUT_COLS = [
    "t_s", "order_lift_oil_pct", "Tmet_C", "Fmet_kg_h", "Toil_C",
    "Qheat_W", "order_lift_H_pct",
]
SETPOINT_COLS = [
    "t_s", "sp_TR_C", "sp_TD_C", "sp_hH_m", "sp_Foil_kg_h",
]
STATE_COLS = [
    "t_s",
    "xR_TG", "xR_DG", "xR_MG", "xR_M", "xR_E", "xR_G", "TR_C",
    "xL_TG", "xL_DG", "xL_MG", "xL_M", "xL_E", "xL_G",
    "xH_M", "xH_E", "xH_G", "hH_m", "TD_C", "rpores_um",
    "lifto_pct", "liftH_pct",
    "xDRY_TG", "xDRY_DG", "xDRY_MG", "xDRY_M", "xDRY_E", "xDRY_G",
    "yDRY_TG", "yDRY_DG", "yDRY_MG", "yDRY_M", "yDRY_E", "yDRY_G",
]
TCLEAN_COLS = ["t_s"]

_PARQUET_KW = {"compression": "snappy", "index": False}


def _df(data: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Build a float64 DataFrame; truncate/pad columns to match data width."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    ncols = arr.shape[1]
    cols = columns[:ncols] if ncols <= len(columns) else (
        columns + [f"col_{i}" for i in range(len(columns), ncols)]
    )
    return pd.DataFrame(arr, columns=cols)


def _octave_shifted_uv(uv: np.ndarray) -> np.ndarray:
    """Approximate BDsim.m ``uv(1,:)=[]`` on an already end-trimmed array.

    Octave drops the first input row from the full grid, then drops the last
    time sample. :func:`~bdsim.simulation.run_with` end-trims every channel
    together (fingerprint contract). Shift ``uv`` forward by one and repeat
    the last row so inputs keep the same length as ``t``.
    """
    if uv.shape[0] < 2:
        return uv
    return np.vstack([uv[1:], uv[-1:]])


def results_to_frames(
    results: Results,
    *,
    octave_uv_shift: bool = True,
) -> dict[str, pd.DataFrame]:
    """Convert a :class:`Results` to handbook-schema DataFrames (display units)."""
    r = results.in_display_units()
    uv = _octave_shifted_uv(r.uv) if octave_uv_shift else r.uv
    sv21 = r.sv[:, :21] if r.sv.shape[1] >= 21 else r.sv

    frames: dict[str, pd.DataFrame] = {
        "measurements": _df(
            np.column_stack([r.t, r.pv]), MEASUREMENT_COLS
        ),
        "inputs": _df(
            np.column_stack([r.t, uv]), INPUT_COLS
        ),
        "setpoints": _df(
            np.column_stack([r.t, r.sp]), SETPOINT_COLS
        ),
        "states": _df(
            np.column_stack([r.t, sv21, r.xLend, r.yLend]),
            STATE_COLS,
        ),
    }
    if r.tclean is not None and len(r.tclean) > 0:
        frames["tclean"] = _df(np.asarray(r.tclean, dtype=np.float64), TCLEAN_COLS)
    return frames


def save_parquet(
    results: Results,
    outdir: str | Path = "data/results",
    *,
    octave_uv_shift: bool = True,
) -> dict[str, Path]:
    """Write trajectory tables as snappy Parquet under ``outdir``."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    frames = results_to_frames(results, octave_uv_shift=octave_uv_shift)
    files: dict[str, Path] = {}
    for name, df in frames.items():
        path = out / f"{name}.parquet"
        df.to_parquet(path, **_PARQUET_KW)
        files[name] = path
    return files


def load_parquet_dir(scenario_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load ``*.parquet`` tables from a scenario directory."""
    d = Path(scenario_dir)
    frames: dict[str, pd.DataFrame] = {}
    for name in ("measurements", "inputs", "setpoints", "states", "tclean"):
        path = d / f"{name}.parquet"
        if path.is_file():
            frames[name] = pd.read_parquet(path)
    return frames


def _parse_ascii_numeric_rows(text: str) -> np.ndarray:
    """Parse Octave ASCII tables; recover data glued onto ``%`` header lines."""
    rows: list[list[float]] = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        if raw.startswith("%"):
            # Header may embed the first data row (states file quirk).
            parts = raw.lstrip("%").split()
            floats: list[float] = []
            started = False
            for p in parts:
                try:
                    floats.append(float(p))
                    started = True
                except ValueError:
                    if started:
                        break
                    continue
            if floats:
                rows.append(floats)
            continue
        rows.append([float(x) for x in raw.split()])
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    width = max(len(r) for r in rows)
    out = np.full((len(rows), width), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def load_ascii_gold(scenario_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load Octave gold ASCII (``inputs``, ``states``, …) into handbook schema."""
    d = Path(scenario_dir)
    frames: dict[str, pd.DataFrame] = {}

    mapping = {
        "measurements": MEASUREMENT_COLS,
        "inputs": INPUT_COLS,
        "setpoints": SETPOINT_COLS,
        "states": STATE_COLS,
    }
    for name, cols in mapping.items():
        path = d / name
        if not path.is_file():
            continue
        arr = _parse_ascii_numeric_rows(path.read_text(encoding="utf-8", errors="replace"))
        frames[name] = _df(arr, cols)

    tclean_path = d / "tclean.txt"
    if tclean_path.is_file():
        arr = _parse_ascii_numeric_rows(
            tclean_path.read_text(encoding="utf-8", errors="replace")
        )
        if arr.size:
            frames["tclean"] = _df(arr.reshape(-1), TCLEAN_COLS)

    return frames


def ascii_dir_to_parquet(
    src: str | Path,
    dst: str | Path,
) -> dict[str, Path]:
    """Convert an Octave ASCII gold directory to handbook Parquet files."""
    frames = load_ascii_gold(src)
    out = Path(dst)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    for name, df in frames.items():
        path = out / f"{name}.parquet"
        df.to_parquet(path, **_PARQUET_KW)
        files[name] = path
    return files
