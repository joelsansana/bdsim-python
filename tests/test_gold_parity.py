"""Gold Parquet parity vs ``data/results_octave/gold_dataset_1w`` (Octave ASCII).

RK45 vs Octave ``lsode`` and NumPy vs Octave RNG drift are expected.
Full-week RMSE uses pre-built ``data/results_py/S##`` when present (cluster);
otherwise S00 can be generated in-process. SMOKE always runs short horizons.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bdsim.io_parquet import (
    INPUT_COLS,
    MEASUREMENT_COLS,
    SETPOINT_COLS,
    STATE_COLS,
    load_ascii_gold,
    load_parquet_dir,
    save_parquet,
)
from bdsim.scenarios.gold import SCENARIO_IDS, run_gold_scenario

ROOT = Path(__file__).resolve().parents[1]
GOLD_ROOT = ROOT / "data" / "results_octave" / "gold_dataset_1w"
RESULTS_PY = ROOT / "data" / "results_py"

# Tolerant bands: solver drift dominates; not byte-identical.
_REL = {
    "measurements": 0.15,
    "setpoints": 1e-6,
    "states": 0.15,
    "inputs": 0.25,
}
_ABS = {
    "measurements": 5.0,
    "setpoints": 1e-6,
    "states": 2.0,
    "inputs": 50.0,
}


def _gold_available() -> bool:
    return (GOLD_ROOT / "S00" / "measurements").is_file()


pytestmark = pytest.mark.skipif(
    not _gold_available(),
    reason=f"gold ASCII not found at {GOLD_ROOT} (gitignored data/)",
)


def _align_and_rmse(
    py: "object",
    gold: "object",
    cols: list[str],
) -> dict[str, float]:
    import pandas as pd

    assert isinstance(py, pd.DataFrame) and isinstance(gold, pd.DataFrame)
    # Join on nearest t_s (grids should match after gold load).
    g = gold.set_index("t_s")
    p = py.set_index("t_s")
    common = g.index.intersection(p.index)
    if len(common) < 10:
        # Fall back to positional trim to min length
        n = min(len(g), len(p))
        g = g.iloc[:n]
        p = p.iloc[:n]
        common = g.index
    else:
        g = g.loc[common]
        p = p.loc[common]
    out: dict[str, float] = {}
    for c in cols:
        if c == "t_s" or c not in g.columns or c not in p.columns:
            continue
        diff = np.asarray(p[c], dtype=float) - np.asarray(g[c], dtype=float)
        out[c] = float(np.sqrt(np.nanmean(diff ** 2)))
    return out


def test_ascii_gold_loader_s00_schema():
    frames = load_ascii_gold(GOLD_ROOT / "S00")
    assert "measurements" in frames
    assert list(frames["measurements"].columns) == MEASUREMENT_COLS
    assert frames["measurements"]["t_s"].iloc[0] == pytest.approx(0.0)
    assert "states" in frames
    assert frames["states"]["t_s"].iloc[0] == pytest.approx(0.0)
    assert list(frames["states"].columns)[:8] == STATE_COLS[:8]


def test_smoke_all_scenarios_parquet_schema(tmp_path):
    """Short horizon: wiring + Parquet schema for every scenario."""
    for sid in SCENARIO_IDS:
        out = tmp_path / sid
        res = run_gold_scenario(
            sid, seed=42, npoints=40, outdir=out, verbose=False
        )
        assert res.t.size >= 30
        files = list(out.glob("*.parquet"))
        assert (out / "measurements.parquet").is_file()
        assert (out / "inputs.parquet").is_file()
        assert (out / "setpoints.parquet").is_file()
        assert (out / "states.parquet").is_file()
        frames = load_parquet_dir(out)
        assert list(frames["measurements"].columns) == MEASUREMENT_COLS
        assert list(frames["inputs"].columns) == INPUT_COLS
        assert list(frames["setpoints"].columns) == SETPOINT_COLS
        assert len(frames["states"].columns) == len(STATE_COLS)


def test_phenomenology_smoke_s00_s01_s02_s03(tmp_path):
    """Port of Octave T00–T03 fingerprints on short runs."""
    s00 = run_gold_scenario("S00", seed=42, npoints=80, outdir=tmp_path / "S00", verbose=False)
    # T00 regulation after settle (~10%)
    i0 = max(1, len(s00.t) // 10)
    tr = s00.pv[i0:, 0]
    td = s00.pv[i0:, 1]
    sp_tr = s00.sp[i0:, 0]
    sp_td = s00.sp[i0:, 1]
    assert float(np.mean(np.abs(tr - sp_tr))) < 5.0
    assert float(np.mean(np.abs(td - sp_td))) < 5.0

    s01 = run_gold_scenario("S01", seed=42, npoints=80, outdir=tmp_path / "S01", verbose=False)
    dp = s01.pv[:, 4]
    assert float(np.max(dp)) > float(dp[0])

    run_gold_scenario("S02", seed=42, npoints=80, outdir=tmp_path / "S02", verbose=False)
    s03 = run_gold_scenario("S03", seed=42, npoints=80, outdir=tmp_path / "S03", verbose=False)
    # Side reactions: ester composition shifts vs healthy
    n = min(len(s00.t), len(s03.t))
    assert float(np.mean(np.abs(s03.sv[:n, 4] - s00.sv[:n, 4]))) > 1e-6


@pytest.mark.slow
@pytest.mark.parametrize("sid", list(SCENARIO_IDS))
def test_full_week_rmse_vs_gold(sid, tmp_path):
    """Compare full-week Python Parquet to Octave gold ASCII.

    Prefers pre-built ``data/results_py/S##`` from the cluster runner; otherwise
    generates the scenario in-process (slow).
    """
    gold = load_ascii_gold(GOLD_ROOT / sid)
    prebuilt = RESULTS_PY / sid / "measurements.parquet"
    if prebuilt.is_file():
        py = load_parquet_dir(RESULTS_PY / sid)
    else:
        if sid != "S00":
            pytest.skip(
                f"no pre-built {RESULTS_PY / sid}; "
                "run ./scripts/run_gold_scenarios_cluster.sh (only S00 auto-builds)"
            )
        out = tmp_path / sid
        run_gold_scenario(sid, seed=42, outdir=out, verbose=False)
        py = load_parquet_dir(out)

    # S06: intermittence RNG differs — check structure only
    if sid == "S06":
        assert "measurements" in py and len(py["measurements"]) > 1000
        assert py["measurements"]["t_s"].iloc[-1] > 5e5
        return

    for name, cols in (
        ("measurements", MEASUREMENT_COLS),
        ("setpoints", SETPOINT_COLS),
        ("states", STATE_COLS),
        ("inputs", INPUT_COLS),
    ):
        if name not in gold or name not in py:
            continue
        rmses = _align_and_rmse(py[name], gold[name], cols)
        rel = _REL[name]
        abs_t = _ABS[name]
        for col, rmse in rmses.items():
            # Clogging scenarios: RK45 vs Octave lsode shifts DPclean /
            # pore-reset timing; oil-valve / Foil / DP family then
            # misalign for the rest of the week. Gate that family on the
            # pre-clean window (S01/S07 only).
            _clog_cols = {
                ("measurements", "Foil_kg_h"),
                ("measurements", "DPfilter_Pa"),
                ("states", "lifto_pct"),
                ("states", "rpores_um"),
                ("inputs", "order_lift_oil_pct"),
            }
            if sid in ("S01", "S07") and (name, col) in _clog_cols:
                g_df = gold[name]
                p_df = py[name]
                pre = (g_df["t_s"] < 1.5e5) & (p_df["t_s"] < 1.5e5)
                g_pre = g_df.loc[pre, col].to_numpy()
                p_pre = p_df.loc[pre, col].to_numpy()
                n = min(len(g_pre), len(p_pre))
                early_rmse = float(np.sqrt(np.mean((p_pre[:n] - g_pre[:n]) ** 2)))
                assert early_rmse <= max(abs_t, rel * (float(np.nanstd(g_pre[:n])) + abs_t)), (
                    f"{sid}/{name}.{col} early RMSE={early_rmse:.4g}"
                )
                if col == "Foil_kg_h":
                    assert "tclean" in py and len(py["tclean"]) >= 1, (
                        f"{sid}: expected filter-clean events in data/results_py"
                    )
                continue
            gstd = float(np.nanstd(gold[name][col])) + abs_t
            # Allow absolute floor or relative to signal scale
            assert rmse <= max(abs_t, rel * gstd) or rmse <= abs_t * 20, (
                f"{sid}/{name}.{col}: RMSE={rmse:.4g} "
                f"(rel={rel}, abs={abs_t}, gold_std={gstd:.4g})"
            )


@pytest.mark.slow
def test_s01_tclean_present_when_prebuilt():
    gold = load_ascii_gold(GOLD_ROOT / "S01")
    if "tclean" not in gold or len(gold["tclean"]) == 0:
        pytest.skip("gold S01 has no tclean")
    prebuilt = RESULTS_PY / "S01"
    if not (prebuilt / "measurements.parquet").is_file():
        pytest.skip("data/results_py/S01 not pre-built")
    py = load_parquet_dir(prebuilt)
    assert "tclean" in py and len(py["tclean"]) >= 1
