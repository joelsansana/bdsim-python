"""Octave vs Python gold compare helpers for notebooks/gold/*.ipynb.

Loads week-long gold from ``data/results_octave/gold_dataset_1w/S##`` (ASCII)
and ``data/results_py/S##`` (Parquet), then builds side-by-side Plotly figures
matching the nine upstream BDsim plots in ``bdsim.plots``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bdsim.io_parquet import load_ascii_gold, load_parquet_dir

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_OCTAVE_ROOT = _ROOT / "data" / "results_octave" / "gold_dataset_1w"
_PYTHON_ROOT = _ROOT / "data" / "results_py"

_TEMPLATE = "plotly_white"
Frames = dict[str, pd.DataFrame]


def repo_root() -> Path:
    return _ROOT


def load_pair(scenario_id: str) -> tuple[Frames, Frames]:
    """Load Octave ASCII and Python Parquet frames for one gold scenario."""
    sid = scenario_id.upper()
    oct_dir = _OCTAVE_ROOT / sid
    py_dir = _PYTHON_ROOT / sid
    missing: list[str] = []
    if not oct_dir.is_dir():
        missing.append(str(oct_dir))
    if not py_dir.is_dir():
        missing.append(str(py_dir))
    if missing:
        raise FileNotFoundError(
            "gold data missing (gitignored; generate or sync first):\n  "
            + "\n  ".join(missing)
        )
    return load_ascii_gold(oct_dir), load_parquet_dir(py_dir)


def _th(frames: Frames) -> Any:
    return frames["measurements"]["t_s"].to_numpy() / 3600.0


def _col(frames: Frames, table: str, name: str) -> Any:
    return frames[table][name].to_numpy()


def _dual_base(
    *,
    title: str,
    secondary_y: bool = False,
) -> go.Figure:
    specs = [[{"secondary_y": secondary_y}, {"secondary_y": secondary_y}]]
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=("Octave", "Python"),
        specs=specs,
        horizontal_spacing=0.08,
    )
    fig.update_layout(
        title=title,
        template=_TEMPLATE,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.12, x=0),
    )
    fig.update_xaxes(title_text="t / h", row=1, col=1)
    fig.update_xaxes(title_text="t / h", row=1, col=2)
    return fig


def _add_scatter(
    fig: go.Figure,
    *,
    col: int,
    x: Any,
    y: Any,
    name: str,
    line: dict[str, Any],
    secondary_y: bool = False,
    showlegend: bool = True,
) -> None:
    # Legend once (left panel); right panel mirrors series without duplicating.
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name=name,
            line=line,
            showlegend=showlegend and col == 1,
            legendgroup=name,
        ),
        row=1,
        col=col,
        secondary_y=secondary_y,
    )


def fig_dual_01(oct_f: Frames, py_f: Frames) -> go.Figure:
    """Fig 01 — TR, TD (left); Toil, Qheat (right). Side-by-side engines."""
    fig = _dual_base(
        title="Fig 01 — 2 input variables & 2 state variables",
        secondary_y=True,
    )
    for col, frames in ((1, oct_f), (2, py_f)):
        th = _th(frames)
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "states", "TR_C"),
            name="TR", line=dict(color="red", width=1),
        )
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "states", "TD_C"),
            name="TD", line=dict(color="magenta", width=1),
        )
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "inputs", "Toil_C"),
            name="Toil", line=dict(color="blue", width=1), secondary_y=True,
        )
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "inputs", "Qheat_W"),
            name="Qheat", line=dict(color="green", width=1), secondary_y=True,
        )
    for c in (1, 2):
        fig.update_yaxes(
            title_text="T / °C", range=[45, 63], row=1, col=c, secondary_y=False,
        )
        fig.update_yaxes(
            title_text="Qheat / W", range=[10000, 30000],
            row=1, col=c, secondary_y=True,
        )
    return fig


def fig_dual_02(oct_f: Frames, py_f: Frames) -> go.Figure:
    """Fig 02 — hH and yDRY_E (decanter level & dryer ester)."""
    fig = _dual_base(title="Fig 02 — 2 state variables", secondary_y=True)
    for col, frames in ((1, oct_f), (2, py_f)):
        th = _th(frames)
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "states", "hH_m"),
            name="hH", line=dict(width=1),
        )
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "states", "yDRY_E"),
            name="yDRY_E", line=dict(width=1), secondary_y=True,
        )
    for c in (1, 2):
        fig.update_yaxes(
            title_text="hH / m", range=[0.4, 0.6], row=1, col=c, secondary_y=False,
        )
        fig.update_yaxes(
            title_text="yDRY / -", range=[0.95, 1.0],
            row=1, col=c, secondary_y=True,
        )
    return fig


def fig_dual_03(oct_f: Frames, py_f: Frames) -> go.Figure:
    """Fig 03 — Tmet and oil-valve controller order."""
    fig = _dual_base(title="Fig 03 — 2 input variables", secondary_y=True)
    for col, frames in ((1, oct_f), (2, py_f)):
        th = _th(frames)
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "inputs", "Tmet_C"),
            name="Tmet", line=dict(width=1),
        )
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "inputs", "order_lift_oil_pct"),
            name="order_lift_oil", line=dict(width=1), secondary_y=True,
        )
    for c in (1, 2):
        fig.update_yaxes(
            title_text="Tmet / °C", range=[30, 55], row=1, col=c, secondary_y=False,
        )
        fig.update_yaxes(
            title_text="controller order for oil valve lift / %",
            range=[0, 100], row=1, col=c, secondary_y=True,
        )
    return fig


def fig_dual_04(oct_f: Frames, py_f: Frames) -> go.Figure:
    """Fig 04 — H-valve order and Fmet."""
    fig = _dual_base(title="Fig 04 — 2 input variables", secondary_y=True)
    for col, frames in ((1, oct_f), (2, py_f)):
        th = _th(frames)
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "inputs", "order_lift_H_pct"),
            name="order_lift_H", line=dict(width=1),
        )
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "inputs", "Fmet_kg_h"),
            name="Fmet", line=dict(width=1), secondary_y=True,
        )
    for c in (1, 2):
        fig.update_yaxes(
            title_text="controller order for H valve lift / %",
            range=[-1, 101], row=1, col=c, secondary_y=False,
        )
        fig.update_yaxes(
            title_text="Fmet / (kg/h)", range=[500, 700],
            row=1, col=c, secondary_y=True,
        )
    return fig


def fig_dual_05(oct_f: Frames, py_f: Frames) -> go.Figure:
    """Fig 05 — TR loop: setpoint, measurement, state."""
    fig = _dual_base(title="Fig 05 — state & measurement & setpoint (TR)")
    for col, frames in ((1, oct_f), (2, py_f)):
        th = _th(frames)
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "setpoints", "sp_TR_C"),
            name="setpoint", line=dict(color="black", width=2, dash="dash"),
        )
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "measurements", "TR_C"),
            name="measurement", line=dict(color="blue", width=1),
        )
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "states", "TR_C"),
            name="state", line=dict(color="red", width=1),
        )
    for c in (1, 2):
        fig.update_yaxes(title_text="TR / °C", range=[55, 65], row=1, col=c)
    return fig


def fig_dual_06(oct_f: Frames, py_f: Frames) -> go.Figure:
    """Fig 06 — TD loop: setpoint, measurement, state."""
    fig = _dual_base(title="Fig 06 — state & measurement & setpoint (TD)")
    for col, frames in ((1, oct_f), (2, py_f)):
        th = _th(frames)
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "setpoints", "sp_TD_C"),
            name="setpoint", line=dict(color="black", width=2, dash="dash"),
        )
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "measurements", "TD_C"),
            name="measurement", line=dict(color="blue", width=1),
        )
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "states", "TD_C"),
            name="state", line=dict(color="red", width=1),
        )
    for c in (1, 2):
        fig.update_yaxes(title_text="TD / °C", range=[45, 55], row=1, col=c)
    return fig


def fig_dual_07(oct_f: Frames, py_f: Frames) -> go.Figure:
    """Fig 07 — hH loop: setpoint, measurement, state."""
    fig = _dual_base(title="Fig 07 — state & measurement & setpoint (hH)")
    for col, frames in ((1, oct_f), (2, py_f)):
        th = _th(frames)
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "setpoints", "sp_hH_m"),
            name="setpoint", line=dict(color="black", width=2, dash="dash"),
        )
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "measurements", "hH_m"),
            name="measurement", line=dict(color="blue", width=1),
        )
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "states", "hH_m"),
            name="state", line=dict(color="red", width=1),
        )
    for c in (1, 2):
        fig.update_yaxes(title_text="hH / m", range=[-0.01, 1.0], row=1, col=c)
    return fig


def fig_dual_08(oct_f: Frames, py_f: Frames) -> go.Figure:
    """Fig 08 — Foil loop: measurement & setpoint."""
    fig = _dual_base(title="Fig 08 — measurement & setpoint (Foil)")
    for col, frames in ((1, oct_f), (2, py_f)):
        th = _th(frames)
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "setpoints", "sp_Foil_kg_h"),
            name="setpoint", line=dict(color="black", width=2, dash="dash"),
        )
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "measurements", "Foil_kg_h"),
            name="measurement", line=dict(color="blue", width=1),
        )
    for c in (1, 2):
        fig.update_yaxes(
            title_text="Foil / (kg/h)", range=[2000, 4000], row=1, col=c,
        )
    return fig


def fig_dual_09(oct_f: Frames, py_f: Frames) -> go.Figure:
    """Fig 09 — DPfilter measurement."""
    fig = _dual_base(title="Fig 09 — measurement (DPfilter)")
    for col, frames in ((1, oct_f), (2, py_f)):
        th = _th(frames)
        _add_scatter(
            fig, col=col, x=th, y=_col(frames, "measurements", "DPfilter_Pa"),
            name="measurement", line=dict(color="blue", width=1),
        )
    for c in (1, 2):
        fig.update_yaxes(
            title_text="DP filter / Pa", range=[0, 1.1e5], row=1, col=c,
        )
    return fig


FIG_DUAL = {
    1: fig_dual_01,
    2: fig_dual_02,
    3: fig_dual_03,
    4: fig_dual_04,
    5: fig_dual_05,
    6: fig_dual_06,
    7: fig_dual_07,
    8: fig_dual_08,
    9: fig_dual_09,
}
