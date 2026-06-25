"""
Plotting and CSV-export helpers (Plotly-based).

Produces the same nine figures as the upstream ``BDsim.m`` plotting block
but as interactive HTML figures (zoom, pan, hover). Saves standalone
``.html`` files plus a single ``index.html`` with all nine embedded.

CSV dumps are written with the same column headers as the upstream so
existing MATLAB post-processing scripts can consume them unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import Results


# Column headers taken verbatim from upstream BDsim.m
_INPUT_HEADER = (
    "% t/s  order_lift_oil/%  Tmet/C  Fmet/(kg/h)  Toil/C  Qheat/W  "
    "order_lift_H/%\r\n"
)
_STATES_HEADER = (
    "% t/s  xR_TG  xR_DG  xR_MG  xR_M  xR_E  xR_G  TR/C  "
    "xL_TG  xL_DG  xL_MG  xL_M  xL_E  xL_G  "
    "xH_M  xH_E  xH_G  hH/m  TD/C  rpores/μm  lifto/%  liftH/%  "
    "xDRY_TG  xDRY_DG  xDRY_MG  xDRY_M  xDRY_E  xDRY_G  "
    "yDRY_TG  yDRY_DG  yDRY_MG  yDRY_M  yDRY_E  yDRY_G\r\n"
)
_MEAS_HEADER = "% t/s  TR/C  TD/C  hH/m  Foil/(kg/h)  DPfilter/Pa\r\n"
_SP_HEADER = "% t/s  sp1/C  sp2/C  sp3/m  sp4/(kg/h)\r\n"


# Plotly default template — light background, clean gridlines
_PLOTLY_TEMPLATE = "plotly_white"


# ---------------------------------------------------------------------------
# CSV writer (unchanged from matplotlib version — Plotly change is
# visualisations only)
# ---------------------------------------------------------------------------

def save_csv(results: Results, outdir: str | Path = "results") -> dict[str, Path]:
    """Write all trajectories to ASCII files matching the upstream format."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    r = results.in_display_units()
    files: dict[str, Path] = {}

    p = out / "inputs"
    p.write_text(_INPUT_HEADER)
    np.savetxt(p, np.column_stack([r.t, r.uv]), fmt="%14.6e", comments="")
    files["inputs"] = p

    p = out / "states"
    p.write_text(_STATES_HEADER)
    np.savetxt(p, np.column_stack([r.t, r.sv, r.xLend, r.yLend]),
               fmt="%14.6e", comments="")
    files["states"] = p

    p = out / "measurements"
    p.write_text(_MEAS_HEADER)
    np.savetxt(p, np.column_stack([r.t, r.pv]), fmt="%14.6e", comments="")
    files["measurements"] = p

    p = out / "setpoints"
    p.write_text(_SP_HEADER)
    np.savetxt(p, np.column_stack([r.t, r.sp]), fmt="%14.6e", comments="")
    files["setpoints"] = p

    return files


# ---------------------------------------------------------------------------
# Plotly figure factory functions
# ---------------------------------------------------------------------------

def _t_h(results: Results) -> np.ndarray:
    """Time vector in hours (matches upstream plotting units)."""
    return results.t / 3600.0


def _save(fig: go.Figure, path: Path) -> Path:
    """Save a Plotly figure as standalone HTML."""
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path


def _fig01_tr_td_toil_qheat(r) -> go.Figure:
    """Figure 1 — TR, TD on left axis; Toil, Qheat on right axis.

    Note: ``r`` is already in display units (°C / kg/h / etc.) because
    :func:`plot_all` calls :meth:`Results.in_display_units` before
    invoking this factory. Do NOT subtract 273.15 again here — that
    would give the "−200 °C" double-conversion bug.
    """
    th = _t_h(r)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=th, y=r.sv[:, 6], name="TR",
                             line=dict(color="red", width=1)), secondary_y=False)
    fig.add_trace(go.Scatter(x=th, y=r.sv[:, 17], name="TD",
                             line=dict(color="magenta", width=1)), secondary_y=False)
    fig.add_trace(go.Scatter(x=th, y=r.uv[:, 3], name="Toil",
                             line=dict(color="blue", width=1)), secondary_y=True)
    fig.add_trace(go.Scatter(x=th, y=r.uv[:, 4], name="Qheat",
                             line=dict(color="green", width=1)), secondary_y=True)

    fig.update_layout(
        title="2 input variables & 2 state variables",
        template=_PLOTLY_TEMPLATE,
        hovermode="x unified",
        xaxis_title="t / h",
    )
    fig.update_yaxes(title_text="T / °C", range=[45, 63], secondary_y=False)
    fig.update_yaxes(title_text="Qheat / W", range=[10000, 30000], secondary_y=True)
    return fig


def _fig02_hh_ydry(r) -> go.Figure:
    """Figure 2 — hH and yDRY (decanter interface & product purity)."""
    th = _t_h(r)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=th, y=r.sv[:, 16], name="hH",
                             line=dict(width=1)), secondary_y=False)
    fig.add_trace(go.Scatter(x=th, y=r.yLend[:, 4], name="yDRY_E",
                             line=dict(width=1)), secondary_y=True)

    fig.update_layout(
        title="2 state variables",
        template=_PLOTLY_TEMPLATE,
        hovermode="x unified",
        xaxis_title="t / h",
    )
    fig.update_yaxes(title_text="hH / m", range=[0.4, 0.6], secondary_y=False)
    fig.update_yaxes(title_text="yDRY / -", range=[0.95, 1.0], secondary_y=True)
    return fig


def _fig03_tmet_oil_valve(r) -> go.Figure:
    """Figure 3 — Tmet on left; oil-valve order on right.

    ``r`` is already in display units (see :func:`_fig01_tr_td_toil_qheat`).
    """
    th = _t_h(r)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=th, y=r.uv[:, 1], name="Tmet",
                             line=dict(width=1)), secondary_y=False)
    fig.add_trace(go.Scatter(x=th, y=r.uv[:, 0], name="order_lift_oil",
                             line=dict(width=1)), secondary_y=True)

    fig.update_layout(
        title="2 input variables",
        template=_PLOTLY_TEMPLATE,
        hovermode="x unified",
        xaxis_title="t / h",
    )
    fig.update_yaxes(title_text="Tmet / °C", range=[30, 55], secondary_y=False)
    fig.update_yaxes(title_text="controller order for oil valve lift / %",
                     range=[0, 100], secondary_y=True)
    return fig


def _fig04_h_valve_fmet(r) -> go.Figure:
    """Figure 4 — H-valve order on left; Fmet on right."""
    th = _t_h(r)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=th, y=r.uv[:, 5], name="order_lift_H",
                             line=dict(width=1)), secondary_y=False)
    fig.add_trace(go.Scatter(x=th, y=r.uv[:, 2], name="Fmet",
                             line=dict(width=1)), secondary_y=True)

    fig.update_layout(
        title="2 input variables",
        template=_PLOTLY_TEMPLATE,
        hovermode="x unified",
        xaxis_title="t / h",
    )
    fig.update_yaxes(title_text="controller order for H valve lift / %",
                     range=[-1, 101], secondary_y=False)
    fig.update_yaxes(title_text="Fmet / (kg/h)", range=[500, 700], secondary_y=True)
    return fig


def _fig05_tr_loop(r) -> go.Figure:
    """Figure 5 — TR loop: setpoint, measurement, state.

    ``r`` is already in display units (see :func:`_fig01_tr_td_toil_qheat`).
    """
    th = _t_h(r)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=th, y=r.sp[:, 0], name="setpoint",
                             line=dict(color="black", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=th, y=r.pv[:, 0], name="measurement",
                             line=dict(color="blue", width=1)))
    fig.add_trace(go.Scatter(x=th, y=r.sv[:, 6], name="state",
                             line=dict(color="red", width=1)))
    fig.update_layout(
        title="state & measurement & setpoint",
        template=_PLOTLY_TEMPLATE,
        hovermode="x unified",
        xaxis_title="t / h",
        yaxis_title="TR / °C",
        yaxis=dict(range=[55, 65]),
    )
    return fig


def _fig06_td_loop(r) -> go.Figure:
    """Figure 6 — TD loop: setpoint, measurement, state.

    ``r`` is already in display units (see :func:`_fig01_tr_td_toil_qheat`).
    """
    th = _t_h(r)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=th, y=r.sp[:, 1], name="setpoint",
                             line=dict(color="black", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=th, y=r.pv[:, 1], name="measurement",
                             line=dict(color="blue", width=1)))
    fig.add_trace(go.Scatter(x=th, y=r.sv[:, 17], name="state",
                             line=dict(color="red", width=1)))
    fig.update_layout(
        title="state & measurement & setpoint",
        template=_PLOTLY_TEMPLATE,
        hovermode="x unified",
        xaxis_title="t / h",
        yaxis_title="TD / °C",
        yaxis=dict(range=[45, 55]),
    )
    return fig


def _fig07_hh_loop(r) -> go.Figure:
    """Figure 7 — hH loop: setpoint, measurement, state."""
    th = _t_h(r)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=th, y=r.sp[:, 2], name="setpoint",
                             line=dict(color="black", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=th, y=r.pv[:, 2], name="measurement",
                             line=dict(color="blue", width=1)))
    fig.add_trace(go.Scatter(x=th, y=r.sv[:, 16], name="state",
                             line=dict(color="red", width=1)))
    fig.update_layout(
        title="state & measurement & setpoint",
        template=_PLOTLY_TEMPLATE,
        hovermode="x unified",
        xaxis_title="t / h",
        yaxis_title="hH / m",
        yaxis=dict(range=[-0.01, 1.0]),
    )
    return fig


def _fig08_foil_loop(r) -> go.Figure:
    """Figure 8 — Foil loop: measurement, setpoint."""
    th = _t_h(r)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=th, y=r.sp[:, 3], name="setpoint",
                             line=dict(color="black", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=th, y=r.pv[:, 3], name="measurement",
                             line=dict(color="blue", width=1)))
    fig.update_layout(
        title="measurement & setpoint",
        template=_PLOTLY_TEMPLATE,
        hovermode="x unified",
        xaxis_title="t / h",
        yaxis_title="Foil / (kg/h)",
        yaxis=dict(range=[2000, 4000]),
    )
    return fig


def _fig09_dpfilter(r) -> go.Figure:
    """Figure 9 — DPfilter measurement."""
    th = _t_h(r)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=th, y=r.pv[:, 4], name="measurement",
                             line=dict(color="blue", width=1)))
    fig.update_layout(
        title="measurement",
        template=_PLOTLY_TEMPLATE,
        hovermode="x unified",
        xaxis_title="t / h",
        yaxis_title="DP filter / Pa",
        yaxis=dict(range=[0, 1.1e5]),
    )
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_FIGS = [
    ("fig01_tr_td_toil_qheat", _fig01_tr_td_toil_qheat),
    ("fig02_hh_ydry",          _fig02_hh_ydry),
    ("fig03_tmet_oil_valve",   _fig03_tmet_oil_valve),
    ("fig04_h_valve_fmet",     _fig04_h_valve_fmet),
    ("fig05_tr_loop",          _fig05_tr_loop),
    ("fig06_td_loop",          _fig06_td_loop),
    ("fig07_hh_loop",          _fig07_hh_loop),
    ("fig08_foil_loop",        _fig08_foil_loop),
    ("fig09_dpfilter",         _fig09_dpfilter),
]


def plot_all(results: Results, outdir: str | Path = "results",
             show: bool = False) -> list[Path]:
    """Replicate the upstream nine-figure block in Plotly.

    Saves each figure as a standalone ``.html`` file plus a combined
    ``index.html`` for easy browsing. Returns the list of HTML paths.
    The ``show`` parameter is a no-op for Plotly (open the HTML files
    in a browser).
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    r = results.in_display_units()

    paths: list[Path] = []
    figures: list[tuple[str, go.Figure]] = []

    for name, factory in _FIGS:
        fig = factory(r)
        figures.append((name, fig))
        path = _save(fig, out / f"{name}.html")
        paths.append(path)

    # Combined index.html that shows all nine figures in sequence
    _save_index(figures, out / "index.html")

    return paths


def _save_index(figures: list[tuple[str, go.Figure]], path: Path) -> None:
    """Write a single HTML page that embeds all nine figures inline."""
    parts: list[str] = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>bdsim — results</title>",
        "<style>body{font-family:sans-serif;margin:24px;max-width:1200px;}",
        ".fig{margin-bottom:32px;border-bottom:1px solid #eee;padding-bottom:16px;}",
        "h2{margin-top:24px;color:#333;}</style>",
        "</head><body>",
        "<h1>bdsim — biodiesel plant simulation results</h1>",
        f"<p>Generated from a {len(figures)}-figure Plotly block. "
        "Each figure is also saved as a standalone HTML file in this folder.</p>",
    ]
    for name, fig in figures:
        parts.append(f'<div class="fig">')
        parts.append(f"<h2>{name}</h2>")
        # include_plotlyjs='inline' avoids the CDN dependency for offline viewing
        parts.append(fig.to_html(include_plotlyjs="cdn", full_html=False))
        parts.append("</div>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts))