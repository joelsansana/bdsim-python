"""One-shot generator for S00–S07_compare.ipynb (run from repo root if regenerating)."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

SHARED_SETUP = """\
**Shared setup** (all gold scenarios):
- Horizon: 1 week (`tf = 604800 s`), `dt = 30 s`, seed `42`
- Controllers: all four PIDs AUTO, `nic = 4`; nominal SPs unless overridden — `TR = 60.4 °C`, `TD = 50 °C`, `hH = 0.5 m`, `Foil = 3000 kg/h`
- Sensor + ARMAX noise off; `use_demo_schedules = False` (calendars match Octave gold)
- Legacy 21-wide `ProcessFaults` (dynamic fouling / quality / wear / spectra masters **off**) for upstream ASCII comparability
"""

# Per-scenario brief fields
BRIEFS: dict[str, dict[str, str]] = {
    "S00": {
        "title": "S00 — Healthy baseline",
        "purpose": "Reference closed-loop plant with no process, sensor, or valve faults.",
        "knobs": "No process / sensor / valve faults; constant setpoints at nominal values.",
        "why": (
            "Proves core ODE + PID + I/O without confounding faults. "
            "Use as the visual baseline for every other scenario."
        ),
        "port": (
            "Left (Octave) and right (Python) should track closely on all channels; "
            "loops regulate after settle; valves/flows steady; DP near baseline."
        ),
    },
    "S01": {
        "title": "S01 — Filter clogging + cleaning",
        "purpose": "Isolates filter pore clogging and automatic cleaning when DP hits the threshold.",
        "knobs": (
            "`clog_fraction = 5.95e-7`, `DPclean = 1e5 Pa`, `filter_std = 0` "
            "(deterministic clogging)."
        ),
        "why": (
            "Pore radius shrinks → `DPfilter` rises; at `DPclean` the filter cleans "
            "(`tclean`) and DP resets. Validates clogging ODE + clean event timing."
        ),
        "port": (
            "Both panels should show the same DP sawtooth and clean events (fig 09); "
            "oil-valve / Foil respond as DP loads the filter (figs 03, 08)."
        ),
    },
    "S02": {
        "title": "S02 — HEX linear fouling (mode 1)",
        "purpose": "Isolates deterministic heat-exchanger fouling (linear heat-transfer factor decay).",
        "knobs": "`fouling = 1`, `foulingpar = [3e-7]` (mode-1 linear fouling).",
        "why": (
            "Heat-transfer factor decays over the week; the TR loop must raise `Qheat` "
            "to hold temperatures. Shows actuator effort under fouling without sensor faults."
        ),
        "port": (
            "`Qheat` trends up vs S00 on both engines (fig 01); TR/TD remain regulated "
            "(figs 05–06)."
        ),
    },
    "S03": {
        "title": "S03 — Side reactions / kinetic loss",
        "purpose": "Isolates reduced observed transesterification rate (side-reaction / kinetic loss).",
        "knobs": "`ratio_robs_r = 0.9` (10% lower observed main rate vs nominal).",
        "why": (
            "Shifts reactor and dryer ester composition without sensor or valve faults. "
            "Composition channels (fig 02) are the primary signature."
        ),
        "port": (
            "`yDRY_E` (and related dryer shape) differs from healthy on both sides; "
            "temperature/level loops otherwise calm."
        ),
    },
    "S04": {
        "title": "S04 — Sequential sensor fault calendar",
        "purpose": "Exercises each of the five PV fault modes on a fixed day calendar.",
        "knobs": (
            "Quiet noise; windows in days from `t = 0`:\n"
            "- **d1–2:** `DPfilter` bias `+5e3 Pa`\n"
            "- **d2–3.5:** `TD` ramp drift `5e-6 · (t − 2d)`\n"
            "- **d3.5–4.5:** `TR` scale `a = 1.05`\n"
            "- **d4.5–5.5:** `Foil` stuck at `3000 kg/h`\n"
            "- **d5.5–6.5:** `hH` loss of signal"
        ),
        "why": (
            "One fault mode at a time on a known calendar — validates sensor fault "
            "wiring (bias, drift, scale, stuck, loss) without process faults."
        ),
        "port": (
            "Figs 05–09 show the active fault window on the affected measurement on "
            "both engines; healthy channels stay close Octave↔Python."
        ),
    },
    "S05": {
        "title": "S05 — Oil-valve stiction + Foil SP steps",
        "purpose": "Isolates oil-valve stiction under Foil setpoint steps.",
        "knobs": (
            "Foil SP `3000 → 3100 kg/h` at day 1, back to `3000` at day 4; "
            "oil-valve stiction `S = 3%`, slip jump `J = 5%`."
        ),
        "why": (
            "Setpoint move plus sticky actuator → stick–slip on oil lift / Foil "
            "tracking. Separates actuator nonlinearity from sensor faults."
        ),
        "port": (
            "Fig 08 shows matching SP steps; fig 03 oil-lift order shows stiction "
            "signature (not smooth healthy) on both sides."
        ),
    },
    "S06": {
        "title": "S06 — Intermittent TD sensor fault",
        "purpose": "Isolates intermittent bias on the decanter temperature measurement.",
        "knobs": (
            "From day 1: TD bias `b = 0.5`, `is_intermit` on TD, "
            "`tmax_interm = 3600 s`."
        ),
        "why": (
            "Intermittent (not continuous) bias on TD — the hard port edge because "
            "NumPy RNG ≠ Octave intermittence masks."
        ),
        "port": (
            "Fig 06: measurement jumps/holds vs state/SP on both sides. "
            "**Timing of intermittence windows may differ** engine-to-engine; "
            "structure (bias present, loops still try to regulate) matters more than "
            "sample-identical masks."
        ),
    },
    "S07": {
        "title": "S07 — Combined multi-fault",
        "purpose": "Stresses concurrent process + sensor effects in one fixed gold week.",
        "knobs": (
            "Clogging as S01 (`clog_fraction = 5.95e-7`, `DPclean = 1e5 Pa`) + "
            "fouling as S02 (`fouling = 1`, `foulingpar = [3e-7]`) + "
            "`ratio_robs_r = 0.95` + TD bias `0.5 K` on **day 3–4 only**."
        ),
        "why": (
            "Closer to a busy plant week while remaining a reproducible gold profile: "
            "clogging cleans, fouling raises heat duty, kinetics shift composition, "
            "and a short TD bias window hits the sensor."
        ),
        "port": (
            "Both sides: DP sawtooth + cleans (fig 09), rising `Qheat` (fig 01), "
            "composition shift (fig 02), clear TD bias window on fig 06 days 3–4."
        ),
    },
}

# Base figure captions (What / Compared); Expect is scenario-tuned below.
FIG_WHAT = {
    1: (
        "**What:** reactor `TR` and decanter `TD` (°C, left axis); oil feed `Toil` (°C) "
        "and heat duty `Qheat` (W, right axis)."
    ),
    2: (
        "**What:** decanter heavy-phase level `hH` (m, left); dryer product ester "
        "mass fraction `yDRY_E` (–, right)."
    ),
    3: (
        "**What:** methanol feed temperature `Tmet` (°C, left); oil-valve controller "
        "order `order_lift_oil` (%, right)."
    ),
    4: (
        "**What:** heavy-phase valve controller order `order_lift_H` (%, left); "
        "methanol mass flow `Fmet` (kg/h, right)."
    ),
    5: (
        "**What:** TR control loop — setpoint (dashed), measurement (blue), state (red), °C."
    ),
    6: (
        "**What:** TD control loop — setpoint (dashed), measurement (blue), state (red), °C."
    ),
    7: (
        "**What:** hH control loop — setpoint (dashed), measurement (blue), state (red), m."
    ),
    8: (
        "**What:** Foil control loop — setpoint (dashed) and measurement (blue), kg/h."
    ),
    9: (
        "**What:** filter differential pressure `DPfilter` measurement, Pa."
    ),
}

COMPARED = (
    "**Compared:** left = Octave ASCII gold; right = Python Parquet. "
    "Same series, colors, and axis ranges as the upstream BDsim plot block."
)

# Default expect (healthy-ish); overridden per scenario where needed
DEFAULT_EXPECT = {
    1: "**Expect:** temperatures regulated; `Qheat` / `Toil` shapes match on both panels.",
    2: "**Expect:** `hH` held near SP; `yDRY_E` near high purity; panels agree.",
    3: "**Expect:** `Tmet` and oil-valve order track; panels agree.",
    4: "**Expect:** H-valve order and `Fmet` steady/regulated; panels agree.",
    5: "**Expect:** after settle, measurement and state track SP; panels agree.",
    6: "**Expect:** after settle, measurement and state track SP; panels agree.",
    7: "**Expect:** after settle, measurement and state track SP; panels agree.",
    8: "**Expect:** Foil measurement tracks SP; panels agree.",
    9: "**Expect:** DP near baseline (no clogging); panels agree.",
}

SCENARIO_EXPECT: dict[str, dict[int, str]] = {
    "S00": {},  # defaults
    "S01": {
        3: (
            "**Expect:** oil-valve order opens more as DP rises, then eases after cleans; "
            "same pattern both sides."
        ),
        8: (
            "**Expect:** Foil may dip/recover around clean events; timing should match."
        ),
        9: (
            "**Expect:** DP sawtooth rising to ~`DPclean`, then reset at cleans — "
            "same events both sides."
        ),
    },
    "S02": {
        1: (
            "**Expect:** `Qheat` trends upward over the week (fouling) while TR/TD hold; "
            "both engines show the same rise."
        ),
        5: "**Expect:** TR still tracks SP despite rising heat duty; panels agree.",
    },
    "S03": {
        2: (
            "**Expect:** `yDRY_E` / purity shape shifted vs healthy baseline; "
            "Octave and Python should match each other."
        ),
    },
    "S04": {
        5: (
            "**Expect:** days 3.5–4.5 — TR measurement scaled (~×1.05) vs state/SP; "
            "window visible on both panels."
        ),
        6: (
            "**Expect:** days 2–3.5 — TD measurement drifts (ramp bias); "
            "then returns; both panels."
        ),
        7: (
            "**Expect:** days 5.5–6.5 — hH measurement loses signal (flat/held); "
            "state still moves; both panels."
        ),
        8: (
            "**Expect:** days 4.5–5.5 — Foil measurement stuck at 3000 kg/h; both panels."
        ),
        9: (
            "**Expect:** days 1–2 — DP measurement bias +5 kPa; both panels."
        ),
    },
    "S05": {
        3: (
            "**Expect:** oil-lift order shows stick–slip around Foil SP steps "
            "(not smooth); both sides."
        ),
        8: (
            "**Expect:** Foil SP steps 3000→3100 (day 1) and back (day 4); "
            "measurement lags/sticks with stiction; both panels."
        ),
    },
    "S06": {
        6: (
            "**Expect:** from day 1, TD measurement shows intermittent +0.5 bias "
            "vs state/SP. Window timing may differ Octave↔Python (RNG); structure should match."
        ),
    },
    "S07": {
        1: (
            "**Expect:** rising `Qheat` from fouling (as S02) plus other disturbances; "
            "both panels."
        ),
        2: (
            "**Expect:** composition shift from `ratio_robs_r = 0.95`; panels agree."
        ),
        6: (
            "**Expect:** clear TD measurement bias window on **days 3–4** only; both panels."
        ),
        9: (
            "**Expect:** DP clogging sawtooth + cleans (as S01); both panels."
        ),
    },
}


def _md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")],
    }


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")],
    }


def _brief_md(sid: str) -> str:
    b = BRIEFS[sid]
    return "\n".join([
        f"# {b['title']}",
        "",
        "Octave (left) vs Python (right) — original nine BDsim figures.",
        "",
        f"**Purpose.** {b['purpose']}",
        "",
        SHARED_SETUP,
        "",
        f"**This scenario’s knobs.** {b['knobs']}",
        "",
        f"**Why these knobs.** {b['why']}",
        "",
        f"**Port check.** {b['port']}",
        "",
        "Data: `data/results_octave/gold_dataset_1w/" + sid + "/` and "
        "`data/results_py/" + sid + "/`.",
    ])


def _fig_md(sid: str, n: int) -> str:
    expect = SCENARIO_EXPECT.get(sid, {}).get(n, DEFAULT_EXPECT[n])
    return "\n".join([
        f"## Fig {n:02d}",
        "",
        FIG_WHAT[n],
        "",
        COMPARED,
        "",
        expect,
    ])


def build_notebook(sid: str) -> dict:
    cells: list[dict] = [
        _md(_brief_md(sid)),
        _code(
            "import sys\n"
            "from pathlib import Path\n"
            "\n"
            "# Allow `from _compare import …` when the kernel cwd is the repo root\n"
            "_gold = Path.cwd() / \"notebooks\" / \"gold\"\n"
            "if not (_gold / \"_compare.py\").is_file():\n"
            "    _gold = Path.cwd()  # notebook opened with cwd = notebooks/gold\n"
            "if str(_gold) not in sys.path:\n"
            "    sys.path.insert(0, str(_gold))\n"
            "\n"
            "from _compare import (\n"
            "    load_pair,\n"
            "    fig_dual_01, fig_dual_02, fig_dual_03, fig_dual_04, fig_dual_05,\n"
            "    fig_dual_06, fig_dual_07, fig_dual_08, fig_dual_09,\n"
            ")\n"
            f"\noct, py = load_pair({sid!r})\n"
            f"print('loaded', {sid!r}, "
            "'oct rows', len(oct['measurements']), "
            "'py rows', len(py['measurements']))"
        ),
    ]
    factories = [
        "fig_dual_01", "fig_dual_02", "fig_dual_03", "fig_dual_04", "fig_dual_05",
        "fig_dual_06", "fig_dual_07", "fig_dual_08", "fig_dual_09",
    ]
    for n, fn in enumerate(factories, start=1):
        cells.append(_md(_fig_md(sid, n)))
        cells.append(_code(f"{fn}(oct, py).show()"))
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


def main() -> None:
    for sid in BRIEFS:
        path = HERE / f"{sid}_compare.ipynb"
        nb = build_notebook(sid)
        # nbformat-friendly: strip trailing newline-only last source fragment issues
        path.write_text(json.dumps(nb, indent=1) + "\n")
        print("wrote", path.name)


if __name__ == "__main__":
    main()
