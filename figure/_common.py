"""Shared data checks, rate guides, and figure export."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from ._style import (  # Sets the writable cache before importing pyplot.
    BLUE, CORRIDOR_COLOR, CRITICAL_COLOR, INK,
    NARROW_COLORS, ORANGE, PATH_COLORS, TEAL, WIDE_COLORS,
    configure_style, save_figure, style_axis,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import patheffects
import numpy as np


BETAS = (.3, .5, .7)
MOTIVATION_BETAS = (1., .8, .6, .4)
MOTIVATION_COLORS = ("#B13E4B", ORANGE, TEAL, BLUE)
HORIZON = 250_000
REPLICATIONS = 500
CHECKPOINT_STEP = 100
PATH_HORIZON = 50_000
COVERAGE_WINDOW = 1_000
GUIDE_HORIZONS = np.array([50_000, 100_000, 150_000, 200_000, 250_000])
COMPONENTS = ("baseline", "truncation", "total", "sum")
COMPONENT_LABELS = {
    "baseline": "Baseline", "truncation": "Truncation",
    "total": "Total", "sum": "Baseline + truncation",
}
COMPONENT_COLORS = {"baseline": BLUE, "truncation": ORANGE,
                    "total": TEAL, "sum": INK}
PAPER_FIGSIZE = (6.5, 0.35 * 6.5)  # Matches the 11-pt manuscript at 1-inch margins.

GUIDE_POLICY = (
    "Theory fixes the exponent before plotting. Only height is fitted, using "
    "the geometric mean of estimate / T**exponent at the prespecified horizons "
    "50000, 100000, 150000, 200000, 250000. Theoretical rates are displayed from 50000 "
    "through 250000. No exponent fitting, run selection, or resimulation."
)

# Theoretical-rate guides: a continuous translucent ribbon in the series colour.
# zorder 2.2 puts it above the grid (1.5) and Monte Carlo bands (1), below the
# spines (2.5) and every estimate line (3-5), so no estimate is covered.
GUIDE_RIBBON = {"linewidth": 3.6, "alpha": .38, "solid_capstyle": "butt",
                "solid_joinstyle": "round", "zorder": 2.2}
GUIDE_LEGEND_GRAY = ".25"
GUIDE_RENDERING = (
    "Each theoretical rate is a continuous translucent ribbon in its series "
    "colour, beneath the estimate; no dashes, no white underlay."
)


def guide_handle(color, label):
    """Legend entry drawn exactly like the in-panel guide ribbon."""
    style = {k: v for k, v in GUIDE_RIBBON.items() if k != "zorder"}
    return Line2D([], [], color=color, label=label, **style)


def configure_simulation_style():
    """Use near-paper-sized text at height=0.35\\textwidth, without rescaling."""
    plt.rcParams.update({
        "font.size": 9.2,
        "axes.titlesize": 9.5,
        "axes.labelsize": 10.0,
        "axes.titlepad": 5.0,
        "axes.labelpad": 2.5,
        "legend.fontsize": 9.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
    })
    if plt.rcParams["text.usetex"]:
        # Match the newpx text and mathematics in main.tex.
        plt.rcParams["text.latex.preamble"] = (
            r"\usepackage{amsmath}\usepackage{newpxtext,newpxmath}")
    else:
        # STIX ships with Matplotlib and fits the paper's compact panel layout.
        plt.rcParams.update({"font.serif": ["STIXGeneral"], "mathtext.fontset": "stix"})


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def write_rows(path, rows):
    if not rows:
        raise ValueError(f"No plotted values for {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fingerprint(path):
    path = Path(path)
    path = path.resolve()
    label = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else "/".join(path.parts[-2:])
    return {"path": label,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def select(rows, beta, algorithm=None):
    selected = [row for row in rows if abs(float(row["beta"]) - beta) < 1e-9
                and (algorithm is None or row["algorithm"] == algorithm)
                and row.get("oracle", "fixed_score") == "fixed_score"]
    if not selected:
        raise ValueError(f"Missing beta={beta}, algorithm={algorithm}")
    return sorted(selected, key=lambda row: int(row["t"]))


def arrays(rows, component):
    keys = [f"{component}_{suffix}" for suffix in ("mean", "ci_low", "ci_high")]
    t = np.array([int(row["t"]) for row in rows])
    y, lo, hi = [np.array([float(row[key]) for row in rows]) for key in keys]
    if not (np.all(np.diff(t) > 0) and np.all(t > 0)):
        raise ValueError("Summary horizons must be positive, unique, and ordered")
    if t[-1] != HORIZON or np.any(t % CHECKPOINT_STEP):
        raise ValueError(f"Summary horizons must be multiples of 100 through {HORIZON}")
    if {int(row["replications"]) for row in rows} != {REPLICATIONS}:
        raise ValueError(f"Every summary must use all {REPLICATIONS} replications")
    if not np.all(np.isin(GUIDE_HORIZONS, t)):
        raise ValueError("Summary is missing prespecified guide anchors")
    if not (np.all(np.isfinite([y, lo, hi])) and np.all(lo <= y) and np.all(y <= hi)):
        raise ValueError(f"Invalid intervals for {component}")
    if np.any(y < 0):
        raise ValueError(f"Negative disagreement estimate for {component}")
    se_key = f"{component}_se"
    if se_key in rows[0]:
        se = np.array([float(row[se_key]) for row in rows])
        if not (np.all(np.isfinite(se)) and np.all(se >= 0)):
            raise ValueError(f"Invalid standard error for {component}")
    return t, y, lo, hi


def checked_line(ax, x, y, checks, description, **kwargs):
    line, = ax.plot(x, y, **kwargs)
    px, py = line.get_data()
    np.testing.assert_array_equal(px, x)
    np.testing.assert_array_equal(py, y)
    checks.append({"array": description, "points": len(x), "exact_match": True})
    return line


def checked_band(ax, x, lo, hi, checks, description, **kwargs):
    band = ax.fill_between(x, lo, hi, **kwargs)
    vertices = band.get_paths()[0].vertices
    n = len(x)
    np.testing.assert_array_equal(vertices[1:n + 1, 0], x)
    np.testing.assert_array_equal(vertices[1:n + 1, 1], lo)
    np.testing.assert_array_equal(vertices[n + 2:2*n + 2, 0], x[::-1])
    np.testing.assert_array_equal(vertices[n + 2:2*n + 2, 1], hi[::-1])
    checks.append({"array": description, "points": n, "exact_match": True})
    return band


def fixed_guide(t, y, exponent, source):
    eligible = np.flatnonzero(np.isin(t, GUIDE_HORIZONS))
    if (not np.array_equal(t[eligible], GUIDE_HORIZONS)
            or np.any(y[eligible] <= 0)):
        raise ValueError(f"Five positive prespecified guide anchors required: {source}")
    shape = t.astype(float)**exponent
    height = float(np.exp(np.mean(np.log(y[eligible] / shape[eligible]))))
    guide = height * shape
    displayed = t >= t[eligible[0]]
    audit = {"source": source, "exponent": exponent, "height": height,
             "fit_horizons": t[eligible].tolist(),
             "fit_values": y[eligible].tolist(), "exponent_fitted": False}
    return guide, displayed, audit


def draw_guide(ax, t, guide, displayed, color, checks, description):
    """One continuous translucent ribbon beneath the estimate (no gaps)."""
    checked_line(ax, t[displayed], guide[displayed], checks, description,
                 color=color, linestyle="-", **GUIDE_RIBBON)


def record_series(plotted, panel, series, beta, t, y, lo, hi, keep, guide=None, displayed=None):
    for i, horizon in enumerate(t):
        plotted.append({"panel": panel, "series": series, "beta": beta,
                        "t": int(horizon), "mean": float(y[i]),
                        "ci_low": float(lo[i]), "ci_high": float(hi[i]),
                        "displayed": bool(keep[i]),
                        "rate_guide": "" if guide is None else float(guide[i]),
                        "guide_displayed": False if displayed is None else bool(displayed[i])})


def interval_label():
    return r"95\% MC interval" if plt.rcParams["text.usetex"] else "95% MC interval"


def finish(fig, stem, output, checks, plotted, sources, guides, panels,
           *, estimand="Expected cumulative disagreement with the population score oracle",
           caption=None, generator=None):
    if not caption:
        raise ValueError(f"A caption is required for {stem}")
    audit = save_figure(fig, stem, output)
    write_rows(output / f"{stem}_plotted_data.csv", plotted)
    caption_file = f"{stem}_caption.md"
    (output / caption_file).write_text(caption + "\n", encoding="utf-8")
    audit.update({"checks": checks,
                  "every_plotted_array_matches_source": all(c["exact_match"] for c in checks),
                  "numerical_sources": [fingerprint(path) for path in sources],
                  "generator": fingerprint(generator or __file__),
                  "shared_sources": [fingerprint(__file__), fingerprint(Path(__file__).with_name("_style.py"))],
                  "guides": guides,
                  "guide_policy": GUIDE_POLICY, "guide_rendering": GUIDE_RENDERING,
                  "guide_ribbon": GUIDE_RIBBON, "panels": panels,
                  "estimand": estimand,
                  "study": "arXiv:2611.21542, version 1",
                  "horizon": HORIZON, "replications": REPLICATIONS,
                  "checkpoint_step": CHECKPOINT_STEP,
                  "caption": caption, "caption_file": caption_file,
                  "visual_inspection": "Inspect final exported PNG before publication"})
    (output / f"{stem}_manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    plt.close(fig)
    return audit


