"""Paper typography, colors, and checked PDF, SVG, and PNG exports."""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "online-cp-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "online-cp-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

INK = "#202020"
TEAL = "#2F7F72"
BLUE = "#315DA8"
ORANGE = "#D18432"
WIDE_COLORS = {0.1: "#9ECAE1", 0.2: "#6BAED6", 0.3: BLUE, 0.4: TEAL}
NARROW_COLORS = {0.6: "#E4AE67", 0.7: ORANGE, 0.8: "#C65F33", 0.9: "#B13E4B"}
CRITICAL_COLOR = "#76679C"
PATH_COLORS = {"wide": BLUE, "narrow": ORANGE}
CORRIDOR_COLOR = "#B13E4B"


@functools.lru_cache(maxsize=1)
def usetex_available() -> bool:
    """Check the dependencies needed for the existing paper typography."""
    if any(shutil.which(c) is None for c in ("latex", "dvipng", "gs", "kpsewhich")):
        return False
    for package in ("type1ec.sty", "newpxtext.sty", "newpxmath.sty"):
        try:
            result = subprocess.run(
                ["kpsewhich", package], capture_output=True, text=True, timeout=20
            )
        except (OSError, subprocess.SubprocessError):
            return False
        if result.returncode or not result.stdout.strip():
            return False
    return True


def configure_style(force_usetex: bool | None = None) -> bool:
    """Apply the common style and return whether external LaTeX is enabled."""
    usetex = usetex_available() if force_usetex is None else force_usetex
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "text.usetex": usetex,
        "text.latex.preamble": r"\usepackage{amsmath,amssymb,amsfonts}",
        "mathtext.fontset": "cm",
        "font.size": 8.2,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.8,
        "legend.fontsize": 7.1,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "svg.hashsalt": "online-cp-simulations",
        "savefig.facecolor": "white",
        "path.simplify": False,
    })
    return usetex


def style_axis(ax, *, log_grid: bool = False) -> None:
    """Apply the quiet open-frame axes used in the simulation panels."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", color="0.87", linewidth=0.45)
    if log_grid:
        ax.grid(True, which="minor", color="0.93", linewidth=0.30)
    ax.tick_params(length=3, width=0.6)


def save_figure(fig, stem: str, out: Path | str) -> dict:
    """Check text bounds; write PDF, 600-dpi PNG, SVG and a compact QA record.

    Paths are ``out/stem.ext``.  The caller retains ownership of the figure
    and should close it after export.  No automatic tight crop changes sizes.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    dormant = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            hidden = not ax.get_visible() or not ax.axison or not axis.get_visible()
            low, high = sorted(axis.get_view_interval())
            if hidden:
                dormant.update((id(axis.label), id(axis.get_offset_text())))
            for tick in (*axis.get_major_ticks(), *axis.get_minor_ticks()):
                # Locators cache ticks beyond the visible interval, especially
                # on logarithmic axes. Axis.draw() never renders those labels.
                if hidden or not low <= tick.get_loc() <= high:
                    dormant.update((id(tick.label1), id(tick.label2)))
    for artist in fig.findobj(Text):
        if id(artist) in dormant or not artist.get_visible() or not artist.get_text():
            continue
        # Hidden axes retain dormant tick labels, which are not rendered.
        if artist.axes is not None and not artist.axes.get_visible():
            continue
        bounds = artist.get_window_extent(renderer)
        if (bounds.x0 < -1 or bounds.y0 < -1
                or bounds.x1 > fig.bbox.x1 + 1 or bounds.y1 > fig.bbox.y1 + 1):
            outside.append(artist.get_text())
    if outside:
        raise RuntimeError(f"Text outside {stem} artboard: {outside}")
    files = {}
    for extension in ("pdf", "png", "svg"):
        path = out / f"{stem}.{extension}"
        kwargs = {"dpi": 600} if extension == "png" else {}
        if extension == "pdf":
            kwargs["metadata"] = {
                "Title": stem.replace("_", " "),
                "Creator": "online-cp simulation pipeline",
                "CreationDate": None,
                "ModDate": None,
            }
        elif extension == "svg":
            kwargs["metadata"] = {"Date": None}
        fig.savefig(path, **kwargs)
        files[extension] = path.name
    audit = {
        "files": files,
        "text_outside_artboard": outside,
        "width_inches": float(fig.get_figwidth()),
        "height_inches": float(fig.get_figheight()),
        "png_dpi": 600,
        "pdf_fonttype": 42,
        "latex_typography": bool(plt.rcParams["text.usetex"]),
        "matplotlib_version": matplotlib.__version__,
    }
    (out / f"{stem}_qa.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit
