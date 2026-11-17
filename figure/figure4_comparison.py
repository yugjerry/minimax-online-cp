"""ACI and truncated split conformal prediction under the same constraint."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figure._common import (
    BETAS, BLUE, CHECKPOINT_STEP, GUIDE_LEGEND_GRAY, Line2D, ORANGE, PAPER_FIGSIZE, Patch,
    arrays, checked_band, checked_line, draw_guide, finish, fixed_guide, guide_handle,
    interval_label, np, plt, read_rows, record_series, select, style_axis,
)


def aci_figure(root, output):
    source = root / "matched" / "summary.csv"
    rows = read_rows(source)
    fig, axes = plt.subplots(1, 3, figsize=PAPER_FIGSIZE)
    checks, plotted, guides, panels = [], [], [], []
    for ax, beta, label in zip(axes, BETAS, "abc"):
        for algorithm, color in (("cp", BLUE), ("aci", ORANGE)):
            t, y, lo, hi = arrays(select(rows, beta, algorithm), "risk")
            keep = t >= CHECKPOINT_STEP
            if np.any(lo[keep] <= 0):
                raise ValueError("Positive lower intervals required on logarithmic axes")
            checked_band(ax, t[keep], lo[keep], hi[keep], checks,
                         f"beta={beta}, {algorithm}: MC interval", color=color, alpha=.16, linewidth=0)
            checked_line(ax, t[keep], y[keep], checks, f"beta={beta}: {algorithm}",
                         color=color, linewidth=1.8, zorder=3)
            exponent = max(.5, beta) if algorithm == "cp" else (1+beta)/2
            guide, displayed, audit = fixed_guide(t, y, exponent,
                f"matched/summary.csv: algorithm={algorithm}, beta={beta}")
            guides.append(audit)
            draw_guide(ax, t, guide, displayed, color, checks, f"beta={beta}, {algorithm}: guide")
            record_series(plotted, label, algorithm, beta, t, y, lo, hi, keep, guide, displayed)
        cp_rate = r"\sqrt{T}" if beta <= .5 else fr"T^{{{beta:g}}}"
        ax.text(.045, .95, "Conformal: " + f"${cp_rate}$", color=BLUE,
                transform=ax.transAxes, va="top", fontsize=9.0)
        ax.text(.045, .80, "ACI: " + fr"$T^{{{(1+beta)/2:g}}}$", color=ORANGE,
                transform=ax.transAxes, va="top", fontsize=9.0)
        ax.set(xscale="log", yscale="log", xlim=(CHECKPOINT_STEP, int(t[-1])), xlabel=r"$T$",
               title=f"({label}) " + fr"$\beta={beta:.1f}$")
        style_axis(ax, log_grid=True)
        panels.append({"panel": label, "beta": beta, "kappa": 2.,
                       "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
                       "scales": ["log", "log"]})
    axes[0].set_ylabel("Expected disagreement")
    handles = [Line2D([], [], color=BLUE, linewidth=1.8, label="Truncated split conformal"),
               Line2D([], [], color=ORANGE, linewidth=1.8, label="ACI"),
               Patch(facecolor=".5", alpha=.2, label=interval_label()),
               guide_handle(GUIDE_LEGEND_GRAY, "Theoretical rates")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, .015), fontsize=9.0, handlelength=1.8,
               columnspacing=1.0, handletextpad=.4)
    fig.subplots_adjust(left=.085, right=.985, top=.88, bottom=.27, wspace=.32)
    caption = (
        "Truncated unsmoothed split conformal and ACI under matched coverage "
        "constraints, with alpha=0.1, kappa=2, and beta=0.3,0.5,0.7. "
        "ACI starts at q_1=0.9 and uses eta_t=t^{-(1-beta)}. Curves show expected "
        "cumulative disagreement with the population score oracle through "
        "T=250000 on logarithmic axes, with pointwise 95% Monte Carlo intervals "
        "over 500 replications. Observations are shared between algorithms "
        "within each replication. The theoretical rates have exponent max(1/2,beta) "
        "for conformal and (1+beta)/2 for ACI. Only their heights are fitted at "
        "the five prespecified horizons from 50000 through 250000."
    )
    return finish(fig, "simulation_aci_comparison", output, checks, plotted,
                  [source], guides, panels, caption=caption, generator=__file__)

if __name__ == "__main__":
    from figure._cli import main
    main(["aci"])
