"""Baseline disagreement, truncation cost, and total disagreement."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figure._common import (
    BETAS, CHECKPOINT_STEP, COMPONENTS, COMPONENT_COLORS, COMPONENT_LABELS, Line2D, ORANGE,
    PAPER_FIGSIZE, Patch, arrays, checked_band, checked_line, draw_guide, finish, fixed_guide,
    guide_handle, interval_label, np, plt, read_rows, record_series, select, style_axis,
)


def decomposition_figure(root, output):
    source = root / "baseline" / "summary.csv"
    rows = read_rows(source)
    fig, axes = plt.subplots(1, 3, figsize=PAPER_FIGSIZE)
    checks, plotted, guides, panels = [], [], [], []
    for ax, beta, label, regime in zip(axes, BETAS, "abc", ("Wide", "Critical", "Narrow")):
        selected = select(rows, beta)
        wide = beta < .5
        max_interval = 0.
        for component in COMPONENTS:
            t, y, lo, hi = arrays(selected, component)
            keep = t >= CHECKPOINT_STEP
            if not wide and np.any(lo[keep] <= 0):
                raise ValueError("Positive lower intervals required on logarithmic axes")
            color = COMPONENT_COLORS[component]
            checked_band(ax, t[keep], lo[keep], hi[keep], checks,
                         f"beta={beta}, {component}: MC interval", color=color,
                         alpha=.10 if component == "sum" else .14, linewidth=0)
            checked_line(ax, t[keep], y[keep], checks, f"beta={beta}: {component}",
                         color=color, linewidth=1.1 if component == "sum" else 1.7,
                         linestyle=(0, (2, 2)) if component == "sum" else "-",
                         zorder=4 if component == "sum" else 3)
            guide = displayed = None
            if component == "truncation":
                exponent = 0. if beta < .5 else beta
                guide, displayed, audit = fixed_guide(t, y, exponent,
                    f"baseline/summary.csv: truncation, beta={beta}")
                guides.append(audit)
                draw_guide(ax, t, guide, displayed, color, checks, f"beta={beta}: truncation guide")
            record_series(plotted, label, component, beta, t, y, lo, hi, keep, guide, displayed)
            max_interval = max(max_interval, float(hi.max()))
        ax.set(xscale="log", yscale="linear" if wide else "log",
               xlim=(CHECKPOINT_STEP, int(t[-1])), xlabel=r"$T$",
               title=f"({label}) {regime}: " + fr"$\beta={beta:.1f}$")
        if wide:
            ax.set_ylim(0, 1.08*max_interval)
        style_axis(ax, log_grid=True)
        panels.append({"panel": label, "beta": beta, "xlim": list(ax.get_xlim()),
                       "ylim": list(ax.get_ylim()),
                       "scales": ["log", "linear" if wide else "log"],
                       "all_summary_values_displayed": True})
    axes[0].set_ylabel("Expected disagreement")
    handles = [Line2D([], [], color=COMPONENT_COLORS[c],
                      linestyle=(0, (2, 2)) if c == "sum" else "-",
                      linewidth=1.1 if c == "sum" else 1.7, label=COMPONENT_LABELS[c])
               for c in COMPONENTS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, .085), fontsize=9.0, handlelength=1.8,
               columnspacing=1.3, handletextpad=.4)
    fig.legend(handles=[Patch(facecolor=".5", alpha=.2, label=interval_label()),
                        guide_handle(ORANGE, "Theoretical truncation rates")],
               loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, .005), fontsize=9.0, handlelength=1.8, columnspacing=1.8)
    fig.subplots_adjust(left=.085, right=.985, top=.88, bottom=.36, wspace=.32)
    caption = (
        "Disagreement decomposition for truncated unsmoothed split conformal "
        "with alpha=0.1, kappa=0.5, and beta=0.3,0.5,0.7 through T=250000. "
        "Curves show expected baseline, truncation, and total disagreement and "
        "the sum of baseline and truncation, computed within each replication. "
        "The wide-constraint panel uses a linear vertical axis to include zero "
        "and confidence intervals reaching zero; the critical and narrow panels "
        "use logarithmic vertical axes. All horizontal axes are logarithmic. "
        "Every recorded mean and interval is "
        "displayed without a positive floor. Pointwise 95% Monte Carlo intervals "
        "use all 500 replications. The theoretical truncation rates have exponents 0, 0.5, "
        "and 0.7. Only their heights are fitted at the five prespecified horizons."
    )
    return finish(fig, "simulation_decomposition", output, checks, plotted,
                  [source], guides, panels, caption=caption, generator=__file__)

if __name__ == "__main__":
    from figure._cli import main
    main(["decomposition"])
