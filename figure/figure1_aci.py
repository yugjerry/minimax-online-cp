"""ACI coverage paths and time-averaged oracle disagreement."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figure._common import (
    CHECKPOINT_STEP, COVERAGE_WINDOW, GUIDE_LEGEND_GRAY, HORIZON, INK, Line2D,
    MOTIVATION_BETAS, MOTIVATION_COLORS, PAPER_FIGSIZE, Patch, REPLICATIONS, arrays,
    checked_band, checked_line, draw_guide, finish, fixed_guide, guide_handle, interval_label,
    np, patheffects, plt, read_rows, record_series, select, style_axis,
)


def motivation_figure(root, output):
    """Compare ACI coverage paths and disagreement at the manuscript size."""
    source = root / "motivation" / "summary.csv"
    path_source = root / "motivation" / "paths.csv"
    summaries, path_rows = read_rows(source), read_rows(path_source)
    fig, axes = plt.subplots(1, 3, figsize=PAPER_FIGSIZE)
    checks, plotted, guides = [], [], []
    expected_path_t = np.union1d(
        np.arange(1, COVERAGE_WINDOW + 1),
        np.arange(CHECKPOINT_STEP, HORIZON + 1, CHECKPOINT_STEP),
    )
    max_risk_interval = 0.
    for beta, color in zip(MOTIVATION_BETAS, MOTIVATION_COLORS):
        selected = select(path_rows, beta)
        if {int(row["rep"]) for row in selected} != {0}:
            raise ValueError("Motivation paths must use the prespecified replication 0")
        t = np.array([int(row["t"]) for row in selected])
        deviation = np.array([float(row["D"]) for row in selected])
        coverage = np.array([float(row["coverage"]) for row in selected])
        np.testing.assert_array_equal(t, expected_path_t)
        if not (np.all(np.isfinite([deviation, coverage]))
                and np.all((0 <= coverage) & (coverage <= 1))):
            raise ValueError(f"Invalid ACI path values for beta={beta}")
        path_keep, coverage_keep = t % CHECKPOINT_STEP == 0, t <= COVERAGE_WINDOW
        checked_line(axes[0], t[path_keep], deviation[path_keep], checks,
                     f"motivation beta={beta}: cumulative coverage path, rep 0",
                     color=color, linewidth=1.05, zorder=3 + beta)
        record_series(plotted, "a", "coverage_deviation_rep_0", beta, t,
                      deviation, deviation, deviation, path_keep)
        checked_line(axes[1], t[coverage_keep], coverage[coverage_keep], checks,
                     f"motivation beta={beta}: conditional coverage, rep 0",
                     color=color, linewidth=.9)
        record_series(plotted, "b", "conditional_coverage_rep_0", beta, t,
                      coverage, coverage, coverage, coverage_keep)

        # R_* columns are cumulative risk summaries, not averages over time.
        t, cumulative, cumulative_lo, cumulative_hi = arrays(select(summaries, beta), "R")
        y, lo, hi = cumulative/t, cumulative_lo/t, cumulative_hi/t
        keep = t >= CHECKPOINT_STEP
        if np.any(lo[keep] <= 0):
            raise ValueError("Positive motivation intervals required on logarithmic axes")
        checked_band(axes[2], t[keep], lo[keep], hi[keep], checks,
                     f"motivation beta={beta}: risk per round MC interval",
                     color=color, alpha=.16, linewidth=0)
        checked_line(axes[2], t[keep], y[keep], checks,
                     f"motivation beta={beta}: mean risk per round",
                     color=color, linewidth=1.7, zorder=3)
        guide = displayed = None
        if beta < 1:
            exponent = -(1-beta)/2
            guide, displayed, audit = fixed_guide(
                t, y, exponent, f"motivation/summary.csv: R_mean/t, beta={beta}")
            guides.append(audit)
            draw_guide(axes[2], t, guide, displayed, color, checks,
                       f"motivation beta={beta}: theoretical rate per round")
        record_series(plotted, "c", "risk_per_round", beta, t, y, lo, hi,
                      keep, guide, displayed)
        max_risk_interval = max(max_risk_interval, float(hi.max()))

    axes[0].axhline(0, color=".35", linewidth=.5)
    axes[0].set(title="(a) Cumulative miscoverage deviation", ylabel=r"$E_t$",
                xlim=(CHECKPOINT_STEP, HORIZON), xlabel=r"$t$ (thousands)")
    ticks = [CHECKPOINT_STEP, HORIZON//2, HORIZON]
    axes[0].set_xticks(ticks, [f"{t / 1000:g}" for t in ticks])
    coverage_t = np.arange(1, COVERAGE_WINDOW + 1)
    target = np.full(COVERAGE_WINDOW, .9)
    target_line = checked_line(axes[1], coverage_t, target, checks, "motivation: target coverage 0.9",
                               color=INK, linestyle=":", linewidth=1.2, zorder=5)
    target_line.set_path_effects([patheffects.withStroke(linewidth=2.2, foreground="white")])
    record_series(plotted, "b", "target_coverage", "", coverage_t, target,
                  target, target, np.ones(COVERAGE_WINDOW, dtype=bool))
    axes[1].set(title="(b) Instantaneous coverage", ylabel=r"$1-p_t$",
                xlim=(1, COVERAGE_WINDOW), ylim=(-.035, 1.045), xlabel=r"$t$")
    axes[1].set_xticks([1, COVERAGE_WINDOW//2, COVERAGE_WINDOW], ["1", "500", "1,000"])
    axes[2].set(title="(c) Time-averaged disagreement", ylabel=r"$\widehat r_t/t$",
                xscale="log", yscale="log", xlim=(CHECKPOINT_STEP, HORIZON), xlabel=r"$t$")
    axes[2].set_xticks([1e2, 1e3, 1e4, 1e5])
    for ax in axes:
        style_axis(ax, log_grid=ax is axes[2])
    handles = [Line2D([], [], color=color, linewidth=1.7, label=fr"$\beta={beta:.1f}$")
               for beta, color in zip(MOTIVATION_BETAS, MOTIVATION_COLORS)]
    handles += [Patch(facecolor=".5", alpha=.2, label=interval_label()),
                guide_handle(GUIDE_LEGEND_GRAY, "Theoretical rates")]
    fig.legend(handles=handles, ncol=6, frameon=False, loc="lower center",
               bbox_to_anchor=(.5, .015), fontsize=9.0, handlelength=1.5,
               columnspacing=.85, handletextpad=.4)
    fig.subplots_adjust(left=.075, right=.975, top=.88, bottom=.30, wspace=.55)
    # Leave room for the long final title at the right edge of the page.
    axes[2].title.set_x(.43)
    panels = [
        {"panel": "a", "displayed_replication": 0, "horizon": HORIZON,
         "checkpoint_step": CHECKPOINT_STEP, "scales": ["linear", "linear"]},
        {"panel": "b", "displayed_replication": 0, "horizon": COVERAGE_WINDOW,
         "checkpoint_step": 1, "scales": ["linear", "linear"]},
        {"panel": "c", "mean_replications": REPLICATIONS, "horizon": HORIZON,
         "scales": ["log", "log"], "source_transformation": "Divide every cumulative R summary by t",
         "maximum_upper_interval": max_risk_interval},
    ]
    caption = (
        "Long-run coverage versus oracle disagreement in ACI. Step sizes are "
        "eta_t=t^{-(1-beta)} for beta=1,0.8,0.6,0.4, with alpha=0.1 and q_1=0.9. "
        "Replication 0 supplies cumulative coverage error at every 100th round "
        "through 250000 and conditional coverage at every round through 1000. "
        "Mean oracle disagreement per round and pointwise 95% Monte Carlo "
        "intervals use all 500 replications through T=250000. The first two "
        "panels use linear axes and the third uses logarithmic axes. "
        "The dotted line in panel (b) marks the target coverage of 0.9. "
        "In panel (c), translucent lines beneath the estimates show the theoretical "
        "rates t^(-(1-beta)/2) for beta=0.8,0.6,0.4. Only their heights are fitted at the five "
        "prespecified horizons. The constant-step curve (beta=1) is empirical."
    )
    return finish(fig, "aci_motivation_R", output, checks, plotted,
                  [source, path_source], guides, panels,
                  estimand="ACI paths and expected cumulative oracle disagreement divided by t",
                  caption=caption, generator=__file__)

if __name__ == "__main__":
    from figure._cli import main
    main(["motivation"])
