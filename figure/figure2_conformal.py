"""Coverage constraints and disagreement of truncated split conformal prediction."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figure._common import (
    CHECKPOINT_STEP, CORRIDOR_COLOR, CRITICAL_COLOR, GUIDE_LEGEND_GRAY, INK, Line2D,
    NARROW_COLORS, PAPER_FIGSIZE, PATH_COLORS, PATH_HORIZON, Patch, REPLICATIONS, WIDE_COLORS,
    arrays, checked_band, checked_line, draw_guide, finish, fixed_guide, guide_handle,
    interval_label, np, plt, read_rows, record_series, select, style_axis,
)


def draw_paths(ax, beta, label, paths, means, checks, plotted):
    selected, mean_rows = select(paths, beta), select(means, beta)
    ids = sorted({int(row["rep"]) for row in selected})[:12]
    if ids != list(range(12)):
        raise ValueError("The path panels must show replications 0 through 11")
    expected_t = np.arange(CHECKPOINT_STEP, PATH_HORIZON + 1, CHECKPOINT_STEP)
    for rep in ids:
        rows = [row for row in selected if int(row["rep"]) == rep]
        t = np.array([int(row["t"]) for row in rows])
        y = np.array([float(row["D"]) for row in rows])
        np.testing.assert_array_equal(t, expected_t)
        if not np.all(np.isfinite(y)):
            raise ValueError("Nonfinite cumulative coverage path")
        if np.any(np.abs(y) > .5 * t**(1-beta) + 1e-7):
            raise ValueError(f"Coverage path violates corridor at beta={beta}, rep={rep}")
        checked_line(ax, t, y, checks, f"beta={beta}: path {rep}",
                     color=".47", alpha=.40, linewidth=.55)
        record_series(plotted, label, f"path_{rep}", beta, t, y, y, y, np.ones(len(t), dtype=bool))
    t = np.array([int(row["t"]) for row in mean_rows])
    mean = np.array([float(row["mean_D"]) for row in mean_rows])
    np.testing.assert_array_equal(t, expected_t)
    if not np.all(np.isfinite(mean)):
        raise ValueError("Nonfinite mean coverage path")
    reps = {int(row["replications"]) for row in mean_rows}
    if reps != {REPLICATIONS}:
        raise ValueError(f"Expected all {REPLICATIONS} saved replications, got {reps}")
    regime = "wide" if beta < .5 else "narrow"
    checked_line(ax, t, mean, checks, f"beta={beta}: all-500 mean",
                 color=PATH_COLORS[regime], linewidth=1.6, zorder=4)
    record_series(plotted, label, "path_mean", beta, t, mean, mean, mean, np.ones(len(t), dtype=bool))
    boundary = .5 * t.astype(float)**(1-beta)
    for sign in (-1, 1):
        y = sign*boundary
        checked_line(ax, t, y, checks, f"beta={beta}: boundary {sign:+d}",
                     color=CORRIDOR_COLOR, linestyle="--", linewidth=.95)
        record_series(plotted, label, f"boundary_{sign:+d}", beta, t, y, y, y, np.ones(len(t), dtype=bool))
    ax.axhline(0, color=".30", linewidth=.5)
    ymax = 1.12 * max(float(boundary.max()), max(abs(float(r["D"])) for r in selected), 1.)
    ax.set(xlim=(CHECKPOINT_STEP, int(t[-1])), ylim=(-ymax, ymax), xlabel=r"$t$",
           title=f"({label}) {regime.capitalize()} constraints\n" + fr"$\beta={beta:.1f}$")
    ticks = sorted({CHECKPOINT_STEP, int(t[-1]//2), int(t[-1])})
    ax.set_xticks(ticks, [f"{v / 1000:g}" for v in ticks])
    ax.set_xlabel(r"$t$ (thousands)")
    ax.set_box_aspect(1)
    style_axis(ax)
    return {"panel": label, "beta": beta, "displayed_replications": ids,
            "mean_replications": 500, "horizon": int(t[-1]),
            "boundary_fully_visible": True}


def conformal_figure(root, output):
    baseline = root / "baseline"
    rows, paths, means = [read_rows(baseline / name) for name in
                          ("summary.csv", "paths.csv", "path_means.csv")]
    fig, axes = plt.subplots(1, 4, figsize=PAPER_FIGSIZE)
    checks, plotted, guides, panels = [], [], [], []
    for ax, beta, label in zip(axes[:2], (.3, .7), ("a", "b")):
        panels.append(draw_paths(ax, beta, label, paths, means, checks, plotted))
    axes[0].set_ylabel(r"$E_t$")
    raw = select(rows, .3)
    bt, by, blo, bhi = arrays(raw, "baseline")
    for beta in sorted({float(r["beta"]) for r in rows}):
        for observed, expected in zip(arrays(select(rows, beta), "baseline"), (bt, by, blo, bhi)):
            np.testing.assert_array_equal(observed, expected)
    for ax, wide, label in zip(axes[2:], (True, False), ("c", "d")):
        betas = sorted({float(row["beta"]) for row in rows if (float(row["beta"]) < .5) == wide})
        max_interval = 0.
        for beta in betas:
            t, y, lo, hi = arrays(select(rows, beta), "total")
            keep = t >= CHECKPOINT_STEP
            if not wide and np.any(lo[keep] <= 0):
                raise ValueError("Positive lower intervals required on logarithmic axes")
            color = (WIDE_COLORS if wide else NARROW_COLORS).get(round(beta, 1), CRITICAL_COLOR)
            checked_band(ax, t[keep], lo[keep], hi[keep], checks,
                         f"panel {label}, beta={beta}: total MC interval", color=color, alpha=.14, linewidth=0)
            checked_line(ax, t[keep], y[keep], checks, f"panel {label}, beta={beta}: total",
                         color=color, linewidth=1.6, label=fr"$\beta={beta:.1f}$", zorder=3)
            guide = displayed = None
            if not wide:
                guide, displayed, audit = fixed_guide(t, y, beta, f"baseline/summary.csv: total, beta={beta}")
                guides.append(audit)
                draw_guide(ax, t, guide, displayed, color, checks, f"panel {label}, beta={beta}: guide")
            record_series(plotted, label, "total", beta, t, y, lo, hi, keep, guide, displayed)
            max_interval = max(max_interval, float(hi.max()))
        keep = bt >= CHECKPOINT_STEP
        if not wide and np.any(blo[keep] <= 0):
            raise ValueError("Positive baseline intervals required on logarithmic axes")
        checked_band(ax, bt[keep], blo[keep], bhi[keep], checks,
                     f"panel {label}: baseline MC interval", color=INK, alpha=.10, linewidth=0)
        checked_line(ax, bt[keep], by[keep], checks, f"panel {label}: raw baseline",
                     color=INK, linestyle=":", linewidth=1.8, zorder=5)
        guide = displayed = None
        if wide:
            guide, displayed, audit = fixed_guide(bt, by, .5, "baseline/summary.csv: baseline, beta=0.3")
            guides.append(audit)
            draw_guide(ax, bt, guide, displayed, INK, checks, f"panel {label}: baseline square-root guide")
        record_series(plotted, label, "baseline", .3, bt, by, blo, bhi, keep, guide, displayed)
        ax.set_xscale("log")
        ax.set_xlim(CHECKPOINT_STEP, int(bt[-1]))
        ax.set_xticks([1e2, 1e3, 1e4, 1e5])
        ax.set_xlabel(r"$T$")
        if wide:
            ax.set_ylabel("Expected disagreement", fontsize=9.0, labelpad=1.0)
            ax.set_title(f"({label}) Wide constraints\n" + r"$\beta<1/2$: $\sqrt{T}$")
            ax.set_ylim(0, 1.30*max(max_interval, float(bhi.max())))
        else:
            ax.set_title(f"({label}) Narrow constraints\n" + r"$\beta\geq1/2$: $T^\beta$")
            ax.set_yscale("log")
            # Reserve the upper third for the beta legend, above every curve.
            ax.set_ylim(top=80 * max_interval)
            ax.set_yticks([1e1, 1e2, 1e3, 1e4, 1e5])
        ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=8.5,
                  handlelength=.9, columnspacing=.45, handletextpad=.3,
                  borderaxespad=.12, labelspacing=.15)
        style_axis(ax, log_grid=True)
        ax.set_box_aspect(1)
        panels.append({"panel": label, "betas": betas, "xlim": list(ax.get_xlim()),
                       "ylim": list(ax.get_ylim()), "scale": ["log", "linear" if wide else "log"]})
    path_handles = [Line2D([], [], color=".47", linewidth=.8, label="Sample paths"),
                    Line2D([], [], color=".2", linewidth=1.7, label="Mean"),
                    Line2D([], [], color=CORRIDOR_COLOR, linestyle="--", label=r"$\pm\kappa t^{1-\beta}$")]
    risk_handles = [Line2D([], [], color=INK, linestyle=":", linewidth=1.8, label="Baseline"),
                    Patch(facecolor=".5", alpha=.2, label=interval_label()),
                    guide_handle(GUIDE_LEGEND_GRAY, "Theoretical rates")]
    for handles, center in ((path_handles, .255), (risk_handles, .752)):
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(center, .015), fontsize=8.5, handlelength=1.5,
                   columnspacing=.7, handletextpad=.4)
    fig.subplots_adjust(left=.105, right=.965, top=.79, bottom=.28, wspace=.42)
    # Leave extra space for the disagreement label between panels (b) and (c).
    position = axes[1].get_position()
    axes[1].set_position([position.x0 - .012, position.y0, position.width, position.height])
    caption = (
        "Truncated unsmoothed split conformal with alpha=0.1 and kappa=0.5. "
        "Panels (a,b) show the first 12 cumulative coverage-error paths and the "
        "mean over all 500 replications for beta=0.3 and 0.7, sampled every 100 "
        "rounds through 50000; dashed boundaries are +/-kappa*t^(1-beta). "
        "Panels (c,d) show expected cumulative disagreement with the population "
        "score oracle through T=250000 for beta=0.1,...,0.9, with the untruncated "
        "baseline and pointwise 95% Monte Carlo intervals over 500 replications. "
        "Panel (c) uses logarithmic x and linear y; panel (d) uses logarithmic "
        "axes. The theoretical rates have exponents 1/2 for the baseline and "
        "beta for the narrow-constraint curves; only their heights are fitted at "
        "the five prespecified horizons from 50000 through 250000."
    )
    return finish(fig, "simulation_conformal", output, checks, plotted,
                  [baseline / n for n in ("summary.csv", "paths.csv", "path_means.csv")],
                  guides, panels, caption=caption, generator=__file__)

if __name__ == "__main__":
    from figure._cli import main
    main(["conformal"])
