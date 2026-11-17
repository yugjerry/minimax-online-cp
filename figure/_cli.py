"""Run the shared simulation, then render selected paper figures."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = {
    "motivation": ("figure1_aci", "motivation_figure"),
    "conformal": ("figure2_conformal", "conformal_figure"),
    "decomposition": ("figure3_decomposition", "decomposition_figure"),
    "aci": ("figure4_comparison", "aci_figure"),
}


def main(selected=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "simu/output/paper")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figure/output")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plots-only", action="store_true",
                        help="Render existing paper summaries without running the simulation")
    parser.add_argument("--no-tex", action="store_true",
                        help="Use Matplotlib's built-in fonts even if LaTeX is available")
    if selected is None:
        parser.add_argument("--figure", choices=FIGURES, action="append")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    data, output = args.input_dir.resolve(), args.output_dir.resolve()
    if not args.plots_only:
        from simu.run import ensure_data
        data = ensure_data(data, workers=args.workers)
    output.mkdir(parents=True, exist_ok=True)

    from ._common import (
        GUIDE_HORIZONS, GUIDE_POLICY, GUIDE_RENDERING, GUIDE_RIBBON,
        HORIZON, REPLICATIONS, configure_simulation_style,
    )
    from ._style import configure_style
    configure_style(force_usetex=False if args.no_tex else None)
    configure_simulation_style()
    names = selected or args.figure or list(FIGURES)
    audits = []
    for name in names:
        module, function = FIGURES[name]
        generator = getattr(importlib.import_module(f"figure.{module}"), function)
        audits.append(generator(data, output))

    # Individual entry points preserve records for figures already exported here.
    records = {}
    for path in sorted(output.glob("*_manifest.json")):
        if path.name == "figures_manifest.json":
            continue
        record = json.loads(path.read_text())
        if "caption_file" in record:
            records[record["caption_file"]] = record
    manifest = {
        "paper": "https://arxiv.org/abs/2611.21542v1",
        "figures": list(records.values()),
        "horizon": HORIZON, "replications": REPLICATIONS,
        "baseline": "unsmoothed split conformal",
        "guide_policy": GUIDE_POLICY, "guide_rendering": GUIDE_RENDERING,
        "guide_ribbon": GUIDE_RIBBON, "guide_horizons": GUIDE_HORIZONS.tolist(),
        "all_array_checks_passed": all(
            a["every_plotted_array_matches_source"] for a in records.values()),
    }
    (output / "figures_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    captions = "# Figure captions\n\n" + "\n\n".join(
        f"## {Path(key).stem.removesuffix('_caption')}\n\n{record['caption']}"
        for key, record in records.items())
    (output / "captions.md").write_text(captions + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "figures": [a["files"] for a in audits],
                      "all_array_checks_passed": manifest["all_array_checks_passed"]}, indent=2))


if __name__ == "__main__":
    main()
