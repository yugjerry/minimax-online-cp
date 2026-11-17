#!/usr/bin/env python3
"""Run the paper experiments, with independent checks and resumable chunks.

All methods share observations within a replication. Replications are independent.
Run from the repository root: python -m simu.run --profile paper --workers 4.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = HERE / "_engine.cpp"
DEFAULT_OUTPUT = HERE / "output" / "paper"
METRICS = ("risk", "realized", "baseline_risk", "truncation", "D")


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def write_json(path, value):
    """Replace a record atomically so interrupted writes leave no partial marker."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_marker(path):
    """Treat missing, unreadable, or malformed cache records as cache misses."""
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Empty output: {path}")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        yield from csv.DictReader(handle)


def compile_engine(out):
    directory = out / "build"
    directory.mkdir(parents=True, exist_ok=True)
    executable, stamp = directory / "engine", directory / "compile.json"
    signature = {"source_sha256": digest(SOURCE), "compiler": os.environ.get("CXX", "c++"),
                 "flags": ["-O3", "-std=c++17", "-Wall", "-Wextra", "-pedantic"]}
    if not (executable.exists() and read_marker(stamp) == signature):
        subprocess.run([signature["compiler"], *signature["flags"], str(SOURCE),
                        "-o", str(executable)], check=True)
        write_json(stamp, signature)
    return executable


def chunk(executable, out, first, count, horizon, seed, hashes):
    directory = out / "chunks" / f"chunk_{first:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    completion = directory / "complete.json"
    parameters = dict(first_rep=first, replications=count, horizon=horizon, seed=seed,
                      source_sha256=hashes)
    filenames = ("checkpoints.csv", "paths.csv", "diagnostics.csv")
    previous = read_marker(completion)
    recorded = previous.get("output_sha256", {})
    if (previous.get("parameters") == parameters and isinstance(recorded, dict)
            and set(recorded) == set(filenames)
            and all((directory / name).exists() and digest(directory / name) == recorded[name]
                    for name in filenames)):
        return dict(first_rep=first, replications=count, cached=True)
    started = time.monotonic()
    with (directory / "run.log").open("w") as log:
        subprocess.run([str(executable), "--out", str(directory), "--first-rep", str(first),
                        "--reps", str(count), "--horizon", str(horizon), "--seed", str(seed)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    write_json(completion, dict(parameters=parameters, seconds=time.monotonic() - started,
                               output_sha256={name: digest(directory / name) for name in filenames}))
    return dict(first_rep=first, replications=count, cached=False)


def statistics(values, nonnegative=True):
    centered = values - values[0]
    mean = float(values[0] + centered.mean())
    se = float(centered.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    low = mean - 1.96 * se
    return dict(mean=mean, se=se, ci_low=max(0.0, low) if nonnegative else low,
                ci_high=mean + 1.96 * se)


def aggregate(out, starts, reps, horizon):
    groups = {}
    seen_diagnostics = set()
    cp_checks = aci_checks = 0
    max_cp = max_aci = 0.0
    path_groups = {}
    sample_paths, motivation_paths = [], []
    motivation_keys = set()
    for first in starts:
        directory = out / "chunks" / f"chunk_{first:04d}"
        for row in read_csv(directory / "checkpoints.csv"):
            key = row["study"], row["algorithm"], float(row["beta"]), int(row["t"])
            rep = int(row["rep"])
            if not 0 <= rep < reps:
                raise AssertionError("Replication outside requested range")
            if key not in groups:
                groups[key] = np.full((reps, len(METRICS)), np.nan)
            values = groups[key]
            if np.isfinite(values[rep, 0]):
                raise AssertionError("Duplicate checkpoint replication")
            values[rep] = [float(row[name]) for name in METRICS]
        for row in read_csv(directory / "diagnostics.csv"):
            rep = int(row["rep"])
            if rep in seen_diagnostics:
                raise AssertionError("Duplicate replication diagnostics")
            seen_diagnostics.add(rep)
            cp_checks += int(row["cp_rounds_checked"])
            aci_checks += int(row["aci_rounds_checked"])
            max_cp = max(max_cp, float(row["max_cp_violation"]))
            max_aci = max(max_aci, float(row["max_aci_violation"]))
        for row in read_csv(directory / "paths.csv"):
            rep, t, beta = int(row["rep"]), int(row["t"]), float(row["beta"])
            if row["study"] == "motivation":
                if rep != 0:
                    raise AssertionError("Motivation path must be the preselected first replication")
                if (beta, t) in motivation_keys:
                    raise AssertionError("Duplicate motivation path row")
                motivation_keys.add((beta, t))
                motivation_paths.append(dict(rep=rep, t=t, beta=beta, D=float(row["D"]),
                                             coverage=float(row["coverage"])))
            elif row["study"] == "baseline":
                key = beta, t
                if key not in path_groups:
                    path_groups[key] = np.full(reps, np.nan)
                values = path_groups[key]
                if not 0 <= rep < reps or np.isfinite(values[rep]):
                    raise AssertionError("Invalid or duplicate path replication")
                values[rep] = float(row["D"])
                if rep < 12 and beta in (.3, .7):
                    sample_paths.append(dict(rep=rep, t=t, beta=beta, D=float(row["D"])))
            else:
                raise AssertionError("Unexpected path study")
    if seen_diagnostics != set(range(reps)) or cp_checks != 12 * reps * horizon or aci_checks != 7 * reps * horizon:
        raise AssertionError("Missing every-round corridor checks")
    if max(max_cp, max_aci) > 1e-10:
        raise AssertionError("Coverage corridor violation")
    expected_states = ({("baseline", "cp", j / 10) for j in range(1, 10)}
                       | {("matched", a, b) for a in ("cp", "aci") for b in (.3, .5, .7)}
                       | {("motivation", "aci", b) for b in (1., .8, .6, .4)})
    times = sorted({key[3] for key in groups})
    prescribed = (set(range(100, 1001, 100)) | set(range(1200, 10001, 200))
                  | set(range(11000, 50001, 1000)) | set(range(60000, 250001, 10000)))
    expected_times = {t for t in prescribed if t <= horizon} | {horizon}
    if set(times) != expected_times or any(t % 100 for t in times) or set(groups) != {
            (*state, t) for state in expected_states for t in times}:
        raise AssertionError("Missing state/checkpoint or non-round checkpoint")
    expected_paths = {(b / 10, t) for b in range(1, 10)
                      for t in range(100, min(50000, horizon) + 1, 100)}
    motivation_times = set(range(1, min(1000, horizon) + 1)) | set(range(100, horizon + 1, 100))
    if set(path_groups) != expected_paths or motivation_keys != {
            (b, t) for b in (1., .8, .6, .4) for t in motivation_times}:
        raise AssertionError("Missing path or unexpected path time")
    summaries = {study: [] for study in ("baseline", "matched", "motivation")}
    comparisons, triangle_slack = [], []
    raw_baselines = {}
    for (study, algorithm, beta, t), values in sorted(groups.items()):
        if not np.isfinite(values).all():
            raise AssertionError("Missing or nonfinite replication")
        risk, realized, baseline, truncation, deviation = values.T
        if np.min(values[:, :4]) < -1e-10:
            raise AssertionError("Negative disagreement")
        if t in raw_baselines:
            np.testing.assert_array_equal(baseline, raw_baselines[t])
        else:
            raw_baselines[t] = baseline
        row = dict(beta=beta, t=t, replications=reps)
        if study == "baseline":
            for name, observed in (("baseline", baseline), ("truncation", truncation),
                                   ("total", risk), ("sum", baseline + truncation)):
                row.update({f"{name}_{k}": v for k, v in statistics(observed).items()})
            row["ratio"] = row["total_mean"] / row["sum_mean"]
        else:
            component = "R" if study == "motivation" else "risk"
            row.update({f"{component}_{k}": v for k, v in statistics(risk).items()})
            if study == "matched":
                row["algorithm"] = algorithm
        summaries[study].append(row)
        if algorithm == "cp":
            slack = baseline + truncation - risk
            triangle_slack.append(float(slack.min()))
            if slack.min() < -1e-7:
                raise AssertionError("Disagreement triangle inequality violated")
        if t == horizon:
            delta = statistics(realized - risk, nonnegative=False)
            comparisons.append(dict(study=study, algorithm=algorithm, beta=beta,
                conditional_risk_mean=float(risk.mean()), realized_mean=float(realized.mean()),
                paired_difference_mean=delta["mean"], paired_difference_se=delta["se"],
                standardized_difference=delta["mean"] / delta["se"] if delta["se"] else None))
    means = []
    for (beta, t), values in sorted(path_groups.items()):
        if not np.isfinite(values).all():
            raise AssertionError("Missing path replication")
        if beta in (.3, .7):
            means.append(dict(beta=beta, t=t, replications=reps, mean_D=float(values.mean())))
    for study, rows in summaries.items():
        (out / study).mkdir(exist_ok=True)
        write_csv(out / study / "summary.csv", rows)
    write_csv(out / "baseline/paths.csv", sorted(sample_paths, key=lambda r: (r["beta"], r["rep"], r["t"])))
    write_csv(out / "baseline/path_means.csv", means)
    write_csv(out / "motivation/paths.csv", sorted(motivation_paths, key=lambda r: (r["beta"], r["t"])))
    # Original replication rows are preserved in chunks; this compact cache also
    # retains the within-replication pairing used for every uncertainty band.
    np.savez_compressed(out / "replications.npz", **{
        f"{study}__{algorithm}__{beta:g}__{t}": values
        for (study, algorithm, beta, t), values in groups.items()})
    return dict(passed=True, replications=reps, horizon=horizon,
                checkpoint_times=times, checkpoint_groups=len(groups),
                cp_rounds_checked=cp_checks, aci_rounds_checked=aci_checks,
                total_corridor_checks=cp_checks + aci_checks,
                max_cp_violation=max_cp, max_aci_violation=max_aci,
                minimum_triangle_slack=min(triangle_slack),
                baseline_shared_exactly_across_all_studies=True,
                paired_sum_intervals=True, replication_cache_columns=METRICS,
                integrated_vs_realized=comparisons,
                monte_carlo_diagnostics_are_descriptive_not_selection_criteria=True)


def source_hashes():
    files = [SOURCE, HERE / "validate.py", HERE / "dgp.py", *sorted((ROOT / "methods").glob("*.py"))]
    return {str(path.relative_to(ROOT)): digest(path) for path in files}


def run_experiment(out: Path, reps=500, horizon=250000, seed=20261030,
                   workers=4, chunk_size=100) -> Path:
    """Validate the implementation, run missing chunks, and save paired summaries."""
    from simu.validate import validate

    if min(reps, horizon, workers, chunk_size) < 1 or horizon % 100:
        raise ValueError("Use positive dimensions and a horizon that is a multiple of 100")
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    executable = compile_engine(out)
    validation_dir = out / "validation"
    validation_dir.mkdir(exist_ok=True)
    write_json(validation_dir / "validation.json", validate(executable, validation_dir))
    hashes = source_hashes()
    starts = list(range(0, reps, chunk_size))
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(chunk, executable, out, first,
                   min(chunk_size, reps - first), horizon, seed, hashes)
                   for first in starts]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(f"Completed replications {result['first_rep']}.."
                  f"{result['first_rep'] + result['replications'] - 1}"
                  + (" (verified cache)" if result["cached"] else ""), flush=True)
    print("Aggregating all replications and checking invariants...", flush=True)
    audit = aggregate(out, starts, reps, horizon)
    write_json(out / "verification.json", audit)
    write_json(out / "manifest.json", dict(
        experiment="Unsmoothed split conformal and ACI; arXiv:2611.21542v1",
        horizon=horizon, replications=reps, seed=seed, chunk_size=chunk_size,
        alpha=.1, baseline_kappa=.5, matched_kappa=2, aci_c=1, aci_B=1, aci_q1=.9,
        baseline_beta=[j / 10 for j in range(1, 10)], matched_beta=[.3, .5, .7],
        motivation_step_exponents=[0., .2, .4, .6],
        split_cp="Unsmoothed: k=t-floor(t/10), kth of all t-1 past scores; +inf if k=t",
        prediction_auxiliary_randomness=False, forced_rounds_update_complete_history=True,
        rng="Independent SplitMix64 data streams per replication; common observations across studies",
        data="X~Uniform[0,1], epsilon~Uniform[-1,1], independent; Y=sin(2*pi*X)+(0.2+0.3*X)*epsilon",
        score="min(abs(Y-sin(2*pi*X))/(0.2+0.3*X),1)", score_law="Uniform[0,1]",
        fixed_oracle="score <= 0.9", risk_estimator="Cumulative exact conditional disagreement",
        truncation_estimator="Sum p_t on forced coverage and 1-p_t on forced misses; p_t=1-clip(q_t,0,1)",
        intervals="Pointwise mean +/-1.96 replication standard errors; paired sum of components",
        source_sha256={**hashes, "simu/run.py": digest(__file__)},
        seconds_this_invocation=time.monotonic() - started, python=sys.version,
        numpy=np.__version__, platform=platform.platform(),
        output_sha256={str(p.relative_to(out)): digest(p) for p in
            [out / "replications.npz", out / "verification.json", *sorted(out.glob("*/summary.csv")),
             out / "baseline/paths.csv", out / "baseline/path_means.csv", out / "motivation/paths.csv"]}))
    print(f"Saved verified numerical results to {out}", flush=True)
    return out


def ensure_data(out: Path | None = None, workers: int = 4) -> Path:
    """Return verified paper data, generating it if absent or out of date."""
    out = Path(out or DEFAULT_OUTPUT).resolve()
    manifest = out / "manifest.json"
    if manifest.exists():
        try:
            record = read_marker(manifest)
            expected = {**source_hashes(), "simu/run.py": digest(__file__)}
            outputs = record["output_sha256"]
            required = {"replications.npz", "verification.json", "baseline/summary.csv",
                        "matched/summary.csv", "motivation/summary.csv", "baseline/paths.csv",
                        "baseline/path_means.csv", "motivation/paths.csv"}
            if (record["replications"] == 500 and record["horizon"] == 250000
                    and record["seed"] == 20261030 and record["source_sha256"] == expected
                    and isinstance(outputs, dict) and set(outputs) == required
                    and all((out / name).is_file() and digest(out / name) == value
                            for name, value in outputs.items())
                    and read_marker(out / "verification.json")["passed"]):
                return out
        except (KeyError, ValueError, OSError):
            pass
    return run_experiment(out, workers=workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("paper", "smoke"), default="paper")
    parser.add_argument("--out", type=Path, help="Default: simu/output/PROFILE")
    parser.add_argument("--reps", type=int, help="Override the profile's replication count")
    parser.add_argument("--horizon", type=int, help="Override the profile's horizon (multiple of 100)")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()
    reps, horizon = (500, 250000) if args.profile == "paper" else (10, 1000)
    try:
        run_experiment(args.out or HERE / "output" / args.profile,
                       reps=args.reps if args.reps is not None else reps,
                       horizon=args.horizon if args.horizon is not None else horizon,
                       seed=args.seed, workers=args.workers, chunk_size=args.chunk_size)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
