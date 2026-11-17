#!/usr/bin/env python3
"""Independently check unsmoothed ranks, conditional losses, ACI, and replay.

The reference uses a sorted Python list, the literal integer rank inequality,
and interval integration for disagreement. A compiled fixture exercises the
engine's tree with ties, admissible error states, and full/empty prediction sets.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from methods import ACI, SplitConformal, TruncatedConformal
from simu.dgp import draw


MASK = (1 << 64) - 1
CONFIGS = [("baseline", "cp", b, .5) for b in range(1, 10)]
CONFIGS += [("matched", algorithm, b, 2.) for b in (3, 5, 7) for algorithm in ("cp", "aci")]
CONFIGS += [("motivation", "aci", b, 2.) for b in (10, 8, 6, 4)]


class RNG:
    def __init__(self, state: int):
        self.state = state & MASK

    def integer(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def uniform(self) -> float:
        return (self.integer() >> 11) * 2.0**-53


def close(observed: float, expected: float, description: str) -> None:
    if observed == expected:
        return
    if not math.isclose(observed, expected, rel_tol=3e-12, abs_tol=3e-11):
        raise AssertionError(f"{description}: {observed} != {expected}")


def xor_mass(a: float, b: float) -> float:
    """Integrate differing threshold membership against Uniform[0,1]."""
    cuts = sorted({0., 1., min(1., max(0., a)), min(1., max(0., b))})
    return sum(right-left for left, right in zip(cuts, cuts[1:])
               if (((left+right)/2 <= a) != ((left+right)/2 <= b)))


def checkpoints(horizon: int) -> set[int]:
    grid = (set(range(100, 1001, 100)) | set(range(1200, 10001, 200))
            | set(range(11000, 50001, 1000)) | set(range(60000, 250001, 10000)))
    return {t for t in grid if t <= horizon} | {horizon}


def run(engine: Path, out: Path, first: int, reps: int, horizon: int,
        seed: int, trace: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    command = [str(engine), "--out", str(out), "--first-rep", str(first), "--reps", str(reps),
               "--horizon", str(horizon), "--seed", str(seed)]
    if trace:
        command.append("--trace")
    subprocess.run(command, check=True, capture_output=True, text=True)


def read(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def key(row: dict) -> tuple:
    return (row["study"], row["algorithm"], int(row["rep"]), int(row["t"]),
            round(10*float(row["beta"])))


def index(rows: list[dict]) -> dict:
    result = {key(row): row for row in rows}
    if len(result) != len(rows):
        raise AssertionError("Duplicated output key")
    return result


def validate_rank_fixture(out: Path) -> dict:
    """Check ranks, adversarial states, and losses beyond the random data paths."""
    source = out / "rank_fixture.cpp"
    source.write_text(r'''#define main simulation_main
#include "_engine.cpp"
#undef main
int main() {
    I64 comparisons=0;
    for (int t:{1,2,9,10,11,19,20,21,99,100,101,500}) {
        for (int mode=0;mode<3;++mode) {
            std::vector<double> values;
            History a(t,123), b(t,987654321);
            for (int j=0;j<t-1;++j) {
                double value=mode==0 ? .5 : mode==1 ? (j%5)/4. : (j+.5)/t;
                values.push_back(value); a.insert(value);
            }
            for (auto it=values.rbegin();it!=values.rend();++it) b.insert(*it);
            LD qa=threshold(a,t), qb=threshold(b,t);
            if (qa!=qb) throw std::runtime_error("Tree priorities/insertion order changed the cutoff");
            std::vector<double> probes={-.1,0.,.1,.5,.9,1.,1.1};
            for (double value:values) {
                probes.push_back(value);
                probes.push_back(std::nextafter(value,-std::numeric_limits<double>::infinity()));
                probes.push_back(std::nextafter(value,std::numeric_limits<double>::infinity()));
            }
            for (double candidate:probes) {
                I64 rank=1;
                for (double old:values) rank+=old>=candidate;
                bool direct_error=10*rank<=t;
                if (direct_error!=(LD(candidate)>qa))
                    throw std::runtime_error("Literal paper rank differs from tree threshold");
                ++comparisons;
            }
        }
    }
    if (branch(0,1,.5L)!=1 || branch(0,5,.5L)!=0 || branch(0,6,.5L)!=-1)
        throw std::runtime_error("Strict corridor boundary test failed");
    I64 state_checks=0, integral_checks=0;
    // Test every admissible previous error count, regardless of which data
    // produced it, and both possible baseline outcomes. Check the resulting
    // error sum directly rather than duplicating the branch conditions.
    for (int b=0;b<=10;++b) for (LD kappa:{.5L,1.L,2.L}) {
        for (int t=1;t<=500;++t) {
            LD bound=kappa*std::pow(LD(t),1-b/10.L);
            LD previous=t==1 ? 0.L : kappa*std::pow(LD(t-1),1-b/10.L);
            for (I64 errors=0;errors<t;++errors) {
                if (std::fabs((10*errors-(t-1))/10.L)>previous) continue;
                int action=branch(errors,t,bound);
                if (b==10 && kappa==.5L) {
                    // The constant half-width admits an exact integer check.
                    int exact=10*errors+10-t>5 ? 1 : 10*errors-t< -5 ? -1 : 0;
                    if (action!=exact) throw std::runtime_error("Exact boundary mismatch");
                }
                for (int baseline_error:{0,1}) {
                    int error=action==1 ? 0 : action==-1 ? 1 : baseline_error;
                    if (std::fabs((10*(errors+error)-t)/10.L)>bound+1e-14L)
                        throw std::runtime_error("Adversarial corridor violation");
                    ++state_checks;
                }
            }
        }
    }
    // Integrate the membership XOR over intervals with constant membership.
    LD infinity=std::numeric_limits<LD>::infinity();
    std::vector<LD> thresholds={-infinity,-.2L,0.L,.05L,.3L,.9L,1.L,1.3L,infinity};
    for (LD a:thresholds) for (LD b:thresholds) {
        std::vector<LD> cuts={0.L,1.L};
        if (a>0 && a<1) cuts.push_back(a);
        if (b>0 && b<1) cuts.push_back(b);
        std::sort(cuts.begin(),cuts.end());
        LD mass=0;
        for (std::size_t i=1;i<cuts.size();++i) {
            LD midpoint=(cuts[i-1]+cuts[i])/2;
            if ((midpoint<=a)!=(midpoint<=b)) mass+=cuts[i]-cuts[i-1];
        }
        if (std::fabs(mass-std::fabs(cdf(a)-cdf(b)))>1e-14L)
            throw std::runtime_error("Conditional disagreement integral mismatch");
        ++integral_checks;
    }
    std::cout<<comparisons<<' '<<state_checks<<' '<<integral_checks<<'\n';
}
''')
    executable = out / "rank_fixture"
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-Wall", "-Wextra",
                    "-pedantic", "-I", str(Path(__file__).resolve().parent), str(source),
                    "-o", str(executable)], check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    rank_checks, state_checks, integral_checks = map(int, result.stdout.split())
    return {"literal_rank_comparisons_with_ties": rank_checks,
            "admissible_state_outcome_checks": state_checks,
            "threshold_pair_integral_checks": integral_checks,
            "integer_rank_boundaries_tested": [9, 10, 11, 19, 20, 21, 99, 100, 101],
            "threshold_independent_of_tree_priorities": True,
            "strict_corridor_boundaries_tested": True}


def validate(engine: Path, out: Path) -> dict:
    fixtures = validate_rank_fixture(out)
    seed, horizon, reps = 997313, 1200, 4
    together = out / "together"
    run(engine, together, 0, reps, horizon, seed, trace=True)
    traces = index(read(together / "trace.csv"))
    records = index(read(together / "checkpoints.csv"))
    paths = index(read(together / "paths.csv"))
    if len(traces) != reps*horizon*len(CONFIGS):
        raise AssertionError("Incomplete trace")
    if len(records) != reps*len(checkpoints(horizon))*len(CONFIGS):
        raise AssertionError("Incomplete checkpoint grid")
    compared = path_checks = forced_plus = forced_minus = 0
    for rep in range(reps):
        split = RNG(seed ^ ((rep * 0xD1B54A32D192ED03) & MASK))
        data = RNG(split.integer())
        past: list[float] = []
        states = {config: {"q": .9, "errors": 0, "risk": 0., "realized": 0,
                           "truncation": 0., "nplus": 0, "nminus": 0}
                  for config in CONFIGS}
        baseline = 0.
        cp_reference = SplitConformal()
        methods = {config: (TruncatedConformal(beta=config[2]/10, kappa=config[3])
                            if config[1] == "cp" else ACI(beta=config[2]/10))
                   for config in CONFIGS}
        for t in range(1, horizon+1):
            k = t-t//10
            cut = math.inf if k > len(past) else past[k-1]
            close(cp_reference.threshold(), cut, "Python split-conformal threshold")
            p = 1-min(1., max(0., cut))
            baseline += xor_mass(cut, .9)
            _, _, score = draw(data)
            # Exact rational alpha=1/10; this test uses no quantile formula.
            raw_error = 10*(1+sum(old >= score for old in past)) <= t
            if raw_error != (score > cut):
                raise AssertionError("Literal rank and sorted-list prediction differ")
            oracle_error = score > .9
            for config, state in states.items():
                study, algorithm, b, kappa = config
                row_key = (study, algorithm, rep, t, b)
                row = traces[row_key]
                power = t**(1-b/10)
                bound = kappa*power
                action = 0
                if algorithm == "cp":
                    cover_candidate = (10*state["errors"]-t)/10
                    miss_candidate = cover_candidate+1
                    action = 1 if miss_candidate > bound else -1 if cover_candidate < -bound else 0
                    effective = math.inf if action == 1 else -math.inf if action == -1 else cut
                else:
                    effective = state["q"]
                method = methods[config]
                close(method.threshold(), effective, "Python method threshold")
                error = score > effective
                if method.update(score) != error:
                    raise AssertionError("Python method and independent reference errors differ")
                loss = xor_mass(effective, .9)
                change = xor_mass(effective, cut) if algorithm == "cp" else 0.
                state["errors"] += error
                state["risk"] += loss
                state["realized"] += error != oracle_error
                state["truncation"] += change
                state["nplus"] += action == 1
                state["nminus"] += action == -1
                deviation = (10*state["errors"]-t)/10
                expected = {"kappa": kappa, "score": score, "baseline_threshold": cut,
                            "baseline_p": p, "threshold": effective, "branch": action,
                            "error": int(error), "oracle_error": int(oracle_error),
                            "risk_increment": loss, "truncation_increment": change,
                            "D": deviation, "history_size": t-1}
                for name, value in expected.items():
                    close(float(row[name]), value, f"{row_key}, {name}")
                    compared += 1
                if abs(deviation) > bound+1e-10:
                    raise AssertionError("Independent every-round corridor check failed")
                if t in checkpoints(horizon):
                    checkpoint = records[row_key]
                    for name in ("risk", "realized", "truncation", "nplus", "nminus"):
                        close(float(checkpoint[name]), state[name], f"{row_key}, cumulative {name}")
                    close(float(checkpoint["D"]), deviation, "Checkpoint deviation")
                    close(float(checkpoint["baseline_risk"]), baseline, "Complete baseline history")
                save_path = ((study == "baseline" and t%100 == 0 and t <= 50000)
                             or (study == "motivation" and rep == 0 and (t <= 1000 or t%100 == 0)))
                if save_path:
                    saved = paths[row_key]
                    close(float(saved["D"]), deviation, "Path deviation")
                    if study == "motivation":
                        close(float(saved["coverage"]), min(1., max(0., effective)), "Pre-update coverage")
                    elif saved["coverage"] != "":
                        raise AssertionError("Baseline path coverage should be blank")
                    path_checks += 1
                elif row_key in paths:
                    raise AssertionError("Unexpected path row")
                if algorithm == "aci":
                    state["q"] += (error-.1)/power
                forced_plus += action == 1
                forced_minus += action == -1
            # Insert every observed score even when several algorithms were forced.
            if cp_reference.update(score) != raw_error:
                raise AssertionError("Python split-conformal update differs from literal rank")
            bisect.insort(past, score)
    if path_checks != len(paths):
        raise AssertionError("Unvalidated path rows")
    if not forced_plus or not forced_minus:
        raise AssertionError("Both forcing branches must be exercised")
    for row in read(together / "diagnostics.csv"):
        if int(row["cp_rounds_checked"]) != 12*horizon or int(row["aci_rounds_checked"]) != 7*horizon:
            raise AssertionError("Wrong diagnostic round counts")
        if float(row["max_cp_violation"]) > 1e-10 or float(row["max_aci_violation"]) > 1e-10:
            raise AssertionError("Engine reported a corridor violation")
    # Chunking must preserve all numerical records, including rep0-only paths.
    names = ("checkpoints.csv", "paths.csv", "diagnostics.csv")
    split_records = {name: [] for name in names}
    for first in (0, 2):
        directory = out / f"split_{first}"
        run(engine, directory, first, 2, horizon, seed)
        for name in names:
            split_records[name].extend(read(directory / name))
    for name in names:
        if split_records[name] != read(together / name):
            raise AssertionError(f"Chunk partition changed {name}")
    replay = out / "replay"
    run(engine, replay, 0, reps, horizon, seed)
    for name in names:
        if (replay / name).read_bytes() != (together / name).read_bytes():
            raise AssertionError(f"Deterministic replay failed: {name}")
    short = out / "short"
    run(engine, short, 7, 1, 17, seed)
    if {int(row["t"]) for row in read(short / "checkpoints.csv")} != {17}:
        raise AssertionError("Small horizon must still be checkpointed")
    if read(short / "paths.csv"):
        raise AssertionError("Unexpected sub-100 paths outside rep0")
    return {"passed": True, "horizon": horizon, "replications": reps,
            "trace_scalar_comparisons": compared, "path_rows_checked": path_checks,
            "forced_coverage_rounds_exercised": forced_plus,
            "forced_miscoverage_rounds_exercised": forced_minus,
            "independent_direct_rank_and_sorted_list": True,
            "conditional_disagreement_by_interval_integration": True,
            "independent_aci_recursion": True,
            "public_python_methods_match_engine": True,
            "separate_dgp_matches_engine_scores": True,
            "all_rounds_corridor_checks": reps*horizon*len(CONFIGS),
            "complete_baseline_history_on_forced_rounds": True,
            "deterministic_replay": True, "chunk_partition_invariance": True,
            "small_horizon_supported": True, **fixtures,
            "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
            "engine_source_sha256": hashlib.sha256(Path(__file__).with_name("_engine.cpp").read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = validate(args.engine.resolve(), args.out.resolve())
    (args.out / "validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print("Independent ordinary-rank, interval-loss, ACI, corridor, path, replay, and chunk checks passed.")


if __name__ == "__main__":
    main()
