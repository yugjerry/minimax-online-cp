# Reproducing the paper

The default commands reproduce the four figures in [arXiv:2611.21542v1](https://arxiv.org/abs/2611.21542v1). All data are simulated; no downloads or fitted models are needed.

## Settings

| Setting | Value |
| --- | --- |
| Replications | 500 |
| Rounds per replication | 250,000 |
| Target miscoverage | 0.1 |
| Random seed | 20261030 |
| Truncation study | $\kappa=0.5$, $\beta=0.1,0.2,\ldots,0.9$ |
| Matched comparison | $\kappa=2$, $\beta=0.3,0.5,0.7$ |
| ACI initialization and steps | $q_1=0.9$, $\eta_t=t^{-(1-\beta)}$ |
| ACI motivation | $\beta=1,0.8,0.6,0.4$ |
| Chunk size | 100 replications |

The smoke profile uses 10 replications and 1,000 rounds in `simu/output/smoke/`. The paper profile writes to `simu/output/paper/`. Figure scripts require the full paper settings.

## Data and methods

`simu/dgp.py` states the regression model, computes the score from each generated response, and provides the score CDF for evaluation. Its C++ counterpart in `simu/_engine.cpp` preserves the same random draws. SplitMix64 gives each replication its own data stream, so worker count and chunk boundaries do not change the observations. Random tree priorities affect only the internal order-statistic tree, not predictions.

At round $t$, split conformal prediction takes the $k$th smallest of all $t-1$ past scores, where $k=t-\lfloor t/10\rfloor$. If $k=t$, the threshold is $+\infty$. Membership includes equality with the threshold; there is no smoothing randomness. The first nine prediction sets are therefore the whole response space.

Truncation uses the errors of the truncated prediction sets to decide whether the next set must be full or empty. The baseline continues to receive every observation. ACI updates the threshold after observing the response and does not clip it to the score support.

The three Python classes have the same interface:

```python
from methods import TruncatedConformal

method = TruncatedConformal(alpha=0.1, beta=0.7, kappa=0.5)
q = method.threshold()       # Construct {y: score(x, y) <= q} before observing y.
error = method.update(0.8)   # Supply the observed score after y is revealed.
```

`SplitConformal` uses a sorted list for clarity. The full simulation uses an order-statistic tree to handle the paper's horizon efficiently. An independent validator checks the engine against both literal rank calculations and the public Python methods before simulation.

## Alignment with the paper

| Method | Paper formulation | Implementation checks |
| --- | --- | --- |
| Split conformal | Section 1.1, the displayed rank definition | Uses only the previous $t-1$ scores; the strict rank inequality gives $k=t-\lfloor\alpha t\rfloor$; ties are included. |
| Truncated split conformal | Section 2.2, the three-branch truncation definition | Tests $S_{t-1}+1-\alpha>\kappa t^{1-\beta}$ and $S_{t-1}-\alpha<-\kappa t^{1-\beta}$; uses its own miscoverage count and retains every baseline observation. |
| Threshold ACI | Section 3, the threshold update | Tests miscoverage against $q_t$, then adds $ct^{-(1-\beta)}(\mathrm{err}_t-\alpha)$, starting at $t=1$; the threshold is not projected. |

The independent validator in `simu/validate.py` checks 2,137,700 admissible previous-state/next-outcome cases, 8,181 tied-rank comparisons, and 81 independent disagreement integrals before comparing complete Python and C++ trajectories. These checks run before the simulation.

Python interprets the decimal values of $\alpha$ and $\kappa$ exactly in conformal rank and truncation comparisons. This prevents conversion to floating point from turning a strict inequality into equality. Fractional powers in the boundary remain numerical approximations.

ACI compares each score with its stored floating-point threshold. Accumulated roundoff can change a decision at an exact tie with the ideal real-arithmetic threshold. For example, with $\alpha=0.1$, $\beta=1$, $c=1$, and $q_1=0.9$, nine updates on score 1 give a stored threshold just below 1 instead of exactly 1. No comparison tolerance is added, since that would change the prescribed membership test.

## Disagreement and uncertainty

For a threshold $q$, the known score law gives miscoverage probability $p=1-\operatorname{clip}(q,0,1)$. Its conditional disagreement with the population oracle is $|p-0.1|$.

The cost of truncation relative to the baseline is $p$ on forced coverage, $1-p$ on forced miscoverage, and zero otherwise, where $p$ is the baseline's current miscoverage probability. This differs from disagreement with the population oracle, which is $0.1$ on forced coverage and $0.9$ on forced miscoverage.

Each replication sums conditional disagreement probabilities. Figures average these totals across replications and show pointwise mean ± 1.96 Monte Carlo standard errors. Nonnegative intervals are clipped at zero. The baseline-plus-truncation interval is computed from paired sums within replications.

Theoretical guide exponents are specified before plotting. Only the height is normalized, using the geometric mean of estimate divided by the theoretical power at 50,000, 100,000, 150,000, 200,000, and 250,000 rounds. These guides are not fitted rate estimates.

Estimate curves include every saved checkpoint in the prescribed time window. Logarithmic panels require positive estimates and interval endpoints and raise an error otherwise. Linear panels retain zero values.

## Commands and outputs

```sh
# Use the dependency versions tested for this release.
python -m pip install -r requirements-lock.txt

# Run the simulation and all figures.
python -m simu.run --profile paper --workers 4
python -m figure

# Render existing results without running the simulation.
python -m figure --plots-only

# Use a separate destination for an experiment.
python -m simu.run --profile paper --out /tmp/online-cp-data
python -m figure --input-dir /tmp/online-cp-data --output-dir /tmp/online-cp-figures

# Render with built-in fonts.
python -m figure --plots-only --no-tex
```

`CXX` selects the C++17 compiler. The default is `c++`. LaTeX rendering requires the paper's `newpxtext` and `newpxmath` packages and the supporting tools checked by `figure/_style.py`. Without them, the figures use built-in fonts; numerical values are unchanged.

Simulation outputs include raw checkpoint rows in `chunks/`, paired replication values in `replications.npz`, and summaries in `baseline/`, `matched/`, and `motivation/`. `manifest.json` records settings, software versions, and source/output hashes. `verification.json` records every-round constraint checks and the disagreement triangle inequality. The independent audit is in `validation/validation.json`.

Figure outputs include four PDF/SVG/PNG exports, plotted CSVs, captions, and numerical/layout audits. The names match the paper: `aci_motivation_R`, `simulation_conformal`, `simulation_decomposition`, and `simulation_aci_comparison`.

Re-running checks the source, settings, and output hashes before reusing completed chunks. The full results include 2,375,000,000 method-round checks of the long-run coverage bounds. Replay and chunk-partition checks confirm that changing the execution schedule does not select different observations.

Floating-point results can differ slightly across compilers and platforms. The same seed and settings reproduce the experiment; byte-identical plots also depend on the plotting and font versions.
