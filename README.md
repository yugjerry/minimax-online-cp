# The cost of enforcing long-run coverage

Code for **The cost of enforcing long-run coverage: a phase transition in online conformal prediction**, by Yu Gui, Rina Foygel Barber, and Cong Ma.

[Paper on arXiv](https://arxiv.org/abs/2611.21542)

## Motivation

Split conformal prediction provides valid prediction sets when the data are exchangeable. For i.i.d. data with a nonatomic score distribution, its instantaneous training-conditional coverage approaches the target level. When the data distribution changes, online methods can instead guarantee long-run empirical coverage. How much does this protection cost when the data are in fact i.i.d.?

We study algorithms that satisfy

$$
\left|\frac{1}{T}\sum_{t=1}^{T}(\mathrm{err}_t-\alpha)\right|
\leq \kappa T^{-\beta}
\qquad\text{for every }T\geq 1
$$

on every data sequence, where $\mathrm{err}_t$ indicates miscoverage. Larger $\beta$ requires tighter control of long-run empirical coverage.

## Main result

For a fixed score with a nonatomic distribution under i.i.d. sampling, we compare prediction sets with oracles whose coverage given the past is exactly $1-\alpha$. Disagreement counts responses that belong to exactly one of the two prediction sets. The algorithm does not know the data distribution.

Over the class of i.i.d. laws with a nonatomic score, the minimax expected cumulative disagreement with the closest exact oracle has order

$$
T^{\max\{1/2,\beta\}}=
\begin{cases}
\sqrt{T}, & 0\leq\beta\leq 1/2,\\
T^\beta, & 1/2<\beta\leq 1.
\end{cases}
$$

Here $\alpha\in(0,1)$ and $\kappa\geq 1/2$. The constraint preserves the unconstrained rate for $\beta\leq 1/2$ and raises it for $\beta>1/2$.

**Truncated split conformal prediction** attains this rate. It keeps the baseline prediction set whenever both possible outcomes respect the constraint. Otherwise, it outputs the full response space or the empty set to force coverage or miscoverage.

For comparison, consider threshold ACI with a fixed score in $[0,B]$, an initial threshold $q_1\in[0,B]$, and steps $\eta_t=ct^{-(1-\beta)}$, where $c>0$. It satisfies the constraint with $\kappa=(B+c)/c$. Under i.i.d. sampling, if the score density is bounded above and away from zero throughout $[0,B]$, its expected disagreement has order $T^{(1+\beta)/2}$ for $0<\beta<1$. This exceeds the minimax rate under the same guarantee.

## Repository

```text
methods/                 One Python implementation per method
  split_conformal.py     Split conformal prediction
  truncated.py           Truncation of a baseline prediction set
  aci.py                 Quantile-threshold ACI
simu/
  dgp.py                 Data-generating process and score
  run.py                 Simulation, summaries, and numerical checks
  validate.py            Independent checks of the simulation engine
  output/                Generated simulation data
figure/
  figure1_aci.py          ACI motivation
  figure2_conformal.py    Coverage paths and oracle disagreement
  figure3_decomposition.py  The cost of truncation
  figure4_comparison.py   ACI and truncated split conformal prediction
  output/                Generated figures and plotted values
```

The Python methods expose the algorithms directly. The simulation uses a C++ engine for the full experiment and checks its predictions against the Python implementations.

## Reproduce

Requirements: Python 3.10 or later and a C++17 compiler (`c++`, or set `CXX`). From this directory:

```sh
python -m pip install -r requirements.txt
python -m simu.run --profile smoke
python -m simu.run --profile paper --workers 4
python -m figure
```

The smoke run checks the implementation on a smaller experiment. The paper run uses 500 independent replications and 250,000 rounds. Completed simulation chunks are reused after their parameters and hashes have been checked.

Each figure also has an end-to-end command. It generates the paper data if needed, then writes PDF, SVG, PNG, and plotted values to `figure/output/`:

```sh
python -m figure.figure1_aci
python -m figure.figure2_conformal
python -m figure.figure3_decomposition
python -m figure.figure4_comparison
```

LaTeX is optional. With the paper's LaTeX fonts installed, the plots use `newpxtext` and `newpxmath`; otherwise they use Matplotlib's built-in text rendering. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for parameters, output files, and checks.

## Numerical experiments

We use the heteroscedastic regression model

$$
X_t\sim\mathrm{Unif}[0,1],\qquad
Y_t=\sin(2\pi X_t)+(0.2+0.3X_t)\varepsilon_t,\qquad
\varepsilon_t\sim\mathrm{Unif}[-1,1],
$$

independently over time, with $X_t$ and $\varepsilon_t$ independent. The fixed score is the clipped absolute standardized residual, which is uniform on $[0,1]$. We set $\alpha=0.1$ and compare with the population prediction set having score threshold $0.9$.

| Figure | Comparison |
| --- | --- |
| 1 | ACI with constant and decaying steps: cumulative coverage deviation, instantaneous training-conditional coverage, and disagreement per round. |
| 2 | Truncated split conformal prediction with $\kappa=0.5$: coverage paths and cumulative oracle disagreement across $\beta$. |
| 3 | Baseline disagreement, truncation cost, and total disagreement for $\beta=0.3,0.5,0.7$. |
| 4 | ACI and truncated split conformal prediction under the same $(\beta,2)$ guarantee. |

The known score distribution is used only to evaluate disagreement. It does not choose the conformal thresholds or update ACI. All methods receive the same observations within each replication, and every observation enters the conformal history, including forced rounds.

The experiments illustrate the phase transition and the greater disagreement of ACI under the matched constraint. They evaluate one data distribution; the minimax statement is proved in the paper. Bands are pointwise 95% Monte Carlo intervals. Theoretical guide exponents are fixed in advance.

## Citation

```bibtex
@article{gui2026cost,
  title = {The cost of enforcing long-run coverage: a phase transition in online conformal prediction},
  author = {Gui, Yu and Barber, Rina Foygel and Ma, Cong},
  journal = {arXiv preprint arXiv:2611.21542},
  year = {2026}
}
```
