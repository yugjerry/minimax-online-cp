"""Unsmoothed split conformal prediction using every past score.

At round t, include a candidate score s when
    (1 + number of past scores >= s) / t > alpha.
The equivalent threshold is the (t - floor(alpha*t))-th smallest past score,
or infinity when that order statistic exceeds the history size. Ties are
included. The sorted list keeps this reference implementation easy to inspect;
the simulation uses an independently checked order-statistic tree.
"""
from bisect import insort
from fractions import Fraction
from math import inf, isfinite


class SplitConformal:
    def __init__(self, alpha: float = 0.1):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        self.alpha = alpha
        self._alpha = Fraction(str(alpha))  # Preserve decimal rank boundaries.
        self.history: list[float] = []

    def threshold(self) -> float:
        """Return q for the next prediction set {y: score(x, y) <= q}."""
        t = len(self.history) + 1
        k = t - (self._alpha.numerator * t // self._alpha.denominator)
        return inf if k > len(self.history) else self.history[k - 1]

    def update(self, score: float) -> bool:
        """Observe a finite score, add it to history, and return miscoverage."""
        if not isfinite(score):
            raise ValueError("score must be finite")
        error = score > self.threshold()
        insort(self.history, score)
        return error
