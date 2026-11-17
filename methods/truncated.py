"""Truncate split conformal sets to enforce the long-run coverage constraint.

Keep the baseline set unless a possible next error could leave
[-kappa*t**(1-beta), kappa*t**(1-beta)]. Use the whole response space when
forcing coverage and the empty set when forcing miscoverage. Every observed
score enters the baseline history, including scores from forced rounds.
"""
from fractions import Fraction
from math import inf, isfinite

from .split_conformal import SplitConformal


class TruncatedConformal:
    def __init__(self, alpha: float = 0.1, beta: float = 0.7, kappa: float = 0.5):
        if not 0 <= beta <= 1 or not isfinite(kappa) or kappa < 0.5:
            raise ValueError("Require beta in [0, 1] and finite kappa >= 0.5")
        self.baseline = SplitConformal(alpha)
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        self.errors = 0

    def branch(self) -> int:
        """Return 1 for forced coverage, -1 for a forced miss, or 0."""
        t = len(self.baseline.history) + 1
        # Keep decimal parameters and centering exact; fractional powers remain
        # floating-point approximations to the paper's real-valued boundary.
        power = t if self.beta == 0 else t ** (1 - self.beta)
        bound = Fraction(str(self.kappa)) * Fraction(str(power))
        centered = self.errors - t * self.baseline._alpha
        if centered + 1 > bound:
            return 1
        if centered < -bound:
            return -1
        return 0

    def threshold(self) -> float:
        action = self.branch()
        return inf if action == 1 else -inf if action == -1 else self.baseline.threshold()

    def update(self, score: float) -> bool:
        """Observe a score and return the truncated set's miscoverage."""
        if not isfinite(score):
            raise ValueError("score must be finite")
        error = score > self.threshold()
        self.baseline.update(score)
        self.errors += error
        return error
