"""Threshold ACI with step size eta_t = c*t**(-(1-beta)).

After predicting {y: score(x, y) <= q_t}, update
q_{t+1} = q_t + eta_t*(error_t-alpha). Thresholds are not clipped: a
threshold outside the score support gives full or empty prediction sets.

Membership compares to the stored floating-point threshold; accumulated
roundoff can change a tie with the ideal real-arithmetic threshold. No
comparison tolerance is added. The paper's coverage certificate assumes
scores and q1 in [0, B], giving kappa = (B+c)/c.
"""
from math import isfinite


class ACI:
    def __init__(self, alpha: float = 0.1, beta: float = 0.7,
                 c: float = 1.0, q1: float = 0.9):
        if not 0 < alpha < 1 or not 0 <= beta <= 1:
            raise ValueError("Require alpha in (0, 1) and beta in [0, 1]")
        if not isfinite(c) or c <= 0 or not isfinite(q1):
            raise ValueError("Require finite c > 0 and finite q1")
        self.alpha, self.beta, self.c = alpha, beta, c
        self.q, self.t = q1, 1

    def threshold(self) -> float:
        """Return the threshold before observing the next response."""
        return self.q

    def update(self, score: float) -> bool:
        """Observe a finite score, update the threshold, and return miscoverage."""
        if not isfinite(score):
            raise ValueError("score must be finite")
        error = score > self.q
        self.q += self.c * self.t ** (self.beta - 1) * (error - self.alpha)
        self.t += 1
        return error
