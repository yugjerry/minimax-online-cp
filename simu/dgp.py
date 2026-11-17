"""The data-generating process in the paper's numerical experiments.

Independently over time, X ~ Uniform[0,1], epsilon ~ Uniform[-1,1],
Y = sin(2*pi*X) + (0.2+0.3*X)*epsilon, with X independent of epsilon.
The fixed clipped standardized-residual score is Uniform[0,1]. Its known
CDF is used only for evaluation, never to fit conformal thresholds or ACI.
"""
import math


def mean(x: float) -> float:
    return math.sin(2 * math.pi * x)


def scale(x: float) -> float:
    return 0.2 + 0.3 * x


def score(x: float, y: float) -> float:
    return min(abs(y - mean(x)) / scale(x), 1.0)


def draw(rng) -> tuple[float, float, float]:
    """Draw (X, Y, score) from an object with a uniform() method.

    Magnitude and sign use separate draws, preserving the paper's random
    streams. Their product is uniform on [-1,1].
    """
    x, magnitude = rng.uniform(), rng.uniform()
    sign = -1.0 if rng.uniform() < 0.5 else 1.0
    y = mean(x) + scale(x) * sign * magnitude
    return x, y, score(x, y)


def score_cdf(q: float) -> float:
    """Population CDF, including full and empty prediction thresholds."""
    return min(1.0, max(0.0, q))
