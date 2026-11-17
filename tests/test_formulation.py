"""Independent checks of the paper's rank, truncation, and ACI formulas."""
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import combinations_with_replacement, product
from math import inf, nextafter
from random import Random
import unittest

from methods import ACI, SplitConformal, TruncatedConformal


class FormulationTests(unittest.TestCase):
    def test_rank_for_every_short_multiset(self):
        # Enumerate histories with ties; evaluate the paper's rank directly.
        for alpha in (0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99):
            target = Fraction(str(alpha))
            for length in range(9):
                for history in combinations_with_replacement((-1., 0., 1.), length):
                    method = SplitConformal(alpha)
                    for score in reversed(history):
                        method.update(score)
                    cutoff = method.threshold()
                    for value in (-1., 0., 1.):
                        for score in (nextafter(value, -inf), value, nextafter(value, inf)):
                            rank = 1 + sum(old >= score for old in history)
                            self.assertEqual(score <= cutoff, Fraction(rank, length + 1) > target)

    def test_first_nine_sets_and_tenth_rank_boundary(self):
        method = SplitConformal(0.1)
        for score in range(9):
            self.assertEqual(method.threshold(), inf)
            method.update(float(score))
        self.assertEqual(method.threshold(), 8.)
        self.assertFalse(method.update(8.))  # Equality at the maximum is covered.

    def test_strict_boundary_does_not_round_to_equality(self):
        method = TruncatedConformal(alpha=0.35714285714285715, beta=1, kappa=0.5)
        target = Fraction("0.35714285714285715")
        for t in range(1, 7):
            method.update(0.)
            self.assertLessEqual(abs(method.errors - t * target), Fraction(1, 2))
        self.assertEqual(method.branch(), -1)
        self.assertTrue(method.update(0.))
        self.assertLessEqual(abs(method.errors - 7 * target), Fraction(1, 2))

    def test_decimal_kappa_preserves_exact_equality(self):
        method = TruncatedConformal(alpha=0.1, beta=1, kappa=0.7)
        method.baseline.history = [0.] * 2
        method.errors = 0
        self.assertEqual(method.branch(), 0)  # Upper candidate is exactly +.7.
        method.baseline.history = [0.] * 6
        self.assertEqual(method.branch(), 0)  # Lower candidate is exactly -.7.

    def test_truncation_agrees_with_high_precision_paper_conditions(self):
        with localcontext() as context:
            context.prec = 60
            for alpha, beta, kappa in product((0.1, 0.3, 0.5, 0.9), (0., 0.5, 1.), (0.5, 0.7, 2.)):
                method = TruncatedConformal(alpha, beta, kappa)
                a, b, k = map(lambda v: Decimal(str(v)), (alpha, beta, kappa))
                for t in range(1, 61):
                    method.baseline.history = [0.] * (t - 1)
                    bound = k * Decimal(t) ** (1 - b)
                    for errors in range(t):
                        method.errors = errors
                        previous = Decimal(errors) - (t - 1) * a
                        upper, lower = previous + 1 - a, previous - a
                        expected = 1 if upper > bound else -1 if lower < -bound else 0
                        self.assertEqual(method.branch(), expected, (alpha, beta, kappa, t, errors))

    def test_aci_exactly_representable_ties(self):
        method = ACI(alpha=0.25, beta=1, c=0.5, q1=0.75)
        q = Fraction(3, 4)
        for _ in range(500):
            score = q if 0 <= q <= 1 else Fraction(0)
            self.assertEqual(method.threshold(), float(q))
            error = score > q
            self.assertEqual(method.update(float(score)), error)
            q += Fraction(1, 2) * (error - Fraction(1, 4))

    def test_aci_decaying_steps_and_summation_by_parts(self):
        random = Random(261121542)
        scores = [random.random() for _ in range(500)]
        with localcontext() as context:
            context.prec = 60
            for alpha, beta, c in product((0.1, 0.25, 0.9), (0.3, 0.5, 0.7), (0.25, 1., 2.)):
                method = ACI(alpha=alpha, beta=beta, c=c, q1=0.9)
                a, b, scale = map(lambda v: Decimal(str(v)), (alpha, beta, c))
                q, errors, displacements, inverse_steps = Decimal("0.9"), 0, [], []
                for t, score in enumerate(scores, 1):
                    self.assertAlmostEqual(method.threshold(), float(q), delta=2e-12)
                    error = Decimal.from_float(score) > q
                    self.assertEqual(method.update(score), error)
                    q += scale * Decimal(t) ** (b - 1) * (int(error) - a)
                    errors += error
                    displacements.append(method.threshold() - 0.9)
                    inverse_steps.append(t ** (1 - beta) / c)
                    self.assertLessEqual(abs(errors - alpha * t), (1 + c) / c * t ** (1 - beta) + 1e-11)
                identity = displacements[-1] * inverse_steps[-1] + sum(
                    d * (left - right) for d, left, right in
                    zip(displacements, inverse_steps, inverse_steps[1:]))
                self.assertAlmostEqual(identity, errors - len(scores) * alpha, delta=2e-10)


if __name__ == "__main__":
    unittest.main()
