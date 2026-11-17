"""Check public methods against set membership and adversarial error paths."""
from fractions import Fraction
from itertools import product
from math import inf, isclose
import unittest

from methods import ACI, SplitConformal, TruncatedConformal


class MethodTests(unittest.TestCase):
    def test_conformal_literal_rank_with_ties(self):
        observations = [0.0, 0.2, 0.2, 0.5, 1.0] * 8
        for alpha in (0.1, 0.25, 0.3):
            method = SplitConformal(alpha)
            a = Fraction(str(alpha))
            for t, observed in enumerate(observations, 1):
                threshold = method.threshold()
                for candidate in (-0.1, 0, 0.2, 0.200001, 0.5, 1, 1.1):
                    rank = 1 + sum(old >= candidate for old in observations[:t - 1])
                    included = Fraction(rank, t) > a
                    self.assertEqual(candidate <= threshold, included)
                method.update(observed)

    def test_truncation_on_every_binary_stream(self):
        # At beta=1, kappa=.5 the strict boundaries are reached repeatedly.
        for alpha, beta in product((0.1, 0.5, 0.9), (0.0, 0.5, 1.0)):
            for stream in product((0.0, 1.0), repeat=10):
                method = TruncatedConformal(alpha, beta, kappa=0.5)
                errors = 0
                for t, value in enumerate(stream, 1):
                    predicted = method.threshold()
                    error = method.update(value)
                    self.assertEqual(error, value > predicted)
                    errors += error
                    self.assertLessEqual(abs(errors - alpha * t),
                                         0.5 * t ** (1 - beta) + 1e-12)
                self.assertEqual(method.baseline.history, sorted(stream))

    def test_strict_truncation_boundary(self):
        method = TruncatedConformal(alpha=0.5, beta=1, kappa=0.5)
        self.assertEqual(method.branch(), 0)  # Both candidate endpoints fit.
        method.update(0.0)
        self.assertEqual(method.threshold(), -inf)
        method.update(1.0)
        self.assertEqual(method.branch(), 0)

    def test_aci_constant_step_identity_and_no_projection(self):
        method = ACI(alpha=0.1, beta=1, c=2, q1=0.9)
        errors = 0
        for t, value in enumerate([1.0] + [0.0] * 30 + [1.0] * 10, 1):
            before = method.threshold()
            error = method.update(value)
            self.assertEqual(error, value > before)
            errors += error
            self.assertTrue(isclose(method.threshold() - 0.9,
                                    2 * (errors - 0.1 * t), abs_tol=1e-12))
            if t == 1:
                self.assertGreater(method.threshold(), 1.0)
        empty = ACI(alpha=0.1, beta=1, c=2, q1=0.0)
        empty.update(0.0)
        self.assertLess(empty.threshold(), 0.0)

    def test_invalid_parameters_and_scores(self):
        for build in (lambda: SplitConformal(0), lambda: ACI(c=0),
                      lambda: TruncatedConformal(kappa=0.4)):
            with self.assertRaises(ValueError):
                build()
        for method in (SplitConformal(), TruncatedConformal(), ACI()):
            with self.assertRaises(ValueError):
                method.update(float("nan"))


if __name__ == "__main__":
    unittest.main()
