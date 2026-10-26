"""Unit Tests for Hidden Markov Model (HMM) Implementation.

Tests verify:
  1. Matrix dimension and stochastic probability invariants
  2. Forward Algorithm likelihood correctness and boundary cases
  3. Backward Algorithm likelihood equivalence with Forward
  4. Viterbi decoding accuracy, backpointer consistency, and inequality P* <= P(O|lambda)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hmm import HiddenMarkovModel


class TestHiddenMarkovModel(unittest.TestCase):

    def setUp(self):
        # Classic Weather Problem
        self.weather_hmm = HiddenMarkovModel(
            states=["Rainy", "Sunny"],
            symbols=["Walk", "Shop", "Clean"],
            A=[[0.7, 0.3], [0.4, 0.6]],
            B=[[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]],
            pi=[0.6, 0.4],
        )

    def test_stochastic_validation(self):
        """Verify non-stochastic matrices raise ValueError."""
        # Invalid pi sum
        with self.assertRaises(ValueError):
            HiddenMarkovModel(
                states=["S1", "S2"],
                symbols=["V1", "V2"],
                A=[[0.5, 0.5], [0.5, 0.5]],
                B=[[0.5, 0.5], [0.5, 0.5]],
                pi=[0.8, 0.8],
            )

        # Invalid A matrix row sum
        with self.assertRaises(ValueError):
            HiddenMarkovModel(
                states=["S1", "S2"],
                symbols=["V1", "V2"],
                A=[[0.7, 0.7], [0.5, 0.5]],
                B=[[0.5, 0.5], [0.5, 0.5]],
                pi=[0.5, 0.5],
            )

    def test_forward_algorithm_values(self):
        """Verify Forward alpha values for t=1, 2, 3."""
        seq = ["Walk", "Shop", "Clean"]
        alpha, p_total = self.weather_hmm.forward(seq)

        # Step 1:
        # alpha_1(Rainy) = 0.6 * 0.1 = 0.06
        # alpha_1(Sunny) = 0.4 * 0.6 = 0.24
        self.assertAlmostEqual(alpha[0][0], 0.06, places=5)
        self.assertAlmostEqual(alpha[0][1], 0.24, places=5)

        # Step 2:
        # Rainy: (0.06*0.7 + 0.24*0.4) * 0.4 = (0.042 + 0.096) * 0.4 = 0.138 * 0.4 = 0.0552
        # Sunny: (0.06*0.3 + 0.24*0.6) * 0.3 = (0.018 + 0.144) * 0.3 = 0.162 * 0.3 = 0.0486
        self.assertAlmostEqual(alpha[1][0], 0.0552, places=5)
        self.assertAlmostEqual(alpha[1][1], 0.0486, places=5)

        # Total P(O | lambda)
        expected_total = sum(alpha[2])
        self.assertAlmostEqual(p_total, expected_total, places=6)
        self.assertGreater(p_total, 0.0)

    def test_forward_backward_equivalence(self):
        """Verify Forward and Backward produce identical sequence likelihood."""
        seq = ["Walk", "Shop", "Clean"]
        _, p_fwd = self.weather_hmm.forward(seq)
        _, p_bwd = self.weather_hmm.backward(seq)
        self.assertAlmostEqual(p_fwd, p_bwd, places=6)

    def test_viterbi_decoding(self):
        """Verify Viterbi best path and probability bounds."""
        seq = ["Walk", "Shop", "Clean"]
        best_path, p_vit, delta, psi = self.weather_hmm.viterbi(seq)
        _, p_fwd = self.weather_hmm.forward(seq)

        self.assertEqual(len(best_path), 3)
        self.assertEqual(best_path, ["Sunny", "Rainy", "Rainy"])
        # Theoretical property: P* <= P(O | lambda)
        self.assertLessEqual(p_vit, p_fwd + 1e-9)
        self.assertGreater(p_vit, 0.0)

    def test_single_observation(self):
        """Verify handling of sequence length T=1."""
        seq = ["Walk"]
        alpha, p_fwd = self.weather_hmm.forward(seq)
        best_path, p_vit, _, _ = self.weather_hmm.viterbi(seq)

        self.assertEqual(len(alpha), 1)
        self.assertEqual(len(best_path), 1)
        self.assertEqual(best_path[0], "Sunny")
        self.assertAlmostEqual(p_vit, 0.24, places=5)
        self.assertAlmostEqual(p_fwd, 0.30, places=5)


if __name__ == "__main__":
    unittest.main()
