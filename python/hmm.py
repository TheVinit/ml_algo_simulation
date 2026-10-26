"""Hidden Markov Model (HMM) Reference Implementation.

Algorithms implemented:
  - Forward Algorithm (Likelihood Evaluation): Computes alpha variables and P(O | lambda)
  - Backward Algorithm (Smoothing): Computes beta variables
  - Viterbi Algorithm (Decoding): Computes delta variables, backpointers psi, and optimal state path Q*
  - Sequence Simulation / Sampling

References:
  - Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition.
  - Viterbi, A. (1967). Error bounds for convolutional codes and an asymptotically optimum decoding algorithm.
"""

from typing import List, Tuple, Dict, Any, Optional
import math


class HiddenMarkovModel:
    """Hidden Markov Model represented by the tuple lambda = (S, V, A, B, pi).

    Attributes:
        states (List[str]): Hidden state space S = {S_1, S_2, ..., S_N}.
        symbols (List[str]): Discrete observation alphabet V = {v_1, v_2, ..., v_M}.
        A (List[List[float]]): State transition probability matrix (N x N), where A[i][j] = P(q_t = S_j | q_{t-1} = S_i).
        B (List[List[float]]): Emission probability matrix (N x M), where B[i][k] = P(O_t = v_k | q_t = S_i).
        pi (List[float]): Initial state distribution vector (N), where pi[i] = P(q_1 = S_i).
    """

    def __init__(
        self,
        states: List[str],
        symbols: List[str],
        A: List[List[float]],
        B: List[List[float]],
        pi: List[float],
        tolerance: float = 1e-4,
    ) -> None:
        self.states = list(states)
        self.symbols = list(symbols)
        self.A = [list(row) for row in A]
        self.B = [list(row) for row in B]
        self.pi = list(pi)
        self.tolerance = tolerance

        self.N = len(self.states)
        self.M = len(self.symbols)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.symbol_to_idx = {v: k for k, v in enumerate(self.symbols)}

        self._validate_stochastic_properties()

    def _validate_stochastic_properties(self) -> None:
        """Validates that initial, transition, and emission matrices are row-stochastic."""
        if len(self.pi) != self.N:
            raise ValueError(f"pi length ({len(self.pi)}) must match states count ({self.N}).")
        if abs(sum(self.pi) - 1.0) > self.tolerance:
            raise ValueError(f"Initial distribution pi must sum to 1.0; got {sum(self.pi)}.")

        if len(self.A) != self.N or any(len(r) != self.N for r in self.A):
            raise ValueError(f"Matrix A must be of shape {self.N}x{self.N}.")
        for i, row in enumerate(self.A):
            if abs(sum(row) - 1.0) > self.tolerance:
                raise ValueError(f"Matrix A row {i} must sum to 1.0; got {sum(row)}.")

        if len(self.B) != self.N or any(len(r) != self.M for r in self.B):
            raise ValueError(f"Matrix B must be of shape {self.N}x{self.M}.")
        for i, row in enumerate(self.B):
            if abs(sum(row) - 1.0) > self.tolerance:
                raise ValueError(f"Matrix B row {i} must sum to 1.0; got {sum(row)}.")

    def forward(self, observations: List[str]) -> Tuple[List[List[float]], float]:
        """Executes the Forward algorithm to compute alpha trellis and sequence likelihood.

        Recurrence:
            alpha_1(i) = pi_i * B_i(o_1)
            alpha_t(j) = [ sum_i alpha_{t-1}(i) * A_ij ] * B_j(o_t)
            P(O | lambda) = sum_i alpha_T(i)

        Args:
            observations: List of observed symbol tokens.

        Returns:
            alpha: Trellis table of shape (T, N) containing forward variables.
            total_likelihood: P(O | lambda).
        """
        T = len(observations)
        if T == 0:
            return [], 0.0

        obs_indices = [self.symbol_to_idx[o] for o in observations]
        alpha = [[0.0] * self.N for _ in range(T)]

        # Base case (t = 1)
        o1 = obs_indices[0]
        for i in range(self.N):
            alpha[0][i] = self.pi[i] * self.B[i][o1]

        # Induction (t = 2 .. T)
        for t in range(1, T):
            ot = obs_indices[t]
            for j in range(self.N):
                incoming_sum = sum(alpha[t - 1][i] * self.A[i][j] for i in range(self.N))
                alpha[t][j] = incoming_sum * self.B[j][ot]

        total_likelihood = sum(alpha[T - 1])
        return alpha, total_likelihood

    def backward(self, observations: List[str]) -> Tuple[List[List[float]], float]:
        """Executes the Backward algorithm to compute beta trellis.

        Recurrence:
            beta_T(i) = 1.0
            beta_t(i) = sum_j [ A_ij * B_j(o_{t+1}) * beta_{t+1}(j) ]
            P(O | lambda) = sum_i [ pi_i * B_i(o_1) * beta_1(i) ]

        Args:
            observations: List of observed symbol tokens.

        Returns:
            beta: Trellis table of shape (T, N) containing backward variables.
            total_likelihood: P(O | lambda).
        """
        T = len(observations)
        if T == 0:
            return [], 0.0

        obs_indices = [self.symbol_to_idx[o] for o in observations]
        beta = [[0.0] * self.N for _ in range(T)]

        # Base case (t = T)
        for i in range(self.N):
            beta[T - 1][i] = 1.0

        # Induction (t = T - 1 down to 1)
        for t in range(T - 2, -1, -1):
            next_o = obs_indices[t + 1]
            for i in range(self.N):
                beta[t][i] = sum(
                    self.A[i][j] * self.B[j][next_o] * beta[t + 1][j] for j in range(self.N)
                )

        o1 = obs_indices[0]
        total_likelihood = sum(self.pi[i] * self.B[i][o1] * beta[0][i] for i in range(self.N))
        return beta, total_likelihood

    def viterbi(
        self, observations: List[str]
    ) -> Tuple[List[str], float, List[List[float]], List[List[int]]]:
        """Executes the Viterbi algorithm to decode the optimal state sequence Q*.

        Recurrence:
            delta_1(i) = pi_i * B_i(o_1),  psi_1(i) = 0
            delta_t(j) = max_i [ delta_{t-1}(i) * A_ij ] * B_j(o_t)
            psi_t(j) = argmax_i [ delta_{t-1}(i) * A_ij ]

        Termination:
            P* = max_i delta_T(i)
            q*_T = argmax_i delta_T(i)

        Backtracking:
            q*_t = psi_{t+1}(q*_{t+1}) for t = T-1 down to 1

        Args:
            observations: List of observed symbol tokens.

        Returns:
            best_path: Optimal hidden state sequence [S_{q*_1}, ..., S_{q*_T}].
            best_prob: Optimal joint probability P*.
            delta: Trellis of shape (T, N) containing maximal subpath metrics.
            psi: Trellis of shape (T, N) containing argmax backpointers.
        """
        T = len(observations)
        if T == 0:
            return [], 0.0, [], []

        obs_indices = [self.symbol_to_idx[o] for o in observations]
        delta = [[0.0] * self.N for _ in range(T)]
        psi = [[0] * self.N for _ in range(T)]

        # Base case (t = 1)
        o1 = obs_indices[0]
        for i in range(self.N):
            delta[0][i] = self.pi[i] * self.B[i][o1]
            psi[0][i] = 0

        # Induction (t = 2 .. T)
        for t in range(1, T):
            ot = obs_indices[t]
            for j in range(self.N):
                max_val = -1.0
                best_i = 0
                for i in range(self.N):
                    candidate = delta[t - 1][i] * self.A[i][j]
                    if candidate > max_val:
                        max_val = candidate
                        best_i = i
                delta[t][j] = max_val * self.B[j][ot]
                psi[t][j] = best_i

        # Termination
        best_prob = max(delta[T - 1])
        last_state = max(range(self.N), key=lambda i: delta[T - 1][i])

        # Backtracking
        path_indices = [0] * T
        path_indices[T - 1] = last_state
        for t in range(T - 2, -1, -1):
            path_indices[t] = psi[t + 1][path_indices[t + 1]]

        best_path = [self.states[idx] for idx in path_indices]
        return best_path, best_prob, delta, psi
