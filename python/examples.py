"""Demonstration and Benchmark Examples for Hidden Markov Models.

This script executes:
  1. The Classic Weather / Activity Model (Rainy/Sunny -> Walk/Shop/Clean)
  2. The Medical Diagnosis Model (Healthy/Fever -> Normal/Cold/Dizzy)

Matches outputs with the interactive visual simulator at index.html.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hmm import HiddenMarkovModel


def print_matrix(title: str, headers: list, rows: list, row_labels: list = None):
    print(f"\n--- {title} ---")
    header_str = " | ".join(f"{h:>10}" for h in (["State"] if row_labels else []) + headers)
    print(header_str)
    print("-" * len(header_str))
    for i, row in enumerate(rows):
        prefix = f"{row_labels[i]:>10} | " if row_labels else ""
        vals = " | ".join(f"{v:10.5f}" if isinstance(v, float) else f"{str(v):>10}" for v in row)
        print(f"{prefix}{vals}")


def run_benchmark(name: str, hmm: HiddenMarkovModel, seq: list):
    print("=" * 70)
    print(f"BENCHMARK: {name}")
    print(f"Observation Sequence: {' -> '.join(seq)}")
    print("=" * 70)

    # 1. Forward Algorithm
    alpha, p_forward = hmm.forward(seq)
    col_headers = [f"t={t+1} ({seq[t]})" for t in range(len(seq))]
    # Transpose alpha for display (rows=states, cols=time)
    alpha_t = [[alpha[t][i] for t in range(len(seq))] for i in range(hmm.N)]
    print_matrix("Forward Trellis Variables (alpha)", col_headers, alpha_t, hmm.states)
    print(f"\n=> Total Sequence Likelihood P(O | lambda) = {p_forward:.6e} ({p_forward:.6f})")

    # 2. Backward Algorithm (Cross-Validation)
    beta, p_backward = hmm.backward(seq)
    print(f"=> Backward Algorithm Likelihood Verification = {p_backward:.6e} ({p_backward:.6f})")
    assert abs(p_forward - p_backward) < 1e-7, "Forward and Backward likelihoods must match!"

    # 3. Viterbi Algorithm
    best_path, p_viterbi, delta, psi = hmm.viterbi(seq)
    delta_t = [[delta[t][i] for t in range(len(seq))] for i in range(hmm.N)]
    print_matrix("Viterbi Trellis Scores (delta)", col_headers, delta_t, hmm.states)

    psi_t = [[hmm.states[psi[t][i]] if t > 0 else "-" for t in range(len(seq))] for i in range(hmm.N)]
    print_matrix("Viterbi Backpointers (psi)", col_headers, psi_t, hmm.states)

    print(f"\n=> Optimal Hidden State Path (Viterbi) Q* = {' -> '.join(best_path)}")
    print(f"=> Optimal Path Probability P* = {p_viterbi:.6e} ({p_viterbi:.6f})")
    print(f"=> Inequality Check: P* ({p_viterbi:.6f}) <= P(O|lambda) ({p_forward:.6f}) : {p_viterbi <= p_forward}")
    print()


def main():
    # Preset 1: Weather / Activity
    weather_hmm = HiddenMarkovModel(
        states=["Rainy", "Sunny"],
        symbols=["Walk", "Shop", "Clean"],
        A=[
            [0.7, 0.3],  # Rainy -> [Rainy, Sunny]
            [0.4, 0.6],  # Sunny -> [Rainy, Sunny]
        ],
        B=[
            [0.1, 0.4, 0.5],  # Rainy -> [Walk, Shop, Clean]
            [0.6, 0.3, 0.1],  # Sunny -> [Walk, Shop, Clean]
        ],
        pi=[0.6, 0.4],
    )
    run_benchmark("Weather & Activity Model", weather_hmm, ["Walk", "Shop", "Clean"])

    # Preset 2: Medical Diagnosis
    medical_hmm = HiddenMarkovModel(
        states=["Healthy", "Fever"],
        symbols=["Normal", "Cold", "Dizzy"],
        A=[
            [0.7, 0.3],  # Healthy -> [Healthy, Fever]
            [0.4, 0.6],  # Fever   -> [Healthy, Fever]
        ],
        B=[
            [0.5, 0.4, 0.1],  # Healthy -> [Normal, Cold, Dizzy]
            [0.1, 0.3, 0.6],  # Fever   -> [Normal, Cold, Dizzy]
        ],
        pi=[0.6, 0.4],
    )
    run_benchmark("Medical Diagnosis Model", medical_hmm, ["Normal", "Cold", "Dizzy"])


if __name__ == "__main__":
    main()
