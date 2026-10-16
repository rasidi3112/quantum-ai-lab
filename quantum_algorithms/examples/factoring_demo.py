from __future__ import annotations

import numpy as np

from quantum_algorithms import QFT, DeutschJozsa, GroverSearch, ShorFactoring


def run_shor() -> None:
    print("\n" + "-" * 60)
    print("  1. Shor's algorithm — factoring via quantum period finding")
    print("-" * 60)
    shor = ShorFactoring(seed=2024)
    for N in (15, 21, 33, 35, 39, 51):
        result = shor.factor(N)
        p, q = sorted(result.factors)
        period = f"r = {result.period_found:2d}" if result.period_found else "      "
        print(f"  {N:3d} = {p:2d} × {q:2d}   a = {result.a_used:2d}   {period}   "
              f"via {result.method}")


def run_grover() -> None:
    print("\n" + "-" * 60)
    print("  2. Grover search — quadratic speed-up over brute force")
    print("-" * 60)
    grover = GroverSearch()
    for n in (4, 6, 8, 10):
        target = (2 ** n) // 3
        result = grover.search(n, target)
        print(f"  N = {2 ** n:5d}  oracle calls = {result.n_iterations:3d} "
              f"(classical ~{2 ** n // 2:4d})  P(target) = {result.found_probability:.4f}")


def run_deutsch_jozsa() -> None:
    print("\n" + "-" * 60)
    print("  3. Deutsch–Jozsa — constant vs balanced in one query")
    print("-" * 60)
    dj = DeutschJozsa(seed=5)
    for kind in ("constant", "balanced", "balanced", "constant"):
        oracle, truth = dj.random_oracle(4, kind)
        verdict = dj.run(oracle, 4).oracle_type
        print(f"  hidden oracle: {truth:9s} → measured: {verdict:9s} "
              f"{'✓' if verdict == truth else '✗'}")


def run_phase_estimation() -> None:
    print("\n" + "-" * 60)
    print("  4. Quantum phase estimation")
    print("-" * 60)
    qft = QFT()
    for phase in (0.25, 0.3125, 0.6875):
        U = np.diag([1.0, np.exp(2j * np.pi * phase)])
        result = qft.phase_estimation(U, np.array([0.0, 1.0]), n_precision=5, seed=0)
        print(f"  true φ = {phase:.4f}   estimated φ = {result.estimated_phase:.4f}")


def main() -> None:
    print("=" * 60)
    print("  Quantum Algorithms — Factoring & Search Demo")
    print("=" * 60)
    run_shor()
    run_grover()
    run_deutsch_jozsa()
    run_phase_estimation()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
