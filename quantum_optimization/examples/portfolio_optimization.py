from __future__ import annotations

import numpy as np

from quantum_optimization import QAOA, MaxCutSolver

ASSETS = ["QBIT", "ATOM", "PHOT", "SPIN", "ION", "TOPO"]
EXPECTED_RETURNS = np.array([0.12, 0.10, 0.07, 0.15, 0.09, 0.11])
VOLATILITY = np.array([0.20, 0.15, 0.08, 0.30, 0.12, 0.18])
CORRELATION = np.array([
    [1.0, 0.3, 0.1, 0.5, 0.2, 0.4],
    [0.3, 1.0, 0.2, 0.3, 0.4, 0.2],
    [0.1, 0.2, 1.0, 0.1, 0.3, 0.1],
    [0.5, 0.3, 0.1, 1.0, 0.2, 0.6],
    [0.2, 0.4, 0.3, 0.2, 1.0, 0.2],
    [0.4, 0.2, 0.1, 0.6, 0.2, 1.0],
])


def portfolio_costs(n: int, risk_aversion: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    covariance = np.outer(VOLATILITY, VOLATILITY) * CORRELATION
    all_bits = np.array([[int(b) for b in format(i, f"0{n}b")] for i in range(2 ** n)])
    costs = np.array([risk_aversion * b @ covariance @ b - EXPECTED_RETURNS @ b for b in all_bits])
    return all_bits, costs


def describe(bits: np.ndarray) -> str:
    return ", ".join(a for a, b in zip(ASSETS, bits) if b)


def run_portfolio() -> None:
    print("\n" + "-" * 60)
    print("  1. Portfolio selection — pick 3 of 6 assets (Markowitz)")
    print("-" * 60)
    n, budget = len(ASSETS), 3
    all_bits, costs = portfolio_costs(n)
    feasible = all_bits.sum(axis=1) == budget
    optimum = int(np.argmin(np.where(feasible, costs, np.inf)))
    f_min, f_max = costs[feasible].min(), costs[feasible].max()
    normalized = (costs - f_min) / (f_max - f_min)

    penalized = normalized + 2.0 * (all_bits.sum(axis=1) - budget) ** 2
    standard = QAOA(n, seed=1).optimize_hamiltonian(
        penalized / penalized.max(), p_layers=4, n_restarts=8, method="L-BFGS-B"
    )
    constrained = QAOA(n, seed=1).optimize_hamiltonian(
        normalized, p_layers=4, n_restarts=8, method="L-BFGS-B",
        mixer="xy", hamming_weight=budget,
    )

    print(f"  Search space: {2 ** n} portfolios, {feasible.sum()} satisfy the budget")
    print(f"  Optimal portfolio (brute force): {describe(all_bits[optimum])}\n")
    print("  Ansatz                         P(valid)   P(optimal)")
    print(f"  {'random guess':29s}  {feasible.mean():7.1%}   {1 / 2 ** n:8.1%}")
    for name, result in (("QAOA, X mixer + penalty", standard),
                         ("QAOA, XY mixer (Dicke start)", constrained)):
        probs = np.abs(result["state"]) ** 2
        print(f"  {name:29s}  {probs[feasible].sum():7.1%}   {probs[optimum]:8.1%}")

    samples = QAOA.sample_solution(constrained["state"], n_samples=2000, seed=7)
    print("\n  Top portfolios sampled from the XY-mixer circuit:")
    for bs, count in list(samples.items())[:4]:
        mark = "  ← optimal" if int(bs, 2) == optimum else ""
        print(f"    {bs}  {count / 2000:6.1%}  {describe([int(b) for b in bs])}{mark}")


def run_maxcut() -> None:
    print("\n" + "-" * 60)
    print("  2. MaxCut on a random 10-node graph")
    print("-" * 60)
    graph = MaxCutSolver.random_graph(10, edge_prob=0.4, seed=3)
    solver = MaxCutSolver(10)
    exact = solver.brute_force_solve(graph, 10)
    for p in (1, 2, 3):
        result = solver.qaoa_solve(graph, p_layers=p, seed=1)
        print(f"  p = {p}:  best cut {result['best_cut']:.0f}/{exact['max_cut']:.0f}   "
              f"approximation ratio ⟨C⟩/C_max = {result['approximation_ratio']:.3f}")


def main() -> None:
    print("=" * 60)
    print("  Quantum Optimization — QAOA Demo")
    print("=" * 60)
    run_portfolio()
    run_maxcut()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
