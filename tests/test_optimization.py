from __future__ import annotations

import numpy as np
import pytest

from quantum_optimization import QAOA, VQE, MaxCutSolver
from quantum_simulation import IsingModel


def test_fast_x_mixer_matches_matrix_exponential(rng):
    graph = MaxCutSolver.random_graph(5, 0.6, seed=2)
    C = QAOA.cost_hamiltonian(graph, 5)
    B = QAOA.mixer_hamiltonian(5)
    gamma, beta = rng.uniform(0, np.pi, 3), rng.uniform(0, np.pi, 3)
    assert np.allclose(QAOA.qaoa_circuit(gamma, beta, C), QAOA.qaoa_circuit(gamma, beta, C, B))


def test_xy_mixer_conserves_hamming_weight():
    B = QAOA.xy_mixer_hamiltonian(5)
    weight = np.diag([bin(i).count("1") for i in range(32)])
    assert np.allclose(B, B.conj().T)
    assert np.allclose(B @ weight, weight @ B)


def test_dicke_state():
    psi = QAOA.dicke_state(4, 2)
    assert np.linalg.norm(psi) == pytest.approx(1.0)
    assert np.count_nonzero(psi) == 6


def test_xy_qaoa_stays_in_feasible_subspace(rng):
    costs = rng.normal(size=64)
    result = QAOA(6, seed=0).optimize_hamiltonian(
        costs, p_layers=2, mixer="xy", hamming_weight=3, maxiter=50
    )
    feasible = np.array([bin(i).count("1") == 3 for i in range(64)])
    assert np.sum(np.abs(result["state"][feasible]) ** 2) == pytest.approx(1.0)


def test_xy_mixer_requires_hamming_weight():
    with pytest.raises(ValueError):
        QAOA(3).optimize_hamiltonian(np.zeros(8), mixer="xy")


def test_maxcut_qaoa_finds_optimum():
    graph = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    result = MaxCutSolver(4).qaoa_solve(graph, p_layers=2, seed=0)
    assert result["best_cut"] == MaxCutSolver.brute_force_solve(graph, 4)["max_cut"] == 4
    assert 0.7 < result["approximation_ratio"] <= 1.0


def test_qaoa_is_reproducible_with_seed():
    graph = MaxCutSolver.random_graph(5, 0.5, seed=1)
    a = QAOA(5, seed=3).optimize(graph, p_layers=1)
    b = QAOA(5, seed=3).optimize(graph, p_layers=1)
    assert np.allclose(a["params"], b["params"])


def test_vqe_reaches_ising_ground_state():
    H = IsingModel(4, 1.0, 1.0).hamiltonian
    result = VQE(4, seed=0).optimize(H, n_layers=3)
    assert result["energy"] == pytest.approx(np.linalg.eigvalsh(H)[0], abs=1e-4)


def test_vqe_uccsd_is_normalized():
    psi = VQE.uccsd_ansatz(4, 2, np.full(5, 0.1))
    assert np.linalg.norm(psi) == pytest.approx(1.0)
