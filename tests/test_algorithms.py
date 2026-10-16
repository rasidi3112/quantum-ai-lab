from __future__ import annotations

from math import gcd

import numpy as np
import pytest

from quantum_algorithms import QFT, DeutschJozsa, GroverSearch, ShorFactoring


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 5])
def test_qft_is_unitary_and_matches_circuit(n_qubits):
    qft = QFT()
    F = qft.qft_matrix(n_qubits)
    assert np.allclose(F @ F.conj().T, np.eye(2 ** n_qubits))
    assert np.allclose(qft.qft_from_gates(n_qubits), F)
    assert np.allclose(qft.iqft_matrix(n_qubits) @ F, np.eye(2 ** n_qubits))


@pytest.mark.parametrize("phase", [0.0, 0.125, 0.3125, 0.5, 0.875])
def test_phase_estimation_recovers_exact_phase(phase):
    U = np.diag([1.0, np.exp(2j * np.pi * phase)])
    result = QFT().phase_estimation(U, np.array([0.0, 1.0]), n_precision=4, seed=0)
    assert result.estimated_phase == pytest.approx(phase)
    assert result.probabilities.max() == pytest.approx(1.0)


@pytest.mark.parametrize("n_qubits", [3, 5, 8])
def test_grover_amplifies_target(n_qubits):
    target = 2 ** n_qubits - 3
    result = GroverSearch().search(n_qubits, target)
    assert result.success
    assert result.found_probability > 0.9
    assert result.n_iterations == GroverSearch.optimal_iterations(2 ** n_qubits)


def test_grover_multi_target():
    result = GroverSearch().multi_target_search(6, [3, 17, 42])
    assert result.success
    assert result.probabilities[[3, 17, 42]].sum() > 0.9


def test_grover_diffusion_is_unitary():
    D = GroverSearch.diffusion_operator(4)
    assert np.allclose(D @ D.conj().T, np.eye(16))


@pytest.mark.parametrize("pattern", range(1, 8))
def test_deutsch_jozsa_balanced(pattern):
    oracle = DeutschJozsa.balanced_oracle(3, pattern)
    assert DeutschJozsa.verify(oracle, 3) == "balanced"
    assert DeutschJozsa(seed=0).run(oracle, 3).oracle_type == "balanced"


@pytest.mark.parametrize("value", [0, 1])
def test_deutsch_jozsa_constant(value):
    oracle = DeutschJozsa.constant_oracle(3, value)
    assert DeutschJozsa(seed=0).run(oracle, 3).oracle_type == "constant"


@pytest.mark.parametrize("N", [15, 21, 33, 35, 39, 51, 55, 77, 91])
def test_shor_factors_semiprimes(N):
    for seed in range(5):
        result = ShorFactoring(seed=seed).factor(N)
        p, q = result.factors
        assert result.success
        assert p * q == N and 1 < p < N
        if result.method == "quantum period finding":
            assert pow(result.a_used, result.period_found, N) == 1
            assert result.period_found % 2 == 0


def test_shor_trivial_cases():
    assert ShorFactoring().factor(22).method == "classical: even"
    assert ShorFactoring().factor(49).method == "classical: perfect power"
    assert not ShorFactoring().factor(1).success


def test_shor_period_finding_matches_classical_order():
    shor = ShorFactoring(seed=3)
    for a, N in [(7, 15), (2, 21), (5, 33)]:
        order = ShorFactoring.classical_order_check(a, N)
        found = [shor.quantum_period_finding(a, N) for _ in range(30)]
        assert any(r > 0 and order % r == 0 for r in found)
        assert gcd(a, N) == 1
