from __future__ import annotations

import numpy as np
import pytest

from quantum_ml import QuantumKernel, QuantumNeuralNetwork, VariationalClassifier


def test_kernel_matrix_is_valid_gram_matrix(rng):
    X = rng.uniform(0, np.pi, size=(8, 2))
    K = QuantumKernel(n_qubits=2, n_layers=2).kernel_matrix(X)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-10
    assert np.all((K >= -1e-12) & (K <= 1 + 1e-12))


def test_feature_map_is_normalized(rng):
    psi = QuantumKernel(n_qubits=3).feature_map(rng.uniform(0, np.pi, 3))
    assert np.linalg.norm(psi) == pytest.approx(1.0)


def test_kernel_svm_separates_simple_data():
    X = np.array([[0.2, 0.3], [0.3, 0.1], [0.1, 0.2], [2.8, 2.9], [2.9, 2.7], [2.7, 3.0]])
    y = np.array([-1, -1, -1, 1, 1, 1])
    preds = QuantumKernel(n_qubits=2, n_layers=1).classify_svm(X, y, X, alpha=0.1)
    assert np.array_equal(preds, y)


def test_variational_classifier_learns(rng):
    X = rng.uniform(0, np.pi, size=(24, 2))
    y = (X[:, 0] > np.pi / 2).astype(int)
    clf = VariationalClassifier(n_qubits=2, n_layers=2, seed=0)
    history = clf.train(X, y, epochs=150, lr=0.5, verbose=False)["history"]
    assert history[-1] < history[0]
    assert clf.accuracy(X, y) >= 0.85


@pytest.mark.parametrize("encoding", ["angle", "amplitude"])
def test_qnn_parameter_shift_matches_finite_difference(rng, encoding):
    qnn = QuantumNeuralNetwork(n_qubits=3, n_layers=2, encoding=encoding, seed=1)
    x = rng.uniform(-1, 1, 3)
    grad = qnn.gradient(x)
    eps = 1e-6
    for i in range(qnn.n_params):
        plus, minus = qnn.weights.copy(), qnn.weights.copy()
        plus[i] += eps
        minus[i] -= eps
        numeric = (qnn.forward(x, plus) - qnn.forward(x, minus)) / (2 * eps)
        assert np.allclose(grad[i], numeric, atol=1e-6)


def test_qnn_outputs_are_physical(rng):
    x = rng.uniform(-1, 1, 4)
    expvals = QuantumNeuralNetwork(n_qubits=4, measurement="expval", seed=0).forward(x)
    probs = QuantumNeuralNetwork(n_qubits=4, measurement="probs", seed=0).forward(x)
    assert np.all(np.abs(expvals) <= 1 + 1e-12)
    assert probs.sum() == pytest.approx(1.0)


def test_random_state_alias_is_deprecated():
    with pytest.warns(DeprecationWarning):
        old = VariationalClassifier(n_qubits=2, random_state=3)
    assert np.allclose(old.params, VariationalClassifier(n_qubits=2, seed=3).params)
