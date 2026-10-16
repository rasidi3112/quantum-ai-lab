from __future__ import annotations

import copy

import numpy as np
import pytest

from quantum_vision import HybridClassifier, QuantumEncoder, Quanvolution
from quantum_vision.examples.bars_and_stripes import bars_and_stripes
from quantum_vision.hybrid_classifier import _cross_entropy


@pytest.mark.parametrize("encoder", ["amplitude_encoding", "angle_encoding", "FRQI", "NEQR"])
def test_encoders_produce_normalized_states(rng, encoder):
    image = rng.uniform(0, 1, size=(4, 4))
    state = getattr(QuantumEncoder, encoder)(image)
    assert np.linalg.norm(state) == pytest.approx(1.0)


def test_amplitude_encoding_roundtrip(rng):
    image = rng.uniform(0, 1, size=(4, 4))
    state = QuantumEncoder.amplitude_encoding(image)
    decoded = QuantumEncoder.decode_amplitude(state, image.shape)
    assert np.allclose(decoded, image / np.linalg.norm(image))


def test_quanvolution_output_shape_and_range(rng):
    quanv = Quanvolution(n_qubits=4, n_layers=1, patch_size=2, seed=0)
    out = quanv.quanvolve(rng.uniform(0, 1, (6, 6)), quanv.random_quantum_filter(seed=0))
    assert out.shape == (3, 3, 4)
    assert np.all((out >= 0) & (out <= 1))


def test_hybrid_classifier_gradients_match_finite_differences(rng):
    x = rng.uniform(0, 1, 6)
    label = 1
    model = HybridClassifier(input_dim=6, hidden_dim=5, n_qubits=3, n_layers=1, seed=2)
    before = copy.deepcopy(model)
    lr = 1e-6
    model._update_step(x[None], np.array([label]), lr)

    for name in ("W1", "W2", "q_params", "W_out", "b_out"):
        analytic = (getattr(before, name) - getattr(model, name)) / lr
        param = getattr(before, name)
        for idx in list(np.ndindex(param.shape))[:6]:
            plus, minus = copy.deepcopy(before), copy.deepcopy(before)
            getattr(plus, name)[idx] += 1e-5
            getattr(minus, name)[idx] -= 1e-5
            numeric = (_cross_entropy(plus.forward(x), label)
                       - _cross_entropy(minus.forward(x), label)) / 2e-5
            assert analytic[idx] == pytest.approx(numeric, abs=1e-6)


def test_hybrid_classifier_learns_bars_and_stripes():
    X, y = bars_and_stripes(64, seed=0)
    X = X.reshape(len(X), -1)
    model = HybridClassifier(input_dim=16, hidden_dim=8, n_qubits=4, n_layers=2, seed=0)
    result = model.train(X, y, epochs=8, lr=0.5, batch_size=8, verbose=False)
    assert result.final_loss < 0.8 * result.train_losses[0]
    assert model.accuracy(X, y) >= 0.7
