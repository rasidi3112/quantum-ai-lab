"""Hybrid classical–quantum image classifier.

``x -> ReLU(W1 x + b1) -> tanh(W2 h + b2) -> VQC probabilities ->
softmax(W_out p + b_out)``.

Every layer is trained end to end. Gradients through the quantum circuit,
with respect to both its parameters and its inputs, use the exact
parameter-shift rule, and the classical layers use ordinary
backpropagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClassifierResult:
    """Training history of :class:`HybridClassifier`.

    Attributes
    ----------
    train_losses : list of float
        Mean cross-entropy per epoch.
    train_accuracies : list of float
        Training accuracy per epoch (measured during the epoch).
    final_loss : float
    final_accuracy : float
    """

    train_losses: list[float] = field(default_factory=list)
    train_accuracies: list[float] = field(default_factory=list)
    final_loss: float = 0.0
    final_accuracy: float = 0.0


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _cross_entropy(probs: np.ndarray, label: int) -> float:
    return float(-np.log(np.clip(probs[label], 1e-15, 1.0)))


_I2 = np.eye(2, dtype=np.complex128)


def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(phi: float) -> np.ndarray:
    return np.diag([np.exp(-1j * phi / 2), np.exp(1j * phi / 2)]).astype(np.complex128)


def _kron_chain(*ops: np.ndarray) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def _apply_single_gate(state: np.ndarray, gate: np.ndarray,
                        qubit: int, n_qubits: int) -> np.ndarray:
    psi = state.reshape([2] * n_qubits)
    psi = np.tensordot(gate, psi, axes=([1], [qubit]))
    psi = np.moveaxis(psi, 0, qubit)
    return psi.reshape(-1)


def _apply_cnot(state: np.ndarray, control: int, target: int,
                n_qubits: int) -> np.ndarray:
    psi = state.reshape([2] * n_qubits).copy()
    idx: list[slice | int] = [slice(None)] * n_qubits
    idx[control] = 1
    target_ax = target - (1 if target > control else 0)
    psi[tuple(idx)] = np.flip(psi[tuple(idx)], axis=target_ax)
    return psi.reshape(-1)


def _parameter_shift_jacobian(x: np.ndarray, q_params: np.ndarray,
                              n_qubits: int, n_layers: int
                              ) -> tuple[np.ndarray, np.ndarray]:
    shift = np.pi / 2
    jac_params = np.zeros((len(q_params), 2 ** n_qubits))
    for k in range(len(q_params)):
        plus, minus = q_params.copy(), q_params.copy()
        plus[k] += shift
        minus[k] -= shift
        jac_params[k] = (_quantum_forward(x, plus, n_qubits, n_layers)
                         - _quantum_forward(x, minus, n_qubits, n_layers)) / 2

    jac_inputs = np.zeros((len(x), 2 ** n_qubits))
    for k in range(len(x)):
        plus, minus = x.copy(), x.copy()
        plus[k] += shift
        minus[k] -= shift
        jac_inputs[k] = (_quantum_forward(plus, q_params, n_qubits, n_layers)
                         - _quantum_forward(minus, q_params, n_qubits, n_layers)) / 2

    return jac_params, jac_inputs


def _quantum_forward(x: np.ndarray, q_params: np.ndarray,
                      n_qubits: int, n_layers: int) -> np.ndarray:
    dim = 2 ** n_qubits
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    for q in range(n_qubits):
        idx = q % len(x)
        state = _apply_single_gate(state, _ry(x[idx]), q, n_qubits)

    params_per_layer = 2 * n_qubits
    for layer in range(n_layers):
        start = layer * params_per_layer
        lp = q_params[start : start + params_per_layer]

        for q in range(n_qubits):
            state = _apply_single_gate(state, _ry(lp[q]), q, n_qubits)
            state = _apply_single_gate(state, _rz(lp[n_qubits + q]), q, n_qubits)

        for q in range(n_qubits - 1):
            state = _apply_cnot(state, q, q + 1, n_qubits)
        if n_qubits > 1:
            state = _apply_cnot(state, n_qubits - 1, 0, n_qubits)

    return np.abs(state) ** 2


class HybridClassifier:
    """MLP encoder, variational quantum circuit and linear softmax head.

    Parameters
    ----------
    input_dim : int, default 64
        Flattened input size.
    hidden_dim : int, default 16
        Hidden ReLU units.
    n_qubits : int, default 4
        Qubits of the variational circuit (its input size).
    n_layers : int, default 2
        Variational layers.
    n_classes : int, default 2
        Number of output classes.
    seed : int, optional
        Seed for initialization and shuffling.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 16,
        n_qubits: int = 4,
        n_layers: int = 2,
        n_classes: int = 2,
        seed: int | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)

        lim1 = np.sqrt(6.0 / (input_dim + hidden_dim))
        self.W1 = self.rng.uniform(-lim1, lim1, (hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim)

        lim2 = np.sqrt(6.0 / (hidden_dim + n_qubits))
        self.W2 = self.rng.uniform(-lim2, lim2, (n_qubits, hidden_dim))
        self.b2 = np.zeros(n_qubits)

        self.n_q_params = n_layers * 2 * n_qubits
        self.q_params = self.rng.normal(0, 0.1, size=self.n_q_params)

        dim_q = 2 ** n_qubits
        lim3 = np.sqrt(6.0 / (dim_q + n_classes))
        self.W_out = self.rng.uniform(-lim3, lim3, (n_classes, dim_q))
        self.b_out = np.zeros(n_classes)


    def _classical_forward(self, x: np.ndarray) -> np.ndarray:
        h = _relu(self.W1 @ x + self.b1)
        return np.tanh(self.W2 @ h + self.b2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Class probabilities for one flattened sample.

        Parameters
        ----------
        x : numpy.ndarray
            Input of length ``input_dim``.

        Returns
        -------
        numpy.ndarray
        """
        encoded = self._classical_forward(x)
        q_probs = _quantum_forward(
            encoded, self.q_params, self.n_qubits, self.n_layers
        )
        logits = self.W_out @ q_probs + self.b_out
        return _softmax(logits)


    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        lr: float = 0.01,
        batch_size: int = 16,
        verbose: bool = True,
    ) -> ClassifierResult:
        """Train all layers with mini-batch gradient descent.

        Parameters
        ----------
        X_train : numpy.ndarray
            Samples of shape ``(n, input_dim)``.
        y_train : numpy.ndarray
            Integer labels.
        epochs : int, default 50
            Passes over the data.
        lr : float, default 0.01
            Learning rate.
        batch_size : int, default 16
            Mini-batch size.
        verbose : bool, default True
            Print progress.

        Returns
        -------
        ClassifierResult
        """
        result = ClassifierResult()
        n_samples = len(X_train)

        if verbose:
            print(f"Training HybridClassifier: "
                  f"MLP({self.input_dim}→{self.hidden_dim}→{self.n_qubits}) + "
                  f"VQC({self.n_qubits}q×{self.n_layers}L) → {self.n_classes} classes")
            print(f"  Samples: {n_samples}, Epochs: {epochs}, "
                  f"lr={lr}, batch={batch_size}")

        for epoch in range(epochs):
            perm = self.rng.permutation(n_samples)
            epoch_loss = 0.0
            correct = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = perm[start:end]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                batch_loss, batch_correct = self._update_step(
                    X_batch, y_batch, lr
                )
                epoch_loss += batch_loss
                correct += batch_correct

            avg_loss = epoch_loss / n_samples
            accuracy = correct / n_samples
            result.train_losses.append(avg_loss)
            result.train_accuracies.append(accuracy)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  Epoch {epoch+1:4d}/{epochs} | "
                      f"Loss: {avg_loss:.4f} | Acc: {accuracy:.1%}")

        result.final_loss = result.train_losses[-1] if result.train_losses else 0.0
        result.final_accuracy = result.train_accuracies[-1] if result.train_accuracies else 0.0

        if verbose:
            print(f"  Training complete. Final acc: {result.final_accuracy:.1%}")

        return result

    def _update_step(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        lr: float,
    ) -> tuple[float, int]:
        batch_loss = 0.0
        correct = 0
        grads = {
            "W1": np.zeros_like(self.W1), "b1": np.zeros_like(self.b1),
            "W2": np.zeros_like(self.W2), "b2": np.zeros_like(self.b2),
            "q": np.zeros_like(self.q_params),
            "W_out": np.zeros_like(self.W_out), "b_out": np.zeros_like(self.b_out),
        }

        for x, y in zip(X_batch, y_batch):
            y = int(y)
            z1 = self.W1 @ x + self.b1
            h = _relu(z1)
            encoded = np.tanh(self.W2 @ h + self.b2)
            q_probs = _quantum_forward(encoded, self.q_params,
                                       self.n_qubits, self.n_layers)
            probs = _softmax(self.W_out @ q_probs + self.b_out)

            batch_loss += _cross_entropy(probs, y)
            correct += int(np.argmax(probs) == y)

            d_logits = probs.copy()
            d_logits[y] -= 1.0
            grads["W_out"] += np.outer(d_logits, q_probs)
            grads["b_out"] += d_logits

            d_q_probs = self.W_out.T @ d_logits
            jac_params, jac_inputs = _parameter_shift_jacobian(
                encoded, self.q_params, self.n_qubits, self.n_layers
            )
            grads["q"] += jac_params @ d_q_probs

            d_z2 = (jac_inputs @ d_q_probs) * (1.0 - encoded ** 2)
            grads["W2"] += np.outer(d_z2, h)
            grads["b2"] += d_z2

            d_z1 = (self.W2.T @ d_z2) * (z1 > 0)
            grads["W1"] += np.outer(d_z1, x)
            grads["b1"] += d_z1

        scale = lr / len(X_batch)
        self.W1 -= scale * grads["W1"]
        self.b1 -= scale * grads["b1"]
        self.W2 -= scale * grads["W2"]
        self.b2 -= scale * grads["b2"]
        self.q_params -= scale * grads["q"]
        self.W_out -= scale * grads["W_out"]
        self.b_out -= scale * grads["b_out"]

        return batch_loss, correct


    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted class of every sample."""
        return np.array([np.argmax(self.forward(x)) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities of every sample."""
        return np.array([self.forward(x) for x in X])

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fraction of correctly classified samples."""
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def __repr__(self) -> str:
        return (
            f"HybridClassifier(input={self.input_dim}, "
            f"hidden={self.hidden_dim}, "
            f"qubits={self.n_qubits}, layers={self.n_layers}, "
            f"classes={self.n_classes})"
        )
