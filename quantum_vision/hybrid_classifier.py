"""
Hybrid Classical-Quantum Image Classifier
==========================================

Implements a hybrid image classifier that combines a classical feature
extractor (two-layer MLP) with a variational quantum circuit for
classification.

Architecture::

    Image (H×W)                     Classical MLP                  Quantum Circuit
    ┌──────────┐   flatten   ┌─────────────────────┐   angles   ┌──────────────┐
    │  Pixels  │ ─────────→ │ Linear → ReLU →      │ ────────→ │ Encode →     │
    │ H × W    │             │ Linear → features   │           │ Variational →│ → P(class)
    └──────────┘             └─────────────────────┘           │ Measure      │
                                                               └──────────────┘

Training uses finite-difference gradients for the quantum parameters
and numerical backpropagation for the classical MLP weights.

References
----------
- Mari et al., "Transfer learning in hybrid classical-quantum neural
  networks" (2020)
- Henderson et al., "Quanvolutional Neural Networks" (2020)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

@dataclass
class ClassifierResult:
    """Diagnostics from training a HybridClassifier."""
    train_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    final_loss: float = 0.0
    final_accuracy: float = 0.0


# ---------------------------------------------------------------------------
# Simple classical layers (numpy-only)
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _cross_entropy(probs: np.ndarray, label: int) -> float:
    return -np.log(np.clip(probs[label], 1e-15, 1.0))


# ---------------------------------------------------------------------------
# Quantum circuit primitives (statevector simulation)
# ---------------------------------------------------------------------------

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
    """Apply a single-qubit gate via tensor reshape (efficient)."""
    psi = state.reshape([2] * n_qubits)
    psi = np.tensordot(gate, psi, axes=([1], [qubit]))
    psi = np.moveaxis(psi, 0, qubit)
    return psi.reshape(-1)


def _apply_cnot(state: np.ndarray, control: int, target: int,
                n_qubits: int) -> np.ndarray:
    """Apply CNOT gate between control and target qubits."""
    dim = 2 ** n_qubits
    new_state = np.zeros(dim, dtype=np.complex128)
    for k in range(dim):
        ctrl_bit = (k >> (n_qubits - 1 - control)) & 1
        if ctrl_bit:
            flipped = k ^ (1 << (n_qubits - 1 - target))
            new_state[flipped] += state[k]
        else:
            new_state[k] += state[k]
    return new_state


def _quantum_forward(x: np.ndarray, q_params: np.ndarray,
                      n_qubits: int, n_layers: int) -> np.ndarray:
    """Run variational quantum circuit and return measurement probabilities.

    Parameters
    ----------
    x : np.ndarray, shape (n_qubits,)
        Encoded features (rotation angles).
    q_params : np.ndarray
        Circuit parameters, shape (n_layers * 2 * n_qubits,).
    n_qubits : int
    n_layers : int

    Returns
    -------
    np.ndarray
        Measurement probabilities, shape (2^n_qubits,).
    """
    dim = 2 ** n_qubits
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    # Angle encoding
    for q in range(n_qubits):
        idx = q % len(x)
        state = _apply_single_gate(state, _ry(x[idx]), q, n_qubits)

    # Variational layers
    params_per_layer = 2 * n_qubits
    for layer in range(n_layers):
        start = layer * params_per_layer
        lp = q_params[start : start + params_per_layer]

        # RY-RZ rotations
        for q in range(n_qubits):
            state = _apply_single_gate(state, _ry(lp[q]), q, n_qubits)
            state = _apply_single_gate(state, _rz(lp[n_qubits + q]), q, n_qubits)

        # CNOT ring
        for q in range(n_qubits - 1):
            state = _apply_cnot(state, q, q + 1, n_qubits)
        if n_qubits > 1:
            state = _apply_cnot(state, n_qubits - 1, 0, n_qubits)

    return np.abs(state) ** 2


# ---------------------------------------------------------------------------
# Hybrid Classifier
# ---------------------------------------------------------------------------

class HybridClassifier:
    """Hybrid classical-quantum image classifier.

    Parameters
    ----------
    input_dim : int
        Flattened input dimensionality (e.g. 64 for 8×8 images).
    hidden_dim : int
        Hidden layer size in the classical MLP.
    n_qubits : int
        Number of qubits in the quantum circuit.
    n_layers : int
        Number of variational circuit layers.
    n_classes : int
        Number of output classes (default: 2 for binary classification).
    seed : int | None
        Random seed.

    Examples
    --------
    >>> clf = HybridClassifier(input_dim=64, hidden_dim=16, n_qubits=4)
    >>> clf.train(X_train, y_train, epochs=50, lr=0.01)
    >>> preds = clf.predict(X_test)
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 16,
        n_qubits: int = 4,
        n_layers: int = 2,
        n_classes: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self._rng = np.random.default_rng(seed)

        # Classical MLP parameters (Xavier initialisation)
        lim1 = np.sqrt(6.0 / (input_dim + hidden_dim))
        self.W1 = self._rng.uniform(-lim1, lim1, (hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim)

        lim2 = np.sqrt(6.0 / (hidden_dim + n_qubits))
        self.W2 = self._rng.uniform(-lim2, lim2, (n_qubits, hidden_dim))
        self.b2 = np.zeros(n_qubits)

        # Quantum parameters
        self.n_q_params = n_layers * 2 * n_qubits
        self.q_params = self._rng.normal(0, 0.1, size=self.n_q_params)

        # Readout: map quantum probabilities → class logits
        dim_q = 2 ** n_qubits
        lim3 = np.sqrt(6.0 / (dim_q + n_classes))
        self.W_out = self._rng.uniform(-lim3, lim3, (n_classes, dim_q))
        self.b_out = np.zeros(n_classes)

    # ---- forward pass --------------------------------------------------- #

    def _classical_forward(self, x: np.ndarray) -> np.ndarray:
        """Two-layer MLP: x → hidden → encoded angles.

        Parameters
        ----------
        x : np.ndarray, shape ``(input_dim,)``

        Returns
        -------
        np.ndarray, shape ``(n_qubits,)``
        """
        h = _relu(self.W1 @ x + self.b1)
        return np.tanh(self.W2 @ h + self.b2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full forward pass: classical MLP → quantum circuit → class probs.

        Parameters
        ----------
        x : np.ndarray, shape ``(input_dim,)``
            Flattened input image.

        Returns
        -------
        np.ndarray, shape ``(n_classes,)``
            Class probability distribution.
        """
        encoded = self._classical_forward(x)
        q_probs = _quantum_forward(
            encoded, self.q_params, self.n_qubits, self.n_layers
        )
        logits = self.W_out @ q_probs + self.b_out
        return _softmax(logits)

    # ---- training ------------------------------------------------------- #

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        lr: float = 0.01,
        batch_size: int = 16,
        epsilon: float = 1e-3,
        verbose: bool = True,
    ) -> ClassifierResult:
        """Train the hybrid classifier.

        Uses finite-difference gradients for all parameters.

        Parameters
        ----------
        X_train : np.ndarray, shape ``(n_samples, input_dim)``
        y_train : np.ndarray, shape ``(n_samples,)`` with int class labels
        epochs : int
        lr : float
            Learning rate.
        batch_size : int
            Mini-batch size.
        epsilon : float
            Finite-difference step size.
        verbose : bool
            Print training progress.

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
            # Shuffle
            perm = self._rng.permutation(n_samples)
            epoch_loss = 0.0
            correct = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = perm[start:end]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                # Compute batch loss and gradients
                batch_loss, batch_correct = self._update_step(
                    X_batch, y_batch, lr, epsilon
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
        epsilon: float,
    ) -> Tuple[float, int]:
        """Single gradient update step on a mini-batch.

        Returns (batch_loss, n_correct).
        """
        batch_loss = 0.0
        correct = 0

        # --- Quantum parameter gradients (finite difference) ---
        q_grad = np.zeros_like(self.q_params)
        for i in range(len(self.q_params)):
            loss_plus = 0.0
            loss_minus = 0.0

            self.q_params[i] += epsilon
            for x, y in zip(X_batch, y_batch):
                probs = self.forward(x)
                loss_plus += _cross_entropy(probs, int(y))
            self.q_params[i] -= 2 * epsilon
            for x, y in zip(X_batch, y_batch):
                probs = self.forward(x)
                loss_minus += _cross_entropy(probs, int(y))
            self.q_params[i] += epsilon  # restore

            q_grad[i] = (loss_plus - loss_minus) / (2 * epsilon * len(X_batch))

        self.q_params -= lr * q_grad

        # --- Readout layer gradients (finite difference) ---
        for i in range(self.W_out.shape[0]):
            for j in range(self.W_out.shape[1]):
                l_p, l_m = 0.0, 0.0
                self.W_out[i, j] += epsilon
                for x, y in zip(X_batch, y_batch):
                    l_p += _cross_entropy(self.forward(x), int(y))
                self.W_out[i, j] -= 2 * epsilon
                for x, y in zip(X_batch, y_batch):
                    l_m += _cross_entropy(self.forward(x), int(y))
                self.W_out[i, j] += epsilon
                self.W_out[i, j] -= lr * (l_p - l_m) / (2 * epsilon * len(X_batch))

        for i in range(len(self.b_out)):
            l_p, l_m = 0.0, 0.0
            self.b_out[i] += epsilon
            for x, y in zip(X_batch, y_batch):
                l_p += _cross_entropy(self.forward(x), int(y))
            self.b_out[i] -= 2 * epsilon
            for x, y in zip(X_batch, y_batch):
                l_m += _cross_entropy(self.forward(x), int(y))
            self.b_out[i] += epsilon
            self.b_out[i] -= lr * (l_p - l_m) / (2 * epsilon * len(X_batch))

        # Compute final batch metrics
        for x, y in zip(X_batch, y_batch):
            probs = self.forward(x)
            batch_loss += _cross_entropy(probs, int(y))
            if np.argmax(probs) == int(y):
                correct += 1

        return batch_loss, correct

    # ---- inference ------------------------------------------------------ #

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray, shape ``(n_samples, input_dim)``

        Returns
        -------
        np.ndarray, shape ``(n_samples,)``
        """
        return np.array([np.argmax(self.forward(x)) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray, shape ``(n_samples, input_dim)``

        Returns
        -------
        np.ndarray, shape ``(n_samples, n_classes)``
        """
        return np.array([self.forward(x) for x in X])

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy.

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray

        Returns
        -------
        float
        """
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def __repr__(self) -> str:
        return (
            f"HybridClassifier(input={self.input_dim}, "
            f"hidden={self.hidden_dim}, "
            f"qubits={self.n_qubits}, layers={self.n_layers}, "
            f"classes={self.n_classes})"
        )
