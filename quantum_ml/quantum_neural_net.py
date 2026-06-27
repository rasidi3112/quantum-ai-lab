"""
Quantum Neural Network (QNN)
============================

Implements a parameterized quantum neural network layer that can serve
as a drop-in replacement for classical neural network layers.

Architecture:
    Input → Encoding → [Variational Layer]×L → Measurement → Output

Encoding strategies:
    - Amplitude encoding: |ψ⟩ = Σ xᵢ|i⟩ / ‖x‖
    - Angle encoding: RY(xᵢ) on qubit i

References:
    Farhi, E., & Neven, H. (2018). Classification with quantum neural networks
    on near term processors. arXiv:1802.06002.
"""

import numpy as np
from typing import Optional, Literal


class QuantumNeuralNetwork:
    """Quantum Neural Network layer for hybrid models.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers.
    encoding : str
        Data encoding strategy: 'amplitude' or 'angle'.
    measurement : str
        Measurement strategy: 'expval' (expectation values) or 'probs' (probabilities).
    """

    # Fundamental gates
    I = np.eye(2, dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        encoding: Literal["amplitude", "angle"] = "angle",
        measurement: Literal["expval", "probs"] = "expval",
        random_state: Optional[int] = None,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.measurement = measurement
        self.dim = 2 ** n_qubits
        self.rng = np.random.RandomState(random_state)

        # Parameters: 3 rotations per qubit per layer (RX, RY, RZ)
        self.n_params = 3 * n_qubits * n_layers
        self.weights = self.rng.uniform(-np.pi, np.pi, self.n_params)

    # ------------------------------------------------------------------
    # Gate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kron_list(matrices: list[np.ndarray]) -> np.ndarray:
        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        return result

    def _rx(self, theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

    def _ry(self, theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    def _rz(self, theta: float) -> np.ndarray:
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=complex)

    def _apply_gate(self, gate: np.ndarray, qubit: int, state: np.ndarray) -> np.ndarray:
        full_gate = self._kron_list(
            [gate if i == qubit else self.I for i in range(self.n_qubits)]
        )
        return full_gate @ state

    def _apply_cnot(self, control: int, target: int, state: np.ndarray) -> np.ndarray:
        n = self.n_qubits
        new_state = np.zeros_like(state)
        for k in range(self.dim):
            ctrl_bit = (k >> (n - 1 - control)) & 1
            if ctrl_bit:
                flipped = k ^ (1 << (n - 1 - target))
                new_state[flipped] += state[k]
            else:
                new_state[k] += state[k]
        return new_state

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def amplitude_encoding(self, x: np.ndarray) -> np.ndarray:
        """Encode data vector into quantum state amplitudes.

        |ψ⟩ = Σᵢ xᵢ|i⟩ / ‖x‖

        The input is zero-padded to 2^n_qubits if necessary.
        """
        state = np.zeros(self.dim, dtype=complex)
        n = min(len(x), self.dim)
        state[:n] = x[:n]

        # Normalize
        norm = np.linalg.norm(state)
        if norm > 1e-10:
            state /= norm
        else:
            state[0] = 1.0

        return state

    def angle_encoding(self, x: np.ndarray) -> np.ndarray:
        """Encode data as rotation angles: RY(xᵢ) applied to |0⟩ state.

        Each feature xᵢ is mapped to a rotation on qubit i.
        """
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0

        for q in range(self.n_qubits):
            idx = q % len(x)
            state = self._apply_gate(self._ry(x[idx]), q, state)

        return state

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input data using the configured strategy."""
        if self.encoding == "amplitude":
            return self.amplitude_encoding(x)
        else:
            return self.angle_encoding(x)

    # ------------------------------------------------------------------
    # Variational circuit
    # ------------------------------------------------------------------

    def _variational_layer(
        self, state: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        """Apply one variational layer: RX-RY-RZ rotations + CNOT entanglement.

        params shape: (3 * n_qubits,) → [rx0, ry0, rz0, rx1, ry1, rz1, ...]
        """
        n = self.n_qubits

        # Rotation sub-layer
        for q in range(n):
            state = self._apply_gate(self._rx(params[3 * q]), q, state)
            state = self._apply_gate(self._ry(params[3 * q + 1]), q, state)
            state = self._apply_gate(self._rz(params[3 * q + 2]), q, state)

        # Entanglement: circular CNOT chain
        for q in range(n - 1):
            state = self._apply_cnot(q, q + 1, state)
        if n > 1:
            state = self._apply_cnot(n - 1, 0, state)

        return state

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def _measure_expval(self, state: np.ndarray) -> np.ndarray:
        """Measure expectation value ⟨Zᵢ⟩ for each qubit."""
        expectations = np.zeros(self.n_qubits)
        for q in range(self.n_qubits):
            z_full = self._kron_list(
                [self.Z if i == q else self.I for i in range(self.n_qubits)]
            )
            expectations[q] = np.real(np.conj(state) @ z_full @ state)
        return expectations

    def _measure_probs(self, state: np.ndarray) -> np.ndarray:
        """Return measurement probabilities |⟨i|ψ⟩|² for all basis states."""
        return np.abs(state) ** 2

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Full forward pass: encode → variational layers → measure.

        Parameters
        ----------
        x : np.ndarray
            Input feature vector.
        weights : np.ndarray, optional
            Circuit weights. Uses self.weights if None.

        Returns
        -------
        np.ndarray
            Output vector (expectation values or probabilities).
        """
        if weights is None:
            weights = self.weights

        # Encode
        state = self.encode(x)

        # Variational layers
        params_per_layer = 3 * self.n_qubits
        for layer in range(self.n_layers):
            start = layer * params_per_layer
            end = start + params_per_layer
            state = self._variational_layer(state, weights[start:end])

        # Measure
        if self.measurement == "expval":
            return self._measure_expval(state)
        else:
            return self._measure_probs(state)

    def gradient(
        self,
        x: np.ndarray,
        weights: Optional[np.ndarray] = None,
        shift: float = np.pi / 2,
    ) -> np.ndarray:
        """Compute gradient via parameter-shift rule.

        ∂f/∂θᵢ = [f(θᵢ + π/2) - f(θᵢ - π/2)] / 2

        Returns
        -------
        np.ndarray
            Gradient of shape (n_params, n_outputs).
        """
        if weights is None:
            weights = self.weights.copy()

        output_size = (
            self.n_qubits if self.measurement == "expval" else self.dim
        )
        grad = np.zeros((len(weights), output_size))

        for i in range(len(weights)):
            w_plus = weights.copy()
            w_minus = weights.copy()
            w_plus[i] += shift
            w_minus[i] -= shift
            grad[i] = (self.forward(x, w_plus) - self.forward(x, w_minus)) / 2.0

        return grad

    def __repr__(self) -> str:
        return (
            f"QuantumNeuralNetwork(n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, encoding='{self.encoding}', "
            f"measurement='{self.measurement}', n_params={self.n_params})"
        )
