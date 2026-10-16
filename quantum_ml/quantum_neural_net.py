"""Quantum neural network layer with exact gradients.

A configurable encoding (angle or amplitude) is followed by layers of
``RX``/``RY``/``RZ`` rotations and a CNOT ring. Gradients are computed with
the parameter-shift rule
``d<O>/d theta = (<O>(theta + pi/2) - <O>(theta - pi/2)) / 2``,
which is exact for rotation gates and works on real hardware.

References
----------
Mitarai, K. et al. "Quantum circuit learning." Phys. Rev. A 98, 032309 (2018).
Schuld, M. et al. "Evaluating analytic gradients on quantum hardware."
Phys. Rev. A 99, 032331 (2019).
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np


class QuantumNeuralNetwork:
    """Parameterized quantum circuit used as a neural-network layer.

    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits.
    n_layers : int, default 2
        Number of variational layers.
    encoding : {"angle", "amplitude"}, default "angle"
        How inputs are loaded into the circuit.
    measurement : {"expval", "probs"}, default "expval"
        Output ``<Z_q>`` per qubit, or all ``2^n`` basis probabilities.
    seed : int, optional
        Seed for the weight initialization.
    random_state : int, optional
        Deprecated alias of ``seed``.
    """

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
        seed: int | None = None,
        *,
        random_state: int | None = None,
    ) -> None:
        if random_state is not None:
            warnings.warn(
                "`random_state` is deprecated, use `seed` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            seed = random_state if seed is None else seed
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.measurement = measurement
        self.dim = 2 ** n_qubits
        self.rng = np.random.default_rng(seed)

        self.n_params = 3 * n_qubits * n_layers
        self.weights = self.rng.uniform(-np.pi, np.pi, self.n_params)


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


    def amplitude_encoding(self, x: np.ndarray) -> np.ndarray:
        """Load ``x`` into the amplitudes of the state.

        Parameters
        ----------
        x : numpy.ndarray
            Input; truncated or zero-padded to ``2^n`` entries and normalized.

        Returns
        -------
        numpy.ndarray
            Normalized state (``|0...0>`` if ``x`` is zero).
        """
        state = np.zeros(self.dim, dtype=complex)
        n = min(len(x), self.dim)
        state[:n] = x[:n]

        norm = np.linalg.norm(state)
        if norm > 1e-10:
            state /= norm
        else:
            state[0] = 1.0

        return state

    def angle_encoding(self, x: np.ndarray) -> np.ndarray:
        """Load ``x`` as ``RY(x_q)`` rotations, one feature per qubit (cyclic).

        Parameters
        ----------
        x : numpy.ndarray
            Input features.

        Returns
        -------
        numpy.ndarray
            Normalized state.
        """
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0

        for q in range(self.n_qubits):
            idx = q % len(x)
            state = self._apply_gate(self._ry(x[idx]), q, state)

        return state

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode ``x`` with the configured encoding."""
        if self.encoding == "amplitude":
            return self.amplitude_encoding(x)
        else:
            return self.angle_encoding(x)


    def _variational_layer(
        self, state: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        n = self.n_qubits

        for q in range(n):
            state = self._apply_gate(self._rx(params[3 * q]), q, state)
            state = self._apply_gate(self._ry(params[3 * q + 1]), q, state)
            state = self._apply_gate(self._rz(params[3 * q + 2]), q, state)

        for q in range(n - 1):
            state = self._apply_cnot(q, q + 1, state)
        if n > 1:
            state = self._apply_cnot(n - 1, 0, state)

        return state


    def _measure_expval(self, state: np.ndarray) -> np.ndarray:
        expectations = np.zeros(self.n_qubits)
        for q in range(self.n_qubits):
            z_full = self._kron_list(
                [self.Z if i == q else self.I for i in range(self.n_qubits)]
            )
            expectations[q] = np.real(np.conj(state) @ z_full @ state)
        return expectations

    def _measure_probs(self, state: np.ndarray) -> np.ndarray:
        return np.abs(state) ** 2


    def forward(
        self,
        x: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate the circuit.

        Parameters
        ----------
        x : numpy.ndarray
            Input features.
        weights : numpy.ndarray, optional
            Weights to use instead of :attr:`weights`.

        Returns
        -------
        numpy.ndarray
            ``<Z_q>`` for every qubit, or basis probabilities, depending on
            ``measurement``.
        """
        if weights is None:
            weights = self.weights

        state = self.encode(x)

        params_per_layer = 3 * self.n_qubits
        for layer in range(self.n_layers):
            start = layer * params_per_layer
            end = start + params_per_layer
            state = self._variational_layer(state, weights[start:end])

        if self.measurement == "expval":
            return self._measure_expval(state)
        else:
            return self._measure_probs(state)

    def gradient(
        self,
        x: np.ndarray,
        weights: np.ndarray | None = None,
        shift: float = np.pi / 2,
    ) -> np.ndarray:
        """Jacobian of the output with respect to the weights (parameter shift).

        Parameters
        ----------
        x : numpy.ndarray
            Input features.
        weights : numpy.ndarray, optional
            Weights to differentiate at; defaults to :attr:`weights`.
        shift : float, default pi/2
            Parameter shift; ``pi/2`` gives exact derivatives.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_params, output_size)``.
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
