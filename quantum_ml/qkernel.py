"""Quantum kernel methods.

Data are embedded with a ZZ feature map (Havlíček et al.), ``|phi(x)> = U(x)|0>``,
and compared through the fidelity kernel ``K(x, y) = |<phi(x)|phi(y)>|^2``.
The kernel matrix can be fed to any kernel method.

References
----------
Havlíček, V. et al. "Supervised learning with quantum-enhanced feature
spaces." Nature 567, 209 (2019).
"""

from __future__ import annotations

import numpy as np


class QuantumKernel:
    """Fidelity kernel from a ZZ feature map.

    Each layer applies Hadamards, ``RZ(x_i)`` on every qubit and
    ``exp(i (pi - x_i)(pi - x_j) Z_i Z_j)`` on every pair. Features are reused
    cyclically when there are more qubits than features.

    Parameters
    ----------
    n_qubits : int, default 2
        Number of qubits.
    n_layers : int, default 2
        Feature-map repetitions.
    """

    I = np.eye(2, dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    def __init__(self, n_qubits: int = 2, n_layers: int = 2) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim = 2 ** n_qubits


    @staticmethod
    def _kron_list(matrices: list[np.ndarray]) -> np.ndarray:
        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        return result

    def _rz(self, theta: float) -> np.ndarray:
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=complex)

    def _apply_single_gate(self, gate: np.ndarray, qubit: int) -> np.ndarray:
        ops = [self.I] * self.n_qubits
        ops[qubit] = gate
        return self._kron_list(ops)

    def _zz_interaction(self, qubit_i: int, qubit_j: int, angle: float) -> np.ndarray:
        dim = self.dim
        op = np.eye(dim, dtype=complex)
        for k in range(dim):
            bit_i = (k >> (self.n_qubits - 1 - qubit_i)) & 1
            bit_j = (k >> (self.n_qubits - 1 - qubit_j)) & 1
            z_i = 1 - 2 * bit_i
            z_j = 1 - 2 * bit_j
            op[k, k] = np.exp(1j * angle * z_i * z_j)
        return op


    def _feature_map_circuit(self, x: np.ndarray) -> np.ndarray:
        n = self.n_qubits
        U = np.eye(self.dim, dtype=complex)

        for _ in range(self.n_layers):
            H_all = self._kron_list([self.H] * n)
            U = H_all @ U

            for i in range(n):
                idx = i % len(x)
                rz_full = self._apply_single_gate(self._rz(x[idx]), i)
                U = rz_full @ U

            for i in range(n):
                for j in range(i + 1, n):
                    idx_i = i % len(x)
                    idx_j = j % len(x)
                    angle = (np.pi - x[idx_i]) * (np.pi - x[idx_j])
                    zz = self._zz_interaction(i, j, angle)
                    U = zz @ U

        return U

    def feature_map(self, x: np.ndarray) -> np.ndarray:
        """Embed a sample as a quantum state.

        Parameters
        ----------
        x : numpy.ndarray
            Feature vector.

        Returns
        -------
        numpy.ndarray
            Normalized state of length ``2^n_qubits``.
        """
        U = self._feature_map_circuit(x)
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0
        return U @ state


    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Kernel value ``|<phi(x1)|phi(x2)>|^2``.

        Parameters
        ----------
        x1, x2 : numpy.ndarray
            Feature vectors.

        Returns
        -------
        float
            Value in ``[0, 1]``.
        """
        state1 = self.feature_map(x1)
        state2 = self.feature_map(x2)
        overlap = np.abs(np.vdot(state1, state2)) ** 2
        return float(overlap)

    def kernel_matrix(self, X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        """Gram matrix between two sets of samples.

        Parameters
        ----------
        X1 : numpy.ndarray
            Samples of shape ``(n1, d)``.
        X2 : numpy.ndarray, optional
            Samples of shape ``(n2, d)``; defaults to ``X1``, in which case symmetry
            is used to halve the work.

        Returns
        -------
        numpy.ndarray
            Matrix of shape ``(n1, n2)``.
        """
        if X2 is None:
            X2 = X1
            symmetric = True
        else:
            symmetric = False

        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))

        for i in range(n1):
            j_start = i if symmetric else 0
            for j in range(j_start, n2):
                K[i, j] = self.evaluate(X1[i], X2[j])
                if symmetric and i != j:
                    K[j, i] = K[i, j]

        return K

    def classify_svm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        alpha: float = 1.0,
    ) -> np.ndarray:
        """Classify with a regularized least-squares kernel classifier.

        Solves ``(K + alpha I) w = y`` on the training set and predicts
        ``sign(K_test w)``. This is kernel ridge classification (a least-squares
        SVM), not a margin-maximizing SVM.

        Parameters
        ----------
        X_train : numpy.ndarray
            Training samples.
        y_train : numpy.ndarray
            Labels in ``{-1, +1}``.
        X_test : numpy.ndarray
            Samples to classify.
        alpha : float, default 1.0
            Ridge regularization strength.

        Returns
        -------
        numpy.ndarray
            Predicted labels in ``{-1, 0, +1}`` (0 only on an exact tie).
        """
        K_train = self.kernel_matrix(X_train)
        K_test = self.kernel_matrix(X_test, X_train)

        n = len(y_train)
        weights = np.linalg.solve(K_train + alpha * np.eye(n), y_train)
        predictions = K_test @ weights

        return np.sign(predictions)
