"""
Quantum Kernel Methods
======================

Implements quantum kernel functions for machine learning using ZZFeatureMap
encoding and statevector simulation.

The quantum kernel is defined as:
    K(x₁, x₂) = |⟨φ(x₁)|φ(x₂)⟩|²

where |φ(x)⟩ is the quantum feature map that encodes classical data into
a quantum Hilbert space.

References:
    Havlíček, V., et al. (2019). Supervised learning with quantum-enhanced
    feature spaces. Nature, 567, 209-212.
"""

import numpy as np
from typing import Optional


class QuantumKernel:
    """Quantum kernel using ZZFeatureMap encoding.

    The ZZFeatureMap encodes classical data x ∈ ℝⁿ into a quantum state
    using layers of Hadamard gates, single-qubit Z-rotations, and
    entangling ZZ interactions:

        U_φ(x) = exp(i Σᵢⱼ (π - xᵢ)(π - xⱼ) ZᵢZⱼ) · exp(i Σᵢ xᵢ Zᵢ) · H⊗ⁿ

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must match feature dimension).
    n_layers : int
        Number of repetitions of the feature map circuit.
    """

    # --- Fundamental gates ---------------------------------------------------
    I = np.eye(2, dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    def __init__(self, n_qubits: int = 2, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim = 2 ** n_qubits

    # ------------------------------------------------------------------
    # Gate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kron_list(matrices: list[np.ndarray]) -> np.ndarray:
        """Compute Kronecker product of a list of matrices."""
        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        return result

    def _rz(self, theta: float) -> np.ndarray:
        """Single-qubit RZ(θ) = exp(-iθZ/2)."""
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=complex)

    def _apply_single_gate(self, gate: np.ndarray, qubit: int) -> np.ndarray:
        """Construct full-system operator for a single-qubit gate on `qubit`."""
        ops = [self.I] * self.n_qubits
        ops[qubit] = gate
        return self._kron_list(ops)

    def _zz_interaction(self, qubit_i: int, qubit_j: int, angle: float) -> np.ndarray:
        """Construct exp(i·angle·ZᵢZⱼ) as a diagonal matrix."""
        dim = self.dim
        op = np.eye(dim, dtype=complex)
        for k in range(dim):
            # Compute eigenvalue of ZᵢZⱼ for computational basis state |k⟩
            bit_i = (k >> (self.n_qubits - 1 - qubit_i)) & 1
            bit_j = (k >> (self.n_qubits - 1 - qubit_j)) & 1
            z_i = 1 - 2 * bit_i  # eigenvalue of Z: +1 for |0⟩, -1 for |1⟩
            z_j = 1 - 2 * bit_j
            op[k, k] = np.exp(1j * angle * z_i * z_j)
        return op

    # ------------------------------------------------------------------
    # Feature map
    # ------------------------------------------------------------------

    def _feature_map_circuit(self, x: np.ndarray) -> np.ndarray:
        """Build the ZZFeatureMap unitary for data vector x.

        Circuit structure per layer:
            1. H⊗ⁿ (Hadamard on all qubits)
            2. RZ(xᵢ) on qubit i  (single-qubit encoding)
            3. For each pair (i,j): exp(i(π - xᵢ)(π - xⱼ) ZᵢZⱼ)  (entangling)
        """
        n = self.n_qubits
        # Start with identity
        U = np.eye(self.dim, dtype=complex)

        for _ in range(self.n_layers):
            # Hadamard layer
            H_all = self._kron_list([self.H] * n)
            U = H_all @ U

            # Single-qubit Z-rotation encoding
            for i in range(n):
                idx = i % len(x)
                rz_full = self._apply_single_gate(self._rz(x[idx]), i)
                U = rz_full @ U

            # ZZ entangling layer
            for i in range(n):
                for j in range(i + 1, n):
                    idx_i = i % len(x)
                    idx_j = j % len(x)
                    angle = (np.pi - x[idx_i]) * (np.pi - x[idx_j])
                    zz = self._zz_interaction(i, j, angle)
                    U = zz @ U

        return U

    def feature_map(self, x: np.ndarray) -> np.ndarray:
        """Map classical data x to quantum state |φ(x)⟩.

        Parameters
        ----------
        x : np.ndarray
            Classical feature vector of shape (n_features,).

        Returns
        -------
        np.ndarray
            Quantum state vector of shape (2^n_qubits,).
        """
        U = self._feature_map_circuit(x)
        # Apply to |0...0⟩
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0
        return U @ state

    # ------------------------------------------------------------------
    # Kernel computation
    # ------------------------------------------------------------------

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute quantum kernel value K(x₁, x₂) = |⟨φ(x₁)|φ(x₂)⟩|².

        Parameters
        ----------
        x1, x2 : np.ndarray
            Feature vectors.

        Returns
        -------
        float
            Kernel value in [0, 1].
        """
        state1 = self.feature_map(x1)
        state2 = self.feature_map(x2)
        overlap = np.abs(np.vdot(state1, state2)) ** 2
        return float(overlap)

    def kernel_matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the quantum kernel matrix.

        Parameters
        ----------
        X1 : np.ndarray
            Data matrix of shape (n_samples_1, n_features).
        X2 : np.ndarray, optional
            Second data matrix. If None, compute K(X1, X1).

        Returns
        -------
        np.ndarray
            Kernel matrix of shape (n_samples_1, n_samples_2).
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
        """Simple kernel-based classification using kernel ridge regression.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_train, n_features)
        y_train : np.ndarray of shape (n_train,) with values in {-1, +1}
        X_test : np.ndarray of shape (n_test, n_features)
        alpha : float
            Regularization parameter.

        Returns
        -------
        np.ndarray
            Predicted labels for X_test.
        """
        K_train = self.kernel_matrix(X_train)
        K_test = self.kernel_matrix(X_test, X_train)

        # Kernel ridge regression: w = (K + αI)⁻¹ y
        n = len(y_train)
        weights = np.linalg.solve(K_train + alpha * np.eye(n), y_train)
        predictions = K_test @ weights

        return np.sign(predictions)
