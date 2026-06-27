"""
Variational Quantum Classifier (VQC)
=====================================

A parameterized quantum circuit classifier that learns to map classical
inputs to class labels through variational optimization.

Architecture:
    1. Data Encoding: angle-encode classical features as rotation angles
    2. Variational Layers: trainable RY-RZ rotations + CNOT entanglement
    3. Measurement: expectation value of Z operator → class prediction

The cost function is minimized using classical optimizers (COBYLA, Nelder-Mead)
with parameter-shift rule or finite differences for gradient estimation.

References:
    Schuld, M., et al. (2020). Circuit-centric quantum classifiers.
    Physical Review A, 101(3), 032308.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Optional, Callable


class VariationalClassifier:
    """Variational Quantum Classifier using parameterized quantum circuits.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    n_layers : int
        Number of variational layers.
    random_state : int, optional
        Seed for reproducibility.
    """

    # Fundamental gates
    I = np.eye(2, dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        random_state: Optional[int] = None,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim = 2 ** n_qubits
        self.rng = np.random.RandomState(random_state)

        # 2 params per qubit per layer (RY + RZ) + 1 bias
        self.n_params = 2 * n_qubits * n_layers + 1
        self.params = self.rng.uniform(-np.pi, np.pi, self.n_params)
        self.training_history: list[float] = []

    # ------------------------------------------------------------------
    # Gate construction
    # ------------------------------------------------------------------

    @staticmethod
    def _kron_list(matrices: list[np.ndarray]) -> np.ndarray:
        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        return result

    def _ry(self, theta: float) -> np.ndarray:
        """RY(θ) rotation gate."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    def _rz(self, theta: float) -> np.ndarray:
        """RZ(θ) rotation gate."""
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=complex)

    def _apply_single_gate(self, gate: np.ndarray, qubit: int) -> np.ndarray:
        ops = [self.I] * self.n_qubits
        ops[qubit] = gate
        return self._kron_list(ops)

    def _apply_cnot(self, control: int, target: int, state: np.ndarray) -> np.ndarray:
        """Apply CNOT between control and target qubits on statevector."""
        n = self.n_qubits
        new_state = np.zeros_like(state)
        for k in range(self.dim):
            ctrl_bit = (k >> (n - 1 - control)) & 1
            if ctrl_bit:
                # Flip target bit
                flipped = k ^ (1 << (n - 1 - target))
                new_state[flipped] += state[k]
            else:
                new_state[k] += state[k]
        return new_state

    # ------------------------------------------------------------------
    # Circuit execution
    # ------------------------------------------------------------------

    def _encode_data(self, x: np.ndarray) -> np.ndarray:
        """Angle-encode classical data into quantum state.

        Applies RY(arctan(xᵢ)) · RZ(xᵢ) to qubit i.
        """
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0  # |0...0⟩

        # Hadamard layer
        for q in range(self.n_qubits):
            gate = self._apply_single_gate(self.H, q)
            state = gate @ state

        # Encode features
        for q in range(self.n_qubits):
            idx = q % len(x)
            ry_gate = self._apply_single_gate(self._ry(np.arctan(x[idx])), q)
            rz_gate = self._apply_single_gate(self._rz(x[idx]), q)
            state = rz_gate @ ry_gate @ state

        return state

    def _variational_layer(self, state: np.ndarray, layer_params: np.ndarray) -> np.ndarray:
        """Apply one variational layer: RY-RZ rotations + CNOT entanglement.

        Parameters
        ----------
        state : np.ndarray
            Current state vector.
        layer_params : np.ndarray
            Parameters of shape (2 * n_qubits,): [ry_0, rz_0, ry_1, rz_1, ...].
        """
        n = self.n_qubits

        # Rotation sub-layer
        for q in range(n):
            ry = self._apply_single_gate(self._ry(layer_params[2 * q]), q)
            rz = self._apply_single_gate(self._rz(layer_params[2 * q + 1]), q)
            state = rz @ ry @ state

        # Entanglement sub-layer (linear connectivity)
        for q in range(n - 1):
            state = self._apply_cnot(q, q + 1, state)

        # Ring closure
        if n > 2:
            state = self._apply_cnot(n - 1, 0, state)

        return state

    def forward(self, x: np.ndarray, params: Optional[np.ndarray] = None) -> float:
        """Run the full VQC circuit and return expectation value of Z₀.

        Parameters
        ----------
        x : np.ndarray
            Input feature vector.
        params : np.ndarray, optional
            Circuit parameters. Uses self.params if None.

        Returns
        -------
        float
            Expectation value ⟨Z₀⟩ in [-1, 1].
        """
        if params is None:
            params = self.params

        state = self._encode_data(x)

        # Apply variational layers
        params_per_layer = 2 * self.n_qubits
        for layer in range(self.n_layers):
            start = layer * params_per_layer
            end = start + params_per_layer
            layer_params = params[start:end]
            state = self._variational_layer(state, layer_params)

        # Measure ⟨Z₀⟩
        z0 = self._apply_single_gate(self.Z, 0)
        expectation = np.real(np.conj(state) @ z0 @ state)

        # Apply bias
        bias = params[-1]
        return float(expectation + bias)

    def predict_proba(self, x: np.ndarray, params: Optional[np.ndarray] = None) -> float:
        """Return probability of class 1."""
        raw = self.forward(x, params)
        return 1.0 / (1.0 + np.exp(-2 * raw))  # sigmoid mapping

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _cost_function(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Binary cross-entropy loss over the dataset."""
        total_loss = 0.0
        for xi, yi in zip(X, y):
            p = self.predict_proba(xi, params)
            p = np.clip(p, 1e-7, 1 - 1e-7)
            total_loss -= yi * np.log(p) + (1 - yi) * np.log(1 - p)
        return total_loss / len(y)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.1,
        method: str = "COBYLA",
        verbose: bool = True,
    ) -> dict:
        """Train the classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,) with values in {0, 1}
        epochs : int
            Maximum number of optimizer iterations.
        lr : float
            Learning rate (used as step size for COBYLA).
        method : str
            Scipy optimization method: 'COBYLA', 'Nelder-Mead', 'Powell'.
        verbose : bool
            Print progress.

        Returns
        -------
        dict
            Training results including final loss and parameters.
        """
        self.training_history = []

        def callback(params):
            loss = self._cost_function(params, X, y)
            self.training_history.append(loss)
            if verbose and len(self.training_history) % 10 == 0:
                print(f"  Iter {len(self.training_history):4d} | Loss: {loss:.6f}")

        if verbose:
            print(f"Training VQC: {self.n_qubits} qubits, {self.n_layers} layers, "
                  f"{self.n_params} parameters")
            print(f"  Method: {method}, Max iterations: {epochs}")

        result = minimize(
            self._cost_function,
            self.params,
            args=(X, y),
            method=method,
            callback=callback,
            options={"maxiter": epochs, "rhobeg": lr},
        )

        self.params = result.x
        final_loss = self._cost_function(self.params, X, y)

        if verbose:
            print(f"  Training complete. Final loss: {final_loss:.6f}")

        return {
            "loss": final_loss,
            "params": self.params.copy(),
            "n_iterations": len(self.training_history),
            "history": self.training_history,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predicted labels in {0, 1}.
        """
        predictions = []
        for x in X:
            p = self.predict_proba(x)
            predictions.append(1 if p >= 0.5 else 0)
        return np.array(predictions)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        preds = self.predict(X)
        return float(np.mean(preds == y))
