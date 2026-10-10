import numpy as np
from scipy.optimize import minimize
from typing import Optional, Callable


class VariationalClassifier:
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

        self.n_params = 2 * n_qubits * n_layers + 1
        self.params = self.rng.uniform(-np.pi, np.pi, self.n_params)
        self.training_history: list[float] = []


    @staticmethod
    def _kron_list(matrices: list[np.ndarray]) -> np.ndarray:
        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        return result

    def _ry(self, theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    def _rz(self, theta: float) -> np.ndarray:
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=complex)

    def _apply_single_gate(self, gate: np.ndarray, qubit: int) -> np.ndarray:
        ops = [self.I] * self.n_qubits
        ops[qubit] = gate
        return self._kron_list(ops)

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


    def _encode_data(self, x: np.ndarray) -> np.ndarray:
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0

        for q in range(self.n_qubits):
            gate = self._apply_single_gate(self.H, q)
            state = gate @ state

        for q in range(self.n_qubits):
            idx = q % len(x)
            ry_gate = self._apply_single_gate(self._ry(np.arctan(x[idx])), q)
            rz_gate = self._apply_single_gate(self._rz(x[idx]), q)
            state = rz_gate @ ry_gate @ state

        return state

    def _variational_layer(self, state: np.ndarray, layer_params: np.ndarray) -> np.ndarray:
        n = self.n_qubits

        for q in range(n):
            ry = self._apply_single_gate(self._ry(layer_params[2 * q]), q)
            rz = self._apply_single_gate(self._rz(layer_params[2 * q + 1]), q)
            state = rz @ ry @ state

        for q in range(n - 1):
            state = self._apply_cnot(q, q + 1, state)

        if n > 2:
            state = self._apply_cnot(n - 1, 0, state)

        return state

    def forward(self, x: np.ndarray, params: Optional[np.ndarray] = None) -> float:
        if params is None:
            params = self.params

        state = self._encode_data(x)

        params_per_layer = 2 * self.n_qubits
        for layer in range(self.n_layers):
            start = layer * params_per_layer
            end = start + params_per_layer
            layer_params = params[start:end]
            state = self._variational_layer(state, layer_params)

        z0 = self._apply_single_gate(self.Z, 0)
        expectation = np.real(np.conj(state) @ z0 @ state)

        bias = params[-1]
        return float(expectation + bias)

    def predict_proba(self, x: np.ndarray, params: Optional[np.ndarray] = None) -> float:
        raw = self.forward(x, params)
        return 1.0 / (1.0 + np.exp(-2 * raw))


    def _cost_function(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
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
        predictions = []
        for x in X:
            p = self.predict_proba(x)
            predictions.append(1 if p >= 0.5 else 0)
        return np.array(predictions)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y))
