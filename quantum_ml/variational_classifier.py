"""Variational quantum classifier (VQC).

Inputs are angle-encoded, processed by layers of trainable ``RY``/``RZ``
rotations and a ring of CNOTs, and read out as ``<Z_0>`` plus a trainable
bias. The circuit is trained with a gradient-free SciPy optimizer on the
binary cross-entropy.

References
----------
Schuld, M., Bocharov, A., Svore, K. & Wiebe, N. "Circuit-centric quantum
classifiers." Phys. Rev. A 101, 032308 (2020).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize


class VariationalClassifier:
    """Binary variational quantum classifier.

    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits; features are reused cyclically.
    n_layers : int, default 2
        Number of variational layers.
    seed : int, optional
        Seed for the parameter initialization.
    random_state : int, optional
        Deprecated alias of ``seed``.

    Attributes
    ----------
    params : numpy.ndarray
        ``2 * n_qubits * n_layers`` rotation angles followed by the bias.
    training_history : list of float
        Loss after every optimizer iteration of the last :meth:`train` call.
    """

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
        self.dim = 2 ** n_qubits
        self.rng = np.random.default_rng(seed)

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

    def forward(self, x: np.ndarray, params: np.ndarray | None = None) -> float:
        """Raw model output ``<Z_0> + bias``.

        Parameters
        ----------
        x : numpy.ndarray
            Feature vector.
        params : numpy.ndarray, optional
            Parameters to use instead of :attr:`params`.

        Returns
        -------
        float
        """
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

    def predict_proba(self, x: np.ndarray, params: np.ndarray | None = None) -> float:
        """Probability of class 1, ``sigmoid(2 * forward(x))``.

        Parameters
        ----------
        x : numpy.ndarray
            Feature vector.
        params : numpy.ndarray, optional
            Parameters to use instead of :attr:`params`.

        Returns
        -------
        float
        """
        raw = self.forward(x, params)
        return float(1.0 / (1.0 + np.exp(-2 * raw)))


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
        """Fit the parameters by minimizing the binary cross-entropy.

        Parameters
        ----------
        X : numpy.ndarray
            Training samples of shape ``(n, d)``.
        y : numpy.ndarray
            Labels in ``{0, 1}``.
        epochs : int, default 100
            Maximum optimizer iterations.
        lr : float, default 0.1
            Initial trust-region radius (``rhobeg``) for COBYLA.
        method : str, default "COBYLA"
            Any :func:`scipy.optimize.minimize` method.
        verbose : bool, default True
            Print progress.

        Returns
        -------
        dict
            ``loss``, ``params``, ``n_iterations`` and ``history``.
        """
        self.training_history = []

        def callback(params: np.ndarray) -> None:
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
        """Predict labels in ``{0, 1}``.

        Parameters
        ----------
        X : numpy.ndarray
            Samples of shape ``(n, d)``.

        Returns
        -------
        numpy.ndarray
        """
        predictions = []
        for x in X:
            p = self.predict_proba(x)
            predictions.append(1 if p >= 0.5 else 0)
        return np.array(predictions)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fraction of correctly classified samples.

        Parameters
        ----------
        X : numpy.ndarray
            Samples.
        y : numpy.ndarray
            True labels.

        Returns
        -------
        float
        """
        preds = self.predict(X)
        return float(np.mean(preds == y))
