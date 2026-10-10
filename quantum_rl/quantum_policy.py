import numpy as np
from typing import Optional, Tuple, List


class QuantumPolicy:
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        n_actions: int = 2,
        seed: Optional[int] = None,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.dim = 2 ** n_qubits
        self._rng = np.random.default_rng(seed)

        self._I = np.eye(2, dtype=np.complex128)
        self._X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        self._CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=np.complex128)


    @staticmethod
    def _ry(theta: float) -> np.ndarray:
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    @staticmethod
    def _rz(phi: float) -> np.ndarray:
        return np.array([
            [np.exp(-1j * phi / 2), 0],
            [0, np.exp(1j * phi / 2)],
        ], dtype=np.complex128)

    def _apply_single_qubit_gate(
        self, state: np.ndarray, gate: np.ndarray, qubit: int
    ) -> np.ndarray:
        n = self.n_qubits
        psi = state.reshape([2] * n)
        psi = np.tensordot(gate, psi, axes=([1], [qubit]))
        psi = np.moveaxis(psi, 0, qubit)
        return psi.reshape(self.dim)

    def _apply_cnot(
        self, state: np.ndarray, control: int, target: int
    ) -> np.ndarray:
        n = self.n_qubits
        psi = state.reshape([2] * n)
        idx_0 = [slice(None)] * n
        idx_1 = [slice(None)] * n
        idx_0[control] = 0
        idx_1[control] = 1

        psi_1 = psi[tuple(idx_1)].copy()
        psi_1 = np.tensordot(self._X, psi_1, axes=([1], [target - (1 if target > control else 0)]))
        target_ax = target - (1 if target > control else 0)
        psi_1 = np.moveaxis(psi_1, 0, target_ax)
        psi[tuple(idx_1)] = psi_1
        return psi.reshape(self.dim)


    def encode_state(
        self, observation: np.ndarray, n_qubits: Optional[int] = None
    ) -> np.ndarray:
        if n_qubits is None:
            n_qubits = self.n_qubits

        state = np.zeros(self.dim, dtype=np.complex128)
        state[0] = 1.0

        for i in range(min(len(observation), n_qubits)):
            angle = np.arctan(observation[i])
            gate = self._ry(angle)
            state = self._apply_single_qubit_gate(state, gate, i)

        return state

    def variational_layer(
        self, state: np.ndarray, params: np.ndarray, n_qubits: Optional[int] = None
    ) -> np.ndarray:
        if n_qubits is None:
            n_qubits = self.n_qubits

        for i in range(n_qubits):
            ry_gate = self._ry(params[i])
            state = self._apply_single_qubit_gate(state, ry_gate, i)
            rz_gate = self._rz(params[n_qubits + i])
            state = self._apply_single_qubit_gate(state, rz_gate, i)

        if n_qubits > 1:
            for i in range(n_qubits):
                control = i
                target = (i + 1) % n_qubits
                state = self._apply_cnot(state, control, target)

        return state

    def forward(
        self, observation: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        state = self.encode_state(observation)

        params_per_layer = 2 * self.n_qubits
        for layer in range(self.n_layers):
            start = layer * params_per_layer
            end = start + params_per_layer
            layer_params = params[start:end]
            state = self.variational_layer(state, layer_params)

        probs = np.abs(state) ** 2

        action_probs = np.zeros(self.n_actions)
        states_per_action = self.dim // self.n_actions
        remainder = self.dim % self.n_actions

        idx = 0
        for a in range(self.n_actions):
            count = states_per_action + (1 if a < remainder else 0)
            action_probs[a] = np.sum(probs[idx : idx + count])
            idx += count

        action_probs = np.clip(action_probs, 1e-10, None)
        action_probs /= action_probs.sum()

        return action_probs

    def select_action(
        self, observation: np.ndarray, params: np.ndarray
    ) -> int:
        probs = self.forward(observation, params)
        action = self._rng.choice(self.n_actions, p=probs)
        return int(action)

    def policy_gradient(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        params: np.ndarray,
        epsilon: float = 1e-3,
    ) -> np.ndarray:
        n = len(params)
        grad = np.zeros(n)

        for i in range(n):
            params_plus = params.copy()
            params_plus[i] += epsilon

            params_minus = params.copy()
            params_minus[i] -= epsilon

            probs_plus = self.forward(observation, params_plus)
            probs_minus = self.forward(observation, params_minus)

            log_prob_plus = np.log(probs_plus[action] + 1e-15)
            log_prob_minus = np.log(probs_minus[action] + 1e-15)

            grad[i] = (log_prob_plus - log_prob_minus) / (2 * epsilon)

        return reward * grad

    @staticmethod
    def n_params(n_qubits: int, n_layers: int) -> int:
        return n_layers * 2 * n_qubits

    def init_params(self, scale: float = 0.1) -> np.ndarray:
        n = self.n_params(self.n_qubits, self.n_layers)
        return self._rng.normal(0, scale, size=n)
