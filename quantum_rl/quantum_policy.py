"""Parameterized quantum circuit used as a stochastic policy.

Observations are angle-encoded with ``RY(2 arctan(x))`` and, with data
re-uploading, encoded again before every variational layer, which makes the
circuit a much more expressive function of its input. The action
distribution is read out by grouping basis-state probabilities.

References
----------
Pérez-Salinas, A. et al. "Data re-uploading for a universal quantum
classifier." Quantum 4, 226 (2020).
Jerbi, S. et al. "Parametrized quantum policies for reinforcement learning."
NeurIPS (2021).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class QuantumPolicy:
    """Quantum policy ``pi(a | s)`` over a discrete action space.

    All methods have batched versions that simulate a whole episode at once.

    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits.
    n_layers : int, default 2
        Number of variational layers.
    n_actions : int, default 2
        Number of discrete actions.
    seed : int, optional
        Seed for action sampling and parameter initialization.
    reupload : bool, default True
        Re-encode the observation before every layer.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        n_actions: int = 2,
        seed: int | None = None,
        reupload: bool = True,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.reupload = reupload
        self.dim = 2 ** n_qubits
        self.rng = np.random.default_rng(seed)

        counts = [self.dim // n_actions + (1 if a < self.dim % n_actions else 0)
                  for a in range(n_actions)]
        self._action_bounds = np.concatenate([[0], np.cumsum(counts)])


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
        psi = self._gate_batch(state.reshape([1] + [2] * self.n_qubits), gate, qubit)
        return psi.reshape(self.dim)

    def _apply_cnot(
        self, state: np.ndarray, control: int, target: int
    ) -> np.ndarray:
        psi = self._cnot_batch(state.reshape([1] + [2] * self.n_qubits), control, target)
        return psi.reshape(self.dim)

    def _gate_batch(self, psi: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
        ax = qubit + 1
        s0 = np.take(psi, 0, axis=ax)
        s1 = np.take(psi, 1, axis=ax)
        g: Any
        if gate.ndim == 3:
            shape = (-1,) + (1,) * (self.n_qubits - 1)
            g = [[gate[:, r, c].reshape(shape) for c in range(2)] for r in range(2)]
        else:
            g = gate
        return np.stack([g[0][0] * s0 + g[0][1] * s1,
                         g[1][0] * s0 + g[1][1] * s1], axis=ax)

    def _cnot_batch(self, psi: np.ndarray, control: int, target: int) -> np.ndarray:
        psi = psi.copy()
        idx: list[slice | int] = [slice(None)] * (self.n_qubits + 1)
        idx[control + 1] = 1
        target_ax = target + 1 - (1 if target > control else 0)
        psi[tuple(idx)] = np.flip(psi[tuple(idx)], axis=target_ax)
        return psi


    def encode_state(
        self, observation: np.ndarray, n_qubits: int | None = None
    ) -> np.ndarray:
        """Encode an observation on ``|0...0>``.

        Parameters
        ----------
        observation : numpy.ndarray
            Observation; one component per qubit.
        n_qubits : int, optional
            Encode only the first ``n_qubits`` components.

        Returns
        -------
        numpy.ndarray
        """
        psi = np.zeros([1] + [2] * self.n_qubits, dtype=np.complex128)
        psi.flat[0] = 1.0
        psi = self._encode_batch(psi, np.atleast_2d(observation), n_qubits)
        return psi.reshape(self.dim)

    def _encode_batch(
        self, psi: np.ndarray, observations: np.ndarray, n_qubits: int | None = None
    ) -> np.ndarray:
        if n_qubits is None:
            n_qubits = self.n_qubits
        for i in range(min(observations.shape[1], n_qubits)):
            angle = 2.0 * np.arctan(observations[:, i])
            c, s = np.cos(angle / 2), np.sin(angle / 2)
            ry = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
            psi = self._gate_batch(psi, ry.astype(np.complex128), i)
        return psi

    def variational_layer(
        self, state: np.ndarray, params: np.ndarray, n_qubits: int | None = None
    ) -> np.ndarray:
        """Apply one layer of ``RY``/``RZ`` rotations and a CNOT ring.

        Parameters
        ----------
        state : numpy.ndarray
            Input state.
        params : numpy.ndarray
            ``2 * n_qubits`` angles (all ``RY`` first, then all ``RZ``).
        n_qubits : int, optional
            Number of qubits the layer acts on.

        Returns
        -------
        numpy.ndarray
        """
        psi = self._layer_batch(state.reshape([1] + [2] * self.n_qubits), params, n_qubits)
        return psi.reshape(self.dim)

    def _layer_batch(
        self, psi: np.ndarray, params: np.ndarray, n_qubits: int | None = None
    ) -> np.ndarray:
        if n_qubits is None:
            n_qubits = self.n_qubits

        for i in range(n_qubits):
            psi = self._gate_batch(psi, self._ry(params[i]), i)
            psi = self._gate_batch(psi, self._rz(params[n_qubits + i]), i)

        if n_qubits > 1:
            for i in range(n_qubits):
                psi = self._cnot_batch(psi, i, (i + 1) % n_qubits)

        return psi

    def forward_batch(
        self, observations: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        """Compute action probabilities for a batch of observations.

        Parameters
        ----------
        observations : numpy.ndarray
            Array of shape ``(batch, obs_dim)``.
        params : numpy.ndarray
            Circuit parameters.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(batch, n_actions)``; each row sums to 1.
        """
        observations = np.atleast_2d(observations)
        psi = np.zeros([len(observations)] + [2] * self.n_qubits, dtype=np.complex128)
        psi.reshape(len(observations), -1)[:, 0] = 1.0
        psi = self._encode_batch(psi, observations)

        params_per_layer = 2 * self.n_qubits
        for layer in range(self.n_layers):
            if self.reupload and layer > 0:
                psi = self._encode_batch(psi, observations)
            start = layer * params_per_layer
            psi = self._layer_batch(psi, params[start:start + params_per_layer])

        probs = np.abs(psi.reshape(len(observations), self.dim)) ** 2
        action_probs = np.add.reduceat(probs, self._action_bounds[:-1], axis=1)
        action_probs = np.clip(action_probs, 1e-10, None)
        return action_probs / action_probs.sum(axis=1, keepdims=True)

    def forward(
        self, observation: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        """Compute action probabilities for one observation.

        Parameters
        ----------
        observation : numpy.ndarray
            Observation vector.
        params : numpy.ndarray
            Circuit parameters.

        Returns
        -------
        numpy.ndarray
        """
        return self.forward_batch(np.asarray(observation)[None, :], params)[0]

    def select_action(
        self, observation: np.ndarray, params: np.ndarray
    ) -> int:
        """Sample an action from ``pi(. | observation)``."""
        probs = self.forward(observation, params)
        action = self.rng.choice(self.n_actions, p=probs)
        return int(action)

    def policy_gradient(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        params: np.ndarray,
        epsilon: float = 1e-3,
    ) -> np.ndarray:
        """REINFORCE gradient ``reward * grad log pi(action | observation)``.

        Parameters
        ----------
        observation : numpy.ndarray
            Observation.
        action : int
            Action taken.
        reward : float
            Return or advantage weighting the gradient.
        params : numpy.ndarray
            Circuit parameters.
        epsilon : float, default 1e-3
            Central finite-difference step.

        Returns
        -------
        numpy.ndarray
        """
        return self.batch_policy_gradient(
            np.asarray(observation)[None, :], np.array([action]),
            np.array([reward]), params, epsilon,
        )

    def batch_policy_gradient(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        params: np.ndarray,
        epsilon: float = 1e-3,
    ) -> np.ndarray:
        """Mean REINFORCE gradient over a batch of steps.

        Computes ``mean_t A_t grad log pi(a_t | s_t)`` with central finite
        differences, simulating the whole batch per evaluation.

        Parameters
        ----------
        observations : numpy.ndarray
            Array of shape ``(T, obs_dim)``.
        actions : numpy.ndarray
            Actions taken.
        advantages : numpy.ndarray
            Advantage of each step.
        params : numpy.ndarray
            Circuit parameters.
        epsilon : float, default 1e-3
            Finite-difference step.

        Returns
        -------
        numpy.ndarray
        """
        rows = np.arange(len(actions))
        grad = np.zeros(len(params))

        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            params_minus = params.copy()
            params_minus[i] -= epsilon

            log_plus = np.log(self.forward_batch(observations, params_plus)[rows, actions] + 1e-15)
            log_minus = np.log(self.forward_batch(observations, params_minus)[rows, actions] + 1e-15)
            grad[i] = np.mean(advantages * (log_plus - log_minus)) / (2 * epsilon)

        return grad

    @staticmethod
    def n_params(n_qubits: int, n_layers: int) -> int:
        """Return the number of circuit parameters, ``2 * n_qubits * n_layers``."""
        return n_layers * 2 * n_qubits

    def init_params(self, scale: float = 0.1) -> np.ndarray:
        """Draw initial parameters from ``N(0, scale^2)``."""
        n = self.n_params(self.n_qubits, self.n_layers)
        return self.rng.normal(0, scale, size=n)
