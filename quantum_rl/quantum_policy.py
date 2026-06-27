"""
Quantum Policy Network
=======================

Implements a variational quantum circuit (VQC) as a policy function for
reinforcement learning. The circuit uses angle encoding for state input
and parameterized rotation + entanglement layers for processing.

Architecture:
    |0⟩ ─── RY(x₀) ─── [RY(θ₀)─RZ(φ₀)] ─── ●─── ... ─── Measure
    |0⟩ ─── RY(x₁) ─── [RY(θ₁)─RZ(φ₁)] ─── X─── ... ─── Measure
    |0⟩ ─── RY(x₂) ─── [RY(θ₂)─RZ(φ₂)] ─── ●─── ... ─── Measure
    |0⟩ ─── RY(x₃) ─── [RY(θ₃)─RZ(φ₃)] ─── X─── ... ─── Measure

Each variational layer consists of:
    1. Single-qubit RY(θ) and RZ(φ) rotations on each qubit
    2. A ring of CNOT gates for entanglement: (0,1), (1,2), ..., (n-1,0)

The measurement probabilities of the computational basis states are used
to derive action probabilities for the RL agent.
"""

import numpy as np
from typing import Optional, Tuple, List


class QuantumPolicy:
    """
    Parameterized quantum circuit acting as a policy function.

    The quantum policy maps classical observations to action probabilities
    through a variational quantum circuit. The circuit parameters are
    optimized using policy gradient methods.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    n_layers : int
        Number of variational layers.
    n_actions : int
        Number of discrete actions available.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers.
    n_actions : int
        Number of actions.
    dim : int
        Hilbert space dimension (2^n_qubits).

    Examples
    --------
    >>> policy = QuantumPolicy(n_qubits=4, n_layers=2, n_actions=2)
    >>> params = np.random.randn(policy.n_params(4, 2)) * 0.1
    >>> obs = np.array([0.1, -0.2, 0.05, 0.3])
    >>> probs = policy.forward(obs, params)
    >>> action = policy.select_action(obs, params)
    """

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

        # Pre-compute gate matrices
        self._I = np.eye(2, dtype=np.complex128)
        self._X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        self._CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=np.complex128)

    # ------------------------------------------------------------------ #
    #  Primitive gate constructors                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ry(theta: float) -> np.ndarray:
        """
        Single-qubit RY rotation gate.

        RY(θ) = [[cos(θ/2), -sin(θ/2)],
                  [sin(θ/2),  cos(θ/2)]]

        Parameters
        ----------
        theta : float
            Rotation angle in radians.

        Returns
        -------
        np.ndarray
            2×2 unitary matrix.
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    @staticmethod
    def _rz(phi: float) -> np.ndarray:
        """
        Single-qubit RZ rotation gate.

        RZ(φ) = [[e^{-iφ/2}, 0],
                  [0, e^{iφ/2}]]

        Parameters
        ----------
        phi : float
            Rotation angle in radians.

        Returns
        -------
        np.ndarray
            2×2 unitary matrix.
        """
        return np.array([
            [np.exp(-1j * phi / 2), 0],
            [0, np.exp(1j * phi / 2)],
        ], dtype=np.complex128)

    def _apply_single_qubit_gate(
        self, state: np.ndarray, gate: np.ndarray, qubit: int
    ) -> np.ndarray:
        """
        Apply a single-qubit gate to a specific qubit in the statevector.

        Uses reshape-transpose-reshape trick for efficiency instead of
        building the full 2^n × 2^n tensor product.

        Parameters
        ----------
        state : np.ndarray
            Statevector of shape (2^n,).
        gate : np.ndarray
            2×2 gate matrix.
        qubit : int
            Target qubit index (0-indexed).

        Returns
        -------
        np.ndarray
            Updated statevector.
        """
        n = self.n_qubits
        # Reshape state into tensor of shape (2, 2, ..., 2)
        psi = state.reshape([2] * n)
        # Apply gate on the target qubit axis
        psi = np.tensordot(gate, psi, axes=([1], [qubit]))
        # Move the new axis back to the correct position
        psi = np.moveaxis(psi, 0, qubit)
        return psi.reshape(self.dim)

    def _apply_cnot(
        self, state: np.ndarray, control: int, target: int
    ) -> np.ndarray:
        """
        Apply a CNOT gate between control and target qubits.

        CNOT flips the target qubit if the control qubit is |1⟩.

        Parameters
        ----------
        state : np.ndarray
            Statevector of shape (2^n,).
        control : int
            Control qubit index.
        target : int
            Target qubit index.

        Returns
        -------
        np.ndarray
            Updated statevector.
        """
        n = self.n_qubits
        psi = state.reshape([2] * n)
        # Extract the part where control qubit = 1
        idx_0 = [slice(None)] * n
        idx_1 = [slice(None)] * n
        idx_0[control] = 0
        idx_1[control] = 1

        # Apply X gate to target qubit, conditioned on control = 1
        psi_1 = psi[tuple(idx_1)].copy()
        psi_1 = np.tensordot(self._X, psi_1, axes=([1], [target - (1 if target > control else 0)]))
        # Adjust axis position
        target_ax = target - (1 if target > control else 0)
        psi_1 = np.moveaxis(psi_1, 0, target_ax)
        psi[tuple(idx_1)] = psi_1
        return psi.reshape(self.dim)

    # ------------------------------------------------------------------ #
    #  Circuit building blocks                                            #
    # ------------------------------------------------------------------ #

    def encode_state(
        self, observation: np.ndarray, n_qubits: Optional[int] = None
    ) -> np.ndarray:
        """
        Encode a classical observation into a quantum state via angle encoding.

        Each observation feature is encoded as an RY rotation on a qubit:
            |ψ⟩ = ∏ᵢ RY(arctan(xᵢ)) |0...0⟩

        Parameters
        ----------
        observation : np.ndarray
            Classical observation vector of shape (obs_dim,).
        n_qubits : int, optional
            Number of qubits to use. Defaults to self.n_qubits.

        Returns
        -------
        np.ndarray
            Encoded quantum state of shape (2^n_qubits,).
        """
        if n_qubits is None:
            n_qubits = self.n_qubits

        # Start from |0...0⟩
        state = np.zeros(self.dim, dtype=np.complex128)
        state[0] = 1.0

        # Encode each observation feature as an RY rotation
        # Use arctan to bound the angle to (-π/2, π/2)
        for i in range(min(len(observation), n_qubits)):
            angle = np.arctan(observation[i])
            gate = self._ry(angle)
            state = self._apply_single_qubit_gate(state, gate, i)

        return state

    def variational_layer(
        self, state: np.ndarray, params: np.ndarray, n_qubits: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply one variational layer: RY-RZ rotations + CNOT entanglement ring.

        Layer structure:
            1. For each qubit i: RY(θ_i) followed by RZ(φ_i)
            2. CNOT ring: CNOT(0,1), CNOT(1,2), ..., CNOT(n-1,0)

        Parameters
        ----------
        state : np.ndarray
            Input quantum state of shape (2^n,).
        params : np.ndarray
            Layer parameters of shape (2 * n_qubits,).
            First n_qubits values are RY angles, next n_qubits are RZ angles.
        n_qubits : int, optional
            Number of qubits. Defaults to self.n_qubits.

        Returns
        -------
        np.ndarray
            Updated quantum state.
        """
        if n_qubits is None:
            n_qubits = self.n_qubits

        # Single-qubit rotations
        for i in range(n_qubits):
            ry_gate = self._ry(params[i])
            state = self._apply_single_qubit_gate(state, ry_gate, i)
            rz_gate = self._rz(params[n_qubits + i])
            state = self._apply_single_qubit_gate(state, rz_gate, i)

        # CNOT entanglement ring
        if n_qubits > 1:
            for i in range(n_qubits):
                control = i
                target = (i + 1) % n_qubits
                state = self._apply_cnot(state, control, target)

        return state

    def forward(
        self, observation: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        """
        Full quantum circuit execution: encoding → variational layers → measurement.

        The output is a probability distribution over computational basis states,
        reduced to action probabilities by grouping/summing.

        Parameters
        ----------
        observation : np.ndarray
            Classical observation vector.
        params : np.ndarray
            All variational parameters, shape (n_layers * 2 * n_qubits,).

        Returns
        -------
        np.ndarray
            Action probability distribution of shape (n_actions,).
        """
        # Encode classical observation
        state = self.encode_state(observation)

        # Apply variational layers
        params_per_layer = 2 * self.n_qubits
        for layer in range(self.n_layers):
            start = layer * params_per_layer
            end = start + params_per_layer
            layer_params = params[start:end]
            state = self.variational_layer(state, layer_params)

        # Measurement probabilities
        probs = np.abs(state) ** 2

        # Map to action probabilities
        # Group basis states into n_actions bins
        action_probs = np.zeros(self.n_actions)
        states_per_action = self.dim // self.n_actions
        remainder = self.dim % self.n_actions

        idx = 0
        for a in range(self.n_actions):
            count = states_per_action + (1 if a < remainder else 0)
            action_probs[a] = np.sum(probs[idx : idx + count])
            idx += count

        # Ensure valid probability distribution
        action_probs = np.clip(action_probs, 1e-10, None)
        action_probs /= action_probs.sum()

        return action_probs

    def select_action(
        self, observation: np.ndarray, params: np.ndarray
    ) -> int:
        """
        Sample an action from the policy's output distribution.

        Parameters
        ----------
        observation : np.ndarray
            Classical observation vector.
        params : np.ndarray
            Variational circuit parameters.

        Returns
        -------
        int
            Selected action index.
        """
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
        """
        Compute policy gradient via finite-difference approximation.

        For each parameter θᵢ:
            ∂log π(a|s;θ) / ∂θᵢ ≈ [log π(a|s;θ+εeᵢ) - log π(a|s;θ-εeᵢ)] / (2ε)

        The gradient of the expected return J(θ) is:
            ∇J(θ) = E[R · ∇log π(a|s;θ)]

        Parameters
        ----------
        observation : np.ndarray
            Classical observation.
        action : int
            Action taken.
        reward : float
            Reward signal (typically discounted return).
        params : np.ndarray
            Current circuit parameters.
        epsilon : float
            Finite-difference step size.

        Returns
        -------
        np.ndarray
            Gradient vector of shape (n_params,).
        """
        n = len(params)
        grad = np.zeros(n)

        for i in range(n):
            # θ + ε·eᵢ
            params_plus = params.copy()
            params_plus[i] += epsilon

            # θ - ε·eᵢ
            params_minus = params.copy()
            params_minus[i] -= epsilon

            # Evaluate log-probabilities
            probs_plus = self.forward(observation, params_plus)
            probs_minus = self.forward(observation, params_minus)

            log_prob_plus = np.log(probs_plus[action] + 1e-15)
            log_prob_minus = np.log(probs_minus[action] + 1e-15)

            # Central difference
            grad[i] = (log_prob_plus - log_prob_minus) / (2 * epsilon)

        # Scale by reward (REINFORCE: ∇J = R · ∇log π)
        return reward * grad

    @staticmethod
    def n_params(n_qubits: int, n_layers: int) -> int:
        """
        Compute total number of trainable parameters.

        Each layer has 2 parameters per qubit (RY + RZ angles).

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        n_layers : int
            Number of variational layers.

        Returns
        -------
        int
            Total parameter count = n_layers × 2 × n_qubits.
        """
        return n_layers * 2 * n_qubits

    def init_params(self, scale: float = 0.1) -> np.ndarray:
        """
        Initialize random circuit parameters.

        Parameters
        ----------
        scale : float
            Standard deviation of the normal distribution.

        Returns
        -------
        np.ndarray
            Random parameter vector.
        """
        n = self.n_params(self.n_qubits, self.n_layers)
        return self._rng.normal(0, scale, size=n)
