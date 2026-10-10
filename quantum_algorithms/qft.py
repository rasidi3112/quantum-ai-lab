from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PhaseEstimationResult:
    estimated_phase: float
    n_precision_bits: int
    measurement_outcome: int
    probabilities: np.ndarray
    true_phase: Optional[float] = None


class QFT:

    @staticmethod
    def qft_matrix(n_qubits: int) -> np.ndarray:
        N = 2 ** n_qubits
        omega = np.exp(2j * np.pi / N)
        indices = np.arange(N)
        return np.power(omega, np.outer(indices, indices)) / np.sqrt(N)


    @staticmethod
    def iqft_matrix(n_qubits: int) -> np.ndarray:
        N = 2 ** n_qubits
        omega = np.exp(-2j * np.pi / N)
        indices = np.arange(N)
        return np.power(omega, np.outer(indices, indices)) / np.sqrt(N)


    def apply_qft(self, state: np.ndarray) -> np.ndarray:
        n = int(np.log2(len(state)))
        assert 2 ** n == len(state), "State length must be a power of 2."
        F = self.qft_matrix(n)
        return F @ state

    def apply_iqft(self, state: np.ndarray) -> np.ndarray:
        n = int(np.log2(len(state)))
        assert 2 ** n == len(state), "State length must be a power of 2."
        F_inv = self.iqft_matrix(n)
        return F_inv @ state


    @staticmethod
    def qft_single_qubit_gates(n_qubits: int
                                ) -> List[Tuple[str, ...]]:
        gates: List[Tuple] = []
        for i in range(n_qubits):
            gates.append(("H", i))
            for j in range(i + 1, n_qubits):
                angle = 2 * np.pi / (2 ** (j - i + 1))
                gates.append(("CP", j, i, angle))

        for i in range(n_qubits // 2):
            gates.append(("SWAP", i, n_qubits - 1 - i))

        return gates


    @staticmethod
    def _hadamard() -> np.ndarray:
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    @staticmethod
    def _controlled_phase(n_qubits: int, control: int, target: int,
                          angle: float) -> np.ndarray:
        N = 2 ** n_qubits
        gate = np.eye(N, dtype=complex)
        for state in range(N):
            bits = format(state, f"0{n_qubits}b")
            if bits[control] == "1" and bits[target] == "1":
                gate[state, state] = np.exp(1j * angle)
        return gate

    def qft_from_gates(self, n_qubits: int) -> np.ndarray:
        N = 2 ** n_qubits
        U = np.eye(N, dtype=complex)
        H = self._hadamard()

        for gate in self.qft_single_qubit_gates(n_qubits):
            if gate[0] == "H":
                qubit = gate[1]
                gate_matrix = np.array([[1]], dtype=complex)
                for q in range(n_qubits):
                    gate_matrix = np.kron(
                        gate_matrix, H if q == qubit else np.eye(2, dtype=complex)
                    )
                U = gate_matrix @ U

            elif gate[0] == "CP":
                control, target, angle = gate[1], gate[2], gate[3]
                cp = self._controlled_phase(n_qubits, control, target, angle)
                U = cp @ U

            elif gate[0] == "SWAP":
                q1, q2 = gate[1], gate[2]
                swap = np.eye(N, dtype=complex)
                for s in range(N):
                    bits = list(format(s, f"0{n_qubits}b"))
                    bits[q1], bits[q2] = bits[q2], bits[q1]
                    t = int("".join(bits), 2)
                    swap[t, s] = 1.0
                    swap[s, s] = 0.0 if t != s else swap[s, s]
                swap = np.zeros((N, N), dtype=complex)
                for s in range(N):
                    bits = list(format(s, f"0{n_qubits}b"))
                    bits[q1], bits[q2] = bits[q2], bits[q1]
                    t = int("".join(bits), 2)
                    swap[t, s] = 1.0
                U = swap @ U

        return U


    def phase_estimation(self, unitary: np.ndarray,
                         eigenstate: np.ndarray,
                         n_precision: int = 4,
                         seed: Optional[int] = None
                         ) -> PhaseEstimationResult:
        rng = np.random.default_rng(seed)
        M = len(eigenstate)
        n_eigen = int(np.log2(M))
        assert 2 ** n_eigen == M

        t = n_precision
        Q = 2 ** t

        full_dim = Q * M

        state = np.zeros(full_dim, dtype=complex)
        for j in range(Q):
            for m in range(M):
                state[j * M + m] = eigenstate[m] / np.sqrt(Q)

        for k in range(t):
            U_power = np.linalg.matrix_power(unitary, 2 ** k)
            new_state = np.zeros_like(state)
            for j in range(Q):
                if (j >> k) & 1:
                    eigen_part = state[j * M:(j + 1) * M]
                    new_state[j * M:(j + 1) * M] = U_power @ eigen_part
                else:
                    new_state[j * M:(j + 1) * M] = state[j * M:(j + 1) * M]
            state = new_state

        iqft = self.iqft_matrix(t)

        state_matrix = state.reshape(Q, M)
        state_matrix = iqft @ state_matrix
        state = state_matrix.flatten()

        probs = np.zeros(Q)
        for j in range(Q):
            probs[j] = np.sum(np.abs(state[j * M:(j + 1) * M]) ** 2)
        probs /= probs.sum()

        measured = int(rng.choice(Q, p=probs))
        phase = measured / Q

        return PhaseEstimationResult(
            estimated_phase=phase,
            n_precision_bits=t,
            measurement_outcome=measured,
            probabilities=probs,
        )
