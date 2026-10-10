from __future__ import annotations

from dataclasses import dataclass, field
from math import floor, pi, sqrt
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass
class GroverResult:
    n_qubits: int = 0
    target_states: List[int] = field(default_factory=list)
    n_iterations: int = 0
    probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    found_state: int = -1
    found_probability: float = 0.0
    success: bool = False


class GroverSearch:

    @staticmethod
    def oracle(target: Union[int, List[int]], n_qubits: int) -> np.ndarray:
        N = 2 ** n_qubits
        oracle_matrix = np.eye(N, dtype=complex)
        targets = [target] if isinstance(target, int) else target
        for t in targets:
            if t < 0 or t >= N:
                raise ValueError(f"Target {t} out of range [0, {N})")
            oracle_matrix[t, t] = -1.0
        return oracle_matrix


    @staticmethod
    def diffusion_operator(n_qubits: int) -> np.ndarray:
        N = 2 ** n_qubits
        s = np.ones(N, dtype=complex) / sqrt(N)
        return 2.0 * np.outer(s, s.conj()) - np.eye(N, dtype=complex)


    @staticmethod
    def optimal_iterations(N: int, M: int = 1) -> int:
        return max(1, floor(pi / 4 * sqrt(N / M)))


    def search(self, n_qubits: int, target: int,
               n_iterations: Optional[int] = None) -> GroverResult:
        N = 2 ** n_qubits
        if n_iterations is None:
            n_iterations = self.optimal_iterations(N, 1)

        state = np.ones(N, dtype=complex) / sqrt(N)

        O = self.oracle(target, n_qubits)
        D = self.diffusion_operator(n_qubits)

        for _ in range(n_iterations):
            state = O @ state
            state = D @ state

        probs = np.abs(state) ** 2
        found = int(np.argmax(probs))

        return GroverResult(
            n_qubits=n_qubits,
            target_states=[target],
            n_iterations=n_iterations,
            probabilities=probs,
            found_state=found,
            found_probability=float(probs[found]),
            success=(found == target),
        )


    def multi_target_search(self, n_qubits: int,
                            targets: List[int],
                            n_iterations: Optional[int] = None
                            ) -> GroverResult:
        N = 2 ** n_qubits
        M = len(targets)
        if n_iterations is None:
            n_iterations = self.optimal_iterations(N, M)

        state = np.ones(N, dtype=complex) / sqrt(N)
        O = self.oracle(targets, n_qubits)
        D = self.diffusion_operator(n_qubits)

        for _ in range(n_iterations):
            state = O @ state
            state = D @ state

        probs = np.abs(state) ** 2
        found = int(np.argmax(probs))

        return GroverResult(
            n_qubits=n_qubits,
            target_states=targets,
            n_iterations=n_iterations,
            probabilities=probs,
            found_state=found,
            found_probability=float(probs[found]),
            success=(found in targets),
        )


    def amplitude_amplification(self, initial_state: np.ndarray,
                                oracle_matrix: np.ndarray,
                                n_iter: int) -> np.ndarray:
        state = initial_state.copy().astype(complex)
        psi0 = initial_state.copy().astype(complex)
        D = 2.0 * np.outer(psi0, psi0.conj()) - np.eye(len(psi0), dtype=complex)

        for _ in range(n_iter):
            state = oracle_matrix @ state
            state = D @ state

        return state
