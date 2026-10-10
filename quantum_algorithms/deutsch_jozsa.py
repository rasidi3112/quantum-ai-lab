from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class DeutschJozsaResult:
    oracle_type: str = ""
    measurement_result: np.ndarray = field(default_factory=lambda: np.array([]))
    probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    n_qubits: int = 0
    is_zero_state: bool = False


class DeutschJozsa:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)


    @staticmethod
    def constant_oracle(n_qubits: int, value: int = 0) -> np.ndarray:
        N = 2 ** n_qubits
        dim = 2 * N
        oracle = np.eye(dim, dtype=complex)

        if value == 1:
            X_ancilla = np.array([[0, 1], [1, 0]], dtype=complex)
            oracle = np.kron(np.eye(N, dtype=complex), X_ancilla)

        return oracle

    @staticmethod
    def balanced_oracle(n_qubits: int, pattern: int) -> np.ndarray:
        N = 2 ** n_qubits
        dim = 2 * N
        oracle = np.eye(dim, dtype=complex)

        for x in range(N):
            fx = bin(x & pattern).count("1") % 2
            if fx == 1:
                idx0 = x * 2
                idx1 = x * 2 + 1
                oracle[idx0, idx0] = 0
                oracle[idx1, idx1] = 0
                oracle[idx0, idx1] = 1
                oracle[idx1, idx0] = 1

        return oracle

    def random_oracle(self, n_qubits: int,
                      oracle_type: Optional[str] = None
                      ) -> Tuple[np.ndarray, str]:
        if oracle_type is None:
            oracle_type = self.rng.choice(["constant", "balanced"])

        if oracle_type == "constant":
            value = int(self.rng.integers(0, 2))
            return self.constant_oracle(n_qubits, value), "constant"
        else:
            pattern = int(self.rng.integers(1, 2 ** n_qubits))
            return self.balanced_oracle(n_qubits, pattern), "balanced"


    def run(self, oracle: np.ndarray, n_qubits: int) -> DeutschJozsaResult:
        N = 2 ** n_qubits
        dim = 2 * N

        state = np.zeros(dim, dtype=complex)
        state[1] = 1.0

        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        H_full = np.array([[1]], dtype=complex)
        for _ in range(n_qubits + 1):
            H_full = np.kron(H_full, H)
        state = H_full @ state

        state = oracle @ state

        H_input = np.array([[1]], dtype=complex)
        for _ in range(n_qubits):
            H_input = np.kron(H_input, H)
        H_with_ancilla = np.kron(H_input, np.eye(2, dtype=complex))
        state = H_with_ancilla @ state

        probs = np.zeros(N)
        for x in range(N):
            amp0 = state[x * 2]
            amp1 = state[x * 2 + 1]
            probs[x] = np.abs(amp0) ** 2 + np.abs(amp1) ** 2

        probs /= probs.sum()

        is_zero = probs[0] > 0.5
        oracle_type = "constant" if is_zero else "balanced"

        return DeutschJozsaResult(
            oracle_type=oracle_type,
            measurement_result=probs,
            probabilities=probs,
            n_qubits=n_qubits,
            is_zero_state=is_zero,
        )


    @staticmethod
    def verify(oracle: np.ndarray, n_qubits: int) -> str:
        N = 2 ** n_qubits

        f_values = []
        for x in range(N):
            idx_in  = x * 2
            idx_out = x * 2 + 1
            fx = 1 if np.abs(oracle[idx_out, idx_in]) > 0.5 else 0
            f_values.append(fx)

        unique = set(f_values)
        if len(unique) == 1:
            return "constant"
        count = sum(f_values)
        if count == N // 2:
            return "balanced"
        return f"unknown (ones={count}/{N})"
