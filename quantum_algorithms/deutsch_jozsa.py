"""Deutsch–Jozsa algorithm.

Decides with a single oracle query whether a Boolean function
``f: {0,1}^n -> {0,1}`` is constant or balanced, a problem that needs
``2^(n-1) + 1`` queries deterministically on a classical computer.

The oracle acts as ``|x>|y> -> |x>|y XOR f(x)>`` on ``n`` input qubits plus one
ancilla (the least-significant qubit).

References
----------
Deutsch, D. & Jozsa, R. "Rapid solution of problems by quantum computation."
Proc. R. Soc. Lond. A 439, 553 (1992).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DeutschJozsaResult:
    """Outcome of a Deutsch–Jozsa run.

    Attributes
    ----------
    oracle_type : str
        ``"constant"`` or ``"balanced"`` as decided by the measurement.
    measurement_result : numpy.ndarray
        Probability of each input-register outcome.
    probabilities : numpy.ndarray
        Same as ``measurement_result``; kept for backwards compatibility.
    n_qubits : int
        Number of input qubits.
    is_zero_state : bool
        Whether ``|0...0>`` was the most likely outcome.
    """

    oracle_type: str = ""
    measurement_result: np.ndarray = field(default_factory=lambda: np.array([]))
    probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    n_qubits: int = 0
    is_zero_state: bool = False


class DeutschJozsa:
    """Statevector simulation of the Deutsch–Jozsa algorithm.

    Parameters
    ----------
    seed : int, optional
        Seed for the random oracle generator.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)


    @staticmethod
    def constant_oracle(n_qubits: int, value: int = 0) -> np.ndarray:
        """Build the oracle of a constant function.

        Parameters
        ----------
        n_qubits : int
            Number of input qubits.
        value : {0, 1}
            Constant output of ``f``.

        Returns
        -------
        numpy.ndarray
            Unitary of shape ``(2^(n+1), 2^(n+1))``.
        """
        N = 2 ** n_qubits
        dim = 2 * N
        oracle = np.eye(dim, dtype=complex)

        if value == 1:
            X_ancilla = np.array([[0, 1], [1, 0]], dtype=complex)
            oracle = np.kron(np.eye(N, dtype=complex), X_ancilla)

        return oracle

    @staticmethod
    def balanced_oracle(n_qubits: int, pattern: int) -> np.ndarray:
        """Build the oracle of the balanced function ``f(x) = popcount(x & pattern) mod 2``.

        Parameters
        ----------
        n_qubits : int
            Number of input qubits.
        pattern : int
            Non-zero bit mask selecting which input bits are XOR-ed together.

        Returns
        -------
        numpy.ndarray
            Unitary of shape ``(2^(n+1), 2^(n+1))``.
        """
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
                      oracle_type: str | None = None
                      ) -> tuple[np.ndarray, str]:
        """Draw a random constant or balanced oracle.

        Parameters
        ----------
        n_qubits : int
            Number of input qubits.
        oracle_type : {"constant", "balanced"}, optional
            Force the oracle type; chosen at random when omitted.

        Returns
        -------
        oracle : numpy.ndarray
            The oracle unitary.
        oracle_type : str
            The type that was generated.
        """
        if oracle_type is None:
            oracle_type = str(self.rng.choice(["constant", "balanced"]))

        if oracle_type == "constant":
            value = int(self.rng.integers(0, 2))
            return self.constant_oracle(n_qubits, value), "constant"
        else:
            pattern = int(self.rng.integers(1, 2 ** n_qubits))
            return self.balanced_oracle(n_qubits, pattern), "balanced"


    def run(self, oracle: np.ndarray, n_qubits: int) -> DeutschJozsaResult:
        """Run the algorithm against an oracle.

        Parameters
        ----------
        oracle : numpy.ndarray
            Oracle unitary on ``n_qubits + 1`` qubits.
        n_qubits : int
            Number of input qubits.

        Returns
        -------
        DeutschJozsaResult
            The measurement statistics and the constant/balanced verdict.
        """
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
        """Classify an oracle by reading off ``f`` directly (classical check).

        Parameters
        ----------
        oracle : numpy.ndarray
            Oracle unitary on ``n_qubits + 1`` qubits.
        n_qubits : int
            Number of input qubits.

        Returns
        -------
        str
            ``"constant"``, ``"balanced"`` or ``"unknown (...)"``.
        """
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
