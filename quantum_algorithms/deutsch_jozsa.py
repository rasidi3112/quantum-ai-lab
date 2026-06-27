"""
Deutsch–Jozsa Algorithm
========================

The Deutsch–Jozsa algorithm (1992) is one of the first examples of a
provable quantum speed-up.  Given a Boolean function
``f : {0,1}^n → {0,1}`` promised to be either:

* **constant** – f(x) is the same for all x, or
* **balanced** – f(x) = 0 for exactly half the inputs and 1 for the rest,

the algorithm determines which case holds with **a single query** to the
quantum oracle, whereas any classical deterministic algorithm needs up to
``2^{n-1} + 1`` queries in the worst case.

Circuit
-------
::

    |0⟩^⊗n ─── H^⊗n ─── U_f ─── H^⊗n ─── Measure
    |1⟩    ─── H     ───     ───         ─── (discard)

If the measurement yields |0⟩^⊗n, f is constant; otherwise f is balanced.

Oracle encoding
---------------
The oracle acts on (n+1) qubits as:

.. math::

    U_f |x\\rangle|y\\rangle = |x\\rangle|y \\oplus f(x)\\rangle

When the ancilla is prepared in ``|−⟩``, this reduces to a phase kickback:

.. math::

    U_f |x\\rangle|{-}\\rangle = (-1)^{f(x)} |x\\rangle|{-}\\rangle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class DeutschJozsaResult:
    """Result of the Deutsch–Jozsa algorithm."""
    oracle_type: str = ""              # "constant" or "balanced"
    measurement_result: np.ndarray = field(default_factory=lambda: np.array([]))
    probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    n_qubits: int = 0
    is_zero_state: bool = False


class DeutschJozsa:
    """Statevector simulation of the Deutsch–Jozsa algorithm.

    Examples
    --------
    >>> dj = DeutschJozsa()
    >>> oracle = dj.balanced_oracle(3, pattern=5)
    >>> result = dj.run(oracle, 3)
    >>> print(result.oracle_type)
    balanced
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    # ----- oracle constructors ----------------------------------------------

    @staticmethod
    def constant_oracle(n_qubits: int, value: int = 0) -> np.ndarray:
        """Create a constant oracle (f(x) = value for all x).

        Parameters
        ----------
        n_qubits : int
            Number of input qubits (excluding the ancilla).
        value : int
            0 or 1.

        Returns
        -------
        np.ndarray
            Unitary matrix of shape ``(2N, 2N)`` with ``N = 2^n``,
            acting on (n + 1) qubits.
        """
        N = 2 ** n_qubits
        dim = 2 * N  # n+1 qubits
        oracle = np.eye(dim, dtype=complex)

        if value == 1:
            # f(x) = 1 for all x  →  flip ancilla for every |x⟩
            X_ancilla = np.array([[0, 1], [1, 0]], dtype=complex)
            oracle = np.kron(np.eye(N, dtype=complex), X_ancilla)

        return oracle

    @staticmethod
    def balanced_oracle(n_qubits: int, pattern: int) -> np.ndarray:
        """Create a balanced oracle defined by a bit pattern.

        ``f(x) = (x · pattern) mod 2`` (inner product mod 2) is balanced
        for any non-zero ``pattern``.

        Parameters
        ----------
        n_qubits : int
        pattern : int
            A non-zero integer in ``[1, 2^n - 1]`` defining the oracle.

        Returns
        -------
        np.ndarray
            Unitary ``(2N, 2N)`` matrix.
        """
        N = 2 ** n_qubits
        dim = 2 * N
        oracle = np.eye(dim, dtype=complex)

        for x in range(N):
            # f(x) = popcount(x & pattern) mod 2
            fx = bin(x & pattern).count("1") % 2
            if fx == 1:
                # Flip the ancilla qubit for input |x⟩
                # |x,0⟩ ↔ |x,1⟩
                idx0 = x * 2      # |x⟩ ⊗ |0⟩
                idx1 = x * 2 + 1  # |x⟩ ⊗ |1⟩
                oracle[idx0, idx0] = 0
                oracle[idx1, idx1] = 0
                oracle[idx0, idx1] = 1
                oracle[idx1, idx0] = 1

        return oracle

    def random_oracle(self, n_qubits: int,
                      oracle_type: Optional[str] = None
                      ) -> Tuple[np.ndarray, str]:
        """Generate a random constant or balanced oracle.

        Parameters
        ----------
        n_qubits : int
        oracle_type : str | None
            ``'constant'``, ``'balanced'``, or None (random choice).

        Returns
        -------
        (oracle, type_str) : tuple
        """
        if oracle_type is None:
            oracle_type = self.rng.choice(["constant", "balanced"])

        if oracle_type == "constant":
            value = int(self.rng.integers(0, 2))
            return self.constant_oracle(n_qubits, value), "constant"
        else:
            pattern = int(self.rng.integers(1, 2 ** n_qubits))
            return self.balanced_oracle(n_qubits, pattern), "balanced"

    # ----- run algorithm ----------------------------------------------------

    def run(self, oracle: np.ndarray, n_qubits: int) -> DeutschJozsaResult:
        """Run the Deutsch–Jozsa algorithm.

        Parameters
        ----------
        oracle : np.ndarray
            The ``(2N, 2N)`` oracle unitary (N = 2^n_qubits).
        n_qubits : int
            Number of input qubits.

        Returns
        -------
        DeutschJozsaResult
        """
        N = 2 ** n_qubits
        dim = 2 * N

        # Step 1: Prepare |0⟩^⊗n ⊗ |1⟩
        state = np.zeros(dim, dtype=complex)
        state[1] = 1.0  # |0...0⟩ ⊗ |1⟩

        # Step 2: Apply H^⊗(n+1)
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        H_full = np.array([[1]], dtype=complex)
        for _ in range(n_qubits + 1):
            H_full = np.kron(H_full, H)
        state = H_full @ state

        # Step 3: Apply oracle
        state = oracle @ state

        # Step 4: Apply H^⊗n to input register (not ancilla)
        H_input = np.array([[1]], dtype=complex)
        for _ in range(n_qubits):
            H_input = np.kron(H_input, H)
        H_with_ancilla = np.kron(H_input, np.eye(2, dtype=complex))
        state = H_with_ancilla @ state

        # Step 5: Measure input register
        # Compute probability of each input-register state (trace out ancilla)
        probs = np.zeros(N)
        for x in range(N):
            amp0 = state[x * 2]      # |x⟩|0⟩
            amp1 = state[x * 2 + 1]  # |x⟩|1⟩
            probs[x] = np.abs(amp0) ** 2 + np.abs(amp1) ** 2

        probs /= probs.sum()  # normalise

        # If P(|0⟩^⊗n) ≈ 1 → constant, otherwise → balanced
        is_zero = probs[0] > 0.5
        oracle_type = "constant" if is_zero else "balanced"

        return DeutschJozsaResult(
            oracle_type=oracle_type,
            measurement_result=probs,
            probabilities=probs,
            n_qubits=n_qubits,
            is_zero_state=is_zero,
        )

    # ----- classical verification -------------------------------------------

    @staticmethod
    def verify(oracle: np.ndarray, n_qubits: int) -> str:
        """Classically verify whether the oracle is constant or balanced.

        Queries the oracle on all ``2^n`` inputs by examining the diagonal
        action on computational-basis states.

        Parameters
        ----------
        oracle : np.ndarray
        n_qubits : int

        Returns
        -------
        str
            ``'constant'`` or ``'balanced'``.
        """
        N = 2 ** n_qubits

        # Evaluate f(x) for each x by checking if the oracle flips the ancilla
        f_values = []
        for x in range(N):
            # Prepare |x⟩|0⟩
            idx_in  = x * 2      # |x,0⟩
            idx_out = x * 2 + 1  # |x,1⟩
            # f(x) = 1 if oracle maps |x,0⟩ → |x,1⟩ component
            fx = 1 if np.abs(oracle[idx_out, idx_in]) > 0.5 else 0
            f_values.append(fx)

        unique = set(f_values)
        if len(unique) == 1:
            return "constant"
        count = sum(f_values)
        if count == N // 2:
            return "balanced"
        return f"unknown (ones={count}/{N})"
