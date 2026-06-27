"""
Quantum Fourier Transform (QFT) & Phase Estimation
===================================================

The QFT is the quantum analogue of the discrete Fourier transform and sits
at the heart of many quantum algorithms (Shor's factoring, phase estimation,
quantum counting, etc.).

QFT definition
--------------
For an *n*-qubit register with ``N = 2^n`` basis states:

.. math::

    \\text{QFT} |j\\rangle = \\frac{1}{\\sqrt{N}}
    \\sum_{k=0}^{N-1} \\omega^{jk} |k\\rangle,
    \\quad \\omega = e^{2\\pi i / N}

Gate decomposition
------------------
The QFT can be decomposed into ``n(n-1)/2`` controlled-phase gates plus
``n`` Hadamard gates, giving an overall gate count of ``O(n²)`` — an
exponential improvement over the classical FFT's ``O(N log N)``.

Phase estimation
----------------
Given a unitary *U* with eigenstate ``|u⟩`` and unknown eigenvalue
``e^{2πiφ}``, quantum phase estimation uses *t* ancilla qubits to
approximate *φ* to *t* bits of precision with high probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PhaseEstimationResult:
    """Result of quantum phase estimation."""
    estimated_phase: float
    n_precision_bits: int
    measurement_outcome: int
    probabilities: np.ndarray
    true_phase: Optional[float] = None


class QFT:
    """Quantum Fourier Transform utilities and phase estimation.

    All operations are implemented as NumPy matrix algebra on complex
    statevectors.

    Examples
    --------
    >>> qft = QFT()
    >>> state = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
    >>> transformed = qft.apply_qft(state)
    >>> print(np.abs(transformed)**2)   # uniform distribution
    """

    # ----- QFT matrix -------------------------------------------------------

    @staticmethod
    def qft_matrix(n_qubits: int) -> np.ndarray:
        """Construct the n-qubit QFT unitary matrix.

        .. math::

            (\\text{QFT})_{jk} = \\frac{1}{\\sqrt{N}} e^{2\\pi i \\, jk / N}

        Parameters
        ----------
        n_qubits : int
            Number of qubits.

        Returns
        -------
        np.ndarray, shape ``(N, N)``
        """
        N = 2 ** n_qubits
        omega = np.exp(2j * np.pi / N)
        indices = np.arange(N)
        return np.power(omega, np.outer(indices, indices)) / np.sqrt(N)

    # ----- Inverse QFT matrix -----------------------------------------------

    @staticmethod
    def iqft_matrix(n_qubits: int) -> np.ndarray:
        """Construct the inverse QFT unitary matrix.

        ``QFT†`` is simply the conjugate transpose of QFT.

        Parameters
        ----------
        n_qubits : int

        Returns
        -------
        np.ndarray, shape ``(N, N)``
        """
        N = 2 ** n_qubits
        omega = np.exp(-2j * np.pi / N)
        indices = np.arange(N)
        return np.power(omega, np.outer(indices, indices)) / np.sqrt(N)

    # ----- apply to state ---------------------------------------------------

    def apply_qft(self, state: np.ndarray) -> np.ndarray:
        """Apply the QFT to a statevector.

        Parameters
        ----------
        state : np.ndarray
            Complex statevector of length ``2**n``.

        Returns
        -------
        np.ndarray
            Transformed statevector.
        """
        n = int(np.log2(len(state)))
        assert 2 ** n == len(state), "State length must be a power of 2."
        F = self.qft_matrix(n)
        return F @ state

    def apply_iqft(self, state: np.ndarray) -> np.ndarray:
        """Apply the inverse QFT to a statevector.

        Parameters
        ----------
        state : np.ndarray

        Returns
        -------
        np.ndarray
        """
        n = int(np.log2(len(state)))
        assert 2 ** n == len(state), "State length must be a power of 2."
        F_inv = self.iqft_matrix(n)
        return F_inv @ state

    # ----- gate decomposition -----------------------------------------------

    @staticmethod
    def qft_single_qubit_gates(n_qubits: int
                                ) -> List[Tuple[str, ...]]:
        """Decompose the QFT into Hadamard and controlled-phase gates.

        Returns a list of gate instructions, each a tuple:
        - ``('H', qubit)`` – Hadamard on *qubit*
        - ``('CP', control, target, angle)`` – controlled phase
        - ``('SWAP', q1, q2)`` – swap qubits

        Parameters
        ----------
        n_qubits : int

        Returns
        -------
        list of tuples
        """
        gates: List[Tuple] = []
        for i in range(n_qubits):
            gates.append(("H", i))
            for j in range(i + 1, n_qubits):
                angle = 2 * np.pi / (2 ** (j - i + 1))
                gates.append(("CP", j, i, angle))

        # Bit-reversal swaps
        for i in range(n_qubits // 2):
            gates.append(("SWAP", i, n_qubits - 1 - i))

        return gates

    # ----- build gate matrices from decomposition ---------------------------

    @staticmethod
    def _hadamard() -> np.ndarray:
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

    @staticmethod
    def _controlled_phase(n_qubits: int, control: int, target: int,
                          angle: float) -> np.ndarray:
        """Build the full n-qubit controlled-phase gate matrix."""
        N = 2 ** n_qubits
        gate = np.eye(N, dtype=complex)
        for state in range(N):
            bits = format(state, f"0{n_qubits}b")
            if bits[control] == "1" and bits[target] == "1":
                gate[state, state] = np.exp(1j * angle)
        return gate

    def qft_from_gates(self, n_qubits: int) -> np.ndarray:
        """Build the QFT unitary by composing individual gates.

        Useful for educational comparison with the direct matrix.

        Parameters
        ----------
        n_qubits : int

        Returns
        -------
        np.ndarray, shape ``(N, N)``
        """
        N = 2 ** n_qubits
        U = np.eye(N, dtype=complex)
        H = self._hadamard()

        for gate in self.qft_single_qubit_gates(n_qubits):
            if gate[0] == "H":
                qubit = gate[1]
                # Embed single-qubit gate into full space
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
                # Build proper SWAP matrix
                swap = np.zeros((N, N), dtype=complex)
                for s in range(N):
                    bits = list(format(s, f"0{n_qubits}b"))
                    bits[q1], bits[q2] = bits[q2], bits[q1]
                    t = int("".join(bits), 2)
                    swap[t, s] = 1.0
                U = swap @ U

        return U

    # ----- phase estimation -------------------------------------------------

    def phase_estimation(self, unitary: np.ndarray,
                         eigenstate: np.ndarray,
                         n_precision: int = 4,
                         seed: Optional[int] = None
                         ) -> PhaseEstimationResult:
        """Quantum phase estimation (QPE).

        Given a unitary *U* and one of its eigenstates ``|u⟩`` with
        ``U|u⟩ = e^{2πiφ}|u⟩``, estimate φ to *n_precision* bits.

        Algorithm:
        1. Prepare ``|0⟩^{⊗t} ⊗ |u⟩`` where t = n_precision.
        2. Apply ``H^{⊗t}`` to the ancilla register.
        3. Apply controlled-``U^{2^k}`` for each ancilla qubit k.
        4. Apply inverse QFT to the ancilla.
        5. Measure the ancilla → binary approximation of φ.

        Parameters
        ----------
        unitary : np.ndarray
            The unitary matrix (shape ``(M, M)``).
        eigenstate : np.ndarray
            An eigenstate of *unitary* (length ``M``).
        n_precision : int
            Number of precision qubits.
        seed : int | None
            For reproducible measurement.

        Returns
        -------
        PhaseEstimationResult
        """
        rng = np.random.default_rng(seed)
        M = len(eigenstate)
        n_eigen = int(np.log2(M))
        assert 2 ** n_eigen == M

        t = n_precision
        Q = 2 ** t

        # Full system: (counting register) ⊗ (eigenstate register)
        # Dimension = Q * M
        full_dim = Q * M

        # Step 1–2: |+⟩^⊗t ⊗ |u⟩
        state = np.zeros(full_dim, dtype=complex)
        for j in range(Q):
            for m in range(M):
                state[j * M + m] = eigenstate[m] / np.sqrt(Q)

        # Step 3: Controlled-U^{2^k}
        # For each counting qubit k, if that qubit is |1⟩, apply U^{2^k}
        # to the eigenstate register.
        for k in range(t):
            U_power = np.linalg.matrix_power(unitary, 2 ** k)
            new_state = np.zeros_like(state)
            for j in range(Q):
                # Check if bit k of j is set
                if (j >> k) & 1:
                    # Apply U^{2^k} to eigenstate part
                    eigen_part = state[j * M:(j + 1) * M]
                    new_state[j * M:(j + 1) * M] = U_power @ eigen_part
                else:
                    new_state[j * M:(j + 1) * M] = state[j * M:(j + 1) * M]
            state = new_state

        # Step 4: inverse QFT on counting register
        iqft = self.iqft_matrix(t)

        # Reshape state as (Q, M), apply iQFT on first axis
        state_matrix = state.reshape(Q, M)
        state_matrix = iqft @ state_matrix
        state = state_matrix.flatten()

        # Step 5: Measure counting register
        # Probability of each counting register outcome (trace out eigenstate)
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
