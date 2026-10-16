"""Grover search and amplitude amplification.

Grover's algorithm finds one of ``M`` marked items among ``N`` with
``O(sqrt(N / M))`` oracle queries, a quadratic speed-up over classical search.
Each iteration applies the phase oracle ``O_f`` followed by the diffusion
operator ``D = 2|s><s| - I``.

References
----------
Grover, L. K. "A fast quantum mechanical algorithm for database search."
STOC (1996).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor, pi, sqrt

import numpy as np


@dataclass
class GroverResult:
    """Outcome of a Grover search.

    Attributes
    ----------
    n_qubits : int
        Register size.
    target_states : list of int
        The marked basis states.
    n_iterations : int
        Number of Grover iterations applied.
    probabilities : numpy.ndarray
        Final measurement distribution over all ``2^n`` basis states.
    found_state : int
        Most likely basis state.
    found_probability : float
        Probability of ``found_state``.
    success : bool
        Whether ``found_state`` is a marked state.
    """

    n_qubits: int = 0
    target_states: list[int] = field(default_factory=list)
    n_iterations: int = 0
    probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    found_state: int = -1
    found_probability: float = 0.0
    success: bool = False


class GroverSearch:
    """Statevector simulation of Grover's search algorithm."""

    @staticmethod
    def oracle(target: int | list[int], n_qubits: int) -> np.ndarray:
        """Phase oracle that flips the sign of the marked states.

        Parameters
        ----------
        target : int or list of int
            Marked basis state(s).
        n_qubits : int
            Register size.

        Returns
        -------
        numpy.ndarray
            Diagonal unitary of shape ``(2^n, 2^n)``.

        Raises
        ------
        ValueError
            If a target lies outside ``[0, 2^n)``.
        """
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
        """Inversion about the mean, ``D = 2|s><s| - I``.

        Parameters
        ----------
        n_qubits : int
            Register size.

        Returns
        -------
        numpy.ndarray
            Unitary of shape ``(2^n, 2^n)``.
        """
        N = 2 ** n_qubits
        s = np.ones(N, dtype=complex) / sqrt(N)
        return 2.0 * np.outer(s, s.conj()) - np.eye(N, dtype=complex)


    @staticmethod
    def optimal_iterations(N: int, M: int = 1) -> int:
        """Optimal number of Grover iterations, ``floor(pi/4 * sqrt(N / M))``.

        Parameters
        ----------
        N : int
            Search-space size.
        M : int, default 1
            Number of marked items.

        Returns
        -------
        int
            Iteration count (at least 1).
        """
        return max(1, floor(pi / 4 * sqrt(N / M)))


    def search(self, n_qubits: int, target: int,
               n_iterations: int | None = None) -> GroverResult:
        """Search for a single marked item.

        Parameters
        ----------
        n_qubits : int
            Register size; the search space has ``2^n`` items.
        target : int
            The marked item.
        n_iterations : int, optional
            Number of Grover iterations; defaults to the optimum.

        Returns
        -------
        GroverResult
        """
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
                            targets: list[int],
                            n_iterations: int | None = None
                            ) -> GroverResult:
        """Search with several marked items.

        Parameters
        ----------
        n_qubits : int
            Register size.
        targets : list of int
            The marked items.
        n_iterations : int, optional
            Number of Grover iterations; defaults to the optimum for ``len(targets)``.

        Returns
        -------
        GroverResult
        """
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
        """Amplify the marked amplitude, starting from an arbitrary initial state.

        Applies ``(2|psi0><psi0| - I) O`` ``n_iter`` times, where ``psi0`` is the
        initial state.

        Parameters
        ----------
        initial_state : numpy.ndarray
            Normalized starting state ``|psi0>``.
        oracle_matrix : numpy.ndarray
            Unitary marking the good subspace.
        n_iter : int
            Number of iterations.

        Returns
        -------
        numpy.ndarray
            The amplified state.
        """
        state = initial_state.copy().astype(complex)
        psi0 = initial_state.copy().astype(complex)
        D = 2.0 * np.outer(psi0, psi0.conj()) - np.eye(len(psi0), dtype=complex)

        for _ in range(n_iter):
            state = oracle_matrix @ state
            state = D @ state

        return state
