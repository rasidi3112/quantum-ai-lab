"""
Grover's Search Algorithm
=========================

Grover's algorithm (1996) provides a quadratic speed-up for unstructured
search.  Given an oracle *f : {0,…,N−1} → {0,1}* that marks *M* solutions,
the algorithm finds a marked item in **O(√(N/M))** queries, compared to
**O(N/M)** classically.

Algorithm outline
-----------------
1.  Initialise an *n*-qubit register in the uniform superposition
    ``|s⟩ = H^⊗n |0⟩^⊗n``.

2.  Repeat **⌊π/4 · √(N/M)⌋** times:
    a.  Apply the *oracle* operator ``O_f``, which flips the phase of
        every marked state: ``|x⟩ → (−1)^{f(x)} |x⟩``.
    b.  Apply the *diffusion* operator (inversion about the mean):
        ``D = 2|s⟩⟨s| − I``.

3.  Measure. The marked state(s) will be observed with high probability.

Complexity
----------
*  Oracle queries: ``O(√N)``  (vs ``O(N)`` classically)
*  Gate complexity: ``O(n √N)`` where ``n = log₂ N``

This module provides both single-target and multi-target search, plus
the more general *amplitude amplification* primitive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor, pi, sqrt
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass
class GroverResult:
    """Container for Grover search results."""
    n_qubits: int = 0
    target_states: List[int] = field(default_factory=list)
    n_iterations: int = 0
    probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    found_state: int = -1
    found_probability: float = 0.0
    success: bool = False


class GroverSearch:
    """Statevector simulation of Grover's quantum search algorithm.

    Examples
    --------
    >>> g = GroverSearch()
    >>> result = g.search(4, target=7)
    >>> print(f"Found |{result.found_state}⟩ with P={result.found_probability:.4f}")
    """


    @staticmethod
    def oracle(target: Union[int, List[int]], n_qubits: int) -> np.ndarray:
        """Construct the Grover oracle matrix O_f.

        The oracle flips the phase of each target (marked) state:

        .. math::

            O_f = I - 2 \\sum_{t \\in T} |t\\rangle\\langle t|

        Parameters
        ----------
        target : int or list of int
            Index/indices of the marked computational-basis state(s).
        n_qubits : int
            Number of qubits (search space size is ``2**n_qubits``).

        Returns
        -------
        np.ndarray, shape ``(N, N)`` with ``N = 2**n_qubits``
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
        """Construct the Grover diffusion operator D = 2|s⟩⟨s| − I.

        ``|s⟩ = H^⊗n|0⟩^⊗n`` is the uniform superposition.

        Parameters
        ----------
        n_qubits : int

        Returns
        -------
        np.ndarray, shape ``(N, N)``
        """
        N = 2 ** n_qubits
        s = np.ones(N, dtype=complex) / sqrt(N)
        return 2.0 * np.outer(s, s.conj()) - np.eye(N, dtype=complex)


    @staticmethod
    def optimal_iterations(N: int, M: int = 1) -> int:
        """Compute the optimal number of Grover iterations.

        .. math::

            k_{opt} = \\left\\lfloor \\frac{\\pi}{4} \\sqrt{N/M} \\right\\rfloor

        Parameters
        ----------
        N : int
            Search-space size.
        M : int
            Number of marked items.

        Returns
        -------
        int
        """
        return max(1, floor(pi / 4 * sqrt(N / M)))


    def search(self, n_qubits: int, target: int,
               n_iterations: Optional[int] = None) -> GroverResult:
        """Run Grover's algorithm to find a single target state.

        Parameters
        ----------
        n_qubits : int
        target : int
            Index of the marked state.
        n_iterations : int | None
            Grover iterations; auto-computed if None.

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
                            targets: List[int],
                            n_iterations: Optional[int] = None
                            ) -> GroverResult:
        """Run Grover's algorithm with multiple marked states.

        Parameters
        ----------
        n_qubits : int
        targets : list of int
            Indices of marked states.
        n_iterations : int | None
            Auto-computed if None (using M = len(targets)).

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
        """General amplitude amplification.

        Applies ``n_iter`` rounds of (D · O) to ``initial_state``, where
        D = 2|ψ₀⟩⟨ψ₀| − I is the reflection about the initial state.

        Parameters
        ----------
        initial_state : np.ndarray
            The starting state |ψ₀⟩.
        oracle_matrix : np.ndarray
            The oracle (phase-flip) unitary.
        n_iter : int
            Number of amplification rounds.

        Returns
        -------
        np.ndarray
            The amplified state vector.
        """
        state = initial_state.copy().astype(complex)
        psi0 = initial_state.copy().astype(complex)
        D = 2.0 * np.outer(psi0, psi0.conj()) - np.eye(len(psi0), dtype=complex)

        for _ in range(n_iter):
            state = oracle_matrix @ state
            state = D @ state

        return state
