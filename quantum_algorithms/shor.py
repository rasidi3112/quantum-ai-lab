"""
Shor's Factoring Algorithm
===========================

Shor's algorithm (1994) factors an integer *N* in polynomial time on a
quantum computer, breaking RSA and similar cryptosystems.

The core quantum subroutine is **quantum period finding** (order finding):
given random *a < N* with ``gcd(a, N) = 1``, find the smallest positive
integer *r* such that ``a^r ≡ 1  (mod N)``.

Once *r* is known (and even), the factors emerge from:

.. math::

    \\gcd(a^{r/2} \\pm 1,\\; N)

Algorithm steps
---------------
1. Pick random ``a`` with ``1 < a < N``.
2. Compute ``gcd(a, N)``; if non-trivial, we're done.
3. Use the quantum period-finding circuit (QPE + modular exponentiation)
   to find the order ``r`` of ``a`` modulo ``N``.
4. If ``r`` is odd or ``a^{r/2} ≡ −1 (mod N)``, go to step 1.
5. Return ``gcd(a^{r/2} − 1, N)`` and ``gcd(a^{r/2} + 1, N)``.

Implementation notes
--------------------
This module provides a *classical simulation* of the quantum algorithm:
statevectors are manipulated via NumPy matrix operations.  For practical
reasons the qubit counts are kept small (N ≤ ~100), but the code faithfully
reproduces the QPE-based period-finding structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd, log2, ceil
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ShorResult:
    """Container for Shor's factoring results."""
    N: int = 0
    factors: Tuple[int, int] = (0, 0)
    a_used: int = 0
    period_found: int = 0
    success: bool = False
    attempts: int = 0
    details: List[str] = field(default_factory=list)


class ShorFactoring:
    """Classical simulation of Shor's quantum factoring algorithm.

    Parameters
    ----------
    seed : int | None
        Random seed for reproducibility.

    Examples
    --------
    >>> shor = ShorFactoring(seed=42)
    >>> result = shor.factor(15)
    >>> print(f"{result.N} = {result.factors[0]} × {result.factors[1]}")
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    # ----- classical order finding (for verification) -----------------------

    @staticmethod
    def classical_order_check(a: int, N: int) -> int:
        """Brute-force period finding: find smallest r with a^r ≡ 1 (mod N).

        Used to verify quantum results.

        Parameters
        ----------
        a : int
        N : int

        Returns
        -------
        int
            The order *r*, or -1 if not found within N steps.
        """
        if gcd(a, N) != 1:
            return -1
        r = 1
        current = a % N
        while current != 1 and r < N:
            current = (current * a) % N
            r += 1
        return r if current == 1 else -1

    # ----- QFT matrix -------------------------------------------------------

    @staticmethod
    def qft_matrix(n: int) -> np.ndarray:
        """Construct the n-qubit QFT unitary matrix.

        .. math::

            (\\text{QFT})_{jk} = \\frac{1}{\\sqrt{N}} \\omega^{jk},
            \\quad \\omega = e^{2\\pi i / N}

        Parameters
        ----------
        n : int
            Number of qubits.

        Returns
        -------
        np.ndarray, shape ``(N, N)`` with ``N = 2**n``
        """
        N = 2 ** n
        omega = np.exp(2j * np.pi / N)
        indices = np.arange(N)
        return np.power(omega, np.outer(indices, indices)) / np.sqrt(N)

    # ----- controlled modular exponentiation --------------------------------

    @staticmethod
    def controlled_modular_exp(a: int, power: int, N: int,
                               n_qubits: int) -> np.ndarray:
        """Construct the unitary for ``|x⟩ → |a^power · x  mod N⟩``.

        Acts on the computational-basis states in the range [0, N-1]
        and maps states ≥ N to themselves (identity padding).

        Parameters
        ----------
        a : int
            Base for modular exponentiation.
        power : int
            Exponent (typically ``2**k`` in QPE).
        N : int
            Modulus.
        n_qubits : int
            Number of qubits in the target register.

        Returns
        -------
        np.ndarray, shape ``(dim, dim)`` with ``dim = 2**n_qubits``
        """
        dim = 2 ** n_qubits
        U = np.zeros((dim, dim), dtype=complex)
        a_pow = pow(a, power, N)  # a^power mod N
        for x in range(dim):
            if x < N:
                y = (a_pow * x) % N
                U[y, x] = 1.0
            else:
                U[x, x] = 1.0  # identity for overflow states
        return U

    # ----- quantum period finding -------------------------------------------

    def quantum_period_finding(self, a: int, N: int) -> int:
        """Simulate QPE-based period finding for ``a mod N``.

        The simulation works as follows:

        1. Prepare the *counting register* (``n_count`` qubits) in uniform
           superposition and the *work register* in ``|1⟩``.
        2. Apply controlled-``U^{2^k}`` gates (modular exponentiation).
        3. Apply inverse QFT to the counting register.
        4. Measure and extract the period via continued fractions.

        Parameters
        ----------
        a : int
        N : int

        Returns
        -------
        int
            Estimated period *r*, or -1 on failure.
        """
        # Register sizes
        n_count = max(4, 2 * ceil(log2(N + 1)))  # precision qubits
        n_work  = max(2, ceil(log2(N + 1)))       # work qubits
        Q = 2 ** n_count
        dim_work = 2 ** n_work

        # ---- Simulate the circuit on the work register ----
        # Instead of building the full (n_count + n_work)-qubit state,
        # we compute the state of the counting register analytically.
        #
        # After the controlled modular exponentiations and tracing out
        # the work register, the counting register state is:
        #
        #   |ψ⟩ = (1/√Q) Σ_j  |j⟩  ⊗  U^j|1⟩
        #
        # Measuring the work register collapses the counting register
        # into a superposition over j values consistent with the observed
        # work state.  The inverse QFT then peaks at multiples of Q/r.

        # Build the state of the full system (counting ⊗ work)
        # For small N this is feasible.
        full_dim = Q * dim_work
        state = np.zeros(full_dim, dtype=complex)

        for j in range(Q):
            # |j⟩ ⊗ U^j|1⟩
            work_state = np.zeros(dim_work, dtype=complex)
            val = pow(a, j, N)
            if val < dim_work:
                work_state[val] = 1.0

            for w in range(dim_work):
                state[j * dim_work + w] += work_state[w]

        state /= np.linalg.norm(state)

        # Measure the work register: pick a random outcome weighted by probs
        work_probs = np.zeros(dim_work)
        for w in range(dim_work):
            for j in range(Q):
                work_probs[w] += np.abs(state[j * dim_work + w]) ** 2
        work_probs /= work_probs.sum()

        measured_work = self.rng.choice(dim_work, p=work_probs)

        # Post-measurement counting-register state
        counting_state = np.zeros(Q, dtype=complex)
        for j in range(Q):
            counting_state[j] = state[j * dim_work + measured_work]
        norm = np.linalg.norm(counting_state)
        if norm < 1e-15:
            return -1
        counting_state /= norm

        # Apply inverse QFT
        qft_mat = self.qft_matrix(n_count)
        iqft_mat = qft_mat.conj().T
        counting_state = iqft_mat @ counting_state

        # Measure the counting register
        probs = np.abs(counting_state) ** 2
        probs /= probs.sum()
        measured = int(self.rng.choice(Q, p=probs))

        if measured == 0:
            return -1

        # Extract period via continued fractions
        return self.continued_fractions(measured, Q, N)

    # ----- continued fractions ----------------------------------------------

    @staticmethod
    def continued_fractions(measured: int, Q: int, N: int) -> int:
        """Extract the period from a QPE measurement using continued fractions.

        We approximate ``measured / Q`` as a fraction ``s / r`` where
        ``r < N``, then return ``r``.

        Parameters
        ----------
        measured : int
            Measured value from counting register.
        Q : int
            Size of counting register ``2**n_count``.
        N : int
            The number being factored.

        Returns
        -------
        int
            Candidate period, or -1 if extraction fails.
        """
        frac = Fraction(measured, Q).limit_denominator(N)
        r = frac.denominator
        return r if r > 0 else -1

    # ----- full factoring pipeline ------------------------------------------

    def factor(self, N: int, max_attempts: int = 20) -> ShorResult:
        """Factor *N* using Shor's algorithm.

        Parameters
        ----------
        N : int
            The composite number to factor (must be > 1 and not prime).
        max_attempts : int
            Maximum number of random *a* values to try.

        Returns
        -------
        ShorResult
        """
        result = ShorResult(N=N)

        # Trivial checks
        if N <= 1:
            result.details.append(f"N={N} is ≤ 1, nothing to factor.")
            return result
        if N % 2 == 0:
            result.factors = (2, N // 2)
            result.success = True
            result.details.append(f"{N} is even → factors are 2 and {N // 2}.")
            return result

        # Check if N is a prime power
        for base in range(2, int(N ** 0.5) + 1):
            power = 2
            while base ** power <= N:
                if base ** power == N:
                    result.factors = (base, N // base)
                    result.success = True
                    result.details.append(
                        f"{N} = {base}^{power}, trivial factoring."
                    )
                    return result
                power += 1

        for attempt in range(1, max_attempts + 1):
            result.attempts = attempt

            a = int(self.rng.integers(2, N))
            result.a_used = a
            result.details.append(f"Attempt {attempt}: a = {a}")

            g = gcd(a, N)
            if g > 1:
                result.factors = (g, N // g)
                result.success = True
                result.details.append(f"  Lucky: gcd({a}, {N}) = {g}")
                return result

            # Quantum period finding
            r = self.quantum_period_finding(a, N)
            result.period_found = r
            result.details.append(f"  Quantum period finding → r = {r}")

            if r <= 0 or r % 2 != 0:
                result.details.append(f"  r={r} is odd or invalid, retrying.")
                continue

            # Also try multiples of r that might be the true period
            candidates = [r]
            # Classical verification of small multiples
            for mult in [1, 2, 3]:
                rc = r * mult
                if pow(a, rc, N) == 1:
                    candidates.append(rc)

            for rc in candidates:
                if rc <= 0 or rc % 2 != 0:
                    continue
                x = pow(a, rc // 2, N)
                if x == N - 1:
                    continue

                f1 = gcd(x - 1, N)
                f2 = gcd(x + 1, N)

                for f in (f1, f2):
                    if 1 < f < N:
                        result.factors = (f, N // f)
                        result.success = True
                        result.period_found = rc
                        result.details.append(
                            f"  Factors: {f} × {N // f}  (r={rc})"
                        )
                        return result

            result.details.append("  No non-trivial factor found, retrying.")

        result.details.append(f"Failed after {max_attempts} attempts.")
        return result
