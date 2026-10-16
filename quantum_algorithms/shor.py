"""Shor's factoring algorithm.

To factor ``N``, pick ``a`` coprime to ``N`` and find the order ``r`` of
``a`` modulo ``N`` (the smallest ``r`` with ``a^r = 1 mod N``) by quantum
period finding. If ``r`` is even and ``a^(r/2) != -1 mod N``, then
``gcd(a^(r/2) +- 1, N)`` are non-trivial factors.

This is a classical simulation: the counting register is prepared and
inverse-Fourier-transformed exactly, and the period is read off with
continued fractions. It reproduces the algorithm's statistics, not its
speed-up.

References
----------
Shor, P. W. "Polynomial-time algorithms for prime factorization and discrete
logarithms on a quantum computer." SIAM J. Comput. 26, 1484 (1997).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import ceil, gcd, log2

import numpy as np


@dataclass
class ShorResult:
    """Outcome of a factoring run.

    Attributes
    ----------
    N : int
        The number to factor.
    factors : tuple of int
        The two factors found, or ``(0, 0)`` on failure.
    a_used : int
        Base of the last attempt.
    period_found : int
        Verified order of ``a`` (0 when the factor came from a classical shortcut).
    success : bool
        Whether a non-trivial factorization was found.
    attempts : int
        Number of random bases tried.
    method : str
        ``"quantum period finding"`` or the classical shortcut that succeeded.
    details : list of str
        Human-readable log of every attempt.
    """

    N: int = 0
    factors: tuple[int, int] = (0, 0)
    a_used: int = 0
    period_found: int = 0
    success: bool = False
    attempts: int = 0
    method: str = ""
    details: list[str] = field(default_factory=list)


class ShorFactoring:
    """Simulation of Shor's algorithm.

    Parameters
    ----------
    seed : int, optional
        Seed for the random bases and simulated measurements.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)


    @staticmethod
    def classical_order_check(a: int, N: int) -> int:
        """Order of ``a`` modulo ``N`` by brute force (reference implementation).

        Parameters
        ----------
        a : int
            Base.
        N : int
            Modulus.

        Returns
        -------
        int
            Smallest ``r > 0`` with ``a^r = 1 mod N``, or ``-1`` if ``gcd(a, N) > 1``.
        """
        if gcd(a, N) != 1:
            return -1
        r = 1
        current = a % N
        while current != 1 and r < N:
            current = (current * a) % N
            r += 1
        return r if current == 1 else -1


    @staticmethod
    def qft_matrix(n: int) -> np.ndarray:
        """Dense QFT matrix on ``n`` qubits.

        Parameters
        ----------
        n : int
            Register size.

        Returns
        -------
        numpy.ndarray
            Unitary of shape ``(2^n, 2^n)``.
        """
        N = 2 ** n
        omega = np.exp(2j * np.pi / N)
        indices = np.arange(N)
        return np.power(omega, np.outer(indices, indices)) / np.sqrt(N)


    @staticmethod
    def controlled_modular_exp(a: int, power: int, N: int,
                               n_qubits: int) -> np.ndarray:
        """Permutation matrix of ``x -> a^power * x mod N`` on the work register.

        Basis states ``x >= N`` are left unchanged so the matrix stays unitary.

        Parameters
        ----------
        a : int
            Base.
        power : int
            Exponent applied to ``a``.
        N : int
            Modulus.
        n_qubits : int
            Work-register size.

        Returns
        -------
        numpy.ndarray
            Unitary of shape ``(2^n, 2^n)``.
        """
        dim = 2 ** n_qubits
        U = np.zeros((dim, dim), dtype=complex)
        a_pow = pow(a, power, N)
        for x in range(dim):
            if x < N:
                y = (a_pow * x) % N
                U[y, x] = 1.0
            else:
                U[x, x] = 1.0
        return U


    def quantum_period_finding(self, a: int, N: int) -> int:
        """Simulate one run of the order-finding circuit.

        The work register is measured first, collapsing the counting register onto
        the arithmetic progression ``{j : a^j mod N = w}``; an inverse QFT (computed
        with an FFT) then concentrates probability near multiples of ``2^t / r``.

        Parameters
        ----------
        a : int
            Base coprime to ``N``.
        N : int
            Modulus.

        Returns
        -------
        int
            Candidate period from continued fractions (a divisor of the true order
            in general), or ``-1`` if the measurement was uninformative.
        """
        n_count = max(4, 2 * ceil(log2(N + 1)))
        n_work  = max(2, ceil(log2(N + 1)))
        Q = 2 ** n_count
        dim_work = 2 ** n_work

        work_values = np.array([pow(a, j, N) for j in range(Q)])
        work_probs = np.bincount(work_values, minlength=dim_work) / Q
        measured_work = self.rng.choice(dim_work, p=work_probs)

        counting_state = (work_values == measured_work).astype(np.complex128)
        counting_state /= np.linalg.norm(counting_state)
        counting_state = np.fft.fft(counting_state) / np.sqrt(Q)

        probs = np.abs(counting_state) ** 2
        probs /= probs.sum()
        measured = int(self.rng.choice(Q, p=probs))

        if measured == 0:
            return -1

        return self.continued_fractions(measured, Q, N)


    @staticmethod
    def continued_fractions(measured: int, Q: int, N: int) -> int:
        """Recover a period candidate from a counting-register measurement.

        Parameters
        ----------
        measured : int
            Measured integer ``m``.
        Q : int
            Counting-register dimension ``2^t``.
        N : int
            Modulus; bounds the denominator.

        Returns
        -------
        int
            Denominator of the best rational approximation to ``m / Q``.
        """
        frac = Fraction(measured, Q).limit_denominator(N)
        r = frac.denominator
        return r if r > 0 else -1


    def factor(self, N: int, max_attempts: int = 20) -> ShorResult:
        """Factor ``N``.

        Even numbers, perfect powers and lucky ``gcd`` hits are handled classically,
        as in the full algorithm. Otherwise the period found by
        :meth:`quantum_period_finding` is extended to the true order and verified
        before it is used.

        Parameters
        ----------
        N : int
            Composite number to factor.
        max_attempts : int, default 20
            Maximum number of random bases to try.

        Returns
        -------
        ShorResult
        """
        result = ShorResult(N=N)

        if N <= 1:
            result.details.append(f"N={N} is ≤ 1, nothing to factor.")
            return result
        if N % 2 == 0:
            result.factors = (2, N // 2)
            result.success = True
            result.method = "classical: even"
            result.details.append(f"{N} is even → factors are 2 and {N // 2}.")
            return result

        for base in range(2, int(N ** 0.5) + 1):
            power = 2
            while base ** power <= N:
                if base ** power == N:
                    result.factors = (base, N // base)
                    result.success = True
                    result.method = "classical: perfect power"
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
                result.period_found = 0
                result.method = "classical: gcd"
                result.details.append(f"  Lucky: gcd({a}, {N}) = {g}")
                return result

            r = self.quantum_period_finding(a, N)
            result.period_found = r
            result.details.append(f"  Quantum period finding → r = {r}")

            if r <= 0:
                result.details.append(f"  r={r} is invalid, retrying.")
                continue

            order = next((r * m for m in range(1, 5) if pow(a, r * m, N) == 1), -1)
            if order <= 0 or order % 2 != 0:
                result.details.append(
                    f"  order={order} is odd or not found, retrying."
                )
                continue

            x = pow(a, order // 2, N)
            if x == N - 1:
                result.details.append(f"  a^(r/2) ≡ -1 (mod {N}), retrying.")
                continue

            for f in (gcd(x - 1, N), gcd(x + 1, N)):
                if 1 < f < N:
                    result.factors = (f, N // f)
                    result.success = True
                    result.period_found = order
                    result.method = "quantum period finding"
                    result.details.append(
                        f"  Factors: {f} × {N // f}  (r={order})"
                    )
                    return result

            result.details.append("  No non-trivial factor found, retrying.")

        result.details.append(f"Failed after {max_attempts} attempts.")
        return result
