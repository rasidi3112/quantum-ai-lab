from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd, log2, ceil
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ShorResult:
    N: int = 0
    factors: Tuple[int, int] = (0, 0)
    a_used: int = 0
    period_found: int = 0
    success: bool = False
    attempts: int = 0
    details: List[str] = field(default_factory=list)


class ShorFactoring:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)


    @staticmethod
    def classical_order_check(a: int, N: int) -> int:
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
        N = 2 ** n
        omega = np.exp(2j * np.pi / N)
        indices = np.arange(N)
        return np.power(omega, np.outer(indices, indices)) / np.sqrt(N)


    @staticmethod
    def controlled_modular_exp(a: int, power: int, N: int,
                               n_qubits: int) -> np.ndarray:
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
        n_count = max(4, 2 * ceil(log2(N + 1)))
        n_work  = max(2, ceil(log2(N + 1)))
        Q = 2 ** n_count
        dim_work = 2 ** n_work


        full_dim = Q * dim_work
        state = np.zeros(full_dim, dtype=complex)

        for j in range(Q):
            work_state = np.zeros(dim_work, dtype=complex)
            val = pow(a, j, N)
            if val < dim_work:
                work_state[val] = 1.0

            for w in range(dim_work):
                state[j * dim_work + w] += work_state[w]

        state /= np.linalg.norm(state)

        work_probs = np.zeros(dim_work)
        for w in range(dim_work):
            for j in range(Q):
                work_probs[w] += np.abs(state[j * dim_work + w]) ** 2
        work_probs /= work_probs.sum()

        measured_work = self.rng.choice(dim_work, p=work_probs)

        counting_state = np.zeros(Q, dtype=complex)
        for j in range(Q):
            counting_state[j] = state[j * dim_work + measured_work]
        norm = np.linalg.norm(counting_state)
        if norm < 1e-15:
            return -1
        counting_state /= norm

        qft_mat = self.qft_matrix(n_count)
        iqft_mat = qft_mat.conj().T
        counting_state = iqft_mat @ counting_state

        probs = np.abs(counting_state) ** 2
        probs /= probs.sum()
        measured = int(self.rng.choice(Q, p=probs))

        if measured == 0:
            return -1

        return self.continued_fractions(measured, Q, N)


    @staticmethod
    def continued_fractions(measured: int, Q: int, N: int) -> int:
        frac = Fraction(measured, Q).limit_denominator(N)
        r = frac.denominator
        return r if r > 0 else -1


    def factor(self, N: int, max_attempts: int = 20) -> ShorResult:
        result = ShorResult(N=N)

        if N <= 1:
            result.details.append(f"N={N} is ≤ 1, nothing to factor.")
            return result
        if N % 2 == 0:
            result.factors = (2, N // 2)
            result.success = True
            result.details.append(f"{N} is even → factors are 2 and {N // 2}.")
            return result

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

            r = self.quantum_period_finding(a, N)
            result.period_found = r
            result.details.append(f"  Quantum period finding → r = {r}")

            if r <= 0 or r % 2 != 0:
                result.details.append(f"  r={r} is odd or invalid, retrying.")
                continue

            candidates = [r]
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
