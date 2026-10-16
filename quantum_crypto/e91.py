"""E91 entanglement-based quantum key distribution.

Alice and Bob share singlet pairs ``|psi-> = (|01> - |10>) / sqrt(2)`` and
measure along randomly chosen analyzer angles. Rounds with equal angles give
perfectly anti-correlated (hence shareable) key bits; the other rounds feed a
CHSH test whose value ``|S| > 2`` certifies entanglement, ideally reaching
Tsirelson's bound ``2 sqrt(2)``.

References
----------
Ekert, A. K. "Quantum cryptography based on Bell's theorem."
Phys. Rev. Lett. 67, 661 (1991).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KET_0 = np.array([1.0, 0.0], dtype=complex)
KET_1 = np.array([0.0, 1.0], dtype=complex)

BELL_PSI_MINUS = (np.kron(KET_0, KET_1) - np.kron(KET_1, KET_0)) / np.sqrt(2)


def _rotation_basis_vectors(theta: float) -> tuple[np.ndarray, np.ndarray]:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    plus_theta  = np.array([ cos, sin], dtype=complex)
    minus_theta = np.array([-sin, cos], dtype=complex)
    return plus_theta, minus_theta


@dataclass
class E91Statistics:
    """Bookkeeping for one E91 run.

    Attributes
    ----------
    n_pairs : int
        Entangled pairs distributed.
    n_key_bits : int
        Raw key length (rounds with matching analyzers).
    chsh_value : float
        Measured CHSH combination ``S``.
    bell_violated : bool
        Whether ``|S| > 2``.
    correlations : dict
        Measured correlator ``E(a, b)`` for each angle pair.
    alice_key, bob_key : list of int
        Raw keys of each party (Bob's bits already flipped).
    """

    n_pairs: int = 0
    n_key_bits: int = 0
    chsh_value: float = 0.0
    bell_violated: bool = False
    correlations: dict[str, float] = field(default_factory=dict)
    alice_key: list[int] = field(default_factory=list)
    bob_key: list[int] = field(default_factory=list)


class E91Protocol:
    """Simulation of the E91 protocol on singlet pairs.

    Parameters
    ----------
    seed : int, optional
        Seed for analyzer choices and measurement outcomes.
    """

    ALICE_ANGLES = (np.pi / 2, 0.0,           3 * np.pi / 8)
    BOB_ANGLES   = (np.pi / 4, -np.pi / 4,    3 * np.pi / 8)

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)


    @staticmethod
    def generate_entangled_pairs(n: int) -> list[np.ndarray]:
        """Create ``n`` copies of the singlet state.

        Parameters
        ----------
        n : int
            Number of pairs.

        Returns
        -------
        list of numpy.ndarray
            Two-qubit states of length 4.
        """
        return [BELL_PSI_MINUS.copy() for _ in range(n)]


    @classmethod
    def choose_measurement_angles(cls) -> tuple[tuple[float, ...],
                                                  tuple[float, ...]]:
        """Return the analyzer angles of Alice and Bob.

        Returns
        -------
        alice_angles, bob_angles : tuple of float
            Three angles each; the first two are used for CHSH and the third is
            shared and used for the key.
        """
        return cls.ALICE_ANGLES, cls.BOB_ANGLES


    def measure_entangled(self, pair: np.ndarray,
                          angle_a: float,
                          angle_b: float) -> tuple[int, int]:
        """Measure a pair along the two analyzer angles.

        Parameters
        ----------
        pair : numpy.ndarray
            Two-qubit state.
        angle_a, angle_b : float
            Bloch-sphere angles (in the x–z plane) of Alice's and Bob's analyzers.

        Returns
        -------
        tuple of int
            Outcomes ``(a, b)`` with ``0`` for "+" and ``1`` for "-".
        """
        a_plus, a_minus = _rotation_basis_vectors(angle_a)
        b_plus, b_minus = _rotation_basis_vectors(angle_b)

        projectors = [
            np.kron(a_plus,  b_plus),
            np.kron(a_plus,  b_minus),
            np.kron(a_minus, b_plus),
            np.kron(a_minus, b_minus),
        ]

        probs = np.array([np.abs(np.vdot(p, pair)) ** 2 for p in projectors])
        probs /= probs.sum()

        outcome_idx = int(self.rng.choice(4, p=probs))
        return outcome_idx >> 1, outcome_idx & 1


    @staticmethod
    def compute_chsh(correlations: dict[tuple[float, float], float]) -> float:
        """Combine correlators into the CHSH value ``S``.

        Parameters
        ----------
        correlations : dict
            ``E(a, b)`` keyed by rounded ``(angle_a, angle_b)`` tuples.

        Returns
        -------
        float
            ``S = E(a1, b1) - E(a1, b2) + E(a2, b1) + E(a2, b2)``.
        """
        a1, a2 = np.pi / 2, 0.0
        b1, b2 = np.pi / 4, -np.pi / 4

        def _E(a: float, b: float) -> float:
            k = (round(a, 10), round(b, 10))
            return correlations.get(k, 0.0)

        S = _E(a1, b1) - _E(a1, b2) + _E(a2, b1) + _E(a2, b2)
        return S

    @staticmethod
    def check_bell_violation(S: float) -> bool:
        """Return whether ``|S|`` exceeds the classical bound of 2."""
        return abs(S) > 2.0


    @staticmethod
    def extract_key(results: list[tuple[int, int]]) -> tuple[list[int],
                                                               list[int]]:
        """Split outcome pairs into Alice's and Bob's raw keys.

        Parameters
        ----------
        results : list of tuple of int
            Outcomes of the key-generating rounds.

        Returns
        -------
        alice_key, bob_key : list of int
        """
        alice_key = [a for a, _ in results]
        bob_key   = [b for _, b in results]
        return alice_key, bob_key


    def run_protocol(self, n_pairs: int = 1000
                     ) -> tuple[np.ndarray, E91Statistics]:
        """Run E91 end to end.

        Parameters
        ----------
        n_pairs : int, default 1000
            Number of entangled pairs; about one in nine becomes a key bit.

        Returns
        -------
        key : numpy.ndarray
            Alice's raw key.
        stats : E91Statistics
            Diagnostics including the CHSH value.
        """
        stats = E91Statistics(n_pairs=n_pairs)
        pairs = self.generate_entangled_pairs(n_pairs)
        alice_angles, bob_angles = self.choose_measurement_angles()

        a_choices = self.rng.integers(0, 3, size=n_pairs)
        b_choices = self.rng.integers(0, 3, size=n_pairs)

        key_results: list[tuple[int, int]] = []
        corr_counts: dict[tuple[float, float], list[int]] = {}

        for i in range(n_pairs):
            aa = alice_angles[a_choices[i]]
            bb = bob_angles[b_choices[i]]
            oa, ob = self.measure_entangled(pairs[i], aa, bb)

            ai, bi = int(a_choices[i]), int(b_choices[i])
            if ai == 2 and bi == 2:
                key_results.append((oa, 1 - ob))
            elif ai in (0, 1) and bi in (0, 1):
                k = (round(aa, 10), round(bb, 10))
                if k not in corr_counts:
                    corr_counts[k] = []
                corr_counts[k].append((1 - 2 * oa) * (1 - 2 * ob))

        correlations_raw: dict[tuple[float, float], float] = {}
        for k, products in corr_counts.items():
            correlations_raw[k] = float(np.mean(products))

        S = self.compute_chsh(correlations_raw)
        stats.chsh_value = S
        stats.bell_violated = self.check_bell_violation(S)
        stats.correlations = {str(k): v for k, v in correlations_raw.items()}

        alice_key, bob_key = self.extract_key(key_results)
        stats.alice_key = alice_key
        stats.bob_key = bob_key
        stats.n_key_bits = len(alice_key)

        return np.array(alice_key, dtype=int), stats
