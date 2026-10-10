from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

KET_0 = np.array([1.0, 0.0], dtype=complex)
KET_1 = np.array([0.0, 1.0], dtype=complex)

BELL_PSI_MINUS = (np.kron(KET_0, KET_1) - np.kron(KET_1, KET_0)) / np.sqrt(2)


def _rotation_basis_vectors(theta: float) -> Tuple[np.ndarray, np.ndarray]:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    plus_theta  = np.array([ cos, sin], dtype=complex)
    minus_theta = np.array([-sin, cos], dtype=complex)
    return plus_theta, minus_theta


@dataclass
class E91Statistics:
    n_pairs: int = 0
    n_key_bits: int = 0
    chsh_value: float = 0.0
    bell_violated: bool = False
    correlations: Dict[str, float] = field(default_factory=dict)
    alice_key: List[int] = field(default_factory=list)
    bob_key: List[int] = field(default_factory=list)


class E91Protocol:
    ALICE_ANGLES = (np.pi / 2, 0.0,           3 * np.pi / 8)
    BOB_ANGLES   = (np.pi / 4, -np.pi / 4,    3 * np.pi / 8)

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)


    @staticmethod
    def generate_entangled_pairs(n: int) -> List[np.ndarray]:
        return [BELL_PSI_MINUS.copy() for _ in range(n)]


    @classmethod
    def choose_measurement_angles(cls) -> Tuple[Tuple[float, ...],
                                                  Tuple[float, ...]]:
        return cls.ALICE_ANGLES, cls.BOB_ANGLES


    def measure_entangled(self, pair: np.ndarray,
                          angle_a: float,
                          angle_b: float) -> Tuple[int, int]:
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

        outcome_idx = self.rng.choice(4, p=probs)
        outcomes = [(0, 0), (0, 1), (1, 0), (1, 1)]
        return outcomes[outcome_idx]


    @staticmethod
    def compute_chsh(correlations: Dict[Tuple[float, float], float]) -> float:
        a1, a2 = np.pi / 2, 0.0
        b1, b2 = np.pi / 4, -np.pi / 4

        def _E(a: float, b: float) -> float:
            k = (round(a, 10), round(b, 10))
            return correlations.get(k, 0.0)

        S = _E(a1, b1) - _E(a1, b2) + _E(a2, b1) + _E(a2, b2)
        return S

    @staticmethod
    def check_bell_violation(S: float) -> bool:
        return abs(S) > 2.0


    @staticmethod
    def extract_key(results: List[Tuple[int, int]]) -> Tuple[List[int],
                                                               List[int]]:
        alice_key = [a for a, _ in results]
        bob_key   = [b for _, b in results]
        return alice_key, bob_key


    def run_protocol(self, n_pairs: int = 1000
                     ) -> Tuple[np.ndarray, E91Statistics]:
        stats = E91Statistics(n_pairs=n_pairs)
        pairs = self.generate_entangled_pairs(n_pairs)
        alice_angles, bob_angles = self.choose_measurement_angles()

        a_choices = self.rng.integers(0, 3, size=n_pairs)
        b_choices = self.rng.integers(0, 3, size=n_pairs)

        key_results: List[Tuple[int, int]] = []
        corr_counts: Dict[Tuple[float, float], List[int]] = {}

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

        correlations_raw: Dict[Tuple[float, float], float] = {}
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
