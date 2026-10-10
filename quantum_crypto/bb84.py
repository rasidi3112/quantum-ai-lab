from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


KET_0 = np.array([1.0, 0.0], dtype=complex)
KET_1 = np.array([0.0, 1.0], dtype=complex)

KET_PLUS  = np.array([1.0,  1.0], dtype=complex) / np.sqrt(2)
KET_MINUS = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2)

RECTILINEAR = 0
DIAGONAL    = 1

PROJ_0 = np.outer(KET_0, KET_0.conj())
PROJ_1 = np.outer(KET_1, KET_1.conj())

PROJ_PLUS  = np.outer(KET_PLUS,  KET_PLUS.conj())
PROJ_MINUS = np.outer(KET_MINUS, KET_MINUS.conj())


@dataclass
class BB84Statistics:
    n_qubits_sent: int = 0
    n_sifted_bits: int = 0
    qber: float = 0.0
    eve_present: bool = False
    error_sample_size: int = 0
    final_key_length: int = 0
    protocol_secure: bool = True
    alice_key: List[int] = field(default_factory=list)
    bob_key: List[int] = field(default_factory=list)


class BB84Protocol:
    QBER_THRESHOLD = 0.11

    def __init__(self, qber_threshold: float = 0.11,
                 seed: Optional[int] = None) -> None:
        self.qber_threshold = qber_threshold
        self.rng = np.random.default_rng(seed)


    def generate_random_bits(self, n: int) -> np.ndarray:
        return self.rng.integers(0, 2, size=n)

    def generate_random_bases(self, n: int) -> np.ndarray:
        return self.rng.integers(0, 2, size=n)


    @staticmethod
    def encode_qubits(bits: np.ndarray,
                      bases: np.ndarray) -> List[np.ndarray]:
        qubits: List[np.ndarray] = []
        for bit, basis in zip(bits, bases):
            if basis == RECTILINEAR:
                qubits.append(KET_0.copy() if bit == 0 else KET_1.copy())
            else:
                qubits.append(KET_PLUS.copy() if bit == 0 else KET_MINUS.copy())
        return qubits


    def measure_qubits(self, qubits: List[np.ndarray],
                       bases: np.ndarray) -> np.ndarray:
        results = np.empty(len(qubits), dtype=int)
        for i, (qubit, basis) in enumerate(zip(qubits, bases)):
            if basis == RECTILINEAR:
                p0 = np.abs(np.vdot(KET_0, qubit)) ** 2
            else:
                p0 = np.abs(np.vdot(KET_PLUS, qubit)) ** 2
            results[i] = 0 if self.rng.random() < p0 else 1
        return results


    @staticmethod
    def sift_keys(alice_bases: np.ndarray,
                  bob_bases: np.ndarray,
                  bob_bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        matching = alice_bases == bob_bases
        return np.where(matching)[0], bob_bits[matching]


    def estimate_error_rate(self, alice_key: np.ndarray,
                            bob_key: np.ndarray,
                            sample_size: Optional[int] = None
                            ) -> Tuple[float, np.ndarray, np.ndarray]:
        n = len(alice_key)
        if sample_size is None:
            sample_size = min(n // 4, 50)
        sample_size = min(sample_size, n)

        indices = self.rng.choice(n, size=sample_size, replace=False)
        mask = np.ones(n, dtype=bool)
        mask[indices] = False

        errors = np.sum(alice_key[indices] != bob_key[indices])
        qber = errors / sample_size if sample_size > 0 else 0.0

        return qber, alice_key[mask], bob_key[mask]


    @staticmethod
    def privacy_amplification(key: np.ndarray,
                              compression_ratio: float = 0.5
                              ) -> np.ndarray:
        key_str = "".join(str(b) for b in key)
        digest = hashlib.sha256(key_str.encode()).hexdigest()

        bits = "".join(format(int(c, 16), "04b") for c in digest)

        target_len = max(1, int(len(key) * compression_ratio))
        target_len = min(target_len, len(bits))

        return np.array([int(b) for b in bits[:target_len]], dtype=int)


    def _eve_intercept_resend(self, qubits: List[np.ndarray]
                              ) -> List[np.ndarray]:
        eve_bases = self.generate_random_bases(len(qubits))
        eve_bits = self.measure_qubits(qubits, eve_bases)
        return self.encode_qubits(eve_bits, eve_bases)


    def run_protocol(self, n_qubits: int = 256,
                     eve_present: bool = False
                     ) -> Tuple[np.ndarray, BB84Statistics]:
        stats = BB84Statistics(n_qubits_sent=n_qubits, eve_present=eve_present)

        alice_bits  = self.generate_random_bits(n_qubits)
        alice_bases = self.generate_random_bases(n_qubits)
        qubits = self.encode_qubits(alice_bits, alice_bases)

        if eve_present:
            qubits = self._eve_intercept_resend(qubits)

        bob_bases = self.generate_random_bases(n_qubits)
        bob_bits  = self.measure_qubits(qubits, bob_bases)

        matching_idx, sifted_bob = self.sift_keys(alice_bases, bob_bases, bob_bits)
        sifted_alice = alice_bits[matching_idx]
        stats.n_sifted_bits = len(sifted_alice)

        if stats.n_sifted_bits < 4:
            stats.protocol_secure = False
            return np.array([], dtype=int), stats

        sample_size = max(1, stats.n_sifted_bits // 4)
        stats.error_sample_size = sample_size
        qber, alice_remaining, bob_remaining = self.estimate_error_rate(
            sifted_alice, sifted_bob, sample_size
        )
        stats.qber = qber

        if qber > self.qber_threshold:
            stats.protocol_secure = False
            stats.alice_key = alice_remaining.tolist()
            stats.bob_key = bob_remaining.tolist()
            return np.array([], dtype=int), stats

        final_key = self.privacy_amplification(alice_remaining, 0.5)

        stats.final_key_length = len(final_key)
        stats.protocol_secure = True
        stats.alice_key = alice_remaining.tolist()
        stats.bob_key = bob_remaining.tolist()

        return final_key, stats
