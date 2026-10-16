"""BB84 quantum key distribution.

Alice encodes random bits in randomly chosen rectilinear (``|0>, |1>``) or
diagonal (``|+>, |->``) bases; Bob measures in random bases. After publicly
comparing bases (sifting), they estimate the quantum bit error rate (QBER)
on a sacrificed sample and compress the rest with 2-universal Toeplitz
hashing (privacy amplification).

An intercept-resend eavesdropper who guesses bases at random introduces a
QBER of about 25%, far above the ~11% abort threshold.

References
----------
Bennett, C. H. & Brassard, G. "Quantum cryptography: Public key distribution
and coin tossing." Proc. IEEE ICCSSP, 175 (1984).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg as la

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
    """Bookkeeping for one BB84 run.

    Attributes
    ----------
    n_qubits_sent : int
        Qubits transmitted by Alice.
    n_sifted_bits : int
        Bits kept after basis reconciliation.
    qber : float
        Error rate measured on the sacrificed sample.
    eve_present : bool
        Whether an intercept-resend attack was simulated.
    error_sample_size : int
        Number of sifted bits disclosed to estimate the QBER.
    final_key_length : int
        Length of the key after privacy amplification.
    protocol_secure : bool
        ``False`` if the QBER exceeded the threshold and the run was aborted.
    keys_match : bool
        Whether Alice's and Bob's final keys are identical.
    alice_key, bob_key : list of int
        Sifted keys after removing the disclosed sample.
    """

    n_qubits_sent: int = 0
    n_sifted_bits: int = 0
    qber: float = 0.0
    eve_present: bool = False
    error_sample_size: int = 0
    final_key_length: int = 0
    protocol_secure: bool = True
    keys_match: bool = False
    alice_key: list[int] = field(default_factory=list)
    bob_key: list[int] = field(default_factory=list)


class BB84Protocol:
    """Simulation of the BB84 protocol, optionally under attack.

    Parameters
    ----------
    qber_threshold : float, default 0.11
        Abort the protocol when the estimated QBER exceeds this value.
    seed : int, optional
        Seed for every random choice made by Alice, Bob and Eve.
    """

    QBER_THRESHOLD = 0.11

    def __init__(self, qber_threshold: float = 0.11,
                 seed: int | None = None) -> None:
        self.qber_threshold = qber_threshold
        self.rng = np.random.default_rng(seed)


    def generate_random_bits(self, n: int) -> np.ndarray:
        """Draw ``n`` uniformly random bits.

        Parameters
        ----------
        n : int
            Number of bits.

        Returns
        -------
        numpy.ndarray
            Integer array of 0s and 1s.
        """
        return self.rng.integers(0, 2, size=n)

    def generate_random_bases(self, n: int) -> np.ndarray:
        """Draw ``n`` random bases (0 = rectilinear, 1 = diagonal).

        Parameters
        ----------
        n : int
            Number of bases.

        Returns
        -------
        numpy.ndarray
            Integer array of 0s and 1s.
        """
        return self.rng.integers(0, 2, size=n)


    @staticmethod
    def encode_qubits(bits: np.ndarray,
                      bases: np.ndarray) -> list[np.ndarray]:
        """Encode bits as single-qubit states in the given bases.

        Parameters
        ----------
        bits : numpy.ndarray
            Bits to send.
        bases : numpy.ndarray
            Basis for each bit (0 = rectilinear, 1 = diagonal).

        Returns
        -------
        list of numpy.ndarray
            One 2-component state per bit.
        """
        qubits: list[np.ndarray] = []
        for bit, basis in zip(bits, bases):
            if basis == RECTILINEAR:
                qubits.append(KET_0.copy() if bit == 0 else KET_1.copy())
            else:
                qubits.append(KET_PLUS.copy() if bit == 0 else KET_MINUS.copy())
        return qubits


    def measure_qubits(self, qubits: list[np.ndarray],
                       bases: np.ndarray) -> np.ndarray:
        """Measure each qubit in the given basis (Born rule).

        Parameters
        ----------
        qubits : list of numpy.ndarray
            Received states.
        bases : numpy.ndarray
            Measurement basis for each qubit.

        Returns
        -------
        numpy.ndarray
            Measured bits.
        """
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
                  bob_bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Keep the positions where Alice and Bob chose the same basis.

        Parameters
        ----------
        alice_bases, bob_bases : numpy.ndarray
            Basis choices of each party.
        bob_bits : numpy.ndarray
            Bob's measurement results.

        Returns
        -------
        indices : numpy.ndarray
            Positions with matching bases.
        sifted_bits : numpy.ndarray
            Bob's bits at those positions.
        """
        matching = alice_bases == bob_bases
        return np.where(matching)[0], bob_bits[matching]


    def estimate_error_rate(self, alice_key: np.ndarray,
                            bob_key: np.ndarray,
                            sample_size: int | None = None
                            ) -> tuple[float, np.ndarray, np.ndarray]:
        """Estimate the QBER on a random sample and discard that sample.

        Parameters
        ----------
        alice_key, bob_key : numpy.ndarray
            Sifted keys.
        sample_size : int, optional
            Bits to disclose; defaults to ``min(n // 4, 50)``.

        Returns
        -------
        qber : float
            Fraction of disagreeing bits in the sample.
        alice_remaining, bob_remaining : numpy.ndarray
            Keys with the disclosed bits removed.
        """
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
                              compression_ratio: float = 0.5,
                              seed: int | None = None,
                              ) -> np.ndarray:
        """Compress a key with a random binary Toeplitz matrix.

        Toeplitz hashing is a 2-universal hash family, the standard choice for
        privacy amplification in QKD. Both parties must use the same ``seed``, which
        can be announced publicly.

        Parameters
        ----------
        key : numpy.ndarray
            Input key bits of length ``n``.
        compression_ratio : float, default 0.5
            Output length is ``max(1, int(n * compression_ratio))``.
        seed : int, optional
            Seed that defines the Toeplitz matrix.

        Returns
        -------
        numpy.ndarray
            The amplified key.
        """
        key = np.asarray(key, dtype=np.int64)
        n = len(key)
        if n == 0:
            return np.array([], dtype=int)
        m = max(1, int(n * compression_ratio))
        rng = np.random.default_rng(seed)
        diagonals = rng.integers(0, 2, size=n + m - 1)
        toeplitz = la.toeplitz(diagonals[:m], diagonals[m - 1:])
        return (toeplitz @ key) % 2


    def _eve_intercept_resend(self, qubits: list[np.ndarray]
                              ) -> list[np.ndarray]:
        eve_bases = self.generate_random_bases(len(qubits))
        eve_bits = self.measure_qubits(qubits, eve_bases)
        return self.encode_qubits(eve_bits, eve_bases)


    def run_protocol(self, n_qubits: int = 256,
                     eve_present: bool = False
                     ) -> tuple[np.ndarray, BB84Statistics]:
        """Run BB84 end to end.

        Parameters
        ----------
        n_qubits : int, default 256
            Qubits sent by Alice.
        eve_present : bool, default False
            Simulate an intercept-resend eavesdropper.

        Returns
        -------
        key : numpy.ndarray
            Alice's final key (empty if the run was aborted).
        stats : BB84Statistics
            Diagnostics of the run.
        """
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

        hash_seed = int(self.rng.integers(0, 2 ** 63 - 1))
        final_key = self.privacy_amplification(alice_remaining, 0.5, hash_seed)
        bob_final = self.privacy_amplification(bob_remaining, 0.5, hash_seed)

        stats.final_key_length = len(final_key)
        stats.protocol_secure = True
        stats.keys_match = bool(np.array_equal(final_key, bob_final))
        stats.alice_key = alice_remaining.tolist()
        stats.bob_key = bob_remaining.tolist()

        return final_key, stats
