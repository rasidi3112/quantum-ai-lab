"""
BB84 Quantum Key Distribution Protocol
=======================================

The BB84 protocol (Bennett & Brassard, 1984) is the first and most widely
studied quantum key distribution scheme.  It enables two parties—Alice and
Bob—to establish a shared secret key whose security is guaranteed by the
laws of quantum mechanics rather than computational assumptions.

Protocol overview
-----------------
1. **Preparation** – Alice picks random bits and random bases (rectilinear
   ``+`` or diagonal ``×``), then encodes each bit as a qubit:

   * Rectilinear basis: ``|0⟩`` for 0, ``|1⟩`` for 1
   * Diagonal basis:    ``|+⟩ = (|0⟩+|1⟩)/√2`` for 0,
                         ``|−⟩ = (|0⟩−|1⟩)/√2`` for 1

2. **Transmission** – Alice sends the qubits to Bob over a quantum channel.

3. **Measurement** – Bob independently picks random bases and measures each
   qubit.  If his basis matches Alice's, he gets the correct bit with
   certainty; otherwise the outcome is uniformly random.

4. **Sifting** – Alice and Bob publicly compare bases (but *not* bit values)
   and keep only the bits where their bases matched.

5. **Error estimation** – They sacrifice a random subset of sifted bits to
   estimate the quantum bit error rate (QBER).  A QBER > ~11 % signals
   potential eavesdropping.

6. **Privacy amplification** – A universal-hash function compresses the
   remaining key to remove any partial information an eavesdropper might
   possess.

Eavesdropper (Eve) simulation
-----------------------------
When ``eve_present=True`` the protocol simulates an *intercept-resend*
attack: Eve randomly picks bases, measures, and re-sends to Bob.  This
introduces a QBER of approximately 25 %, well above the safety threshold.

Mathematical notes
------------------
*  Measurement probability in matching basis: ``P(correct) = 1``
*  Measurement probability in mismatched basis: ``P(correct) = 0.5``
*  Expected QBER under intercept-resend: ``0.5 × 0.5 = 0.25`` (25 %)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Qubit basis states
# ---------------------------------------------------------------------------

# Computational (rectilinear) basis
KET_0 = np.array([1.0, 0.0], dtype=complex)  # |0⟩
KET_1 = np.array([0.0, 1.0], dtype=complex)  # |1⟩

# Hadamard (diagonal) basis
KET_PLUS  = np.array([1.0,  1.0], dtype=complex) / np.sqrt(2)  # |+⟩
KET_MINUS = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2)  # |−⟩

# Basis labels
RECTILINEAR = 0  # + basis
DIAGONAL    = 1  # × basis

# Measurement projectors
PROJ_0 = np.outer(KET_0, KET_0.conj())
PROJ_1 = np.outer(KET_1, KET_1.conj())

PROJ_PLUS  = np.outer(KET_PLUS,  KET_PLUS.conj())
PROJ_MINUS = np.outer(KET_MINUS, KET_MINUS.conj())


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BB84Statistics:
    """Container for BB84 protocol run statistics."""
    n_qubits_sent: int = 0
    n_sifted_bits: int = 0
    qber: float = 0.0
    eve_present: bool = False
    error_sample_size: int = 0
    final_key_length: int = 0
    protocol_secure: bool = True
    alice_key: List[int] = field(default_factory=list)
    bob_key: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BB84 Protocol implementation
# ---------------------------------------------------------------------------

class BB84Protocol:
    """Full simulation of the BB84 QKD protocol.

    All quantum operations use pure NumPy statevector simulation.

    Parameters
    ----------
    qber_threshold : float
        Maximum tolerable QBER before the key is discarded (default 0.11).
    seed : int | None
        Random seed for reproducibility.

    Examples
    --------
    >>> proto = BB84Protocol(seed=42)
    >>> key, stats = proto.run_protocol(500, eve_present=False)
    >>> print(f"Shared key length: {len(key)}, QBER: {stats.qber:.4f}")
    """

    QBER_THRESHOLD = 0.11  # ~11 % safety threshold

    def __init__(self, qber_threshold: float = 0.11,
                 seed: Optional[int] = None) -> None:
        self.qber_threshold = qber_threshold
        self.rng = np.random.default_rng(seed)

    # ----- primitive helpers ------------------------------------------------

    def generate_random_bits(self, n: int) -> np.ndarray:
        """Generate a random binary string of length *n*.

        Returns
        -------
        np.ndarray of int, shape ``(n,)``
            Each element is 0 or 1.
        """
        return self.rng.integers(0, 2, size=n)

    def generate_random_bases(self, n: int) -> np.ndarray:
        """Generate a random basis string of length *n*.

        Returns
        -------
        np.ndarray of int, shape ``(n,)``
            0 → rectilinear (+), 1 → diagonal (×).
        """
        return self.rng.integers(0, 2, size=n)

    # ----- encoding ---------------------------------------------------------

    @staticmethod
    def encode_qubits(bits: np.ndarray,
                      bases: np.ndarray) -> List[np.ndarray]:
        """Encode classical bits into qubit state vectors.

        Encoding table::

            basis=0 (rectilinear):  bit 0 → |0⟩,  bit 1 → |1⟩
            basis=1 (diagonal):     bit 0 → |+⟩,  bit 1 → |−⟩

        Parameters
        ----------
        bits : array-like of int
        bases : array-like of int (same length)

        Returns
        -------
        list of np.ndarray
            Each element is a 2-component complex state vector.
        """
        qubits: List[np.ndarray] = []
        for bit, basis in zip(bits, bases):
            if basis == RECTILINEAR:
                qubits.append(KET_0.copy() if bit == 0 else KET_1.copy())
            else:
                qubits.append(KET_PLUS.copy() if bit == 0 else KET_MINUS.copy())
        return qubits

    # ----- measurement ------------------------------------------------------

    def measure_qubits(self, qubits: List[np.ndarray],
                       bases: np.ndarray) -> np.ndarray:
        """Measure each qubit in the given basis.

        Born rule simulation:
            ``P(outcome=0) = |⟨proj_0 | ψ⟩|²``

        Parameters
        ----------
        qubits : list of state vectors
        bases : array of basis choices (0 = rectilinear, 1 = diagonal)

        Returns
        -------
        np.ndarray of int
            Measurement outcomes (0 or 1).
        """
        results = np.empty(len(qubits), dtype=int)
        for i, (qubit, basis) in enumerate(zip(qubits, bases)):
            if basis == RECTILINEAR:
                p0 = np.abs(np.vdot(KET_0, qubit)) ** 2
            else:
                p0 = np.abs(np.vdot(KET_PLUS, qubit)) ** 2
            results[i] = 0 if self.rng.random() < p0 else 1
        return results

    # ----- sifting ----------------------------------------------------------

    @staticmethod
    def sift_keys(alice_bases: np.ndarray,
                  bob_bases: np.ndarray,
                  bob_bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sift the raw key: keep only positions where bases match.

        Returns
        -------
        (matching_indices, sifted_bits) : tuple of np.ndarray
        """
        matching = alice_bases == bob_bases
        return np.where(matching)[0], bob_bits[matching]

    # ----- error estimation -------------------------------------------------

    def estimate_error_rate(self, alice_key: np.ndarray,
                            bob_key: np.ndarray,
                            sample_size: Optional[int] = None
                            ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Estimate the QBER from a random sample of sifted bits.

        The sampled bits are *removed* from the key (consumed).

        Parameters
        ----------
        alice_key, bob_key : np.ndarray
            Sifted key bits for each party.
        sample_size : int | None
            Number of bits to sample.  Defaults to min(len/4, 50).

        Returns
        -------
        (qber, alice_remaining, bob_remaining) : tuple
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

    # ----- privacy amplification --------------------------------------------

    @staticmethod
    def privacy_amplification(key: np.ndarray,
                              compression_ratio: float = 0.5
                              ) -> np.ndarray:
        """Hash-based privacy amplification.

        Compresses the key using SHA-256 to remove partial information
        an eavesdropper may possess.

        Parameters
        ----------
        key : np.ndarray of int
            The sifted (and error-estimated) key bits.
        compression_ratio : float
            Fraction of original key length to retain (0, 1].

        Returns
        -------
        np.ndarray of int
            The shortened, privacy-amplified key.
        """
        key_str = "".join(str(b) for b in key)
        digest = hashlib.sha256(key_str.encode()).hexdigest()

        # Convert hex digest to bits
        bits = "".join(format(int(c, 16), "04b") for c in digest)

        target_len = max(1, int(len(key) * compression_ratio))
        target_len = min(target_len, len(bits))

        return np.array([int(b) for b in bits[:target_len]], dtype=int)

    # ----- eavesdropper simulation ------------------------------------------

    def _eve_intercept_resend(self, qubits: List[np.ndarray]
                              ) -> List[np.ndarray]:
        """Simulate Eve's intercept-resend attack.

        Eve measures each qubit in a random basis and re-sends a fresh
        qubit consistent with her measurement outcome.

        Returns
        -------
        list of np.ndarray
            The qubits that Bob will receive (potentially disturbed).
        """
        eve_bases = self.generate_random_bases(len(qubits))
        eve_bits = self.measure_qubits(qubits, eve_bases)
        # Re-encode in Eve's bases
        return self.encode_qubits(eve_bits, eve_bases)

    # ----- full protocol run ------------------------------------------------

    def run_protocol(self, n_qubits: int = 256,
                     eve_present: bool = False
                     ) -> Tuple[np.ndarray, BB84Statistics]:
        """Run the complete BB84 QKD protocol.

        Parameters
        ----------
        n_qubits : int
            Number of qubits Alice sends.
        eve_present : bool
            If True, simulate an intercept-resend eavesdropper.

        Returns
        -------
        (final_key, stats) : tuple
            ``final_key`` is a numpy array of bits (may be empty if the
            protocol aborts).  ``stats`` contains full run diagnostics.
        """
        stats = BB84Statistics(n_qubits_sent=n_qubits, eve_present=eve_present)

        # Step 1 — Alice prepares qubits
        alice_bits  = self.generate_random_bits(n_qubits)
        alice_bases = self.generate_random_bases(n_qubits)
        qubits = self.encode_qubits(alice_bits, alice_bases)

        # Step 2 — (optional) Eve intercepts
        if eve_present:
            qubits = self._eve_intercept_resend(qubits)

        # Step 3 — Bob measures
        bob_bases = self.generate_random_bases(n_qubits)
        bob_bits  = self.measure_qubits(qubits, bob_bases)

        # Step 4 — Sift
        matching_idx, sifted_bob = self.sift_keys(alice_bases, bob_bases, bob_bits)
        sifted_alice = alice_bits[matching_idx]
        stats.n_sifted_bits = len(sifted_alice)

        if stats.n_sifted_bits < 4:
            stats.protocol_secure = False
            return np.array([], dtype=int), stats

        # Step 5 — Error estimation
        sample_size = max(1, stats.n_sifted_bits // 4)
        stats.error_sample_size = sample_size
        qber, alice_remaining, bob_remaining = self.estimate_error_rate(
            sifted_alice, sifted_bob, sample_size
        )
        stats.qber = qber

        # Security check
        if qber > self.qber_threshold:
            stats.protocol_secure = False
            stats.alice_key = alice_remaining.tolist()
            stats.bob_key = bob_remaining.tolist()
            return np.array([], dtype=int), stats

        # Step 6 — Privacy amplification
        final_key = self.privacy_amplification(alice_remaining, 0.5)

        stats.final_key_length = len(final_key)
        stats.protocol_secure = True
        stats.alice_key = alice_remaining.tolist()
        stats.bob_key = bob_remaining.tolist()

        return final_key, stats
