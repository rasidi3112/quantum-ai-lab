"""
Quantum One-Time Pad (Quantum OTP)
==================================

Combines quantum key distribution (BB84 or E91) with the classical one-time
pad (OTP) cipher to achieve *information-theoretic* security.

Security guarantee
------------------
Shannon proved (1949) that the OTP is perfectly secret when:

1. The key is at least as long as the message.
2. The key is truly random.
3. The key is used only once.

QKD provides condition (2) via the laws of quantum mechanics, while the
protocol enforces (3) by generating a fresh key for every message.

Encryption / Decryption
-----------------------
Both operations are simply bitwise XOR:

.. math::

    c_i = m_i \\oplus k_i \\qquad m_i = c_i \\oplus k_i

where *m* is the plaintext, *k* the key, and *c* the ciphertext.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from quantum_crypto.bb84 import BB84Protocol
from quantum_crypto.e91 import E91Protocol


@dataclass
class CommunicationResult:
    """Result container for a full secure-communication pipeline."""
    plaintext: str
    ciphertext_bits: np.ndarray
    decrypted_text: str
    key_bits: np.ndarray
    key_length: int
    protocol_used: str
    success: bool
    error_message: str = ""


class QuantumOTP:
    """Quantum One-Time Pad: QKD + classical OTP encryption.

    Parameters
    ----------
    seed : int | None
        Random seed for reproducibility.

    Examples
    --------
    >>> otp = QuantumOTP(seed=42)
    >>> result = otp.secure_communicate("Hello!")
    >>> print(result.decrypted_text)
    Hello!
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed

    # ----- text ↔ binary ---------------------------------------------------

    @staticmethod
    def encode_message(text: str) -> np.ndarray:
        """Convert a text string to a binary (bit) array.

        Each character is encoded as 8-bit ASCII.

        Parameters
        ----------
        text : str

        Returns
        -------
        np.ndarray of int, shape ``(8 * len(text),)``
        """
        bits = []
        for ch in text:
            byte = format(ord(ch), "08b")
            bits.extend(int(b) for b in byte)
        return np.array(bits, dtype=int)

    @staticmethod
    def decode_message(binary: np.ndarray) -> str:
        """Convert a binary (bit) array back to a text string.

        Parameters
        ----------
        binary : array-like of int
            Length must be a multiple of 8.

        Returns
        -------
        str
        """
        binary = np.asarray(binary, dtype=int)
        chars = []
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            if len(byte) < 8:
                break
            value = int("".join(str(b) for b in byte), 2)
            chars.append(chr(value))
        return "".join(chars)

    # ----- XOR encryption / decryption --------------------------------------

    @staticmethod
    def encrypt(message_bits: np.ndarray,
                quantum_key: np.ndarray) -> np.ndarray:
        """Encrypt binary message with a quantum-derived key (XOR).

        Parameters
        ----------
        message_bits : np.ndarray of int
        quantum_key : np.ndarray of int
            Must be at least as long as *message_bits*.

        Returns
        -------
        np.ndarray of int
            The ciphertext bits.

        Raises
        ------
        ValueError
            If the key is shorter than the message.
        """
        if len(quantum_key) < len(message_bits):
            raise ValueError(
                f"Key length ({len(quantum_key)}) < message length "
                f"({len(message_bits)}). OTP requires key ≥ message."
            )
        return np.bitwise_xor(message_bits, quantum_key[:len(message_bits)])

    @staticmethod
    def decrypt(ciphertext: np.ndarray,
                quantum_key: np.ndarray) -> np.ndarray:
        """Decrypt ciphertext with the same quantum key (XOR).

        Parameters
        ----------
        ciphertext : np.ndarray of int
        quantum_key : np.ndarray of int

        Returns
        -------
        np.ndarray of int
            The plaintext bits.
        """
        return np.bitwise_xor(ciphertext, quantum_key[:len(ciphertext)])

    # ----- full pipeline ----------------------------------------------------

    def secure_communicate(self, message: str,
                           protocol: str = "bb84"
                           ) -> CommunicationResult:
        """End-to-end secure communication pipeline.

        1. Encode the plaintext to binary.
        2. Generate a quantum key via the chosen QKD protocol.
        3. Encrypt with XOR.
        4. Decrypt with XOR.
        5. Verify integrity.

        Parameters
        ----------
        message : str
            The plaintext message to send securely.
        protocol : str
            ``'bb84'`` or ``'e91'``.

        Returns
        -------
        CommunicationResult
        """
        message_bits = self.encode_message(message)
        needed = len(message_bits)

        # Generate enough key material
        # We request more qubits than needed because sifting + privacy amp
        # reduce the usable key length.
        if protocol.lower() == "bb84":
            qkd = BB84Protocol(seed=self.seed)
            # Need roughly 8× qubits after sifting + amplification
            n_qubits = max(needed * 10, 512)
            key, stats = qkd.run_protocol(n_qubits, eve_present=False)

            # If key is too short, run again with more qubits
            attempts = 0
            while len(key) < needed and attempts < 5:
                n_qubits *= 2
                qkd = BB84Protocol(seed=None)  # fresh randomness
                key, stats = qkd.run_protocol(n_qubits, eve_present=False)
                attempts += 1

        elif protocol.lower() == "e91":
            qkd_e91 = E91Protocol(seed=self.seed)
            n_pairs = max(needed * 15, 1024)
            key, stats = qkd_e91.run_protocol(n_pairs)

            attempts = 0
            while len(key) < needed and attempts < 5:
                n_pairs *= 2
                qkd_e91 = E91Protocol(seed=None)
                key, stats = qkd_e91.run_protocol(n_pairs)
                attempts += 1
        else:
            return CommunicationResult(
                plaintext=message,
                ciphertext_bits=np.array([], dtype=int),
                decrypted_text="",
                key_bits=np.array([], dtype=int),
                key_length=0,
                protocol_used=protocol,
                success=False,
                error_message=f"Unknown protocol: '{protocol}'. Use 'bb84' or 'e91'."
            )

        if len(key) < needed:
            return CommunicationResult(
                plaintext=message,
                ciphertext_bits=np.array([], dtype=int),
                decrypted_text="",
                key_bits=key,
                key_length=len(key),
                protocol_used=protocol,
                success=False,
                error_message=(
                    f"Could not generate enough key material. "
                    f"Need {needed} bits, got {len(key)}."
                )
            )

        # Encrypt → transmit → decrypt
        ciphertext = self.encrypt(message_bits, key)
        decrypted_bits = self.decrypt(ciphertext, key)
        decrypted_text = self.decode_message(decrypted_bits)

        return CommunicationResult(
            plaintext=message,
            ciphertext_bits=ciphertext,
            decrypted_text=decrypted_text,
            key_bits=key[:needed],
            key_length=needed,
            protocol_used=protocol,
            success=(decrypted_text == message),
        )
