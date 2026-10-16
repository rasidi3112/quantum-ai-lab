"""One-time-pad encryption with keys from quantum key distribution.

The message is UTF-8 encoded, XOR-ed with key bits produced by BB84 or E91,
and decrypted with the same key. With a truly random, never-reused key at
least as long as the message, the one-time pad is information-theoretically
secure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from quantum_crypto.bb84 import BB84Protocol
from quantum_crypto.e91 import E91Protocol


@dataclass
class CommunicationResult:
    """Outcome of :meth:`QuantumOTP.secure_communicate`.

    Attributes
    ----------
    plaintext : str
        Original message.
    ciphertext_bits : numpy.ndarray
        Encrypted bits.
    decrypted_text : str
        Message recovered by the receiver.
    key_bits : numpy.ndarray
        Key bits consumed.
    key_length : int
        Number of key bits consumed.
    protocol_used : str
        ``"bb84"`` or ``"e91"``.
    success : bool
        Whether the decrypted text equals the plaintext.
    error_message : str
        Reason for failure, if any.
    """

    plaintext: str
    ciphertext_bits: np.ndarray
    decrypted_text: str
    key_bits: np.ndarray
    key_length: int
    protocol_used: str
    success: bool
    error_message: str = ""


class QuantumOTP:
    """One-time pad keyed by a quantum key distribution protocol.

    Parameters
    ----------
    seed : int, optional
        Seed passed to the first key-distribution run.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed


    @staticmethod
    def encode_message(text: str) -> np.ndarray:
        """Convert text to bits via UTF-8.

        Parameters
        ----------
        text : str
            Message; any Unicode text is supported.

        Returns
        -------
        numpy.ndarray
            Bits, most significant bit of each byte first.
        """
        data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        return np.unpackbits(data).astype(int)

    @staticmethod
    def decode_message(binary: np.ndarray) -> str:
        """Convert bits back to text via UTF-8.

        Parameters
        ----------
        binary : numpy.ndarray
            Bits; a trailing partial byte is ignored.

        Returns
        -------
        str
            Decoded text (invalid sequences become U+FFFD).
        """
        binary = np.asarray(binary, dtype=np.uint8)
        usable = len(binary) - len(binary) % 8
        data = np.packbits(binary[:usable]).tobytes()
        return str(data.decode("utf-8", errors="replace"))


    @staticmethod
    def encrypt(message_bits: np.ndarray,
                quantum_key: np.ndarray) -> np.ndarray:
        """XOR message bits with the key.

        Parameters
        ----------
        message_bits : numpy.ndarray
            Plaintext bits.
        quantum_key : numpy.ndarray
            Key bits; must be at least as long as the message.

        Returns
        -------
        numpy.ndarray
            Ciphertext bits.

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
        """XOR ciphertext bits with the key to recover the plaintext bits.

        Parameters
        ----------
        ciphertext : numpy.ndarray
            Encrypted bits.
        quantum_key : numpy.ndarray
            The key used for encryption.

        Returns
        -------
        numpy.ndarray
            Plaintext bits.
        """
        return np.bitwise_xor(ciphertext, quantum_key[:len(ciphertext)])


    def secure_communicate(self, message: str,
                           protocol: Literal["bb84", "e91"] = "bb84"
                           ) -> CommunicationResult:
        """Distribute a key and send a message with it.

        Runs the chosen QKD protocol, enlarging the run (up to five times) until
        enough key bits are available, then encrypts and decrypts the message.

        Parameters
        ----------
        message : str
            Text to send.
        protocol : {"bb84", "e91"}, default "bb84"
            Key-distribution protocol.

        Returns
        -------
        CommunicationResult
        """
        message_bits = self.encode_message(message)
        needed = len(message_bits)

        if protocol.lower() == "bb84":
            qkd = BB84Protocol(seed=self.seed)
            n_qubits = max(needed * 10, 512)
            key, _ = qkd.run_protocol(n_qubits, eve_present=False)

            attempts = 0
            while len(key) < needed and attempts < 5:
                n_qubits *= 2
                qkd = BB84Protocol(seed=None)
                key, _ = qkd.run_protocol(n_qubits, eve_present=False)
                attempts += 1

        elif protocol.lower() == "e91":
            qkd_e91 = E91Protocol(seed=self.seed)
            n_pairs = max(needed * 15, 1024)
            key, _ = qkd_e91.run_protocol(n_pairs)

            attempts = 0
            while len(key) < needed and attempts < 5:
                n_pairs *= 2
                qkd_e91 = E91Protocol(seed=None)
                key, _ = qkd_e91.run_protocol(n_pairs)
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
