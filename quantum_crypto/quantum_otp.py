from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from quantum_crypto.bb84 import BB84Protocol
from quantum_crypto.e91 import E91Protocol


@dataclass
class CommunicationResult:
    plaintext: str
    ciphertext_bits: np.ndarray
    decrypted_text: str
    key_bits: np.ndarray
    key_length: int
    protocol_used: str
    success: bool
    error_message: str = ""


class QuantumOTP:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed


    @staticmethod
    def encode_message(text: str) -> np.ndarray:
        bits = []
        for ch in text:
            byte = format(ord(ch), "08b")
            bits.extend(int(b) for b in byte)
        return np.array(bits, dtype=int)

    @staticmethod
    def decode_message(binary: np.ndarray) -> str:
        binary = np.asarray(binary, dtype=int)
        chars = []
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            if len(byte) < 8:
                break
            value = int("".join(str(b) for b in byte), 2)
            chars.append(chr(value))
        return "".join(chars)


    @staticmethod
    def encrypt(message_bits: np.ndarray,
                quantum_key: np.ndarray) -> np.ndarray:
        if len(quantum_key) < len(message_bits):
            raise ValueError(
                f"Key length ({len(quantum_key)}) < message length "
                f"({len(message_bits)}). OTP requires key ≥ message."
            )
        return np.bitwise_xor(message_bits, quantum_key[:len(message_bits)])

    @staticmethod
    def decrypt(ciphertext: np.ndarray,
                quantum_key: np.ndarray) -> np.ndarray:
        return np.bitwise_xor(ciphertext, quantum_key[:len(ciphertext)])


    def secure_communicate(self, message: str,
                           protocol: str = "bb84"
                           ) -> CommunicationResult:
        message_bits = self.encode_message(message)
        needed = len(message_bits)

        if protocol.lower() == "bb84":
            qkd = BB84Protocol(seed=self.seed)
            n_qubits = max(needed * 10, 512)
            key, stats = qkd.run_protocol(n_qubits, eve_present=False)

            attempts = 0
            while len(key) < needed and attempts < 5:
                n_qubits *= 2
                qkd = BB84Protocol(seed=None)
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
