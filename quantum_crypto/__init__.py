"""
Quantum Cryptography Module
============================

Implements quantum key distribution (QKD) protocols and quantum-secure
encryption primitives using statevector simulation with NumPy.

Protocols:
    - BB84Protocol: Bennett-Brassard 1984 QKD protocol
    - E91Protocol:  Ekert 1991 entanglement-based QKD protocol
    - QuantumOTP:   Quantum One-Time Pad encryption

All quantum operations are simulated via pure NumPy/SciPy matrix algebra—
no external quantum computing libraries are required.
"""

from quantum_crypto.bb84 import BB84Protocol
from quantum_crypto.e91 import E91Protocol
from quantum_crypto.quantum_otp import QuantumOTP

__all__ = ["BB84Protocol", "E91Protocol", "QuantumOTP"]
__version__ = "1.0.0"
