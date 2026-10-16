"""Quantum key distribution and quantum-secured encryption.

Includes the prepare-and-measure BB84 protocol, the entanglement-based E91
protocol with a CHSH Bell test, and a one-time pad that consumes the
distributed keys.
"""

from __future__ import annotations

from quantum_crypto.bb84 import BB84Protocol
from quantum_crypto.e91 import E91Protocol
from quantum_crypto.quantum_otp import QuantumOTP

__all__ = ["BB84Protocol", "E91Protocol", "QuantumOTP"]
__version__ = "1.1.0"
