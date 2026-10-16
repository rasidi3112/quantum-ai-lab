"""Quantum computer vision.

Includes image-to-state encoders (amplitude, angle, FRQI, NEQR),
quanvolutional filters, and a hybrid MLP -> variational circuit classifier
trained end to end with the parameter-shift rule.
"""

from __future__ import annotations

from quantum_vision.hybrid_classifier import HybridClassifier
from quantum_vision.quantum_encoder import QuantumEncoder
from quantum_vision.quanvolution import Quanvolution

__all__ = [
    "Quanvolution",
    "QuantumEncoder",
    "HybridClassifier",
]

__version__ = "1.1.0"
