"""
Quantum Vision Module
=====================

This module implements quantum-enhanced computer vision techniques including
quanvolutional neural networks, quantum image encoding schemes, and hybrid
classical-quantum image classifiers.

Key Components:
    - Quanvolution: Quantum convolutional filters for image feature extraction
    - QuantumEncoder: Multiple quantum image encoding strategies (FRQI, NEQR, etc.)
    - HybridClassifier: Classical-quantum hybrid image classifier

The quanvolutional approach replaces classical convolution kernels with
parameterized quantum circuits that can capture correlations unreachable
by small classical filters.

References:
    - Henderson et al., "Quanvolutional Neural Networks" (2020)
    - Le et al., "A Flexible Representation of Quantum Images" (2011)
    - Zhang et al., "NEQR: A Novel Enhanced Quantum Representation" (2013)
"""

from quantum_vision.quanvolution import Quanvolution
from quantum_vision.quantum_encoder import QuantumEncoder
from quantum_vision.hybrid_classifier import HybridClassifier

__all__ = [
    "Quanvolution",
    "QuantumEncoder",
    "HybridClassifier",
]

__version__ = "0.1.0"
