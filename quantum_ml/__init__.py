"""Quantum machine learning.

Includes quantum kernels built from a ZZ feature map, a variational quantum
classifier and a quantum neural network layer with parameter-shift
gradients.
"""

from __future__ import annotations

from quantum_ml.qkernel import QuantumKernel
from quantum_ml.quantum_neural_net import QuantumNeuralNetwork
from quantum_ml.variational_classifier import VariationalClassifier

__all__ = ["QuantumKernel", "VariationalClassifier", "QuantumNeuralNetwork"]
__version__ = "1.1.0"
