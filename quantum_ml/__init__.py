"""
Quantum Machine Learning Module
================================

Implements quantum machine learning algorithms using statevector simulation:

- QuantumKernel: Quantum kernel methods with ZZFeatureMap
- VariationalClassifier: Variational Quantum Classifier (VQC)
- QuantumNeuralNetwork: Parameterized quantum neural network layers

All implementations use pure NumPy/SciPy — no quantum hardware required.

References:
    - Havlíček et al. (2019). Supervised learning with quantum-enhanced feature spaces.
    - Schuld & Petruccione (2021). Machine Learning with Quantum Computers.
"""

from quantum_ml.qkernel import QuantumKernel
from quantum_ml.variational_classifier import VariationalClassifier
from quantum_ml.quantum_neural_net import QuantumNeuralNetwork

__all__ = ["QuantumKernel", "VariationalClassifier", "QuantumNeuralNetwork"]
__version__ = "1.0.0"
