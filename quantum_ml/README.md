# Quantum Machine Learning

> *Quantum-enhanced feature spaces and variational classifiers*

## Overview

This module implements quantum machine learning algorithms that leverage quantum Hilbert spaces for classification and feature extraction. All algorithms use pure NumPy statevector simulation.

## Components

### `QuantumKernel` — Quantum Kernel Methods
Implements quantum kernel estimation using the ZZFeatureMap encoding. The quantum kernel is defined as:

$$K(x_1, x_2) = |\langle\phi(x_1)|\phi(x_2)\rangle|^2$$

```python
from quantum_ml import QuantumKernel

qk = QuantumKernel(n_qubits=2, n_layers=2)
K = qk.kernel_matrix(X_train)
predictions = qk.classify_svm(X_train, y_train, X_test)
```

### `VariationalClassifier` — Variational Quantum Classifier
A parameterized quantum circuit that learns classification through variational optimization:

```python
from quantum_ml import VariationalClassifier

clf = VariationalClassifier(n_qubits=4, n_layers=2)
result = clf.train(X_train, y_train, epochs=100)
accuracy = clf.accuracy(X_test, y_test)
```

### `QuantumNeuralNetwork` — Quantum Neural Network Layer
Configurable QNN with multiple encoding strategies and measurement options:

```python
from quantum_ml import QuantumNeuralNetwork

qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2, encoding='angle')
output = qnn.forward(x_input)
grad = qnn.gradient(x_input)  # Parameter-shift rule
```

## References

1. Havlíček, V., et al. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*, 567, 209-212.
2. Schuld, M., & Petruccione, F. (2021). *Machine Learning with Quantum Computers*. Springer.
3. Farhi, E., & Neven, H. (2018). Classification with quantum neural networks on near term processors. *arXiv:1802.06002*.
