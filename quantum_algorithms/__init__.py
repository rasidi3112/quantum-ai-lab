"""
Quantum Algorithms Module
==========================

Pure-NumPy statevector simulations of landmark quantum algorithms.

Algorithms:
    - GroverSearch:   Grover's unstructured search / amplitude amplification
    - ShorFactoring:  Shor's integer factoring algorithm
    - QFT:            Quantum Fourier Transform & Phase Estimation
    - DeutschJozsa:   Deutsch–Jozsa algorithm

All quantum operations are implemented as matrix algebra on complex NumPy
arrays—no external quantum computing frameworks required.
"""

from quantum_algorithms.grover import GroverSearch
from quantum_algorithms.shor import ShorFactoring
from quantum_algorithms.qft import QFT
from quantum_algorithms.deutsch_jozsa import DeutschJozsa

__all__ = ["GroverSearch", "ShorFactoring", "QFT", "DeutschJozsa"]
__version__ = "1.0.0"
