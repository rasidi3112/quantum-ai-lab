"""Textbook quantum algorithms as statevector simulations.

Includes Shor's factoring algorithm, Grover search and amplitude
amplification, the quantum Fourier transform with phase estimation, and the
Deutsch–Jozsa algorithm.
"""

from __future__ import annotations

from quantum_algorithms.deutsch_jozsa import DeutschJozsa
from quantum_algorithms.grover import GroverSearch
from quantum_algorithms.qft import QFT
from quantum_algorithms.shor import ShorFactoring

__all__ = ["GroverSearch", "ShorFactoring", "QFT", "DeutschJozsa"]
__version__ = "1.1.0"
