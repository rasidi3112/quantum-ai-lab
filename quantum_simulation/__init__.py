"""
Quantum Simulation Module
=========================

Provides statevector-based simulation of quantum many-body systems, including:

* **IsingModel** – Transverse-field Ising model on a 1-D chain (or ring).
* **HubbardModel** – Single-band Fermi-Hubbard model via Jordan-Wigner mapping.
* **TrotterEvolution** – Product-formula time-evolution (1st, 2nd, 4th order).

All operators are built as dense NumPy matrices so the module requires only
``numpy``, ``scipy``, and (optionally) ``matplotlib`` for plotting.
"""

from quantum_simulation.ising_model import IsingModel
from quantum_simulation.hubbard_model import HubbardModel
from quantum_simulation.trotter import TrotterEvolution

__all__ = ["IsingModel", "HubbardModel", "TrotterEvolution"]
__version__ = "0.1.0"
