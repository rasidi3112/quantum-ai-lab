"""Quantum many-body simulation.

Includes the transverse-field Ising chain, the Fermi–Hubbard model via the
Jordan–Wigner transformation, and Trotter–Suzuki product formulas for time
evolution.
"""

from __future__ import annotations

from quantum_simulation.hubbard_model import HubbardModel
from quantum_simulation.ising_model import IsingModel
from quantum_simulation.trotter import TrotterEvolution

__all__ = ["IsingModel", "HubbardModel", "TrotterEvolution"]
__version__ = "1.1.0"
