"""
Quantum Optimization Module
============================

Variational and hybrid quantum-classical optimization algorithms, including:

* **QAOA** – Quantum Approximate Optimization Algorithm for combinatorial problems.
* **VQE** – Variational Quantum Eigensolver with multiple ansatz options.
* **MaxCutSolver** – End-to-end MaxCut pipeline (graph → Hamiltonian → solution).

All circuits are simulated via dense statevector evolution using NumPy / SciPy.
"""

from quantum_optimization.qaoa import QAOA
from quantum_optimization.vqe import VQE
from quantum_optimization.max_cut import MaxCutSolver

__all__ = ["QAOA", "VQE", "MaxCutSolver"]
__version__ = "0.1.0"
