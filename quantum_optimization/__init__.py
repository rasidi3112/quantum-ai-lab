"""Variational quantum optimization.

Includes QAOA with the standard transverse-field mixer and a
Hamming-weight-preserving XY mixer, the variational quantum eigensolver, and
a MaxCut toolkit.
"""

from __future__ import annotations

from quantum_optimization.max_cut import MaxCutSolver
from quantum_optimization.qaoa import QAOA
from quantum_optimization.vqe import VQE

__all__ = ["QAOA", "VQE", "MaxCutSolver"]
__version__ = "1.1.0"
