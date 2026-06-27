"""
Quantum Chaos Module
=====================

A pure NumPy/SciPy implementation of quantum chaos diagnostics,
including the quantum kicked top, level spacing statistics, and
out-of-time-order correlators (OTOC).

Key Classes:
    - KickedTop: Quantum kicked top model with Husimi Q visualization
    - LevelSpacing: Nearest-neighbor level spacing statistics
    - LyapunovEstimator: OTOC-based quantum Lyapunov exponent estimation
"""

from .kicked_top import KickedTop
from .level_spacing import LevelSpacing
from .lyapunov import LyapunovEstimator

__all__ = ["KickedTop", "LevelSpacing", "LyapunovEstimator"]
__version__ = "0.1.0"
