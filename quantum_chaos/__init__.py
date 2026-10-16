"""Signatures of quantum chaos.

Includes the quantum kicked top, random-matrix level-spacing statistics and
out-of-time-order correlators (OTOCs) for estimating quantum Lyapunov
exponents.
"""

from __future__ import annotations

from .kicked_top import KickedTop
from .level_spacing import LevelSpacing
from .lyapunov import LyapunovEstimator

__all__ = ["KickedTop", "LevelSpacing", "LyapunovEstimator"]
__version__ = "1.1.0"
