"""Quantum reinforcement learning.

Includes a data-re-uploading quantum policy, a hybrid classical–quantum
REINFORCE agent, and dependency-free CartPole and FrozenLake environments.
``QuantumEnvironment`` is an alias of :class:`CartPoleEnv`.
"""

from __future__ import annotations

from quantum_rl.hybrid_agent import HybridAgent
from quantum_rl.quantum_environment import CartPoleEnv, FrozenLakeEnv
from quantum_rl.quantum_policy import QuantumPolicy

QuantumEnvironment = CartPoleEnv

__all__ = [
    "QuantumPolicy",
    "HybridAgent",
    "QuantumEnvironment",
    "CartPoleEnv",
    "FrozenLakeEnv",
]

__version__ = "1.1.0"
