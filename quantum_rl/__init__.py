"""
Quantum Reinforcement Learning Module
======================================

This module implements hybrid classical-quantum reinforcement learning
algorithms using variational quantum circuits as policy networks.

Key Components:
    - QuantumPolicy: Parameterized quantum circuit acting as a policy function
    - HybridAgent: Classical-quantum hybrid RL agent using REINFORCE
    - QuantumEnvironment: Built-in environments (CartPole, FrozenLake)

The approach uses variational quantum circuits (VQCs) where:
    1. Classical observations are encoded into quantum states
    2. Parameterized gates process the quantum information
    3. Measurements produce action probabilities

References:
    - Jerbi et al., "Parametrized quantum policies for reinforcement learning" (2021)
    - Skolik et al., "Quantum agents in the Gym" (2022)
    - Lockwood & Si, "Reinforcement Learning with Quantum Variational Circuits" (2020)
"""

from quantum_rl.quantum_policy import QuantumPolicy
from quantum_rl.hybrid_agent import HybridAgent
from quantum_rl.quantum_environment import CartPoleEnv, FrozenLakeEnv

# Convenience alias
QuantumEnvironment = CartPoleEnv

__all__ = [
    "QuantumPolicy",
    "HybridAgent",
    "QuantumEnvironment",
    "CartPoleEnv",
    "FrozenLakeEnv",
]

__version__ = "0.1.0"
