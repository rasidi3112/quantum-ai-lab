from quantum_rl.quantum_policy import QuantumPolicy
from quantum_rl.hybrid_agent import HybridAgent
from quantum_rl.quantum_environment import CartPoleEnv, FrozenLakeEnv

QuantumEnvironment = CartPoleEnv

__all__ = [
    "QuantumPolicy",
    "HybridAgent",
    "QuantumEnvironment",
    "CartPoleEnv",
    "FrozenLakeEnv",
]

__version__ = "0.1.0"
