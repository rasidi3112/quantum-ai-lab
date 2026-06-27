# Quantum AI Laboratory — Architecture

## Overview

The Quantum AI Laboratory is a monorepo containing 8 self-contained research modules, each exploring a different facet of quantum computing and artificial intelligence. All modules share a common mathematical foundation built on statevector simulation.

## Design Principles

### 1. Pure Simulation
All quantum operations are simulated using NumPy/SciPy without any dependency on quantum hardware or quantum SDKs. This makes the codebase:
- **Portable**: Runs on any machine with Python 3.10+
- **Debuggable**: Full access to quantum state at every step
- **Educational**: Transparent implementation of every gate and operation

### 2. Statevector Representation
Quantum states are represented as complex NumPy arrays:
```python
# |0⟩ state for 1 qubit
state = np.array([1.0 + 0j, 0.0 + 0j])

# |00⟩ state for 2 qubits
state = np.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j])

# General n-qubit state: 2^n dimensional complex vector
state = np.zeros(2**n, dtype=complex)
state[0] = 1.0  # |00...0⟩
```

### 3. Gate Operations as Matrices
Quantum gates are represented as unitary matrices:
```python
# Single-qubit gates
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)     # Hadamard
X = np.array([[0, 1], [1, 0]])                     # Pauli-X
Z = np.array([[1, 0], [0, -1]])                    # Pauli-Z
RY = lambda θ: np.array([[np.cos(θ/2), -np.sin(θ/2)],
                          [np.sin(θ/2),  np.cos(θ/2)]])

# Multi-qubit: Kronecker product
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
```

### 4. Modular Independence
Each module is self-contained with its own:
- `__init__.py` with clean exports
- `README.md` with documentation
- `examples/` with runnable demos

## Module Dependency Graph

```
                    ┌─────────────┐
                    │  NumPy/SciPy │
                    │  Foundation  │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
    ┌───────▼──────┐ ┌─────▼─────┐ ┌─────▼──────┐
    │  Algorithms  │ │ Simulation│ │   Quantum   │
    │  (Core)      │ │ (Physics) │ │     ML      │
    └───┬───┬──────┘ └──┬────┬──┘ └──┬──────┬───┘
        │   │           │    │       │      │
        │   │     ┌─────┘    │       │      │
   ┌────▼┐ ┌▼─────▼──┐  ┌───▼───┐ ┌─▼──┐ ┌─▼────┐
   │Crypt│ │Optimiz.  │  │ Chaos │ │ RL │ │Vision│
   └─────┘ └──────────┘  └───────┘ └────┘ └──────┘
```

## Computational Complexity

| Module | Space Complexity | Time Complexity | Max Practical Size |
|--------|-----------------|-----------------|-------------------|
| Algorithms | O(2ⁿ) | O(2ⁿ · poly(n)) | ~20 qubits |
| Simulation | O(2ⁿ) | O(2²ⁿ) | ~12 sites |
| ML | O(2ⁿ) | O(2ⁿ · params) | ~10 qubits |
| Optimization | O(2ⁿ) | O(2ⁿ · iterations) | ~16 qubits |
| Chaos | O((2j+1)²) | O((2j+1)³) | j ≈ 50 |
| Crypto | O(1) per qubit | O(n) | Unlimited |
| RL | O(2ⁿ) | O(episodes · 2ⁿ) | ~8 qubits |
| Vision | O(2ⁿ) | O(patches · 2ⁿ) | ~8 qubits |

## Data Flow

### Typical Quantum Algorithm Pipeline
```
Classical Input → Encoding → Quantum Circuit → Measurement → Classical Output
     x ∈ ℝⁿ        |ψ(x)⟩     U(θ)|ψ(x)⟩     ⟨ψ|M|ψ⟩       y ∈ ℝᵐ
```

### Variational Algorithm Pipeline
```
                    ┌──────────────────────────┐
                    │    Classical Optimizer    │
                    │   (COBYLA / L-BFGS-B)    │
                    └────────┬─────────┬───────┘
                        params θ    gradient ∇L
                             │         ↑
                    ┌────────▼─────────┤
                    │  Quantum Circuit │
                    │    U(θ)|ψ₀⟩     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Measurement   │
                    │    ⟨O⟩ = f(θ)   │
                    └─────────────────┘
```

## Testing Strategy

Each module can be verified independently:
```bash
# Run all module examples
python -m quantum_ml.examples.iris_classification
python -m quantum_chaos.examples.chaos_visualization
python -m quantum_crypto.examples.secure_channel_demo
python -m quantum_algorithms.examples.factoring_demo
python -m quantum_simulation.examples.phase_transition_demo
python -m quantum_optimization.examples.portfolio_optimization
python -m quantum_rl.examples.cartpole_quantum
python -m quantum_vision.examples.mnist_quantum
```
