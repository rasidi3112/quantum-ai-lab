"""
Variational Quantum Eigensolver (VQE)
======================================

Implements VQE with two ansatz options:

1. **Hardware-efficient ansatz** – alternating layers of single-qubit
   RY-RZ rotations and CNOT entangling gates (ladder topology).

2. **Simplified UCCSD-like ansatz** – parameterised single and double
   excitation operators applied via Trotterised unitary.

The energy :math:`E(\\boldsymbol\\theta) = \\langle 0|U^\\dagger(\\boldsymbol\\theta)\\,
H\\,U(\\boldsymbol\\theta)|0\\rangle` is minimised using a classical
optimiser (default: L-BFGS-B).

References
----------
* Peruzzo *et al.*, Nature Communications **5**, 4213 (2014).
* Kandala *et al.*, Nature **549**, 242 (2017).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la
from scipy.optimize import minimize


# ======================================================================
# Gate primitives (dense matrices)
# ======================================================================

_I2 = np.eye(2, dtype=np.complex128)

def _ry(theta: float) -> NDArray[np.complex128]:
    """Single-qubit RY(θ) rotation."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)

def _rz(theta: float) -> NDArray[np.complex128]:
    """Single-qubit RZ(θ) rotation."""
    return np.array(
        [[np.exp(-1j * theta / 2), 0],
         [0, np.exp(1j * theta / 2)]],
        dtype=np.complex128,
    )

def _rx(theta: float) -> NDArray[np.complex128]:
    """Single-qubit RX(θ) rotation."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)

def _cnot_matrix(n_qubits: int, control: int, target: int) -> NDArray[np.complex128]:
    """Full CNOT matrix on *n_qubits* with given control and target."""
    dim = 2 ** n_qubits
    U = np.zeros((dim, dim), dtype=np.complex128)
    for basis in range(dim):
        bits = list(format(basis, f"0{n_qubits}b"))
        if bits[control] == '1':
            bits[target] = '0' if bits[target] == '1' else '1'
        new_basis = int(''.join(bits), 2)
        U[new_basis, basis] = 1.0
    return U

def _kron_chain(*ops: NDArray[np.complex128]) -> NDArray[np.complex128]:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def _apply_single_qubit(
    gate: NDArray[np.complex128],
    qubit: int,
    n_qubits: int,
) -> NDArray[np.complex128]:
    """Full matrix for a single-qubit gate on the given qubit."""
    ops = [_I2] * n_qubits
    ops[qubit] = gate
    return _kron_chain(*ops)


# ======================================================================
# VQE class
# ======================================================================

class VQE:
    """Variational Quantum Eigensolver.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.convergence_history: List[float] = []
        self.optimal_params: Optional[NDArray[np.float64]] = None
        self.optimal_energy: Optional[float] = None
        self.optimal_state: Optional[NDArray[np.complex128]] = None

    # ----------------------------------------------------------------
    # Ansätze
    # ----------------------------------------------------------------

    @staticmethod
    def hardware_efficient_ansatz(
        n_qubits: int,
        n_layers: int,
        params: NDArray[np.float64],
    ) -> NDArray[np.complex128]:
        r"""Hardware-efficient ansatz: RY-RZ layers + CNOT ladder.

        Each layer has 2 parameters per qubit (RY, RZ) followed by a
        CNOT ladder connecting qubit *i* → qubit *i+1*.

        Total parameters = ``2 * n_qubits * n_layers``.

        Parameters
        ----------
        n_qubits : int
        n_layers : int
        params : ndarray, shape (2 * n_qubits * n_layers,)

        Returns
        -------
        ndarray, shape (2^n,)
            State vector :math:`U(\\theta)|0\\rangle`.
        """
        dim = 2 ** n_qubits
        psi = np.zeros(dim, dtype=np.complex128)
        psi[0] = 1.0  # |00…0⟩

        idx = 0
        for layer in range(n_layers):
            # Single-qubit rotations
            for q in range(n_qubits):
                U_ry = _apply_single_qubit(_ry(params[idx]), q, n_qubits)
                psi = U_ry @ psi
                idx += 1
                U_rz = _apply_single_qubit(_rz(params[idx]), q, n_qubits)
                psi = U_rz @ psi
                idx += 1

            # CNOT ladder
            for q in range(n_qubits - 1):
                U_cx = _cnot_matrix(n_qubits, q, q + 1)
                psi = U_cx @ psi

        return psi

    @staticmethod
    def uccsd_ansatz(
        n_qubits: int,
        n_electrons: int,
        params: NDArray[np.float64],
    ) -> NDArray[np.complex128]:
        r"""Simplified UCCSD-like ansatz.

        Generates single and double excitation operators as anti-Hermitian
        generators and exponentiates them.  This is a *simplified* version
        that uses a Hartree-Fock reference with the first *n_electrons*
        qubits set to |1⟩.

        Number of parameters = n_singles + n_doubles where
        n_singles = n_occ × n_virt, n_doubles = n_occ*(n_occ-1)/2 * n_virt*(n_virt-1)/2.

        Parameters
        ----------
        n_qubits : int
        n_electrons : int
        params : ndarray

        Returns
        -------
        ndarray, shape (2^n,)
        """
        dim = 2 ** n_qubits

        # Hartree-Fock reference: first n_electrons qubits in |1⟩
        hf_index = 0
        for q in range(n_electrons):
            hf_index |= (1 << (n_qubits - 1 - q))
        psi = np.zeros(dim, dtype=np.complex128)
        psi[hf_index] = 1.0

        n_occ = n_electrons
        n_virt = n_qubits - n_electrons

        # Build generator T - T†
        T = np.zeros((dim, dim), dtype=np.complex128)

        idx = 0

        # Single excitations  |occ⟩ → |virt⟩
        for i in range(n_occ):
            for a in range(n_occ, n_qubits):
                if idx >= len(params):
                    break
                # Simple excitation operator |a⟩⟨i|  (in computational basis)
                for basis in range(dim):
                    bits = list(format(basis, f"0{n_qubits}b"))
                    if bits[i] == '1' and bits[a] == '0':
                        new_bits = bits.copy()
                        new_bits[i] = '0'
                        new_bits[a] = '1'
                        new_basis = int(''.join(new_bits), 2)
                        T[new_basis, basis] += params[idx]
                idx += 1

        # Double excitations  |occ_i, occ_j⟩ → |virt_a, virt_b⟩
        for i in range(n_occ):
            for j in range(i + 1, n_occ):
                for a in range(n_occ, n_qubits):
                    for b in range(a + 1, n_qubits):
                        if idx >= len(params):
                            break
                        for basis in range(dim):
                            bits = list(format(basis, f"0{n_qubits}b"))
                            if (bits[i] == '1' and bits[j] == '1'
                                    and bits[a] == '0' and bits[b] == '0'):
                                new_bits = bits.copy()
                                new_bits[i] = '0'
                                new_bits[j] = '0'
                                new_bits[a] = '1'
                                new_bits[b] = '1'
                                new_basis = int(''.join(new_bits), 2)
                                T[new_basis, basis] += params[idx]
                        idx += 1

        # Anti-Hermitian generator
        generator = T - T.conj().T
        U = la.expm(generator)
        psi = U @ psi
        return psi

    # ----------------------------------------------------------------
    # Energy evaluation
    # ----------------------------------------------------------------

    @staticmethod
    def energy_evaluation(
        params: NDArray[np.float64],
        hamiltonian: NDArray[np.complex128],
        ansatz_fn: Callable[..., NDArray[np.complex128]],
        *ansatz_args: Any,
    ) -> float:
        r"""Compute :math:`\langle\psi(\theta)|H|\psi(\theta)\rangle`.

        Parameters
        ----------
        params : ndarray
            Variational parameters.
        hamiltonian : ndarray
            Hamiltonian matrix.
        ansatz_fn : callable
            Function that returns the state vector given params.
        *ansatz_args
            Extra positional arguments for *ansatz_fn* (inserted before params).

        Returns
        -------
        float
            Energy expectation value.
        """
        psi = ansatz_fn(*ansatz_args, params)
        return float(np.real(psi.conj() @ hamiltonian @ psi))

    # ----------------------------------------------------------------
    # Optimisation
    # ----------------------------------------------------------------

    def optimize(
        self,
        hamiltonian: NDArray[np.complex128],
        ansatz: str = "hardware_efficient",
        n_layers: int = 2,
        n_electrons: int = 1,
        method: str = "L-BFGS-B",
        maxiter: int = 500,
        initial_params: Optional[NDArray[np.float64]] = None,
    ) -> Dict[str, Any]:
        """Run VQE optimisation.

        Parameters
        ----------
        hamiltonian : ndarray
            Hamiltonian to find the ground-state energy of.
        ansatz : {'hardware_efficient', 'uccsd'}
        n_layers : int
            Layers for hardware-efficient ansatz.
        n_electrons : int
            Electrons for UCCSD ansatz.
        method : str
            Classical optimiser.
        maxiter : int
        initial_params : ndarray, optional

        Returns
        -------
        dict
            ``{'energy', 'params', 'state', 'history', 'result'}``
        """
        self.convergence_history = []

        if ansatz == "hardware_efficient":
            n_params = 2 * self.n_qubits * n_layers
            ansatz_fn = self.hardware_efficient_ansatz
            ansatz_args: tuple = (self.n_qubits, n_layers)
        elif ansatz == "uccsd":
            n_occ = n_electrons
            n_virt = self.n_qubits - n_electrons
            n_singles = n_occ * n_virt
            n_doubles = (n_occ * (n_occ - 1) // 2) * (n_virt * (n_virt - 1) // 2)
            n_params = n_singles + n_doubles
            ansatz_fn = self.uccsd_ansatz  # type: ignore[assignment]
            ansatz_args = (self.n_qubits, n_electrons)
        else:
            raise ValueError(f"Unknown ansatz: {ansatz}")

        if initial_params is None:
            initial_params = np.random.uniform(-0.1, 0.1, n_params)

        def objective(params: NDArray[np.float64]) -> float:
            e = self.energy_evaluation(params, hamiltonian, ansatz_fn, *ansatz_args)
            self.convergence_history.append(e)
            return e

        result = minimize(objective, initial_params, method=method,
                          options={"maxiter": maxiter})

        final_state = ansatz_fn(*ansatz_args, result.x)
        final_energy = float(np.real(final_state.conj() @ hamiltonian @ final_state))

        self.optimal_params = result.x
        self.optimal_energy = final_energy
        self.optimal_state  = final_state

        return {
            "energy": final_energy,
            "params": result.x,
            "state": final_state,
            "history": list(self.convergence_history),
            "result": result,
        }

    # ----------------------------------------------------------------
    # Parameter landscape
    # ----------------------------------------------------------------

    def parameter_landscape(
        self,
        hamiltonian: NDArray[np.complex128],
        base_params: NDArray[np.float64],
        param_idx: int,
        param_range: NDArray[np.float64],
        ansatz: str = "hardware_efficient",
        n_layers: int = 2,
        n_electrons: int = 1,
    ) -> NDArray[np.float64]:
        """1-D energy scan over one parameter.

        Parameters
        ----------
        hamiltonian : ndarray
        base_params : ndarray
            Reference parameter vector.
        param_idx : int
            Index of the parameter to scan.
        param_range : ndarray
            Values to scan.
        ansatz : str
        n_layers : int
        n_electrons : int

        Returns
        -------
        ndarray
            Energy values.
        """
        if ansatz == "hardware_efficient":
            ansatz_fn = self.hardware_efficient_ansatz
            ansatz_args: tuple = (self.n_qubits, n_layers)
        else:
            ansatz_fn = self.uccsd_ansatz  # type: ignore[assignment]
            ansatz_args = (self.n_qubits, n_electrons)

        energies = np.zeros(len(param_range))
        for i, val in enumerate(param_range):
            params = base_params.copy()
            params[param_idx] = val
            energies[i] = self.energy_evaluation(
                params, hamiltonian, ansatz_fn, *ansatz_args
            )
        return energies

    def __repr__(self) -> str:
        return f"VQE(n_qubits={self.n_qubits})"
