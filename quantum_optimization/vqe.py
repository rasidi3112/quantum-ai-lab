"""Variational quantum eigensolver (VQE).

Minimizes ``<psi(theta)|H|psi(theta)>`` over a parameterized ansatz to
approximate the ground state of ``H``. Two ansätze are provided: a
hardware-efficient ``RY``/``RZ`` + CNOT-ladder circuit, and a simplified
unitary coupled-cluster singles-and-doubles ansatz.

References
----------
Peruzzo, A. et al. "A variational eigenvalue solver on a photonic quantum
processor." Nat. Commun. 5, 4213 (2014).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la
from scipy.optimize import minimize

_I2 = np.eye(2, dtype=np.complex128)


def _ry(theta: float) -> NDArray[np.complex128]:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(theta: float) -> NDArray[np.complex128]:
    return np.array(
        [[np.exp(-1j * theta / 2), 0],
         [0, np.exp(1j * theta / 2)]],
        dtype=np.complex128,
    )


def _rx(theta: float) -> NDArray[np.complex128]:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _cnot_matrix(n_qubits: int, control: int, target: int) -> NDArray[np.complex128]:
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
    out: NDArray[np.complex128] = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op).astype(np.complex128, copy=False)
    return out


def _apply_single_qubit(
    gate: NDArray[np.complex128],
    qubit: int,
    n_qubits: int,
) -> NDArray[np.complex128]:
    ops = [_I2] * n_qubits
    ops[qubit] = gate
    return _kron_chain(*ops)


class VQE:
    """Variational quantum eigensolver.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    seed : int, optional
        Seed for random initial parameters.

    Attributes
    ----------
    convergence_history : list of float
        Energy at every objective evaluation of the last run.
    optimal_params, optimal_energy, optimal_state
        Results of the last :meth:`optimize` call.
    """

    def __init__(self, n_qubits: int, seed: int | None = None) -> None:
        self.n_qubits = n_qubits
        self.rng = np.random.default_rng(seed)
        self.dim = 2 ** n_qubits
        self.convergence_history: list[float] = []
        self.optimal_params: NDArray[np.float64] | None = None
        self.optimal_energy: float | None = None
        self.optimal_state: NDArray[np.complex128] | None = None


    @staticmethod
    def hardware_efficient_ansatz(
        n_qubits: int,
        n_layers: int,
        params: NDArray[np.float64],
    ) -> NDArray[np.complex128]:
        """Hardware-efficient ansatz state.

        Each layer applies ``RY`` and ``RZ`` on every qubit followed by a CNOT
        ladder, starting from ``|0...0>``.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        n_layers : int
            Number of layers.
        params : numpy.ndarray
            ``2 * n_qubits * n_layers`` angles.

        Returns
        -------
        numpy.ndarray
        """
        dim = 2 ** n_qubits
        psi = np.zeros(dim, dtype=np.complex128)
        psi[0] = 1.0

        idx = 0
        for _ in range(n_layers):
            for q in range(n_qubits):
                U_ry = _apply_single_qubit(_ry(params[idx]), q, n_qubits)
                psi = U_ry @ psi
                idx += 1
                U_rz = _apply_single_qubit(_rz(params[idx]), q, n_qubits)
                psi = U_rz @ psi
                idx += 1

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
        """Simplified UCCSD state ``exp(T - T^dagger)|HF>``.

        Single and double excitations from the first ``n_electrons`` qubits
        (occupied) to the rest (virtual) are applied as qubit excitations, without
        Jordan–Wigner parity signs, so this is a UCCSD-inspired ansatz rather than
        the exact fermionic one.

        Parameters
        ----------
        n_qubits : int
            Number of spin orbitals.
        n_electrons : int
            Number of electrons in the Hartree–Fock reference.
        params : numpy.ndarray
            Singles amplitudes followed by doubles amplitudes.

        Returns
        -------
        numpy.ndarray
        """
        dim = 2 ** n_qubits

        hf_index = 0
        for q in range(n_electrons):
            hf_index |= (1 << (n_qubits - 1 - q))
        psi = np.zeros(dim, dtype=np.complex128)
        psi[hf_index] = 1.0

        n_occ = n_electrons

        T = np.zeros((dim, dim), dtype=np.complex128)

        idx = 0

        for i in range(n_occ):
            for a in range(n_occ, n_qubits):
                if idx >= len(params):
                    break
                for basis in range(dim):
                    bits = list(format(basis, f"0{n_qubits}b"))
                    if bits[i] == '1' and bits[a] == '0':
                        new_bits = bits.copy()
                        new_bits[i] = '0'
                        new_bits[a] = '1'
                        new_basis = int(''.join(new_bits), 2)
                        T[new_basis, basis] += params[idx]
                idx += 1

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

        generator = T - T.conj().T
        U = la.expm(generator)
        psi = U @ psi
        return psi


    @staticmethod
    def energy_evaluation(
        params: NDArray[np.float64],
        hamiltonian: NDArray[np.complex128],
        ansatz_fn: Callable[..., NDArray[np.complex128]],
        *ansatz_args: Any,
    ) -> float:
        """Energy of the ansatz state for given parameters.

        Parameters
        ----------
        params : numpy.ndarray
            Ansatz parameters.
        hamiltonian : numpy.ndarray
            Hermitian matrix.
        ansatz_fn : callable
            Ansatz constructor, called as ``ansatz_fn(*ansatz_args, params)``.
        *ansatz_args
            Leading arguments of ``ansatz_fn``.

        Returns
        -------
        float
        """
        psi = ansatz_fn(*ansatz_args, params)
        return float(np.real(psi.conj() @ hamiltonian @ psi))


    def optimize(
        self,
        hamiltonian: NDArray[np.complex128],
        ansatz: Literal["hardware_efficient", "uccsd"] = "hardware_efficient",
        n_layers: int = 2,
        n_electrons: int = 1,
        method: str = "L-BFGS-B",
        maxiter: int = 500,
        initial_params: NDArray[np.float64] | None = None,
    ) -> dict[str, Any]:
        """Minimize the energy of a Hamiltonian.

        Parameters
        ----------
        hamiltonian : numpy.ndarray
            Hermitian matrix of dimension ``2^n_qubits``.
        ansatz : {"hardware_efficient", "uccsd"}, default "hardware_efficient"
            Ansatz family.
        n_layers : int, default 2
            Layers of the hardware-efficient ansatz.
        n_electrons : int, default 1
            Electrons for the UCCSD ansatz.
        method : str, default "L-BFGS-B"
            Any :func:`scipy.optimize.minimize` method.
        maxiter : int, default 500
            Optimizer iteration limit.
        initial_params : numpy.ndarray, optional
            Starting parameters; small random values by default.

        Returns
        -------
        dict
            ``energy``, ``params``, ``state``, ``history`` and the raw SciPy
            ``result``.

        Raises
        ------
        ValueError
            For an unknown ansatz.
        """
        self.convergence_history = []

        if ansatz == "hardware_efficient":
            n_params = 2 * self.n_qubits * n_layers
            ansatz_fn: Callable[..., NDArray[np.complex128]] = self.hardware_efficient_ansatz
            ansatz_args: tuple = (self.n_qubits, n_layers)
        elif ansatz == "uccsd":
            n_occ = n_electrons
            n_virt = self.n_qubits - n_electrons
            n_singles = n_occ * n_virt
            n_doubles = (n_occ * (n_occ - 1) // 2) * (n_virt * (n_virt - 1) // 2)
            n_params = n_singles + n_doubles
            ansatz_fn = self.uccsd_ansatz
            ansatz_args = (self.n_qubits, n_electrons)
        else:
            raise ValueError(f"Unknown ansatz: {ansatz}")

        if initial_params is None:
            initial_params = self.rng.uniform(-0.1, 0.1, n_params)

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


    def parameter_landscape(
        self,
        hamiltonian: NDArray[np.complex128],
        base_params: NDArray[np.float64],
        param_idx: int,
        param_range: NDArray[np.float64],
        ansatz: Literal["hardware_efficient", "uccsd"] = "hardware_efficient",
        n_layers: int = 2,
        n_electrons: int = 1,
    ) -> NDArray[np.float64]:
        """Energy as one parameter is scanned with the others fixed.

        Parameters
        ----------
        hamiltonian : numpy.ndarray
            Hermitian matrix.
        base_params : numpy.ndarray
            Parameters held fixed.
        param_idx : int
            Index of the scanned parameter.
        param_range : numpy.ndarray
            Values to scan.
        ansatz : {"hardware_efficient", "uccsd"}, default "hardware_efficient"
            Ansatz family.
        n_layers : int, default 2
            Layers of the hardware-efficient ansatz.
        n_electrons : int, default 1
            Electrons for the UCCSD ansatz.

        Returns
        -------
        numpy.ndarray
        """
        if ansatz == "hardware_efficient":
            ansatz_fn: Callable[..., NDArray[np.complex128]] = self.hardware_efficient_ansatz
            ansatz_args: tuple = (self.n_qubits, n_layers)
        else:
            ansatz_fn = self.uccsd_ansatz
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
