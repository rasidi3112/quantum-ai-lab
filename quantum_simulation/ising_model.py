from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la


sigma_x: NDArray[np.complex128] = np.array([[0, 1], [1, 0]], dtype=np.complex128)

sigma_y: NDArray[np.complex128] = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

sigma_z: NDArray[np.complex128] = np.array([[1, 0], [0, -1]], dtype=np.complex128)

identity: NDArray[np.complex128] = np.eye(2, dtype=np.complex128)


def tensor_product(*ops: NDArray[np.complex128]) -> NDArray[np.complex128]:
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


class IsingModel:
    def __init__(
        self,
        n_sites: int,
        J: float = 1.0,
        h: float = 1.0,
        periodic: bool = True,
    ) -> None:
        if n_sites < 2:
            raise ValueError("Need at least 2 sites for the Ising chain.")
        self.n_sites = n_sites
        self.J = J
        self.h = h
        self.periodic = periodic
        self.dim = 2 ** n_sites
        self._hamiltonian: Optional[NDArray[np.complex128]] = None


    @staticmethod
    def build_hamiltonian(
        n_sites: int,
        J: float,
        h: float,
        periodic: bool = True,
    ) -> NDArray[np.complex128]:
        dim = 2 ** n_sites
        H = np.zeros((dim, dim), dtype=np.complex128)

        n_bonds = n_sites if periodic else n_sites - 1
        for bond in range(n_bonds):
            i = bond
            j = (bond + 1) % n_sites
            ops = [identity] * n_sites
            ops[i] = sigma_z
            ops[j] = sigma_z
            H -= J * tensor_product(*ops)

        for i in range(n_sites):
            ops = [identity] * n_sites
            ops[i] = sigma_x
            H -= h * tensor_product(*ops)

        return H

    @property
    def hamiltonian(self) -> NDArray[np.complex128]:
        if self._hamiltonian is None:
            self._hamiltonian = self.build_hamiltonian(
                self.n_sites, self.J, self.h, self.periodic
            )
        return self._hamiltonian


    @staticmethod
    def ground_state(
        H: NDArray[np.complex128],
    ) -> Tuple[float, NDArray[np.complex128]]:
        eigenvalues, eigenvectors = la.eigh(H)
        return float(eigenvalues[0]), eigenvectors[:, 0]

    @staticmethod
    def energy_spectrum(
        H: NDArray[np.complex128],
        n_states: int = 10,
    ) -> NDArray[np.float64]:
        eigenvalues = la.eigh(H, eigvals_only=True)
        return eigenvalues[: min(n_states, len(eigenvalues))]


    @staticmethod
    def magnetization(
        state: NDArray[np.complex128],
        n_sites: int,
    ) -> NDArray[np.float64]:
        mag = np.zeros(n_sites, dtype=np.float64)
        for i in range(n_sites):
            ops = [identity] * n_sites
            ops[i] = sigma_z
            Sz_i = tensor_product(*ops)
            mag[i] = np.real(state.conj() @ Sz_i @ state)
        return mag

    @staticmethod
    def correlation_function(
        state: NDArray[np.complex128],
        i: int,
        j: int,
        n_sites: int,
    ) -> float:
        ops = [identity] * n_sites
        ops[i] = sigma_z
        ops[j] = sigma_z
        ZZ = tensor_product(*ops)
        return float(np.real(state.conj() @ ZZ @ state))


    @staticmethod
    def phase_diagram(
        J_range: NDArray[np.float64],
        h_range: NDArray[np.float64],
        n_sites: int,
        periodic: bool = True,
    ) -> NDArray[np.float64]:
        order = np.zeros((len(J_range), len(h_range)), dtype=np.float64)
        for iJ, J in enumerate(J_range):
            for ih, h in enumerate(h_range):
                H = IsingModel.build_hamiltonian(n_sites, J, h, periodic)
                _, psi0 = IsingModel.ground_state(H)
                mag = IsingModel.magnetization(psi0, n_sites)
                order[iJ, ih] = np.mean(np.abs(mag))
        return order


    @staticmethod
    def time_evolve(
        state: NDArray[np.complex128],
        H: NDArray[np.complex128],
        t: float,
        dt: float,
    ) -> NDArray[np.complex128]:
        n_steps = int(np.ceil(t / dt))
        U = la.expm(-1j * H * dt)
        psi = state.copy().astype(np.complex128)
        for _ in range(n_steps):
            psi = U @ psi
        psi /= la.norm(psi)
        return psi


    def get_ground_state(self) -> Tuple[float, NDArray[np.complex128]]:
        return self.ground_state(self.hamiltonian)

    def get_magnetization(self) -> NDArray[np.float64]:
        _, psi0 = self.get_ground_state()
        return self.magnetization(psi0, self.n_sites)

    def get_energy_spectrum(self, n_states: int = 10) -> NDArray[np.float64]:
        return self.energy_spectrum(self.hamiltonian, n_states)

    def __repr__(self) -> str:
        bc = "PBC" if self.periodic else "OBC"
        return (
            f"IsingModel(n_sites={self.n_sites}, J={self.J}, h={self.h}, {bc})"
        )
