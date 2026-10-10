from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la


_I2 = np.eye(2, dtype=np.complex128)
_X  = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y  = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z  = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _kron_chain(*ops: NDArray[np.complex128]) -> NDArray[np.complex128]:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def _jordan_wigner_creation(site: int, n_modes: int) -> NDArray[np.complex128]:
    sigma_plus = (_X - 1j * _Y) / 2.0

    ops: list[NDArray[np.complex128]] = []
    for k in range(n_modes):
        if k < site:
            ops.append(_Z)
        elif k == site:
            ops.append(sigma_plus)
        else:
            ops.append(_I2)
    return _kron_chain(*ops)


def _jordan_wigner_annihilation(site: int, n_modes: int) -> NDArray[np.complex128]:
    return _jordan_wigner_creation(site, n_modes).conj().T


def _number_operator(site: int, n_modes: int) -> NDArray[np.complex128]:
    c_dag = _jordan_wigner_creation(site, n_modes)
    c = _jordan_wigner_annihilation(site, n_modes)
    return c_dag @ c


class HubbardModel:
    def __init__(
        self,
        n_sites: int,
        t_hop: float = 1.0,
        U: float = 2.0,
        periodic: bool = False,
    ) -> None:
        if n_sites < 2:
            raise ValueError("Need at least 2 sites.")
        self.n_sites = n_sites
        self.t_hop = t_hop
        self.U = U
        self.periodic = periodic
        self.n_modes = 2 * n_sites
        self.dim = 2 ** self.n_modes
        self._hamiltonian: Optional[NDArray[np.complex128]] = None

    def _mode(self, site: int, spin: int) -> int:
        return site * 2 + spin


    @staticmethod
    def build_hamiltonian(
        n_sites: int,
        t_hop: float,
        U: float,
        periodic: bool = False,
    ) -> NDArray[np.complex128]:
        n_modes = 2 * n_sites
        dim = 2 ** n_modes
        H = np.zeros((dim, dim), dtype=np.complex128)

        def mode(site: int, spin: int) -> int:
            return site * 2 + spin

        n_bonds = n_sites if periodic else n_sites - 1
        for bond in range(n_bonds):
            site_i = bond
            site_j = (bond + 1) % n_sites
            for spin in (0, 1):
                mi = mode(site_i, spin)
                mj = mode(site_j, spin)
                c_dag_i = _jordan_wigner_creation(mi, n_modes)
                c_j     = _jordan_wigner_annihilation(mj, n_modes)
                hop = c_dag_i @ c_j
                H -= t_hop * (hop + hop.conj().T)

        for site in range(n_sites):
            n_up   = _number_operator(mode(site, 0), n_modes)
            n_down = _number_operator(mode(site, 1), n_modes)
            H += U * (n_up @ n_down)

        return H

    @property
    def hamiltonian(self) -> NDArray[np.complex128]:
        if self._hamiltonian is None:
            self._hamiltonian = self.build_hamiltonian(
                self.n_sites, self.t_hop, self.U, self.periodic
            )
        return self._hamiltonian

    @staticmethod
    def ground_state_energy(
        H: NDArray[np.complex128],
    ) -> Tuple[float, NDArray[np.complex128]]:
        eigenvalues, eigenvectors = la.eigh(H)
        return float(eigenvalues[0]), eigenvectors[:, 0]

    def density_profile(
        self,
        state: NDArray[np.complex128],
    ) -> NDArray[np.float64]:
        density = np.zeros(self.n_sites, dtype=np.float64)
        for site in range(self.n_sites):
            for spin in (0, 1):
                n_op = _number_operator(self._mode(site, spin), self.n_modes)
                density[site] += np.real(state.conj() @ n_op @ state)
        return density

    def double_occupancy(
        self,
        state: NDArray[np.complex128],
    ) -> NDArray[np.float64]:
        docc = np.zeros(self.n_sites, dtype=np.float64)
        for site in range(self.n_sites):
            n_up   = _number_operator(self._mode(site, 0), self.n_modes)
            n_down = _number_operator(self._mode(site, 1), self.n_modes)
            docc[site] = np.real(state.conj() @ (n_up @ n_down) @ state)
        return docc

    def spectral_function(
        self,
        H: NDArray[np.complex128],
        omega_range: NDArray[np.float64],
        eta: float = 0.1,
        site: int = 0,
        spin: int = 0,
    ) -> NDArray[np.float64]:
        eigenvalues, eigenvectors = la.eigh(H)
        E0 = eigenvalues[0]
        psi0 = eigenvectors[:, 0]

        mode_idx = self._mode(site, spin)
        c     = _jordan_wigner_annihilation(mode_idx, self.n_modes)
        c_dag = _jordan_wigner_creation(mode_idx, self.n_modes)

        A = np.zeros_like(omega_range, dtype=np.float64)

        c_dag_psi0 = c_dag @ psi0
        c_psi0 = c @ psi0

        for m, Em in enumerate(eigenvalues):
            weight_p = np.abs(eigenvectors[:, m].conj() @ c_dag_psi0) ** 2
            A += (1.0 / np.pi) * eta / ((omega_range - (Em - E0)) ** 2 + eta ** 2) * weight_p

            weight_h = np.abs(eigenvectors[:, m].conj() @ c_psi0) ** 2
            A += (1.0 / np.pi) * eta / ((omega_range + (Em - E0)) ** 2 + eta ** 2) * weight_h

        return A

    def get_ground_state(self) -> Tuple[float, NDArray[np.complex128]]:
        return self.ground_state_energy(self.hamiltonian)

    def __repr__(self) -> str:
        bc = "PBC" if self.periodic else "OBC"
        return (
            f"HubbardModel(n_sites={self.n_sites}, t={self.t_hop}, "
            f"U={self.U}, {bc})"
        )
