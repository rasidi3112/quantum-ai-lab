"""Transverse-field Ising chain.

``H = -J sum_i Z_i Z_(i+1) - h sum_i X_i``

The chain has a quantum phase transition at ``h = J`` between a ferromagnet
(``h < J``) and a paramagnet (``h > J``). It is exactly solvable through a
mapping to free fermions, which the test suite uses to verify the
exact-diagonalization results here.

References
----------
Pfeuty, P. "The one-dimensional Ising model with a transverse field."
Ann. Phys. 57, 79 (1970).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la

sigma_x: NDArray[np.complex128] = np.array([[0, 1], [1, 0]], dtype=np.complex128)

sigma_y: NDArray[np.complex128] = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

sigma_z: NDArray[np.complex128] = np.array([[1, 0], [0, -1]], dtype=np.complex128)

identity: NDArray[np.complex128] = np.eye(2, dtype=np.complex128)


def tensor_product(*ops: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Kronecker product of the given operators, left to right."""
    result: NDArray[np.complex128] = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op).astype(np.complex128, copy=False)
    return result


class IsingModel:
    """Transverse-field Ising chain of ``n_sites`` spins.

    Parameters
    ----------
    n_sites : int
        Number of spins (at least 2).
    J : float, default 1.0
        Nearest-neighbour ZZ coupling (ferromagnetic for ``J > 0``).
    h : float, default 1.0
        Transverse field.
    periodic : bool, default True
        Periodic boundary conditions.
    """

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
        self._hamiltonian: NDArray[np.complex128] | None = None


    @staticmethod
    def build_hamiltonian(
        n_sites: int,
        J: float,
        h: float,
        periodic: bool = True,
    ) -> NDArray[np.complex128]:
        """Dense Hamiltonian ``-J sum Z_i Z_(i+1) - h sum X_i``.

        Parameters
        ----------
        n_sites : int
            Number of spins.
        J : float
            ZZ coupling.
        h : float
            Transverse field.
        periodic : bool, default True
            Periodic boundary conditions.

        Returns
        -------
        numpy.ndarray
            Matrix of shape ``(2^n, 2^n)``.
        """
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
        """The Hamiltonian matrix, built on first access and cached."""
        if self._hamiltonian is None:
            self._hamiltonian = self.build_hamiltonian(
                self.n_sites, self.J, self.h, self.periodic
            )
        return self._hamiltonian


    @staticmethod
    def ground_state(
        H: NDArray[np.complex128],
    ) -> tuple[float, NDArray[np.complex128]]:
        """Lowest eigenpair of a Hermitian matrix.

        Parameters
        ----------
        H : numpy.ndarray
            Hermitian matrix.

        Returns
        -------
        energy : float
        state : numpy.ndarray
        """
        eigenvalues, eigenvectors = la.eigh(H)
        return float(eigenvalues[0]), eigenvectors[:, 0]

    @staticmethod
    def energy_spectrum(
        H: NDArray[np.complex128],
        n_states: int = 10,
    ) -> NDArray[np.float64]:
        """Lowest ``n_states`` eigenvalues.

        Parameters
        ----------
        H : numpy.ndarray
            Hermitian matrix.
        n_states : int, default 10
            Number of levels.

        Returns
        -------
        numpy.ndarray
        """
        eigenvalues = la.eigh(H, eigvals_only=True)
        return eigenvalues[: min(n_states, len(eigenvalues))]


    @staticmethod
    def magnetization(
        state: NDArray[np.complex128],
        n_sites: int,
    ) -> NDArray[np.float64]:
        """Site-resolved magnetization ``<Z_i>``.

        Parameters
        ----------
        state : numpy.ndarray
            Statevector.
        n_sites : int
            Number of spins.

        Returns
        -------
        numpy.ndarray
            Array of length ``n_sites``. For a symmetric ground state this vanishes;
            use :meth:`correlation_function` to detect order.
        """
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
        """Spin–spin correlation ``<Z_i Z_j>``.

        Parameters
        ----------
        state : numpy.ndarray
            Statevector.
        i, j : int
            Sites.
        n_sites : int
            Number of spins.

        Returns
        -------
        float
        """
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
        """Mean ``|<Z_i>|`` of the ground state over a ``(J, h)`` grid.

        Parameters
        ----------
        J_range, h_range : numpy.ndarray
            Grid values.
        n_sites : int
            Number of spins.
        periodic : bool, default True
            Periodic boundary conditions.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(J_range), len(h_range))``.
        """
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
        """Exact time evolution ``exp(-iHt)|psi>`` in equal steps.

        Parameters
        ----------
        state : numpy.ndarray
            Initial state.
        H : numpy.ndarray
            Hamiltonian.
        t : float
            Total time.
        dt : float
            Maximum step; the step is shrunk so that the steps add up to ``t``.

        Returns
        -------
        numpy.ndarray
        """
        n_steps = max(1, int(np.ceil(t / dt)))
        U = la.expm(-1j * H * (t / n_steps))
        psi = state.copy().astype(np.complex128)
        for _ in range(n_steps):
            psi = U @ psi
        psi /= la.norm(psi)
        return psi


    def get_ground_state(self) -> tuple[float, NDArray[np.complex128]]:
        """Ground-state energy and vector of this chain."""
        return self.ground_state(self.hamiltonian)

    def get_magnetization(self) -> NDArray[np.float64]:
        """Site magnetizations ``<Z_i>`` of this chain's ground state."""
        _, psi0 = self.get_ground_state()
        return self.magnetization(psi0, self.n_sites)

    def get_energy_spectrum(self, n_states: int = 10) -> NDArray[np.float64]:
        """Lowest ``n_states`` energies of this chain."""
        return self.energy_spectrum(self.hamiltonian, n_states)

    def __repr__(self) -> str:
        bc = "PBC" if self.periodic else "OBC"
        return (
            f"IsingModel(n_sites={self.n_sites}, J={self.J}, h={self.h}, {bc})"
        )
