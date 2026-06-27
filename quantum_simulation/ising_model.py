"""
Transverse-Field Ising Model
=============================

Implements the 1-D transverse-field Ising model (TFIM) on *n* sites:

.. math::

    H = -J \\sum_{\\langle i,j \\rangle} \\sigma_z^{(i)} \\sigma_z^{(j)}
        - h \\sum_i \\sigma_x^{(i)}

where *J* is the nearest-neighbour coupling and *h* the transverse field.

The model exhibits a quantum phase transition at h/J = 1 (in the
thermodynamic limit) between a ferromagnetic phase (h < J) and a
paramagnetic phase (h > J).

Key features
------------
* Exact construction of the full 2^n × 2^n Hamiltonian.
* Ground-state computation via ``scipy.linalg.eigh``.
* Observables: magnetisation, two-point correlators, energy spectrum.
* Phase-diagram scan over (J, h) parameter space.
* Unitary time evolution via matrix exponential.

Limitations: full exact diagonalisation is feasible for n ≲ 14 sites on a
laptop (2^14 = 16 384 dimensional Hilbert space).
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la

# ---------------------------------------------------------------------------
# Pauli matrices and identity  (2 × 2, complex128)
# ---------------------------------------------------------------------------

#: Pauli-X (bit-flip) matrix
sigma_x: NDArray[np.complex128] = np.array([[0, 1], [1, 0]], dtype=np.complex128)

#: Pauli-Y matrix
sigma_y: NDArray[np.complex128] = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

#: Pauli-Z (phase-flip) matrix
sigma_z: NDArray[np.complex128] = np.array([[1, 0], [0, -1]], dtype=np.complex128)

#: 2 × 2 identity
identity: NDArray[np.complex128] = np.eye(2, dtype=np.complex128)


def tensor_product(*ops: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Construct a multi-qubit operator via successive Kronecker products.

    Parameters
    ----------
    *ops : ndarray
        Sequence of square matrices (typically 2×2 single-qubit operators).

    Returns
    -------
    ndarray
        The Kronecker (tensor) product ``ops[0] ⊗ ops[1] ⊗ … ⊗ ops[-1]``.

    Examples
    --------
    >>> tensor_product(sigma_z, sigma_z)  # σ_z ⊗ σ_z on 2 qubits
    """
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


class IsingModel:
    """Transverse-field Ising model on a 1-D chain (or ring).

    Parameters
    ----------
    n_sites : int
        Number of spin-½ sites.
    J : float
        Nearest-neighbour Ising coupling strength.
    h : float
        Transverse magnetic field strength.
    periodic : bool, optional
        If ``True`` (default), use periodic (ring) boundary conditions.
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
        self._hamiltonian: Optional[NDArray[np.complex128]] = None

    # ------------------------------------------------------------------
    # Hamiltonian construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_hamiltonian(
        n_sites: int,
        J: float,
        h: float,
        periodic: bool = True,
    ) -> NDArray[np.complex128]:
        r"""Construct the full Hamiltonian matrix.

        .. math::

            H = -J \sum_{\langle i,j\rangle} \sigma_z^{(i)}\sigma_z^{(j)}
                -h \sum_i \sigma_x^{(i)}

        Parameters
        ----------
        n_sites : int
            Number of sites.
        J, h : float
            Coupling and transverse-field strengths.
        periodic : bool
            Whether the chain has periodic boundary conditions.

        Returns
        -------
        ndarray, shape (2^n, 2^n)
            The Hamiltonian as a dense Hermitian matrix.
        """
        dim = 2 ** n_sites
        H = np.zeros((dim, dim), dtype=np.complex128)

        # --- ZZ interaction terms ---
        n_bonds = n_sites if periodic else n_sites - 1
        for bond in range(n_bonds):
            i = bond
            j = (bond + 1) % n_sites
            # Build σ_z^{(i)} ⊗ σ_z^{(j)} acting on the full Hilbert space
            ops = [identity] * n_sites
            ops[i] = sigma_z
            ops[j] = sigma_z
            H -= J * tensor_product(*ops)

        # --- Transverse field terms ---
        for i in range(n_sites):
            ops = [identity] * n_sites
            ops[i] = sigma_x
            H -= h * tensor_product(*ops)

        return H

    @property
    def hamiltonian(self) -> NDArray[np.complex128]:
        """Lazily-built Hamiltonian matrix (cached)."""
        if self._hamiltonian is None:
            self._hamiltonian = self.build_hamiltonian(
                self.n_sites, self.J, self.h, self.periodic
            )
        return self._hamiltonian

    # ------------------------------------------------------------------
    # Ground state & spectrum
    # ------------------------------------------------------------------

    @staticmethod
    def ground_state(
        H: NDArray[np.complex128],
    ) -> Tuple[float, NDArray[np.complex128]]:
        """Find the ground state via exact diagonalisation.

        Parameters
        ----------
        H : ndarray
            Hermitian Hamiltonian matrix.

        Returns
        -------
        E0 : float
            Ground-state energy.
        psi0 : ndarray
            Ground-state vector (normalised).
        """
        eigenvalues, eigenvectors = la.eigh(H)
        return float(eigenvalues[0]), eigenvectors[:, 0]

    @staticmethod
    def energy_spectrum(
        H: NDArray[np.complex128],
        n_states: int = 10,
    ) -> NDArray[np.float64]:
        """Return the lowest *n_states* eigenvalues.

        Parameters
        ----------
        H : ndarray
            Hamiltonian matrix.
        n_states : int
            Number of eigenvalues to return.

        Returns
        -------
        ndarray, shape (n_states,)
            Sorted eigenvalues.
        """
        eigenvalues = la.eigh(H, eigvals_only=True)
        return eigenvalues[: min(n_states, len(eigenvalues))]

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    @staticmethod
    def magnetization(
        state: NDArray[np.complex128],
        n_sites: int,
    ) -> NDArray[np.float64]:
        r"""Compute ⟨σ_z⟩ for each site.

        Parameters
        ----------
        state : ndarray, shape (2^n,)
            Quantum state vector.
        n_sites : int
            Number of sites.

        Returns
        -------
        ndarray, shape (n_sites,)
            Expectation value ⟨ψ| σ_z^{(i)} |ψ⟩ for each site *i*.
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
        r"""Two-point correlator ⟨σ_z^{(i)} σ_z^{(j)}⟩.

        Parameters
        ----------
        state : ndarray
            Quantum state vector.
        i, j : int
            Site indices (0-based).
        n_sites : int
            Number of sites.

        Returns
        -------
        float
            The correlation value.
        """
        ops = [identity] * n_sites
        ops[i] = sigma_z
        ops[j] = sigma_z
        ZZ = tensor_product(*ops)
        return float(np.real(state.conj() @ ZZ @ state))

    # ------------------------------------------------------------------
    # Phase diagram
    # ------------------------------------------------------------------

    @staticmethod
    def phase_diagram(
        J_range: NDArray[np.float64],
        h_range: NDArray[np.float64],
        n_sites: int,
        periodic: bool = True,
    ) -> NDArray[np.float64]:
        r"""Compute the order parameter |⟨σ_z⟩| across parameter space.

        The *absolute* average magnetisation
        :math:`m = \frac{1}{n}\sum_i |⟨σ_z^{(i)}⟩|`
        is used as the order parameter.

        Parameters
        ----------
        J_range, h_range : array-like
            1-D arrays of J and h values to scan.
        n_sites : int
            Number of sites.
        periodic : bool
            Boundary conditions.

        Returns
        -------
        ndarray, shape (len(J_range), len(h_range))
            Order parameter grid.
        """
        order = np.zeros((len(J_range), len(h_range)), dtype=np.float64)
        for iJ, J in enumerate(J_range):
            for ih, h in enumerate(h_range):
                H = IsingModel.build_hamiltonian(n_sites, J, h, periodic)
                _, psi0 = IsingModel.ground_state(H)
                mag = IsingModel.magnetization(psi0, n_sites)
                order[iJ, ih] = np.mean(np.abs(mag))
        return order

    # ------------------------------------------------------------------
    # Time evolution
    # ------------------------------------------------------------------

    @staticmethod
    def time_evolve(
        state: NDArray[np.complex128],
        H: NDArray[np.complex128],
        t: float,
        dt: float,
    ) -> NDArray[np.complex128]:
        r"""Evolve a state under *H* from 0 to *t* in steps of *dt*.

        Uses the matrix exponential:

        .. math::

            |ψ(t+dt)⟩ = e^{-i H \, dt} |ψ(t)⟩

        Parameters
        ----------
        state : ndarray
            Initial state vector.
        H : ndarray
            Hamiltonian matrix.
        t : float
            Total evolution time.
        dt : float
            Time step.

        Returns
        -------
        ndarray
            Final state vector (normalised).
        """
        n_steps = int(np.ceil(t / dt))
        U = la.expm(-1j * H * dt)
        psi = state.copy().astype(np.complex128)
        for _ in range(n_steps):
            psi = U @ psi
        # Re-normalise to mitigate floating-point drift
        psi /= la.norm(psi)
        return psi

    # ------------------------------------------------------------------
    # Convenience methods on instance
    # ------------------------------------------------------------------

    def get_ground_state(self) -> Tuple[float, NDArray[np.complex128]]:
        """Ground-state energy and vector for *this* model instance."""
        return self.ground_state(self.hamiltonian)

    def get_magnetization(self) -> NDArray[np.float64]:
        """Per-site ⟨σ_z⟩ of the ground state."""
        _, psi0 = self.get_ground_state()
        return self.magnetization(psi0, self.n_sites)

    def get_energy_spectrum(self, n_states: int = 10) -> NDArray[np.float64]:
        """Lowest eigenvalues for *this* model instance."""
        return self.energy_spectrum(self.hamiltonian, n_states)

    def __repr__(self) -> str:
        bc = "PBC" if self.periodic else "OBC"
        return (
            f"IsingModel(n_sites={self.n_sites}, J={self.J}, h={self.h}, {bc})"
        )
