"""
Fermi-Hubbard Model (1-D, single-band)
=======================================

Implements the single-band Hubbard model on a 1-D chain:

.. math::

    H = -t \\sum_{\\langle i,j \\rangle, \\sigma}
            \\bigl(c^\\dagger_{i\\sigma} c_{j\\sigma} + \\text{h.c.}\\bigr)
        + U \\sum_i n_{i\\uparrow} n_{i\\downarrow}

Fermions are mapped to qubits via the **Jordan-Wigner transformation**:

.. math::

    c^\\dagger_j = \\biggl(\\prod_{k<j} Z_k\\biggr)
                   \\frac{X_j - i Y_j}{2}

The resulting Hamiltonian is a :math:`2^{2n} \\times 2^{2n}` matrix (two
spin species on *n* sites → 2n qubits).  Practical limits: n ≤ 6.

Key features
------------
* Proper fermionic anti-commutation via Jordan-Wigner strings.
* Creation / annihilation operators as sparse-compatible dense matrices.
* Ground-state energy, density profiles, double occupancy.
* Spectral function :math:`A(\\omega)` via the Lehmann representation.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la


# ======================================================================
# Single-qubit operators
# ======================================================================

_I2 = np.eye(2, dtype=np.complex128)
_X  = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y  = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z  = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _kron_chain(*ops: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Kronecker-product chain helper."""
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


# ======================================================================
# Jordan-Wigner helpers
# ======================================================================

def _jordan_wigner_creation(site: int, n_modes: int) -> NDArray[np.complex128]:
    r"""Build the creation operator :math:`c^\dagger_j` on *n_modes* qubits.

    Jordan-Wigner mapping:

    .. math::

        c^\\dagger_j = \\biggl(\\prod_{k<j} Z_k\\biggr)
                       \\frac{X_j - i\\,Y_j}{2}

    Parameters
    ----------
    site : int
        The fermionic mode index (0-based).
    n_modes : int
        Total number of fermionic modes (= total qubits).

    Returns
    -------
    ndarray, shape (2^n_modes, 2^n_modes)
    """
    # (X - iY) / 2  =  |0><1|  =  σ_+ (raising operator)
    sigma_plus = (_X - 1j * _Y) / 2.0

    ops: list[NDArray[np.complex128]] = []
    for k in range(n_modes):
        if k < site:
            ops.append(_Z)       # Jordan-Wigner string
        elif k == site:
            ops.append(sigma_plus)
        else:
            ops.append(_I2)
    return _kron_chain(*ops)


def _jordan_wigner_annihilation(site: int, n_modes: int) -> NDArray[np.complex128]:
    r"""Annihilation operator :math:`c_j` (Hermitian conjugate of creation)."""
    return _jordan_wigner_creation(site, n_modes).conj().T


def _number_operator(site: int, n_modes: int) -> NDArray[np.complex128]:
    r"""Number operator :math:`n_j = c^\dagger_j c_j`."""
    c_dag = _jordan_wigner_creation(site, n_modes)
    c = _jordan_wigner_annihilation(site, n_modes)
    return c_dag @ c


class HubbardModel:
    """Single-band Fermi-Hubbard model on a 1-D chain.

    The mode ordering convention is:

        qubit index = site_index * 2 + spin   (spin: 0 = ↑, 1 = ↓)

    So for *n_sites* spatial sites we have 2 * n_sites qubits (modes).

    Parameters
    ----------
    n_sites : int
        Number of spatial sites (up to ~6 for exact diag).
    t_hop : float
        Nearest-neighbour hopping amplitude.
    U : float
        On-site Coulomb repulsion.
    periodic : bool
        Periodic (ring) boundary conditions.
    """

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
        self.n_modes = 2 * n_sites          # total fermionic modes (qubits)
        self.dim = 2 ** self.n_modes        # Hilbert-space dimension
        self._hamiltonian: Optional[NDArray[np.complex128]] = None

    # ------------------------------------------------------------------
    # Mode-index helpers
    # ------------------------------------------------------------------

    def _mode(self, site: int, spin: int) -> int:
        """Return qubit index for (site, spin).  spin: 0=↑, 1=↓."""
        return site * 2 + spin

    # ------------------------------------------------------------------
    # Hamiltonian construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_hamiltonian(
        n_sites: int,
        t_hop: float,
        U: float,
        periodic: bool = False,
    ) -> NDArray[np.complex128]:
        r"""Build the full Hubbard Hamiltonian.

        .. math::

            H = -t \sum_{\langle i,j \rangle,\sigma}
                    (c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.})
                + U \sum_i n_{i\uparrow} n_{i\downarrow}

        Parameters
        ----------
        n_sites : int
            Spatial sites.
        t_hop : float
            Hopping amplitude.
        U : float
            On-site repulsion.
        periodic : bool
            Periodic boundary conditions.

        Returns
        -------
        ndarray, shape (2^{2n}, 2^{2n})
        """
        n_modes = 2 * n_sites
        dim = 2 ** n_modes
        H = np.zeros((dim, dim), dtype=np.complex128)

        def mode(site: int, spin: int) -> int:
            return site * 2 + spin

        # Hopping term  -t Σ (c†_iσ c_jσ + h.c.)
        n_bonds = n_sites if periodic else n_sites - 1
        for bond in range(n_bonds):
            site_i = bond
            site_j = (bond + 1) % n_sites
            for spin in (0, 1):                 # ↑ and ↓
                mi = mode(site_i, spin)
                mj = mode(site_j, spin)
                c_dag_i = _jordan_wigner_creation(mi, n_modes)
                c_j     = _jordan_wigner_annihilation(mj, n_modes)
                hop = c_dag_i @ c_j
                H -= t_hop * (hop + hop.conj().T)   # + h.c.

        # On-site interaction  U Σ n_i↑ n_i↓
        for site in range(n_sites):
            n_up   = _number_operator(mode(site, 0), n_modes)
            n_down = _number_operator(mode(site, 1), n_modes)
            H += U * (n_up @ n_down)

        return H

    @property
    def hamiltonian(self) -> NDArray[np.complex128]:
        """Lazily built (and cached) Hamiltonian."""
        if self._hamiltonian is None:
            self._hamiltonian = self.build_hamiltonian(
                self.n_sites, self.t_hop, self.U, self.periodic
            )
        return self._hamiltonian

    # ------------------------------------------------------------------
    # Ground state
    # ------------------------------------------------------------------

    @staticmethod
    def ground_state_energy(
        H: NDArray[np.complex128],
    ) -> Tuple[float, NDArray[np.complex128]]:
        """Exact diagonalisation → ground-state energy and vector.

        Returns
        -------
        E0 : float
        psi0 : ndarray
        """
        eigenvalues, eigenvectors = la.eigh(H)
        return float(eigenvalues[0]), eigenvectors[:, 0]

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def density_profile(
        self,
        state: NDArray[np.complex128],
    ) -> NDArray[np.float64]:
        r"""Particle density ⟨n_{i↑} + n_{i↓}⟩ per spatial site.

        Parameters
        ----------
        state : ndarray, shape (dim,)
            State vector.

        Returns
        -------
        ndarray, shape (n_sites,)
        """
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
        r"""Double occupancy ⟨n_{i↑} n_{i↓}⟩ per site.

        Parameters
        ----------
        state : ndarray
            State vector.

        Returns
        -------
        ndarray, shape (n_sites,)
        """
        docc = np.zeros(self.n_sites, dtype=np.float64)
        for site in range(self.n_sites):
            n_up   = _number_operator(self._mode(site, 0), self.n_modes)
            n_down = _number_operator(self._mode(site, 1), self.n_modes)
            docc[site] = np.real(state.conj() @ (n_up @ n_down) @ state)
        return docc

    # ------------------------------------------------------------------
    # Spectral function
    # ------------------------------------------------------------------

    def spectral_function(
        self,
        H: NDArray[np.complex128],
        omega_range: NDArray[np.float64],
        eta: float = 0.1,
        site: int = 0,
        spin: int = 0,
    ) -> NDArray[np.float64]:
        r"""Local spectral function via Lehmann representation.

        .. math::

            A_j(\omega) = -\frac{1}{\pi} \mathrm{Im}\,G_j(\omega)

        where the retarded Green's function is

        .. math::

            G_j(\omega) = \sum_m
              \frac{|\langle m|c_j|0\rangle|^2}
                   {\omega - (E_m - E_0) + i\eta}
            + \frac{|\langle m|c^\dagger_j|0\rangle|^2}
                   {\omega + (E_m - E_0) + i\eta}

        Parameters
        ----------
        H : ndarray
            Hamiltonian (must already be built).
        omega_range : ndarray
            Frequency grid.
        eta : float
            Lorentzian broadening (half-width).
        site : int
            Spatial site index for the local Green's function.
        spin : int
            Spin species (0=↑, 1=↓).

        Returns
        -------
        ndarray, shape like omega_range
            Spectral weight :math:`A(\omega)`.
        """
        eigenvalues, eigenvectors = la.eigh(H)
        E0 = eigenvalues[0]
        psi0 = eigenvectors[:, 0]

        mode_idx = self._mode(site, spin)
        c     = _jordan_wigner_annihilation(mode_idx, self.n_modes)
        c_dag = _jordan_wigner_creation(mode_idx, self.n_modes)

        A = np.zeros_like(omega_range, dtype=np.float64)

        # Particle part  ⟨m|c†|0⟩
        c_dag_psi0 = c_dag @ psi0
        # Hole part  ⟨m|c|0⟩
        c_psi0 = c @ psi0

        for m, Em in enumerate(eigenvalues):
            # Particle contribution
            weight_p = np.abs(eigenvectors[:, m].conj() @ c_dag_psi0) ** 2
            A += (1.0 / np.pi) * eta / ((omega_range - (Em - E0)) ** 2 + eta ** 2) * weight_p

            # Hole contribution
            weight_h = np.abs(eigenvectors[:, m].conj() @ c_psi0) ** 2
            A += (1.0 / np.pi) * eta / ((omega_range + (Em - E0)) ** 2 + eta ** 2) * weight_h

        return A

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_ground_state(self) -> Tuple[float, NDArray[np.complex128]]:
        """Ground-state energy & vector for *this* instance."""
        return self.ground_state_energy(self.hamiltonian)

    def __repr__(self) -> str:
        bc = "PBC" if self.periodic else "OBC"
        return (
            f"HubbardModel(n_sites={self.n_sites}, t={self.t_hop}, "
            f"U={self.U}, {bc})"
        )
