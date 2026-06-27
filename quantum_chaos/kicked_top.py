"""
Quantum Kicked Top
===================

Implements the quantum kicked top, a paradigmatic model for studying the
transition from regular to chaotic dynamics in quantum systems. The model
describes a spin-j angular momentum vector subject to periodic kicks.

Model
-----
The Floquet (one-period) operator for the kicked top is:

    U = exp(-i k/(2j) Jz²) · exp(-i p Jy)

where:
- j is the spin quantum number (Hilbert space dimension = 2j+1)
- k is the kicking strength (chaos parameter)
- p is the precession angle
- Jx, Jy, Jz are angular momentum operators for spin-j

For small k, the dynamics are regular (integrable limit). As k increases,
the phase space develops mixed regions and eventually becomes fully chaotic.

The Husimi Q representation provides a phase-space portrait:

    Q(θ, φ) = (2j+1)/(4π) |⟨θ,φ|ψ⟩|²

where |θ,φ⟩ is a spin coherent state on the Bloch sphere.

References
----------
[1] Haake, F., Kuś, M. & Scharf, R. "Classical and quantum chaos for a
    kicked top." Z. Phys. B 65, 381–395 (1987).
[2] Haake, F. "Quantum Signatures of Chaos." Springer, 3rd ed. (2010).
"""

import numpy as np
from scipy.linalg import expm
from typing import Optional


class KickedTop:
    """Quantum kicked top model for spin-j systems.

    Parameters
    ----------
    j : float
        Spin quantum number (half-integer or integer, e.g. 10, 20, 50).
        The Hilbert space dimension is 2j+1.
    k : float, optional
        Kicking strength / chaos parameter (default: 3.0).
        k ≈ 0: regular, k ≈ 3: mixed, k ≈ 6: fully chaotic.
    p : float, optional
        Precession angle around y-axis (default: π/2).

    Examples
    --------
    >>> import numpy as np
    >>> from quantum_chaos import KickedTop
    >>> kt = KickedTop(j=20, k=3.0)
    >>> state = kt.coherent_state(np.pi/4, np.pi/3)
    >>> evolved = kt.evolve(state, n_kicks=100)
    """

    def __init__(
        self,
        j: float,
        k: float = 3.0,
        p: float = np.pi / 2,
    ) -> None:
        if j < 0.5 or (2 * j) != int(2 * j):
            raise ValueError(
                f"j must be a positive half-integer or integer, got {j}"
            )

        self.j = j
        self.k = k
        self.p = p
        self.dim = int(2 * j + 1)

        # Build angular momentum operators
        self.Jx, self.Jy, self.Jz = self._build_angular_momentum()

        # Build the Floquet operator
        self._floquet = self._build_floquet_operator()

    # ------------------------------------------------------------------
    # Angular momentum operators
    # ------------------------------------------------------------------

    def _build_angular_momentum(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct Jx, Jy, Jz matrices for spin-j.

        Uses the standard basis |j, m⟩ with m = -j, -j+1, ..., j.
        Convention: index 0 corresponds to m = j (highest weight).

        Matrix elements:
            ⟨j,m'|Jz|j,m⟩ = m δ_{m',m}
            ⟨j,m'|J+|j,m⟩ = √(j(j+1) - m(m+1)) δ_{m',m+1}
            Jx = (J+ + J-)/2,  Jy = (J+ - J-)/(2i)

        Returns
        -------
        Jx, Jy, Jz : tuple of np.ndarray
            Angular momentum matrices of dimension (2j+1) × (2j+1).
        """
        j = self.j
        dim = self.dim

        # m values from j down to -j (standard convention)
        m_vals = np.arange(j, -j - 1, -1, dtype=np.float64)

        # Jz is diagonal
        Jz = np.diag(m_vals).astype(np.complex128)

        # Raising operator J+: ⟨m+1|J+|m⟩ = √(j(j+1) - m(m+1))
        # In our ordering, m+1 is at index one before m
        Jp = np.zeros((dim, dim), dtype=np.complex128)
        for idx in range(1, dim):
            m = m_vals[idx]  # m value at index idx
            Jp[idx - 1, idx] = np.sqrt(j * (j + 1) - m * (m + 1))

        # Lowering operator J- = (J+)†
        Jm = Jp.conj().T

        # Jx = (J+ + J-)/2, Jy = (J+ - J-)/(2i)
        Jx = (Jp + Jm) / 2.0
        Jy = (Jp - Jm) / (2.0j)

        return Jx, Jy, Jz

    # ------------------------------------------------------------------
    # Floquet operator
    # ------------------------------------------------------------------

    def _build_floquet_operator(self) -> np.ndarray:
        """Build the Floquet (one-period) operator.

        U = exp(-i k/(2j) Jz²) · exp(-i p Jy)

        The first factor represents the nonlinear kick and the second
        factor represents free precession around the y-axis.

        Returns
        -------
        np.ndarray
            Unitary Floquet operator of dimension (2j+1) × (2j+1).
        """
        j = self.j

        # Torsion (kick): exp(-i k/(2j) Jz²)
        U_kick = expm(-1j * self.k / (2.0 * j) * self.Jz @ self.Jz)

        # Precession: exp(-i p Jy)
        U_prec = expm(-1j * self.p * self.Jy)

        return U_kick @ U_prec

    @property
    def floquet_operator(self) -> np.ndarray:
        """The Floquet operator U."""
        return self._floquet.copy()

    @property
    def floquet_eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the Floquet operator (quasi-energies)."""
        return np.linalg.eigvals(self._floquet)

    @property
    def quasi_energies(self) -> np.ndarray:
        """Quasi-energies ε defined by eigenvalues e^{-iε}.

        Returns
        -------
        np.ndarray
            Quasi-energies in [-π, π), sorted.
        """
        evals = self.floquet_eigenvalues
        quasi_e = -np.angle(evals)
        return np.sort(quasi_e)

    # ------------------------------------------------------------------
    # State preparation
    # ------------------------------------------------------------------

    def coherent_state(self, theta: float, phi: float) -> np.ndarray:
        """Construct a spin coherent state |θ, φ⟩.

        The spin coherent state is defined as:

            |θ,φ⟩ = R(θ,φ)|j,j⟩ = exp(-iφJz) exp(-iθJy) |j,j⟩

        where |j,j⟩ is the highest-weight state.

        Alternatively, in the |j,m⟩ basis:

            ⟨j,m|θ,φ⟩ = C(j,m) (cos θ/2)^{j+m} (sin θ/2)^{j-m} e^{-i(j-m)φ}

        where C(j,m) = √(2j choose j-m).

        Parameters
        ----------
        theta : float
            Polar angle ∈ [0, π].
        phi : float
            Azimuthal angle ∈ [0, 2π).

        Returns
        -------
        np.ndarray
            Normalized spin coherent state of dimension 2j+1.
        """
        j = self.j
        dim = self.dim
        state = np.zeros(dim, dtype=np.complex128)

        c = np.cos(theta / 2)
        s = np.sin(theta / 2)

        for idx in range(dim):
            m = j - idx  # m goes from j down to -j
            # Binomial coefficient: C(2j, j-m)
            binom = 1.0
            jmm = int(j - m)
            jpm = int(j + m)
            for r in range(1, jmm + 1):
                binom *= (jpm + r) / r

            state[idx] = (
                np.sqrt(binom)
                * c ** jpm
                * s ** jmm
                * np.exp(-1j * jmm * phi)
            )

        # Normalize (should already be normalized, but ensure numerics)
        state /= np.linalg.norm(state)
        return state

    # ------------------------------------------------------------------
    # Time evolution
    # ------------------------------------------------------------------

    def evolve(
        self, state: np.ndarray, n_kicks: int
    ) -> np.ndarray:
        """Evolve a state through n kicks of the Floquet operator.

        |ψ(n)⟩ = U^n |ψ(0)⟩

        Parameters
        ----------
        state : np.ndarray
            Initial state of dimension 2j+1.
        n_kicks : int
            Number of kicks to apply.

        Returns
        -------
        np.ndarray
            Final state after n kicks.
        """
        for _ in range(n_kicks):
            state = self._floquet @ state
        return state

    def evolve_trajectory(
        self, state: np.ndarray, n_kicks: int
    ) -> np.ndarray:
        """Evolve and record the state at each kick.

        Parameters
        ----------
        state : np.ndarray
            Initial state.
        n_kicks : int
            Number of kicks.

        Returns
        -------
        np.ndarray of shape (n_kicks+1, dim)
            Trajectory of states, including the initial state.
        """
        trajectory = np.zeros((n_kicks + 1, self.dim), dtype=np.complex128)
        trajectory[0] = state.copy()
        for t in range(n_kicks):
            state = self._floquet @ state
            trajectory[t + 1] = state.copy()
        return trajectory

    # ------------------------------------------------------------------
    # Husimi Q representation
    # ------------------------------------------------------------------

    def husimi_q(
        self,
        state: np.ndarray,
        theta_grid: np.ndarray,
        phi_grid: np.ndarray,
    ) -> np.ndarray:
        """Compute the Husimi Q function on a spherical grid.

        Q(θ,φ) = (2j+1)/(4π) |⟨θ,φ|ψ⟩|²

        Parameters
        ----------
        state : np.ndarray
            Quantum state of dimension 2j+1.
        theta_grid : np.ndarray of shape (n_theta,)
            Polar angles.
        phi_grid : np.ndarray of shape (n_phi,)
            Azimuthal angles.

        Returns
        -------
        np.ndarray of shape (n_theta, n_phi)
            Husimi Q function values on the grid.
        """
        n_theta = len(theta_grid)
        n_phi = len(phi_grid)
        Q = np.zeros((n_theta, n_phi))

        prefactor = (2 * self.j + 1) / (4 * np.pi)

        for i, theta in enumerate(theta_grid):
            for k, phi in enumerate(phi_grid):
                cs = self.coherent_state(theta, phi)
                overlap = np.vdot(cs, state)
                Q[i, k] = prefactor * np.abs(overlap) ** 2

        return Q

    def husimi_q_fast(
        self,
        state: np.ndarray,
        n_theta: int = 100,
        n_phi: int = 200,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Husimi Q on a regular grid (convenience wrapper).

        Parameters
        ----------
        state : np.ndarray
            Quantum state.
        n_theta : int
            Number of theta points (default: 100).
        n_phi : int
            Number of phi points (default: 200).

        Returns
        -------
        theta_grid, phi_grid, Q : tuple
            Grid arrays and Q function values.
        """
        theta_grid = np.linspace(0, np.pi, n_theta)
        phi_grid = np.linspace(0, 2 * np.pi, n_phi)
        Q = self.husimi_q(state, theta_grid, phi_grid)
        return theta_grid, phi_grid, Q

    # ------------------------------------------------------------------
    # Classical limit expectations
    # ------------------------------------------------------------------

    def expectation_values(
        self, state: np.ndarray
    ) -> tuple[float, float, float]:
        """Compute ⟨Jx⟩, ⟨Jy⟩, ⟨Jz⟩ normalized by j.

        Returns the Bloch sphere coordinates (x, y, z) of the state.

        Parameters
        ----------
        state : np.ndarray
            Quantum state.

        Returns
        -------
        tuple of float
            (⟨Jx⟩/j, ⟨Jy⟩/j, ⟨Jz⟩/j).
        """
        jx = float(np.real(state.conj() @ self.Jx @ state)) / self.j
        jy = float(np.real(state.conj() @ self.Jy @ state)) / self.j
        jz = float(np.real(state.conj() @ self.Jz @ state)) / self.j
        return jx, jy, jz

    def __repr__(self) -> str:
        return (
            f"KickedTop(j={self.j}, k={self.k}, p={self.p:.4f}, "
            f"dim={self.dim})"
        )
