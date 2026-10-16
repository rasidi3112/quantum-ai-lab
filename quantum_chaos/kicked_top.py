"""The quantum kicked top.

A spin ``j`` is rotated about the y axis and then kicked by a twist about z,
giving the Floquet operator

``U = exp(-i k / (2j) Jz^2) exp(-i p Jy)``.

For small kick strength ``k`` the dynamics is regular; around ``k = 3`` it is
mixed and for ``k >~ 6`` fully chaotic. In the semiclassical limit
``j -> infinity`` it reduces to a classical map on the sphere.

References
----------
Haake, F., Kuś, M. & Scharf, R. "Classical and quantum chaos for a kicked
top." Z. Phys. B 65, 381 (1987).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.special import gammaln


class KickedTop:
    """Quantum kicked top of spin ``j``.

    Parameters
    ----------
    j : float
        Spin quantum number, a positive integer or half-integer; the Hilbert
        space has dimension ``2j + 1``.
    k : float, default 3.0
        Kick (twist) strength.
    p : float, default pi/2
        Rotation angle about the y axis per period.

    Raises
    ------
    ValueError
        If ``j`` is not a positive (half-)integer.
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

        self.Jx, self.Jy, self.Jz = self._build_angular_momentum()
        self._floquet = self._build_floquet_operator()


    def _build_angular_momentum(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        j = self.j
        dim = self.dim

        m_vals = np.arange(j, -j - 1, -1, dtype=np.float64)

        Jz = np.diag(m_vals).astype(np.complex128)

        Jp = np.zeros((dim, dim), dtype=np.complex128)
        for idx in range(1, dim):
            m = m_vals[idx]
            Jp[idx - 1, idx] = np.sqrt(j * (j + 1) - m * (m + 1))

        Jm = Jp.conj().T

        Jx = (Jp + Jm) / 2.0
        Jy = (Jp - Jm) / (2.0j)

        return Jx, Jy, Jz


    def _build_floquet_operator(self) -> np.ndarray:
        j = self.j

        U_kick = expm(-1j * self.k / (2.0 * j) * self.Jz @ self.Jz)

        U_prec = expm(-1j * self.p * self.Jy)

        return U_kick @ U_prec

    @property
    def floquet_operator(self) -> np.ndarray:
        """One-period evolution operator ``U`` (a copy)."""
        return self._floquet.copy()

    @property
    def floquet_eigenvalues(self) -> np.ndarray:
        """Eigenvalues of ``U``, all on the unit circle."""
        return np.linalg.eigvals(self._floquet)

    @property
    def quasi_energies(self) -> np.ndarray:
        """Sorted quasi-energies ``-arg(lambda)`` in ``[-pi, pi)``."""
        evals = self.floquet_eigenvalues
        quasi_e = -np.angle(evals)
        return np.sort(quasi_e)


    def coherent_state(self, theta: float, phi: float) -> np.ndarray:
        """Spin coherent state pointing along ``(theta, phi)``.

        Parameters
        ----------
        theta : float
            Polar angle measured from +z.
        phi : float
            Azimuthal angle.

        Returns
        -------
        numpy.ndarray
            Normalized state in the ``|j, m>`` basis ordered from ``m = j`` down to
            ``m = -j``.
        """
        j = self.j
        dim = self.dim
        state = np.zeros(dim, dtype=np.complex128)

        c = np.cos(theta / 2)
        s = np.sin(theta / 2)

        for idx in range(dim):
            m = j - idx
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

        state /= np.linalg.norm(state)
        return state


    def evolve(
        self, state: np.ndarray, n_kicks: int
    ) -> np.ndarray:
        """Apply ``n_kicks`` periods of the Floquet operator.

        Parameters
        ----------
        state : numpy.ndarray
            Initial state.
        n_kicks : int
            Number of periods.

        Returns
        -------
        numpy.ndarray
            Evolved state.
        """
        for _ in range(n_kicks):
            state = self._floquet @ state
        return state

    def evolve_trajectory(
        self, state: np.ndarray, n_kicks: int
    ) -> np.ndarray:
        """Evolve and record the state after every kick.

        Parameters
        ----------
        state : numpy.ndarray
            Initial state.
        n_kicks : int
            Number of periods.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_kicks + 1, 2j + 1)``; row 0 is the initial state.
        """
        trajectory = np.zeros((n_kicks + 1, self.dim), dtype=np.complex128)
        trajectory[0] = state.copy()
        for t in range(n_kicks):
            state = self._floquet @ state
            trajectory[t + 1] = state.copy()
        return trajectory


    def husimi_q(
        self,
        state: np.ndarray,
        theta_grid: np.ndarray,
        phi_grid: np.ndarray,
    ) -> np.ndarray:
        """Husimi Q function ``Q = (2j+1)/(4 pi) |<theta, phi|psi>|^2`` on a grid.

        The overlaps with all coherent states are computed at once, so even fine
        grids take milliseconds. ``Q`` integrates to 1 over the sphere.

        Parameters
        ----------
        state : numpy.ndarray
            State in the ``|j, m>`` basis.
        theta_grid : numpy.ndarray
            Polar angles.
        phi_grid : numpy.ndarray
            Azimuthal angles.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(theta_grid), len(phi_grid))``.
        """
        n = self.dim - 1
        idx = np.arange(self.dim)
        log_binom = gammaln(n + 1) - gammaln(idx + 1) - gammaln(n - idx + 1)
        c = np.cos(np.asarray(theta_grid) / 2)[:, None]
        s = np.sin(np.asarray(theta_grid) / 2)[:, None]
        radial = np.exp(0.5 * log_binom) * c ** (n - idx) * s ** idx
        phases = np.exp(1j * np.outer(idx, phi_grid))
        overlap = (radial * state) @ phases

        prefactor = (2 * self.j + 1) / (4 * np.pi)
        return prefactor * np.abs(overlap) ** 2

    def husimi_q_fast(
        self,
        state: np.ndarray,
        n_theta: int = 100,
        n_phi: int = 200,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Husimi Q function on a uniform ``(theta, phi)`` grid.

        Parameters
        ----------
        state : numpy.ndarray
            State in the ``|j, m>`` basis.
        n_theta : int, default 100
            Points in ``theta`` over ``[0, pi]``.
        n_phi : int, default 200
            Points in ``phi`` over ``[0, 2 pi]``.

        Returns
        -------
        theta_grid, phi_grid, Q : numpy.ndarray
        """
        theta_grid = np.linspace(0, np.pi, n_theta)
        phi_grid = np.linspace(0, 2 * np.pi, n_phi)
        Q = self.husimi_q(state, theta_grid, phi_grid)
        return theta_grid, phi_grid, Q


    def expectation_values(
        self, state: np.ndarray
    ) -> tuple[float, float, float]:
        """Spin expectation values normalized by ``j``.

        Parameters
        ----------
        state : numpy.ndarray
            State in the ``|j, m>`` basis.

        Returns
        -------
        tuple of float
            ``(<Jx>, <Jy>, <Jz>) / j``, a point in the unit ball.
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
