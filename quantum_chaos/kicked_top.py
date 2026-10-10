import numpy as np
from scipy.linalg import expm
from typing import Optional


class KickedTop:
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
        return self._floquet.copy()

    @property
    def floquet_eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvals(self._floquet)

    @property
    def quasi_energies(self) -> np.ndarray:
        evals = self.floquet_eigenvalues
        quasi_e = -np.angle(evals)
        return np.sort(quasi_e)


    def coherent_state(self, theta: float, phi: float) -> np.ndarray:
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
        for _ in range(n_kicks):
            state = self._floquet @ state
        return state

    def evolve_trajectory(
        self, state: np.ndarray, n_kicks: int
    ) -> np.ndarray:
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
        theta_grid = np.linspace(0, np.pi, n_theta)
        phi_grid = np.linspace(0, 2 * np.pi, n_phi)
        Q = self.husimi_q(state, theta_grid, phi_grid)
        return theta_grid, phi_grid, Q


    def expectation_values(
        self, state: np.ndarray
    ) -> tuple[float, float, float]:
        jx = float(np.real(state.conj() @ self.Jx @ state)) / self.j
        jy = float(np.real(state.conj() @ self.Jy @ state)) / self.j
        jz = float(np.real(state.conj() @ self.Jz @ state)) / self.j
        return jx, jy, jz

    def __repr__(self) -> str:
        return (
            f"KickedTop(j={self.j}, k={self.k}, p={self.p:.4f}, "
            f"dim={self.dim})"
        )
