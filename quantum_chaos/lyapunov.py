import numpy as np
from scipy.linalg import expm
from scipy.optimize import curve_fit
from typing import Optional


class LyapunovEstimator:
    def __init__(self) -> None:
        pass


    @staticmethod
    def _time_evolve_operator(
        H: np.ndarray, O: np.ndarray, t: float
    ) -> np.ndarray:
        U = expm(-1j * H * t)
        U_dag = U.conj().T
        return U_dag @ O @ U

    @staticmethod
    def _time_evolve_operator_eig(
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        O: np.ndarray,
        t: float,
    ) -> np.ndarray:
        V = eigenvectors
        V_dag = V.conj().T
        phases = np.exp(1j * eigenvalues * t)
        phases_neg = np.exp(-1j * eigenvalues * t)

        O_energy = V_dag @ O @ V
        O_evolved_energy = np.outer(phases, phases_neg) * O_energy
        return V @ O_evolved_energy @ V_dag


    def compute_otoc(
        self,
        H: np.ndarray,
        W: np.ndarray,
        V: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        N = H.shape[0]

        eigenvalues, eigenvectors = np.linalg.eigh(H)

        otoc = np.zeros(len(times))

        for idx, t in enumerate(times):
            W_t = self._time_evolve_operator_eig(
                eigenvalues, eigenvectors, W, t
            )

            comm = W_t @ V - V @ W_t

            otoc[idx] = np.real(np.trace(comm.conj().T @ comm)) / N

        return otoc

    def thermal_otoc(
        self,
        H: np.ndarray,
        W: np.ndarray,
        V: np.ndarray,
        beta: float,
        times: np.ndarray,
    ) -> np.ndarray:
        N = H.shape[0]

        eigenvalues, eigenvectors = np.linalg.eigh(H)

        E_shifted = eigenvalues - np.min(eigenvalues)
        boltzmann = np.exp(-beta * E_shifted)
        Z = np.sum(boltzmann)

        rho_diag = boltzmann / Z

        otoc = np.zeros(len(times))

        for idx, t in enumerate(times):
            W_t = self._time_evolve_operator_eig(
                eigenvalues, eigenvectors, W, t
            )

            comm = W_t @ V - V @ W_t

            comm_energy = eigenvectors.conj().T @ (comm.conj().T @ comm) @ eigenvectors
            otoc[idx] = np.real(np.sum(rho_diag * np.diag(comm_energy)))

        return otoc


    @staticmethod
    def estimate_lyapunov(
        times: np.ndarray,
        otoc_values: np.ndarray,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> dict:
        mask = otoc_values > 0
        t_pos = times[mask]
        c_pos = otoc_values[mask]

        if len(c_pos) < 3:
            return {
                "lyapunov_exponent": np.nan,
                "amplitude": np.nan,
                "fit_times": np.array([]),
                "fit_values": np.array([]),
                "r_squared": np.nan,
            }

        c_max = np.max(c_pos)

        if t_start is None:
            idx_start_candidates = np.where(c_pos > 0.01 * c_max)[0]
            if len(idx_start_candidates) > 0:
                t_start = t_pos[idx_start_candidates[0]]
            else:
                t_start = t_pos[0]

        if t_end is None:
            idx_end_candidates = np.where(c_pos > 0.5 * c_max)[0]
            if len(idx_end_candidates) > 0:
                t_end = t_pos[idx_end_candidates[0]]
            else:
                t_end = t_pos[-1]

        fit_mask = (t_pos >= t_start) & (t_pos <= t_end)
        t_fit = t_pos[fit_mask]
        c_fit = c_pos[fit_mask]

        if len(c_fit) < 3:
            t_fit = t_pos
            c_fit = c_pos

        log_c = np.log(c_fit)
        coeffs = np.polyfit(t_fit, log_c, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        lyapunov = slope / 2.0
        amplitude = np.exp(intercept)

        log_c_pred = np.polyval(coeffs, t_fit)
        ss_res = np.sum((log_c - log_c_pred) ** 2)
        ss_tot = np.sum((log_c - np.mean(log_c)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        fit_values = amplitude * np.exp(2 * lyapunov * t_fit)

        return {
            "lyapunov_exponent": float(lyapunov),
            "amplitude": float(amplitude),
            "fit_times": t_fit,
            "fit_values": fit_values,
            "r_squared": float(r_squared),
        }


    @staticmethod
    def scrambling_time(
        times: np.ndarray, otoc_values: np.ndarray, threshold: float = 0.5
    ) -> float:
        c_max = np.max(otoc_values)
        target = threshold * c_max

        idx = np.where(otoc_values >= target)[0]
        if len(idx) > 0:
            return float(times[idx[0]])
        return np.inf

    def __repr__(self) -> str:
        return "LyapunovEstimator()"
