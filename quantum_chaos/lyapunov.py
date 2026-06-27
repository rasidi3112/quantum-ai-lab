"""
Lyapunov Exponent Estimation via OTOC
======================================

Implements Out-of-Time-Order Correlator (OTOC) computation for quantum
Lyapunov exponent estimation. The OTOC is a key diagnostic for quantum
chaos and scrambling:

    C(t) = -⟨[W(t), V(0)]²⟩

where W(t) = e^{iHt} W e^{-iHt} is the Heisenberg-evolved operator.

In chaotic systems, C(t) grows exponentially at early times:

    C(t) ~ e^{2λt}

where λ is the quantum Lyapunov exponent. The Maldacena-Shenker-Stanford
bound constrains:

    λ ≤ 2πk_BT/ℏ

At finite temperature β = 1/(k_BT), the thermal OTOC is:

    C_β(t) = Tr[ρ_β [W(t), V(0)]† [W(t), V(0)]] / Tr[ρ_β]

where ρ_β = exp(-βH) is the thermal density matrix.

References
----------
[1] Maldacena, J., Shenker, S. H. & Stanford, D. "A bound on chaos."
    JHEP 2016, 106 (2016).
[2] Swingle, B. "Unscrambling the physics of out-of-time-order correlators."
    Nature Physics 14, 988–990 (2018).
[3] Larkin, A. I. & Ovchinnikov, Y. N. "Quasiclassical method in the theory
    of superconductivity." JETP 28, 1200 (1969).
"""

import numpy as np
from scipy.linalg import expm
from scipy.optimize import curve_fit
from typing import Optional


class LyapunovEstimator:
    """Estimate quantum Lyapunov exponents from OTOCs.

    Computes out-of-time-order correlators using exact diagonalization
    and matrix exponentials, then extracts the Lyapunov exponent by
    fitting the exponential growth regime.

    Examples
    --------
    >>> import numpy as np
    >>> from quantum_chaos import LyapunovEstimator
    >>> le = LyapunovEstimator()
    >>> # Random GOE Hamiltonian
    >>> N = 50
    >>> H = np.random.randn(N, N)
    >>> H = (H + H.T) / 2
    >>> W = np.random.randn(N, N); W = (W + W.T) / 2
    >>> V = np.random.randn(N, N); V = (V + V.T) / 2
    >>> times = np.linspace(0, 2, 50)
    >>> otoc = le.compute_otoc(H, W, V, times)
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Time evolution
    # ------------------------------------------------------------------

    @staticmethod
    def _time_evolve_operator(
        H: np.ndarray, O: np.ndarray, t: float
    ) -> np.ndarray:
        """Evolve operator O in the Heisenberg picture.

        O(t) = e^{iHt} O e^{-iHt}

        Uses scipy.linalg.expm for the matrix exponential.

        Parameters
        ----------
        H : np.ndarray
            Hamiltonian matrix.
        O : np.ndarray
            Operator to evolve.
        t : float
            Time.

        Returns
        -------
        np.ndarray
            Time-evolved operator O(t).
        """
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
        """Evolve operator using pre-computed eigensystem (faster for many times).

        O(t) = V · diag(e^{iEt}) · V† · O · V · diag(e^{-iEt}) · V†

        Parameters
        ----------
        eigenvalues : np.ndarray
            Eigenvalues of H.
        eigenvectors : np.ndarray
            Column eigenvectors of H.
        O : np.ndarray
            Operator to evolve.
        t : float
            Time.

        Returns
        -------
        np.ndarray
            Time-evolved operator.
        """
        V = eigenvectors
        V_dag = V.conj().T
        phases = np.exp(1j * eigenvalues * t)
        phases_neg = np.exp(-1j * eigenvalues * t)

        # O(t) = V diag(e^{iEt}) V† O V diag(e^{-iEt}) V†
        O_energy = V_dag @ O @ V  # O in energy basis
        O_evolved_energy = np.outer(phases, phases_neg) * O_energy
        return V @ O_evolved_energy @ V_dag

    # ------------------------------------------------------------------
    # OTOC computation
    # ------------------------------------------------------------------

    def compute_otoc(
        self,
        H: np.ndarray,
        W: np.ndarray,
        V: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """Compute the out-of-time-order correlator C(t).

        C(t) = -⟨[W(t), V]²⟩ = -(1/N) Tr([W(t), V]† [W(t), V])

        where the expectation is taken in the infinite-temperature
        (maximally mixed) state ρ = I/N.

        Parameters
        ----------
        H : np.ndarray
            Hamiltonian matrix (N × N, Hermitian).
        W : np.ndarray
            First operator (N × N, Hermitian).
        V : np.ndarray
            Second operator (N × N).
        times : np.ndarray
            Array of time values at which to evaluate the OTOC.

        Returns
        -------
        np.ndarray
            OTOC values C(t) for each time point.
        """
        N = H.shape[0]

        # Pre-diagonalize H for efficiency
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        otoc = np.zeros(len(times))

        for idx, t in enumerate(times):
            # Evolve W: W(t) = e^{iHt} W e^{-iHt}
            W_t = self._time_evolve_operator_eig(
                eigenvalues, eigenvectors, W, t
            )

            # Commutator: [W(t), V]
            comm = W_t @ V - V @ W_t

            # C(t) = (1/N) Tr(comm† comm)  (positive-definite version)
            # This equals -Tr([W(t),V]^2)/N when W,V are Hermitian
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
        """Compute the thermal OTOC at inverse temperature β.

        C_β(t) = Tr[ρ_β C†(t) C(t)] / Tr[ρ_β]

        where C(t) = [W(t), V] and ρ_β = exp(-βH).

        The regularized version uses ρ^{1/4} insertions:

        C_β^{reg}(t) = Tr[ρ^{1/4} W(t) ρ^{1/4} V ρ^{1/4} W(t) ρ^{1/4} V]

        but here we use the simpler thermal average for clarity.

        Parameters
        ----------
        H : np.ndarray
            Hamiltonian.
        W : np.ndarray
            First operator.
        V : np.ndarray
            Second operator.
        beta : float
            Inverse temperature β = 1/(k_B T).
        times : np.ndarray
            Time array.

        Returns
        -------
        np.ndarray
            Thermal OTOC values.
        """
        N = H.shape[0]

        # Compute thermal state
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        # Shift eigenvalues for numerical stability
        E_shifted = eigenvalues - np.min(eigenvalues)
        boltzmann = np.exp(-beta * E_shifted)
        Z = np.sum(boltzmann)  # Partition function

        # Density matrix in energy basis
        rho_diag = boltzmann / Z

        otoc = np.zeros(len(times))

        for idx, t in enumerate(times):
            # Evolve W(t)
            W_t = self._time_evolve_operator_eig(
                eigenvalues, eigenvectors, W, t
            )

            # Commutator [W(t), V]
            comm = W_t @ V - V @ W_t

            # Thermal average: Tr(ρ · comm† · comm)
            # In energy basis: Σ_n ρ_n ⟨n|comm†·comm|n⟩
            comm_energy = eigenvectors.conj().T @ (comm.conj().T @ comm) @ eigenvectors
            otoc[idx] = np.real(np.sum(rho_diag * np.diag(comm_energy)))

        return otoc

    # ------------------------------------------------------------------
    # Lyapunov exponent extraction
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_lyapunov(
        times: np.ndarray,
        otoc_values: np.ndarray,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> dict:
        """Extract the Lyapunov exponent from OTOC data.

        Fits C(t) = A · exp(2λt) in the exponential growth regime.
        The fit is performed in log space: log C(t) = log A + 2λt.

        Parameters
        ----------
        times : np.ndarray
            Time array.
        otoc_values : np.ndarray
            OTOC values C(t).
        t_start : float or None, optional
            Start of the fitting window. If None, uses the first time
            where C(t) exceeds 1% of its maximum.
        t_end : float or None, optional
            End of the fitting window. If None, uses the time where
            C(t) reaches 50% of its maximum (before saturation).

        Returns
        -------
        dict
            'lyapunov_exponent': λ,
            'amplitude': A,
            'fit_times': times used for fitting,
            'fit_values': fitted C(t) values,
            'r_squared': R² goodness of fit.
        """
        # Filter positive OTOC values for log fit
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

        # Auto-detect fitting window
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

        # Select fitting window
        fit_mask = (t_pos >= t_start) & (t_pos <= t_end)
        t_fit = t_pos[fit_mask]
        c_fit = c_pos[fit_mask]

        if len(c_fit) < 3:
            # Not enough points; use all positive data
            t_fit = t_pos
            c_fit = c_pos

        # Log-space linear fit: log(C) = log(A) + 2λt
        log_c = np.log(c_fit)
        coeffs = np.polyfit(t_fit, log_c, 1)
        slope = coeffs[0]  # = 2λ
        intercept = coeffs[1]  # = log(A)

        lyapunov = slope / 2.0
        amplitude = np.exp(intercept)

        # R² for goodness of fit
        log_c_pred = np.polyval(coeffs, t_fit)
        ss_res = np.sum((log_c - log_c_pred) ** 2)
        ss_tot = np.sum((log_c - np.mean(log_c)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Predicted values for plotting
        fit_values = amplitude * np.exp(2 * lyapunov * t_fit)

        return {
            "lyapunov_exponent": float(lyapunov),
            "amplitude": float(amplitude),
            "fit_times": t_fit,
            "fit_values": fit_values,
            "r_squared": float(r_squared),
        }

    # ------------------------------------------------------------------
    # Scrambling time
    # ------------------------------------------------------------------

    @staticmethod
    def scrambling_time(
        times: np.ndarray, otoc_values: np.ndarray, threshold: float = 0.5
    ) -> float:
        """Estimate the scrambling time t*.

        The scrambling time is defined as the time when the OTOC
        reaches a fraction of its saturation value:

            C(t*) = threshold × C_max

        Parameters
        ----------
        times : np.ndarray
            Time array.
        otoc_values : np.ndarray
            OTOC values.
        threshold : float, optional
            Fraction of saturation (default: 0.5).

        Returns
        -------
        float
            Estimated scrambling time. Returns np.inf if threshold not reached.
        """
        c_max = np.max(otoc_values)
        target = threshold * c_max

        idx = np.where(otoc_values >= target)[0]
        if len(idx) > 0:
            return float(times[idx[0]])
        return np.inf

    def __repr__(self) -> str:
        return "LyapunovEstimator()"
