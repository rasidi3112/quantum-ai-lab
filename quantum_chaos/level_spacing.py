"""
Level Spacing Statistics
=========================

Implements nearest-neighbor level spacing analysis, a fundamental diagnostic
for quantum chaos. The distribution of energy level spacings reveals the
nature of the underlying classical dynamics:

- **Integrable systems**: Poisson distribution P(s) = exp(-s)
  Eigenvalues are uncorrelated → level crossings are allowed.

- **Chaotic systems (GOE)**: Wigner surmise P(s) = (πs/2)exp(-πs²/4)
  Eigenvalues repel → level crossings are forbidden.

The Brody distribution interpolates between these limits with parameter q:

    P(s; q) = (1+q) · a · s^q · exp(-a · s^{1+q})

where a = [Γ((q+2)/(q+1))]^{1+q}. q=0 → Poisson, q=1 → Wigner.

The ratio statistic r̃ = min(r_n, 1/r_n) where r_n = s_n/s_{n-1} provides
a complementary diagnostic that does not require spectral unfolding:
    - Poisson: ⟨r̃⟩ ≈ 0.386
    - GOE:     ⟨r̃⟩ ≈ 0.536

References
----------
[1] Mehta, M. L. "Random Matrices." Academic Press, 3rd ed. (2004).
[2] Atas, Y. Y., et al. "Distribution of the ratio of consecutive level
    spacings in random matrix ensembles." Phys. Rev. Lett. 110, 084101 (2013).
[3] Brody, T. A., et al. "Random-matrix physics: spectrum and strength
    fluctuations." Rev. Mod. Phys. 53, 385 (1981).
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gamma as gamma_func
from typing import Optional


class LevelSpacing:
    """Nearest-neighbor level spacing statistics for quantum chaos analysis.

    Analyzes energy level distributions to distinguish integrable from
    chaotic quantum systems.

    Examples
    --------
    >>> import numpy as np
    >>> from quantum_chaos import LevelSpacing
    >>> ls = LevelSpacing()
    >>> # Random GOE matrix (should show Wigner-Dyson statistics)
    >>> H = np.random.randn(200, 200)
    >>> H = (H + H.T) / 2
    >>> result = ls.analyze(H)
    >>> print(f"Brody parameter: {result['brody_q']:.3f}")  # expect ~1.0
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    @staticmethod
    def compute_spacings(
        eigenvalues: np.ndarray, unfold: bool = True
    ) -> np.ndarray:
        """Compute nearest-neighbor level spacings.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Energy eigenvalues (will be sorted).
        unfold : bool, optional
            Whether to unfold the spectrum to unit mean spacing (default: True).
            Unfolding uses a polynomial fit to the integrated density of states.

        Returns
        -------
        np.ndarray
            Nearest-neighbor spacings (length = len(eigenvalues) - 1).
        """
        E = np.sort(np.real(eigenvalues))
        spacings = np.diff(E)

        if unfold:
            spacings = LevelSpacing._unfold_spacings(E)

        return spacings

    @staticmethod
    def _unfold_spacings(
        sorted_eigenvalues: np.ndarray, poly_order: int = 6
    ) -> np.ndarray:
        """Unfold the spectrum using polynomial smoothing of the staircase.

        The unfolding procedure maps eigenvalues E → ε such that the mean
        spacing ⟨s⟩ = 1. This is done by fitting a smooth polynomial N(E)
        to the staircase function (integrated density of states).

        Parameters
        ----------
        sorted_eigenvalues : np.ndarray
            Sorted energy eigenvalues.
        poly_order : int, optional
            Order of the polynomial fit (default: 6).

        Returns
        -------
        np.ndarray
            Unfolded spacings with unit mean.
        """
        E = sorted_eigenvalues
        N = len(E)

        # Staircase function: N(E_i) = i
        staircase = np.arange(N, dtype=np.float64)

        # Fit polynomial to staircase
        coeffs = np.polyfit(E, staircase, poly_order)
        N_smooth = np.polyval(coeffs, E)

        # Unfolded spacings
        spacings = np.diff(N_smooth)

        # Remove any non-positive spacings (numerical artifacts)
        spacings = spacings[spacings > 0]

        return spacings

    # ------------------------------------------------------------------
    # Theoretical distributions
    # ------------------------------------------------------------------

    @staticmethod
    def wigner_surmise(s: np.ndarray) -> np.ndarray:
        """GOE Wigner surmise for nearest-neighbor spacing distribution.

        P(s) = (π/2) s exp(-π s²/4)

        This distribution indicates chaotic (non-integrable) dynamics.
        It exhibits level repulsion: P(0) = 0.

        Parameters
        ----------
        s : np.ndarray
            Normalized spacings.

        Returns
        -------
        np.ndarray
            Probability density values.
        """
        return (np.pi / 2) * s * np.exp(-np.pi * s ** 2 / 4)

    @staticmethod
    def poisson_distribution(s: np.ndarray) -> np.ndarray:
        """Poisson distribution for nearest-neighbor spacings.

        P(s) = exp(-s)

        This distribution indicates integrable (regular) dynamics.
        No level repulsion: P(0) = 1.

        Parameters
        ----------
        s : np.ndarray
            Normalized spacings.

        Returns
        -------
        np.ndarray
            Probability density values.
        """
        return np.exp(-s)

    @staticmethod
    def brody_distribution(s: np.ndarray, q: float) -> np.ndarray:
        """Brody distribution interpolating between Poisson and Wigner.

        P(s; q) = (1+q) · a · s^q · exp(-a · s^{1+q})

        where a = [Γ((q+2)/(q+1))]^{1+q}.

        - q = 0 → Poisson
        - q = 1 → Wigner (approximate)

        Parameters
        ----------
        s : np.ndarray
            Normalized spacings.
        q : float
            Brody parameter in [0, 1].

        Returns
        -------
        np.ndarray
            Probability density values.
        """
        a = gamma_func((q + 2) / (q + 1)) ** (1 + q)
        return (1 + q) * a * s ** q * np.exp(-a * s ** (1 + q))

    @staticmethod
    def gue_surmise(s: np.ndarray) -> np.ndarray:
        """GUE (Gaussian Unitary Ensemble) surmise.

        P(s) = (32/π²) s² exp(-4s²/π)

        For systems without time-reversal symmetry.

        Parameters
        ----------
        s : np.ndarray
            Normalized spacings.

        Returns
        -------
        np.ndarray
            Probability density values.
        """
        return (32 / np.pi ** 2) * s ** 2 * np.exp(-4 * s ** 2 / np.pi)

    # ------------------------------------------------------------------
    # Ratio statistic
    # ------------------------------------------------------------------

    @staticmethod
    def ratio_statistic(eigenvalues: np.ndarray) -> dict:
        """Compute the ratio statistic r̃ for chaos detection.

        The ratio r_n = s_n / s_{n-1} and r̃_n = min(r_n, 1/r_n)
        avoids the need for spectral unfolding.

        Reference values:
            - Poisson: ⟨r̃⟩ ≈ 2 ln(2) - 1 ≈ 0.386
            - GOE:     ⟨r̃⟩ ≈ 0.536
            - GUE:     ⟨r̃⟩ ≈ 0.603

        Parameters
        ----------
        eigenvalues : np.ndarray
            Energy eigenvalues.

        Returns
        -------
        dict
            'r_values': array of r̃ values,
            'r_mean': mean ⟨r̃⟩,
            'diagnosis': 'Poisson', 'GOE', or 'intermediate'.
        """
        E = np.sort(np.real(eigenvalues))
        spacings = np.diff(E)

        # Remove zero spacings (degenerate levels)
        spacings = spacings[spacings > 1e-14]

        if len(spacings) < 2:
            return {"r_values": np.array([]), "r_mean": np.nan, "diagnosis": "N/A"}

        # Consecutive spacing ratios
        ratios = spacings[1:] / spacings[:-1]
        r_tilde = np.minimum(ratios, 1.0 / ratios)

        r_mean = float(np.mean(r_tilde))

        # Diagnose
        if r_mean < 0.44:
            diagnosis = "Poisson (integrable)"
        elif r_mean > 0.50:
            diagnosis = "GOE (chaotic)"
        else:
            diagnosis = "Intermediate (mixed)"

        return {
            "r_values": r_tilde,
            "r_mean": r_mean,
            "diagnosis": diagnosis,
        }

    # ------------------------------------------------------------------
    # Brody parameter fitting
    # ------------------------------------------------------------------

    @staticmethod
    def fit_brody(spacings: np.ndarray) -> tuple[float, float]:
        """Fit the Brody distribution to observed spacings.

        Parameters
        ----------
        spacings : np.ndarray
            Unfolded nearest-neighbor spacings.

        Returns
        -------
        q : float
            Fitted Brody parameter (0 = Poisson, 1 = Wigner).
        q_err : float
            Standard error of q from the fit.
        """
        # Build histogram
        s_max = min(float(np.max(spacings)), 5.0)
        bins = np.linspace(0, s_max, 50)
        hist, bin_edges = np.histogram(spacings, bins=bins, density=True)
        s_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Remove zero bins
        mask = hist > 0
        s_fit = s_centers[mask]
        p_fit = hist[mask]

        def brody_func(s, q):
            a = gamma_func((q + 2) / (q + 1)) ** (1 + q)
            return (1 + q) * a * s ** q * np.exp(-a * s ** (1 + q))

        try:
            popt, pcov = curve_fit(brody_func, s_fit, p_fit, p0=[0.5], bounds=(0, 1))
            q = float(popt[0])
            q_err = float(np.sqrt(pcov[0, 0]))
        except RuntimeError:
            q, q_err = np.nan, np.nan

        return q, q_err

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def analyze(self, hamiltonian: np.ndarray) -> dict:
        """Complete level spacing analysis of a Hamiltonian.

        Computes eigenvalues, unfolds the spectrum, computes spacings,
        fits distributions, and computes the ratio statistic.

        Parameters
        ----------
        hamiltonian : np.ndarray
            Hermitian matrix (the Hamiltonian).

        Returns
        -------
        dict
            Comprehensive analysis results:
            - 'eigenvalues': sorted eigenvalues
            - 'spacings': unfolded spacings
            - 'mean_spacing': mean of unfolded spacings
            - 'brody_q': fitted Brody parameter
            - 'brody_q_err': error on Brody parameter
            - 'ratio_stat': ratio statistic results (dict)
            - 'diagnosis': chaos classification string
        """
        # Diagonalize
        eigenvalues = np.linalg.eigvalsh(hamiltonian)

        # Compute spacings
        spacings = self.compute_spacings(eigenvalues, unfold=True)

        # Fit Brody
        q, q_err = self.fit_brody(spacings)

        # Ratio statistic
        ratio = self.ratio_statistic(eigenvalues)

        # Diagnosis
        if q < 0.3:
            diagnosis = "Poisson (integrable)"
        elif q > 0.7:
            diagnosis = "Wigner-Dyson (chaotic)"
        else:
            diagnosis = f"Intermediate (mixed, q={q:.3f})"

        return {
            "eigenvalues": eigenvalues,
            "spacings": spacings,
            "mean_spacing": float(np.mean(spacings)),
            "brody_q": q,
            "brody_q_err": q_err,
            "ratio_stat": ratio,
            "diagnosis": diagnosis,
        }

    def __repr__(self) -> str:
        return "LevelSpacing()"
