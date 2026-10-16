"""Random-matrix diagnostics of quantum chaos.

Integrable systems show uncorrelated (Poisson) energy levels, whereas chaotic
systems show level repulsion described by random-matrix theory: the
Wigner–Dyson distributions (GOE, GUE). This module provides spacing
distributions, the Brody interpolation, and the unfolding-free ratio
statistic ``<r>`` (0.386 for Poisson, 0.531 for GOE).

References
----------
Bohigas, O., Giannoni, M. J. & Schmit, C. Phys. Rev. Lett. 52, 1 (1984).
Atas, Y. Y. et al. Phys. Rev. Lett. 110, 084101 (2013).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gamma as gamma_func


class LevelSpacing:
    """Level-spacing statistics and chaos diagnosis for a spectrum."""

    def __init__(self) -> None:
        pass


    @staticmethod
    def compute_spacings(
        eigenvalues: np.ndarray, unfold: bool = True
    ) -> np.ndarray:
        """Nearest-neighbour spacings of a spectrum.

        Parameters
        ----------
        eigenvalues : numpy.ndarray
            Energy levels (sorted internally).
        unfold : bool, default True
            Unfold with a polynomial fit of the level staircase so that the mean
            spacing is about 1.

        Returns
        -------
        numpy.ndarray
            Positive spacings.
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
        E = sorted_eigenvalues
        N = len(E)

        staircase = np.arange(N, dtype=np.float64)

        coeffs = np.polyfit(E, staircase, poly_order)
        N_smooth = np.polyval(coeffs, E)

        spacings = np.diff(N_smooth)

        spacings = spacings[spacings > 0]

        return spacings


    @staticmethod
    def wigner_surmise(s: np.ndarray) -> np.ndarray:
        """GOE Wigner surmise ``P(s) = (pi/2) s exp(-pi s^2 / 4)``."""
        return (np.pi / 2) * s * np.exp(-np.pi * s ** 2 / 4)

    @staticmethod
    def poisson_distribution(s: np.ndarray) -> np.ndarray:
        """Poisson spacing distribution ``P(s) = exp(-s)``."""
        return np.exp(-s)

    @staticmethod
    def brody_distribution(s: np.ndarray, q: float) -> np.ndarray:
        """Brody distribution interpolating Poisson (``q = 0``) and GOE (``q = 1``).

        Parameters
        ----------
        s : numpy.ndarray
            Spacings.
        q : float
            Brody parameter in ``[0, 1]``.

        Returns
        -------
        numpy.ndarray
            Normalized density ``P(s)``.
        """
        a = gamma_func((q + 2) / (q + 1)) ** (1 + q)
        return (1 + q) * a * s ** q * np.exp(-a * s ** (1 + q))

    @staticmethod
    def gue_surmise(s: np.ndarray) -> np.ndarray:
        """GUE Wigner surmise ``P(s) = (32/pi^2) s^2 exp(-4 s^2 / pi)``."""
        return (32 / np.pi ** 2) * s ** 2 * np.exp(-4 * s ** 2 / np.pi)


    @staticmethod
    def ratio_statistic(eigenvalues: np.ndarray) -> dict:
        """Mean spacing ratio ``<r>``, which needs no unfolding.

        Parameters
        ----------
        eigenvalues : numpy.ndarray
            Energy levels.

        Returns
        -------
        dict
            ``r_values`` (per-level ``min(r, 1/r)``), ``r_mean`` and a ``diagnosis``
            string (Poisson below 0.44, GOE above 0.50).
        """
        E = np.sort(np.real(eigenvalues))
        spacings = np.diff(E)

        spacings = spacings[spacings > 1e-14]

        if len(spacings) < 2:
            return {"r_values": np.array([]), "r_mean": np.nan, "diagnosis": "N/A"}

        ratios = spacings[1:] / spacings[:-1]
        r_tilde = np.minimum(ratios, 1.0 / ratios)

        r_mean = float(np.mean(r_tilde))

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


    @staticmethod
    def fit_brody(spacings: np.ndarray) -> tuple[float, float]:
        """Fit the Brody parameter to a spacing histogram.

        Parameters
        ----------
        spacings : numpy.ndarray
            Unfolded spacings.

        Returns
        -------
        q : float
            Best-fit Brody parameter (``nan`` if the fit fails).
        q_err : float
            One-sigma uncertainty.
        """
        s_max = min(float(np.max(spacings)), 5.0)
        bins = np.linspace(0, s_max, 50)
        hist, bin_edges = np.histogram(spacings, bins=bins, density=True)
        s_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        mask = hist > 0
        s_fit = s_centers[mask]
        p_fit = hist[mask]

        def brody_func(s: np.ndarray, q: float) -> np.ndarray:
            a = gamma_func((q + 2) / (q + 1)) ** (1 + q)
            return (1 + q) * a * s ** q * np.exp(-a * s ** (1 + q))

        try:
            popt, pcov = curve_fit(brody_func, s_fit, p_fit, p0=[0.5], bounds=(0, 1))
            q = float(popt[0])
            q_err = float(np.sqrt(pcov[0, 0]))
        except RuntimeError:
            q, q_err = np.nan, np.nan

        return q, q_err


    def analyze(self, hamiltonian: np.ndarray) -> dict:
        """Diagonalize a Hamiltonian and diagnose its level statistics.

        Parameters
        ----------
        hamiltonian : numpy.ndarray
            Hermitian matrix.

        Returns
        -------
        dict
            Eigenvalues, unfolded spacings, mean spacing, Brody fit, ratio
            statistic and an overall diagnosis.
        """
        eigenvalues = np.linalg.eigvalsh(hamiltonian)

        spacings = self.compute_spacings(eigenvalues, unfold=True)

        q, q_err = self.fit_brody(spacings)

        ratio = self.ratio_statistic(eigenvalues)

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
