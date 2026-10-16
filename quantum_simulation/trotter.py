"""Trotter–Suzuki product formulas.

Approximates ``exp(-i (H_1 + ... + H_K) t)`` by products of the individual
exponentials. The first-order formula has error ``O(dt^2)`` per step, the
symmetric second-order (Strang) formula ``O(dt^3)`` and Suzuki's fourth-order
recursion ``O(dt^5)``.

References
----------
Suzuki, M. "General theory of fractal path integrals with applications to
many-body theories and statistical physics." J. Math. Phys. 32, 400 (1991).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la


class TrotterEvolution:
    """Trotter–Suzuki propagators and accuracy diagnostics."""

    @staticmethod
    def first_order(
        H_list: list[NDArray[np.complex128]],
        dt: float,
    ) -> NDArray[np.complex128]:
        """First-order step ``prod_k exp(-i H_k dt)``.

        Parameters
        ----------
        H_list : list of numpy.ndarray
            Hamiltonian terms.
        dt : float
            Time step.

        Returns
        -------
        numpy.ndarray
        """
        dim = H_list[0].shape[0]
        U = np.eye(dim, dtype=np.complex128)
        for Hk in H_list:
            U = la.expm(-1j * Hk * dt) @ U
        return U

    @staticmethod
    def second_order(
        H_list: list[NDArray[np.complex128]],
        dt: float,
    ) -> NDArray[np.complex128]:
        """Symmetric second-order (Strang) step.

        Parameters
        ----------
        H_list : list of numpy.ndarray
            Hamiltonian terms.
        dt : float
            Time step.

        Returns
        -------
        numpy.ndarray
        """
        dim = H_list[0].shape[0]
        U = np.eye(dim, dtype=np.complex128)

        for Hk in H_list:
            U = la.expm(-1j * Hk * dt / 2) @ U

        for Hk in reversed(H_list):
            U = la.expm(-1j * Hk * dt / 2) @ U

        return U

    @staticmethod
    def fourth_order(
        H_list: list[NDArray[np.complex128]],
        dt: float,
    ) -> NDArray[np.complex128]:
        """Fourth-order Suzuki step built from five second-order steps.

        Parameters
        ----------
        H_list : list of numpy.ndarray
            Hamiltonian terms.
        dt : float
            Time step.

        Returns
        -------
        numpy.ndarray
        """
        p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        s2_p  = TrotterEvolution.second_order(H_list, p * dt)
        s2_m  = TrotterEvolution.second_order(H_list, (1.0 - 4.0 * p) * dt)

        return s2_p @ s2_p @ s2_m @ s2_p @ s2_p


    @staticmethod
    def evolve(
        state: NDArray[np.complex128],
        H_list: list[NDArray[np.complex128]],
        t_total: float,
        dt: float,
        order: int = 2,
    ) -> NDArray[np.complex128]:
        """Evolve a state with a Trotter–Suzuki formula.

        Parameters
        ----------
        state : numpy.ndarray
            Initial state.
        H_list : list of numpy.ndarray
            Hamiltonian terms.
        t_total : float
            Total time.
        dt : float
            Maximum step; shrunk so the steps add up to ``t_total``.
        order : {1, 2, 4}, default 2
            Order of the product formula.

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        ValueError
            For an unsupported order.
        """
        propagator_fn = {
            1: TrotterEvolution.first_order,
            2: TrotterEvolution.second_order,
            4: TrotterEvolution.fourth_order,
        }
        if order not in propagator_fn:
            raise ValueError(f"Unsupported Trotter order {order}; use 1, 2, or 4.")

        n_steps = max(1, int(np.ceil(t_total / dt)))
        U_step = propagator_fn[order](H_list, t_total / n_steps)

        psi = state.copy().astype(np.complex128)
        for _ in range(n_steps):
            psi = U_step @ psi

        psi /= la.norm(psi)
        return psi


    @staticmethod
    def exact_evolution(
        state: NDArray[np.complex128],
        H_total: NDArray[np.complex128],
        t: float,
    ) -> NDArray[np.complex128]:
        """Exact evolution ``exp(-i H t)|psi>`` for reference.

        Parameters
        ----------
        state : numpy.ndarray
            Initial state.
        H_total : numpy.ndarray
            Full Hamiltonian.
        t : float
            Time.

        Returns
        -------
        numpy.ndarray
        """
        U = la.expm(-1j * H_total * t)
        psi = U @ state
        psi /= la.norm(psi)
        return psi


    @staticmethod
    def error_bound(
        H_list: list[NDArray[np.complex128]],
        dt: float,
        order: int,
    ) -> float:
        """Commutator-based estimate of the error of one Trotter step.

        Parameters
        ----------
        H_list : list of numpy.ndarray
            Hamiltonian terms.
        dt : float
            Time step.
        order : {1, 2, 4}
            Order of the product formula.

        Returns
        -------
        float
            Order-of-magnitude estimate built from the Frobenius norms of pairwise
            commutators (not a rigorous bound for order 4).

        Raises
        ------
        ValueError
            For an unsupported order.
        """
        comm_norm = 0.0
        K = len(H_list)
        for j in range(K):
            for k in range(j + 1, K):
                comm = H_list[j] @ H_list[k] - H_list[k] @ H_list[j]
                comm_norm += la.norm(comm, ord="fro")

        if order == 1:
            return 0.5 * comm_norm * dt ** 2
        elif order == 2:
            return (1.0 / 12.0) * comm_norm * dt ** 3
        elif order == 4:
            return comm_norm * dt ** 5
        else:
            raise ValueError(f"Unsupported order {order}")

    @staticmethod
    def compare_orders(
        state: NDArray[np.complex128],
        H_list: list[NDArray[np.complex128]],
        t: float,
        dt_values: list[float],
    ) -> dict[int, list[float]]:
        """Infidelity ``1 - |<exact|trotter>|^2`` of orders 1, 2 and 4 versus step size.

        Parameters
        ----------
        state : numpy.ndarray
            Initial state.
        H_list : list of numpy.ndarray
            Hamiltonian terms.
        t : float
            Total time.
        dt_values : list of float
            Step sizes to test.

        Returns
        -------
        dict
            Infidelities keyed by order, in the order of ``dt_values``.
        """
        H_total = np.sum(H_list, axis=0)
        psi_exact = TrotterEvolution.exact_evolution(state, H_total, t)

        results: dict[int, list[float]] = {1: [], 2: [], 4: []}
        for dt in dt_values:
            for order in (1, 2, 4):
                psi_trotter = TrotterEvolution.evolve(state, H_list, t, dt, order)
                fidelity = np.abs(psi_exact.conj() @ psi_trotter) ** 2
                results[order].append(1.0 - fidelity)

        return results

    def __repr__(self) -> str:
        return "TrotterEvolution()"
