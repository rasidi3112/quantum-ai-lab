"""
Product-Formula (Trotter-Suzuki) Time Evolution
================================================

Implements Trotter-Suzuki decompositions for simulating quantum time
evolution when the Hamiltonian is a sum of non-commuting terms:

.. math::

    H = \\sum_k H_k

The exact propagator :math:`e^{-iHt}` is approximated by a product of
simpler exponentials:

**First order (Lie-Trotter):**

.. math::

    S_1(\\Delta t) = \\prod_k e^{-i H_k \\Delta t}

**Second order (Suzuki-Trotter):**

.. math::

    S_2(\\Delta t) = \\prod_k e^{-i H_k \\Delta t/2}
                     \\prod_{k'} e^{-i H_{k'} \\Delta t/2}

where the second product runs in *reverse* order.

**Fourth order (Suzuki S4):**

.. math::

    S_4(\\Delta t) = S_2(p\\,\\Delta t)^2 \\; S_2((1-4p)\\Delta t) \\;
                     S_2(p\\,\\Delta t)^2

with :math:`p = 1/(4 - 4^{1/3})`.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la


class TrotterEvolution:
    """Product-formula time evolution engine.

    The class is stateless: all methods are ``@staticmethod`` or
    ``@classmethod`` so they can be used without instantiation, but an
    instance can also be created for convenience.
    """

    # ----------------------------------------------------------------
    # Single-step propagators
    # ----------------------------------------------------------------

    @staticmethod
    def first_order(
        H_list: List[NDArray[np.complex128]],
        dt: float,
    ) -> NDArray[np.complex128]:
        r"""First-order (Lie-Trotter) propagator.

        .. math::

            U_1(dt) = \prod_{k=0}^{K-1} e^{-i H_k \, dt}

        Parameters
        ----------
        H_list : list of ndarray
            Hamiltonian terms :math:`\{H_k\}`.
        dt : float
            Time-step size.

        Returns
        -------
        ndarray
            Unitary propagator matrix.
        """
        dim = H_list[0].shape[0]
        U = np.eye(dim, dtype=np.complex128)
        for Hk in H_list:
            U = la.expm(-1j * Hk * dt) @ U
        return U

    @staticmethod
    def second_order(
        H_list: List[NDArray[np.complex128]],
        dt: float,
    ) -> NDArray[np.complex128]:
        r"""Second-order (symmetric) Suzuki-Trotter propagator.

        .. math::

            S_2(dt) = \prod_{k} e^{-i H_k \, dt/2}
                      \;\prod_{k'} e^{-i H_{k'} \, dt/2}

        where the second product is in reverse order.

        Parameters
        ----------
        H_list : list of ndarray
            Hamiltonian terms.
        dt : float
            Time-step size.

        Returns
        -------
        ndarray
            Unitary propagator matrix.
        """
        dim = H_list[0].shape[0]
        U = np.eye(dim, dtype=np.complex128)

        # Forward sweep with dt/2
        for Hk in H_list:
            U = la.expm(-1j * Hk * dt / 2) @ U

        # Reverse sweep with dt/2
        for Hk in reversed(H_list):
            U = la.expm(-1j * Hk * dt / 2) @ U

        return U

    @staticmethod
    def fourth_order(
        H_list: List[NDArray[np.complex128]],
        dt: float,
    ) -> NDArray[np.complex128]:
        r"""Fourth-order Suzuki (S4) propagator.

        .. math::

            S_4(dt) = S_2(p\,dt)^2 \; S_2((1-4p)\,dt) \; S_2(p\,dt)^2

        with :math:`p = (4 - 4^{1/3})^{-1}`.

        Parameters
        ----------
        H_list : list of ndarray
            Hamiltonian terms.
        dt : float
            Time-step size.

        Returns
        -------
        ndarray
            Unitary propagator matrix.
        """
        p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        s2_p  = TrotterEvolution.second_order(H_list, p * dt)
        s2_m  = TrotterEvolution.second_order(H_list, (1.0 - 4.0 * p) * dt)

        # S4 = S2(p)^2  ·  S2(1-4p)  ·  S2(p)^2
        return s2_p @ s2_p @ s2_m @ s2_p @ s2_p

    # ----------------------------------------------------------------
    # Full time evolution
    # ----------------------------------------------------------------

    @staticmethod
    def evolve(
        state: NDArray[np.complex128],
        H_list: List[NDArray[np.complex128]],
        t_total: float,
        dt: float,
        order: int = 2,
    ) -> NDArray[np.complex128]:
        """Evolve *state* from 0 to *t_total* in steps of *dt*.

        Parameters
        ----------
        state : ndarray
            Initial state vector.
        H_list : list of ndarray
            Hamiltonian terms.
        t_total : float
            Total evolution time.
        dt : float
            Time-step size.
        order : {1, 2, 4}
            Trotter order.

        Returns
        -------
        ndarray
            Final state vector (normalised).
        """
        propagator_fn = {
            1: TrotterEvolution.first_order,
            2: TrotterEvolution.second_order,
            4: TrotterEvolution.fourth_order,
        }
        if order not in propagator_fn:
            raise ValueError(f"Unsupported Trotter order {order}; use 1, 2, or 4.")

        n_steps = int(np.ceil(t_total / dt))
        U_step = propagator_fn[order](H_list, dt)

        psi = state.copy().astype(np.complex128)
        for _ in range(n_steps):
            psi = U_step @ psi

        psi /= la.norm(psi)
        return psi

    # ----------------------------------------------------------------
    # Exact reference evolution
    # ----------------------------------------------------------------

    @staticmethod
    def exact_evolution(
        state: NDArray[np.complex128],
        H_total: NDArray[np.complex128],
        t: float,
    ) -> NDArray[np.complex128]:
        r"""Exact time evolution :math:`e^{-iHt}|ψ⟩`.

        Parameters
        ----------
        state : ndarray
            Initial state.
        H_total : ndarray
            Full Hamiltonian.
        t : float
            Evolution time.

        Returns
        -------
        ndarray
            Evolved state (normalised).
        """
        U = la.expm(-1j * H_total * t)
        psi = U @ state
        psi /= la.norm(psi)
        return psi

    # ----------------------------------------------------------------
    # Error analysis
    # ----------------------------------------------------------------

    @staticmethod
    def error_bound(
        H_list: List[NDArray[np.complex128]],
        dt: float,
        order: int,
    ) -> float:
        r"""Heuristic upper-bound on the per-step Trotter error.

        For the first-order formula the leading error is :math:`O(dt^2)`:

        .. math::

            \epsilon_1 \sim \frac{dt^2}{2} \sum_{j<k} \|[H_j, H_k]\|

        For second order it is :math:`O(dt^3)`, and for fourth order
        :math:`O(dt^5)`.  We use the Frobenius norm of the commutators.

        Parameters
        ----------
        H_list : list of ndarray
            Hamiltonian terms.
        dt : float
            Time step.
        order : int
            Trotter order (1, 2, or 4).

        Returns
        -------
        float
            Estimated error per step.
        """
        # Sum of commutator norms
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
            return comm_norm * dt ** 5   # rough estimate
        else:
            raise ValueError(f"Unsupported order {order}")

    @staticmethod
    def compare_orders(
        state: NDArray[np.complex128],
        H_list: List[NDArray[np.complex128]],
        t: float,
        dt_values: List[float],
    ) -> Dict[int, List[float]]:
        """Compare Trotter orders 1, 2, 4 across several step sizes.

        For each *dt* in *dt_values*, evolve *state* to time *t* using
        first-, second-, and fourth-order formulae and compute the
        fidelity error :math:`1 - |⟨ψ_{exact}|ψ_{trotter}⟩|^2`
        against the exact result.

        Parameters
        ----------
        state : ndarray
            Initial state vector.
        H_list : list of ndarray
            Hamiltonian terms.
        t : float
            Total evolution time.
        dt_values : list of float
            Step sizes to test.

        Returns
        -------
        dict
            ``{order: [error_for_dt0, error_for_dt1, …]}``
        """
        H_total = sum(H_list)
        psi_exact = TrotterEvolution.exact_evolution(state, H_total, t)

        results: Dict[int, List[float]] = {1: [], 2: [], 4: []}
        for dt in dt_values:
            for order in (1, 2, 4):
                psi_trotter = TrotterEvolution.evolve(state, H_list, t, dt, order)
                fidelity = np.abs(psi_exact.conj() @ psi_trotter) ** 2
                results[order].append(1.0 - fidelity)

        return results

    def __repr__(self) -> str:
        return "TrotterEvolution()"
