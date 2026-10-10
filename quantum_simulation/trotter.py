from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la


class TrotterEvolution:

    @staticmethod
    def first_order(
        H_list: List[NDArray[np.complex128]],
        dt: float,
    ) -> NDArray[np.complex128]:
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
        dim = H_list[0].shape[0]
        U = np.eye(dim, dtype=np.complex128)

        for Hk in H_list:
            U = la.expm(-1j * Hk * dt / 2) @ U

        for Hk in reversed(H_list):
            U = la.expm(-1j * Hk * dt / 2) @ U

        return U

    @staticmethod
    def fourth_order(
        H_list: List[NDArray[np.complex128]],
        dt: float,
    ) -> NDArray[np.complex128]:
        p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        s2_p  = TrotterEvolution.second_order(H_list, p * dt)
        s2_m  = TrotterEvolution.second_order(H_list, (1.0 - 4.0 * p) * dt)

        return s2_p @ s2_p @ s2_m @ s2_p @ s2_p


    @staticmethod
    def evolve(
        state: NDArray[np.complex128],
        H_list: List[NDArray[np.complex128]],
        t_total: float,
        dt: float,
        order: int = 2,
    ) -> NDArray[np.complex128]:
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


    @staticmethod
    def exact_evolution(
        state: NDArray[np.complex128],
        H_total: NDArray[np.complex128],
        t: float,
    ) -> NDArray[np.complex128]:
        U = la.expm(-1j * H_total * t)
        psi = U @ state
        psi /= la.norm(psi)
        return psi


    @staticmethod
    def error_bound(
        H_list: List[NDArray[np.complex128]],
        dt: float,
        order: int,
    ) -> float:
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
        H_list: List[NDArray[np.complex128]],
        t: float,
        dt_values: List[float],
    ) -> Dict[int, List[float]]:
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
