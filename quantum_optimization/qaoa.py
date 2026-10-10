from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la
from scipy.optimize import minimize


_I2 = np.eye(2, dtype=np.complex128)
_X  = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Z  = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _kron_chain(*ops: NDArray[np.complex128]) -> NDArray[np.complex128]:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


class QAOA:
    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self._cost_H: Optional[NDArray[np.complex128]] = None
        self._mixer_H: Optional[NDArray[np.complex128]] = None
        self.optimal_params: Optional[NDArray[np.float64]] = None
        self.optimal_state: Optional[NDArray[np.complex128]] = None
        self.optimal_energy: Optional[float] = None


    @staticmethod
    def cost_hamiltonian(
        graph: List[Tuple[int, int]],
        n_qubits: int,
        weights: Optional[List[float]] = None,
    ) -> NDArray[np.complex128]:
        if weights is None:
            weights = [1.0] * len(graph)

        dim = 2 ** n_qubits
        C = np.zeros((dim, dim), dtype=np.complex128)
        I_full = np.eye(dim, dtype=np.complex128)

        for (i, j), w in zip(graph, weights):
            ops = [_I2] * n_qubits
            ops[i] = _Z
            ops[j] = _Z
            ZZ = _kron_chain(*ops)
            C += w * (I_full - ZZ) / 2.0

        return C

    @staticmethod
    def mixer_hamiltonian(n_qubits: int) -> NDArray[np.complex128]:
        dim = 2 ** n_qubits
        B = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(n_qubits):
            ops = [_I2] * n_qubits
            ops[i] = _X
            B += _kron_chain(*ops)
        return B


    @staticmethod
    def qaoa_circuit(
        gamma: NDArray[np.float64],
        beta: NDArray[np.float64],
        cost_H: NDArray[np.complex128],
        mixer_H: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        n = cost_H.shape[0]
        psi = np.ones(n, dtype=np.complex128) / np.sqrt(n)

        p = len(gamma)
        for l in range(p):
            U_C = la.expm(-1j * gamma[l] * cost_H)
            psi = U_C @ psi
            U_B = la.expm(-1j * beta[l] * mixer_H)
            psi = U_B @ psi

        return psi


    @staticmethod
    def expectation_value(
        state: NDArray[np.complex128],
        hamiltonian: NDArray[np.complex128],
    ) -> float:
        return float(np.real(state.conj() @ hamiltonian @ state))


    def optimize(
        self,
        graph: List[Tuple[int, int]],
        p_layers: int = 1,
        method: str = "COBYLA",
        weights: Optional[List[float]] = None,
        maxiter: int = 1000,
        initial_params: Optional[NDArray[np.float64]] = None,
    ) -> Dict[str, Any]:
        cost_H  = self.cost_hamiltonian(graph, self.n_qubits, weights)
        mixer_H = self.mixer_hamiltonian(self.n_qubits)
        self._cost_H  = cost_H
        self._mixer_H = mixer_H

        if initial_params is None:
            initial_params = np.random.uniform(0, np.pi, 2 * p_layers)

        def objective(params: NDArray[np.float64]) -> float:
            gamma = params[:p_layers]
            beta  = params[p_layers:]
            psi = self.qaoa_circuit(gamma, beta, cost_H, mixer_H)
            return -self.expectation_value(psi, cost_H)

        result = minimize(objective, initial_params, method=method,
                          options={"maxiter": maxiter})

        gamma_opt = result.x[:p_layers]
        beta_opt  = result.x[p_layers:]
        psi_opt = self.qaoa_circuit(gamma_opt, beta_opt, cost_H, mixer_H)
        energy = self.expectation_value(psi_opt, cost_H)

        self.optimal_params = result.x
        self.optimal_state  = psi_opt
        self.optimal_energy = energy

        return {
            "energy": energy,
            "params": result.x,
            "state": psi_opt,
            "result": result,
        }


    @staticmethod
    def sample_solution(
        state: NDArray[np.complex128],
        n_samples: int = 1024,
    ) -> Dict[str, int]:
        n_qubits = int(np.log2(len(state)))
        probs = np.abs(state) ** 2
        probs /= probs.sum()

        indices = np.random.choice(len(state), size=n_samples, p=probs)
        counts: Dict[str, int] = {}
        for idx in indices:
            bs = format(idx, f"0{n_qubits}b")
            counts[bs] = counts.get(bs, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))


    @staticmethod
    def energy_landscape(
        graph: List[Tuple[int, int]],
        n_qubits: int,
        gamma_range: NDArray[np.float64],
        beta_range: NDArray[np.float64],
        p: int = 1,
        weights: Optional[List[float]] = None,
    ) -> NDArray[np.float64]:
        cost_H  = QAOA.cost_hamiltonian(graph, n_qubits, weights)
        mixer_H = QAOA.mixer_hamiltonian(n_qubits)

        landscape = np.zeros((len(gamma_range), len(beta_range)))
        for ig, g in enumerate(gamma_range):
            for ib, b in enumerate(beta_range):
                gamma = np.array([g] + [0.0] * (p - 1))
                beta  = np.array([b] + [0.0] * (p - 1))
                psi = QAOA.qaoa_circuit(gamma, beta, cost_H, mixer_H)
                landscape[ig, ib] = QAOA.expectation_value(psi, cost_H)
        return landscape

    def __repr__(self) -> str:
        return f"QAOA(n_qubits={self.n_qubits})"
