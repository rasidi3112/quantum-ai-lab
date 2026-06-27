"""
Quantum Approximate Optimization Algorithm (QAOA)
==================================================

Implements QAOA for combinatorial optimisation problems encoded as
classical Ising Hamiltonians.

The algorithm alternates *p* layers of:

1. **Cost unitary**   :math:`e^{-i\\gamma_l C}`
2. **Mixer unitary**  :math:`e^{-i\\beta_l B}`

where *C* is the cost Hamiltonian derived from the problem and
:math:`B = \\sum_i X_i` is the standard transverse-field mixer.

The initial state is the uniform superposition :math:`|+\\rangle^{\\otimes n}`.
The variational parameters :math:`(\\boldsymbol\\gamma, \\boldsymbol\\beta)`
are optimised classically to minimise :math:`\\langle\\psi|C|\\psi\\rangle`.

References
----------
* Farhi, Goldstone & Gutmann, arXiv:1411.4028 (2014).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la
from scipy.optimize import minimize


# ======================================================================
# Pauli building blocks
# ======================================================================

_I2 = np.eye(2, dtype=np.complex128)
_X  = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Z  = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _kron_chain(*ops: NDArray[np.complex128]) -> NDArray[np.complex128]:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


# ======================================================================
# QAOA class
# ======================================================================

class QAOA:
    """Quantum Approximate Optimization Algorithm.

    A graph (or arbitrary cost Hamiltonian) is provided, and QAOA
    constructs and optimises the variational quantum circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (= nodes in a graph problem).
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self._cost_H: Optional[NDArray[np.complex128]] = None
        self._mixer_H: Optional[NDArray[np.complex128]] = None
        self.optimal_params: Optional[NDArray[np.float64]] = None
        self.optimal_state: Optional[NDArray[np.complex128]] = None
        self.optimal_energy: Optional[float] = None

    # ----------------------------------------------------------------
    # Hamiltonian builders
    # ----------------------------------------------------------------

    @staticmethod
    def cost_hamiltonian(
        graph: List[Tuple[int, int]],
        n_qubits: int,
        weights: Optional[List[float]] = None,
    ) -> NDArray[np.complex128]:
        r"""Build cost Hamiltonian from a graph.

        For MaxCut the cost operator is

        .. math::

            C = \sum_{(i,j)\in E} w_{ij}\,\frac{I - Z_i Z_j}{2}

        Parameters
        ----------
        graph : list of (int, int)
            Edge list.
        n_qubits : int
            Number of qubits / nodes.
        weights : list of float, optional
            Edge weights (default all 1).

        Returns
        -------
        ndarray, shape (2^n, 2^n)
        """
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
        r"""Standard transverse-field mixer :math:`B = \sum_i X_i`.

        Parameters
        ----------
        n_qubits : int

        Returns
        -------
        ndarray
        """
        dim = 2 ** n_qubits
        B = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(n_qubits):
            ops = [_I2] * n_qubits
            ops[i] = _X
            B += _kron_chain(*ops)
        return B

    # ----------------------------------------------------------------
    # Circuit
    # ----------------------------------------------------------------

    @staticmethod
    def qaoa_circuit(
        gamma: NDArray[np.float64],
        beta: NDArray[np.float64],
        cost_H: NDArray[np.complex128],
        mixer_H: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        r"""Apply *p* QAOA layers to the uniform superposition.

        .. math::

            |\psi(\boldsymbol\gamma, \boldsymbol\beta)\rangle
            = \prod_{l=1}^{p} e^{-i\beta_l B}\,e^{-i\gamma_l C}
              \;|+\rangle^{\otimes n}

        Parameters
        ----------
        gamma, beta : ndarray, shape (p,)
            Variational parameters (one per layer).
        cost_H, mixer_H : ndarray
            Cost and mixer Hamiltonians.

        Returns
        -------
        ndarray
            Final state vector.
        """
        n = cost_H.shape[0]
        # |+>^n
        psi = np.ones(n, dtype=np.complex128) / np.sqrt(n)

        p = len(gamma)
        for l in range(p):
            # Cost unitary
            U_C = la.expm(-1j * gamma[l] * cost_H)
            psi = U_C @ psi
            # Mixer unitary
            U_B = la.expm(-1j * beta[l] * mixer_H)
            psi = U_B @ psi

        return psi

    # ----------------------------------------------------------------
    # Expectation value
    # ----------------------------------------------------------------

    @staticmethod
    def expectation_value(
        state: NDArray[np.complex128],
        hamiltonian: NDArray[np.complex128],
    ) -> float:
        """Compute ⟨ψ|H|ψ⟩."""
        return float(np.real(state.conj() @ hamiltonian @ state))

    # ----------------------------------------------------------------
    # Optimisation
    # ----------------------------------------------------------------

    def optimize(
        self,
        graph: List[Tuple[int, int]],
        p_layers: int = 1,
        method: str = "COBYLA",
        weights: Optional[List[float]] = None,
        maxiter: int = 1000,
        initial_params: Optional[NDArray[np.float64]] = None,
    ) -> Dict[str, Any]:
        r"""Optimise QAOA parameters for a given graph.

        Parameters
        ----------
        graph : list of (int, int)
            Edge list.
        p_layers : int
            Number of QAOA layers.
        method : str
            ``scipy.optimize.minimize`` method.
        weights : list of float, optional
            Edge weights.
        maxiter : int
            Maximum iterations for the classical optimiser.
        initial_params : ndarray, optional
            Starting parameters ``[γ_1,…,γ_p, β_1,…,β_p]``.

        Returns
        -------
        dict
            ``{'energy', 'params', 'state', 'result'}``
        """
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
            return -self.expectation_value(psi, cost_H)  # minimise ↔ maximise cut

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

    # ----------------------------------------------------------------
    # Sampling
    # ----------------------------------------------------------------

    @staticmethod
    def sample_solution(
        state: NDArray[np.complex128],
        n_samples: int = 1024,
    ) -> Dict[str, int]:
        """Sample computational-basis bitstrings from the final state.

        Parameters
        ----------
        state : ndarray
            State vector.
        n_samples : int
            Number of samples.

        Returns
        -------
        dict
            ``{bitstring: count}`` sorted by count descending.
        """
        n_qubits = int(np.log2(len(state)))
        probs = np.abs(state) ** 2
        probs /= probs.sum()  # numerical safety

        indices = np.random.choice(len(state), size=n_samples, p=probs)
        counts: Dict[str, int] = {}
        for idx in indices:
            bs = format(idx, f"0{n_qubits}b")
            counts[bs] = counts.get(bs, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    # ----------------------------------------------------------------
    # Energy landscape
    # ----------------------------------------------------------------

    @staticmethod
    def energy_landscape(
        graph: List[Tuple[int, int]],
        n_qubits: int,
        gamma_range: NDArray[np.float64],
        beta_range: NDArray[np.float64],
        p: int = 1,
        weights: Optional[List[float]] = None,
    ) -> NDArray[np.float64]:
        r"""Compute 2-D energy landscape for p=1 QAOA.

        Parameters
        ----------
        graph : edge list
        n_qubits : int
        gamma_range, beta_range : 1-D arrays
        p : int
            Number of layers (only first-layer params are scanned).
        weights : list of float, optional

        Returns
        -------
        ndarray, shape (len(gamma_range), len(beta_range))
            Cost expectation values.
        """
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
