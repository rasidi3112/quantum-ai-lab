"""Quantum approximate optimization algorithm (QAOA).

QAOA prepares ``|gamma, beta> = prod_k exp(-i beta_k B) exp(-i gamma_k C) |psi_0>``
and tunes the angles classically to optimize ``<C>``. Diagonal cost
Hamiltonians are applied as element-wise phases and the X mixer as a product
of single-qubit rotations, so no matrix exponentials are needed on the fast
path.

For problems with a fixed number of ones (e.g. "pick k of n"), the XY ring
mixer started from a Dicke state keeps the state inside the feasible
subspace exactly (the quantum alternating operator ansatz).

References
----------
Farhi, E., Goldstone, J. & Gutmann, S. "A quantum approximate optimization
algorithm." arXiv:1411.4028 (2014).
Hadfield, S. et al. "From the quantum approximate optimization algorithm to
a quantum alternating operator ansatz." Algorithms 12, 34 (2019).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la
from scipy.optimize import minimize

_I2 = np.eye(2, dtype=np.complex128)
_X  = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y  = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z  = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _kron_chain(*ops: NDArray[np.complex128]) -> NDArray[np.complex128]:
    out: NDArray[np.complex128] = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op).astype(np.complex128, copy=False)
    return out


def _diagonal_or_none(H: NDArray[np.complex128]) -> NDArray[np.complex128] | None:
    if H.ndim == 1:
        return H
    diag = np.diagonal(H)
    if np.count_nonzero(H - np.diag(diag)) == 0:
        return diag
    return None


def _apply_x_mixer(
    psi: NDArray[np.complex128], beta: float, n_qubits: int
) -> NDArray[np.complex128]:
    c, s = np.cos(beta), np.sin(beta)
    tensor = psi.reshape([2] * n_qubits)
    for q in range(n_qubits):
        tensor = c * tensor - 1j * s * np.flip(tensor, axis=q)
    return tensor.reshape(-1)


class QAOA:
    """QAOA simulator and optimizer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (problem variables).
    seed : int, optional
        Seed for random initial angles.

    Attributes
    ----------
    optimal_params : numpy.ndarray or None
        Best angles ``[gamma_1..gamma_p, beta_1..beta_p]`` from the last run.
    optimal_state : numpy.ndarray or None
        The corresponding QAOA state.
    optimal_energy : float or None
        ``<C>`` in that state.
    """

    def __init__(self, n_qubits: int, seed: int | None = None) -> None:
        self.n_qubits = n_qubits
        self.rng = np.random.default_rng(seed)
        self.dim = 2 ** n_qubits
        self._cost_H: NDArray[np.complex128] | None = None
        self._mixer_H: NDArray[np.complex128] | None = None
        self.optimal_params: NDArray[np.float64] | None = None
        self.optimal_state: NDArray[np.complex128] | None = None
        self.optimal_energy: float | None = None


    @staticmethod
    def cost_hamiltonian(
        graph: list[tuple[int, int]],
        n_qubits: int,
        weights: list[float] | None = None,
    ) -> NDArray[np.complex128]:
        """MaxCut cost Hamiltonian ``C = sum_(i,j) w_ij (I - Z_i Z_j) / 2``.

        Parameters
        ----------
        graph : list of tuple of int
            Edges.
        n_qubits : int
            Number of nodes.
        weights : list of float, optional
            Edge weights (default 1).

        Returns
        -------
        numpy.ndarray
            Diagonal matrix whose entries are the cut values of each bitstring.
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
        """Transverse-field mixer ``B = sum_i X_i`` as a dense matrix.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.

        Returns
        -------
        numpy.ndarray
        """
        dim = 2 ** n_qubits
        B = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(n_qubits):
            ops = [_I2] * n_qubits
            ops[i] = _X
            B += _kron_chain(*ops)
        return B


    @staticmethod
    def xy_mixer_hamiltonian(n_qubits: int, ring: bool = True) -> NDArray[np.complex128]:
        """XY mixer ``sum_i (X_i X_(i+1) + Y_i Y_(i+1)) / 2``.

        It commutes with the total number of ones, so evolution preserves Hamming
        weight.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        ring : bool, default True
            Couple the last qubit back to the first.

        Returns
        -------
        numpy.ndarray
        """
        dim = 2 ** n_qubits
        B = np.zeros((dim, dim), dtype=np.complex128)
        n_pairs = n_qubits if ring and n_qubits > 2 else n_qubits - 1
        for i in range(n_pairs):
            j = (i + 1) % n_qubits
            for P in (_X, _Y):
                ops = [_I2] * n_qubits
                ops[i] = P
                ops[j] = P
                B += 0.5 * _kron_chain(*ops)
        return B

    @staticmethod
    def dicke_state(n_qubits: int, hamming_weight: int) -> NDArray[np.complex128]:
        """Uniform superposition of all bitstrings with a given Hamming weight.

        Parameters
        ----------
        n_qubits : int
            Number of qubits.
        hamming_weight : int
            Number of ones.

        Returns
        -------
        numpy.ndarray
            Normalized Dicke state.
        """
        weights = np.array([bin(i).count("1") for i in range(2 ** n_qubits)])
        psi = (weights == hamming_weight).astype(np.complex128)
        return psi / np.linalg.norm(psi)


    @staticmethod
    def _evolve(
        gamma: NDArray[np.float64],
        beta: NDArray[np.float64],
        cost_op: NDArray[np.complex128],
        mixer_eig: tuple[NDArray[np.float64], NDArray[np.complex128]] | None,
        psi0: NDArray[np.complex128] | None,
    ) -> NDArray[np.complex128]:
        n = cost_op.shape[0]
        n_qubits = int(round(np.log2(n)))
        psi = np.ones(n, dtype=np.complex128) / np.sqrt(n) if psi0 is None else psi0.copy()

        for l in range(len(gamma)):
            if cost_op.ndim == 1:
                psi = np.exp(-1j * gamma[l] * cost_op) * psi
            else:
                psi = la.expm(-1j * gamma[l] * cost_op) @ psi
            if mixer_eig is None:
                psi = _apply_x_mixer(psi, beta[l], n_qubits)
            else:
                w, V = mixer_eig
                psi = V @ (np.exp(-1j * beta[l] * w) * (V.conj().T @ psi))

        return psi

    @staticmethod
    def qaoa_circuit(
        gamma: NDArray[np.float64],
        beta: NDArray[np.float64],
        cost_H: NDArray[np.complex128],
        mixer_H: NDArray[np.complex128] | None = None,
        initial_state: NDArray[np.complex128] | None = None,
    ) -> NDArray[np.complex128]:
        """Prepare the QAOA state for given angles.

        Parameters
        ----------
        gamma, beta : numpy.ndarray
            Cost and mixer angles, one per layer.
        cost_H : numpy.ndarray
            Cost Hamiltonian, as a matrix or as its diagonal.
        mixer_H : numpy.ndarray, optional
            Mixer Hamiltonian; the fast ``sum_i X_i`` mixer is used when omitted.
        initial_state : numpy.ndarray, optional
            Starting state; defaults to ``|+>^n``.

        Returns
        -------
        numpy.ndarray
        """
        cost_diag = _diagonal_or_none(cost_H)
        cost_op = cost_diag if cost_diag is not None else cost_H
        mixer_eig = None if mixer_H is None else la.eigh(mixer_H)
        return QAOA._evolve(gamma, beta, cost_op, mixer_eig, initial_state)


    @staticmethod
    def expectation_value(
        state: NDArray[np.complex128],
        hamiltonian: NDArray[np.complex128],
    ) -> float:
        """Energy ``<psi|H|psi>`` for a matrix or a diagonal ``H``."""
        if hamiltonian.ndim == 1:
            return float(np.real(np.sum(np.abs(state) ** 2 * hamiltonian)))
        return float(np.real(state.conj() @ hamiltonian @ state))


    def optimize(
        self,
        graph: list[tuple[int, int]],
        p_layers: int = 1,
        method: str = "COBYLA",
        weights: list[float] | None = None,
        maxiter: int = 1000,
        initial_params: NDArray[np.float64] | None = None,
        n_restarts: int = 1,
    ) -> dict[str, Any]:
        """Maximize the cut value of a graph.

        Parameters
        ----------
        graph : list of tuple of int
            Edges.
        p_layers : int, default 1
            Circuit depth ``p``.
        method : str, default "COBYLA"
            Any :func:`scipy.optimize.minimize` method.
        weights : list of float, optional
            Edge weights.
        maxiter : int, default 1000
            Optimizer iteration limit.
        initial_params : numpy.ndarray, optional
            Starting angles for the first restart.
        n_restarts : int, default 1
            Independent random restarts; the best result is kept.

        Returns
        -------
        dict
            ``energy`` (expected cut), ``params``, ``state`` and the raw SciPy
            ``result``.
        """
        cost_H = self.cost_hamiltonian(graph, self.n_qubits, weights)
        return self.optimize_hamiltonian(
            cost_H, p_layers, method, maxiter, initial_params,
            n_restarts=n_restarts, maximize=True,
        )

    def optimize_hamiltonian(
        self,
        cost_H: NDArray[np.complex128],
        p_layers: int = 1,
        method: str = "COBYLA",
        maxiter: int = 1000,
        initial_params: NDArray[np.float64] | None = None,
        n_restarts: int = 1,
        maximize: bool = False,
        mixer: Literal["x", "xy"] = "x",
        hamming_weight: int | None = None,
    ) -> dict[str, Any]:
        """Optimize QAOA for an arbitrary cost Hamiltonian.

        Parameters
        ----------
        cost_H : numpy.ndarray
            Cost Hamiltonian, as a matrix or as the diagonal of cost values.
        p_layers : int, default 1
            Circuit depth ``p``.
        method : str, default "COBYLA"
            Any :func:`scipy.optimize.minimize` method.
        maxiter : int, default 1000
            Optimizer iteration limit.
        initial_params : numpy.ndarray, optional
            Starting angles for the first restart.
        n_restarts : int, default 1
            Independent random restarts; the best result is kept.
        maximize : bool, default False
            Maximize ``<C>`` instead of minimizing it.
        mixer : {"x", "xy"}, default "x"
            ``"x"``: transverse field from ``|+>^n``. ``"xy"``: XY ring mixer from a
            Dicke state, which enforces a fixed Hamming weight.
        hamming_weight : int, optional
            Required when ``mixer="xy"``.

        Returns
        -------
        dict
            ``energy``, ``params``, ``state`` and the raw SciPy ``result``.

        Raises
        ------
        ValueError
            For an unknown mixer or a missing ``hamming_weight``.
        """
        cost_diag = _diagonal_or_none(cost_H)
        cost_op = cost_diag if cost_diag is not None else cost_H
        self._cost_H = cost_H

        if mixer == "x":
            self._mixer_H = None
            mixer_eig, psi0 = None, None
        elif mixer == "xy":
            if hamming_weight is None:
                raise ValueError("The XY mixer needs a hamming_weight constraint.")
            self._mixer_H = self.xy_mixer_hamiltonian(self.n_qubits)
            mixer_eig = la.eigh(self._mixer_H)
            psi0 = self.dicke_state(self.n_qubits, hamming_weight)
        else:
            raise ValueError(f"Unknown mixer '{mixer}'; use 'x' or 'xy'.")

        sign = -1.0 if maximize else 1.0

        def objective(params: NDArray[np.float64]) -> float:
            psi = self._evolve(params[:p_layers], params[p_layers:], cost_op, mixer_eig, psi0)
            return sign * self.expectation_value(psi, cost_op)

        def start(restart: int) -> NDArray[np.float64]:
            if initial_params is not None and restart == 0:
                return np.asarray(initial_params, dtype=np.float64)
            return self.rng.uniform(0, np.pi, 2 * p_layers)

        results = [
            minimize(objective, start(r), method=method, options={"maxiter": maxiter})
            for r in range(max(1, n_restarts))
        ]
        best = min(results, key=lambda r: r.fun)

        psi_opt = self._evolve(best.x[:p_layers], best.x[p_layers:], cost_op, mixer_eig, psi0)
        energy = self.expectation_value(psi_opt, cost_op)

        self.optimal_params = best.x
        self.optimal_state = psi_opt
        self.optimal_energy = energy

        return {
            "energy": energy,
            "params": best.x,
            "state": psi_opt,
            "result": best,
        }


    @staticmethod
    def sample_solution(
        state: NDArray[np.complex128],
        n_samples: int = 1024,
        seed: int | None = None,
    ) -> dict[str, int]:
        """Sample bitstrings from a state.

        Parameters
        ----------
        state : numpy.ndarray
            QAOA state.
        n_samples : int, default 1024
            Number of shots.
        seed : int, optional
            Sampling seed.

        Returns
        -------
        dict
            Counts per bitstring, most frequent first.
        """
        n_qubits = int(np.log2(len(state)))
        probs = np.abs(state) ** 2
        probs /= probs.sum()

        rng = np.random.default_rng(seed)
        indices = rng.choice(len(state), size=n_samples, p=probs)
        counts: dict[str, int] = {}
        for idx in indices:
            bs = format(idx, f"0{n_qubits}b")
            counts[bs] = counts.get(bs, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))


    @staticmethod
    def energy_landscape(
        graph: list[tuple[int, int]],
        n_qubits: int,
        gamma_range: NDArray[np.float64],
        beta_range: NDArray[np.float64],
        p: int = 1,
        weights: list[float] | None = None,
    ) -> NDArray[np.float64]:
        """Compute the expected cut value over a grid of first-layer angles.

        Parameters
        ----------
        graph : list of tuple of int
            Edges.
        n_qubits : int
            Number of nodes.
        gamma_range, beta_range : numpy.ndarray
            Grid values of the first-layer angles (deeper layers are set to zero).
        p : int, default 1
            Circuit depth.
        weights : list of float, optional
            Edge weights.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(gamma_range), len(beta_range))``.
        """
        cost_diag = np.diagonal(QAOA.cost_hamiltonian(graph, n_qubits, weights))

        landscape = np.zeros((len(gamma_range), len(beta_range)))
        for ig, g in enumerate(gamma_range):
            for ib, b in enumerate(beta_range):
                gamma = np.array([g] + [0.0] * (p - 1))
                beta  = np.array([b] + [0.0] * (p - 1))
                psi = QAOA.qaoa_circuit(gamma, beta, cost_diag)
                landscape[ig, ib] = QAOA.expectation_value(psi, cost_diag)
        return landscape

    def __repr__(self) -> str:
        return f"QAOA(n_qubits={self.n_qubits})"
