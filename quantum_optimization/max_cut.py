"""
MaxCut Solver
=============

End-to-end pipeline for the **Maximum Cut** problem on unweighted /
weighted graphs:

1. Generate random graphs.
2. Map MaxCut → Ising Hamiltonian.
3. Solve via brute-force enumeration (classical baseline).
4. Solve via QAOA (quantum approximate).
5. Evaluate and visualise solutions.

The MaxCut cost function is:

.. math::

    C(z) = \\sum_{(i,j)\\in E} \\frac{1 - z_i z_j}{2}
         = \\sum_{(i,j)\\in E} \\frac{I - Z_i Z_j}{2}

where :math:`z_i \\in \\{+1,-1\\}` encodes the partition.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from quantum_optimization.qaoa import QAOA


class MaxCutSolver:
    """MaxCut problem solver with classical and quantum methods.

    Parameters
    ----------
    n_nodes : int
        Number of graph nodes.
    """

    def __init__(self, n_nodes: int) -> None:
        self.n_nodes = n_nodes

    # ----------------------------------------------------------------
    # Graph generation
    # ----------------------------------------------------------------

    @staticmethod
    def random_graph(
        n_nodes: int,
        edge_prob: float = 0.5,
        seed: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """Generate a random Erdős-Rényi graph.

        Parameters
        ----------
        n_nodes : int
            Number of vertices.
        edge_prob : float
            Probability that any pair of nodes is connected.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        list of (int, int)
            Edge list (undirected, each edge listed once).
        """
        rng = np.random.default_rng(seed)
        edges: List[Tuple[int, int]] = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < edge_prob:
                    edges.append((i, j))
        return edges

    # ----------------------------------------------------------------
    # Hamiltonian
    # ----------------------------------------------------------------

    @staticmethod
    def graph_to_hamiltonian(
        graph: List[Tuple[int, int]],
        n_nodes: int,
        weights: Optional[List[float]] = None,
    ) -> NDArray[np.complex128]:
        r"""Convert MaxCut instance to Ising Hamiltonian.

        .. math::

            C = \sum_{(i,j)\in E} w_{ij}\,\frac{I - Z_i Z_j}{2}

        Parameters
        ----------
        graph : edge list
        n_nodes : int
        weights : list of float, optional

        Returns
        -------
        ndarray, shape (2^n, 2^n)
        """
        return QAOA.cost_hamiltonian(graph, n_nodes, weights)

    # ----------------------------------------------------------------
    # Classical brute force
    # ----------------------------------------------------------------

    @staticmethod
    def cut_value(
        bitstring: str,
        graph: List[Tuple[int, int]],
        weights: Optional[List[float]] = None,
    ) -> float:
        """Evaluate the cut value for a given partition.

        Parameters
        ----------
        bitstring : str
            Binary string of length n_nodes (e.g. ``"0110"``).
        graph : edge list
        weights : list of float, optional

        Returns
        -------
        float
            Number (or total weight) of edges crossing the cut.
        """
        if weights is None:
            weights = [1.0] * len(graph)
        value = 0.0
        for (i, j), w in zip(graph, weights):
            if bitstring[i] != bitstring[j]:
                value += w
        return value

    @staticmethod
    def brute_force_solve(
        graph: List[Tuple[int, int]],
        n_nodes: int,
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Find the maximum cut by exhaustive enumeration.

        Parameters
        ----------
        graph : edge list
        n_nodes : int
        weights : list of float, optional

        Returns
        -------
        dict
            ``{'max_cut': float, 'bitstring': str, 'all_cuts': dict}``
        """
        best_cut = -1.0
        best_bs = ""
        all_cuts: Dict[str, float] = {}
        for idx in range(2 ** n_nodes):
            bs = format(idx, f"0{n_nodes}b")
            val = MaxCutSolver.cut_value(bs, graph, weights)
            all_cuts[bs] = val
            if val > best_cut:
                best_cut = val
                best_bs = bs
        return {"max_cut": best_cut, "bitstring": best_bs, "all_cuts": all_cuts}

    # ----------------------------------------------------------------
    # QAOA wrapper
    # ----------------------------------------------------------------

    def qaoa_solve(
        self,
        graph: List[Tuple[int, int]],
        p_layers: int = 1,
        weights: Optional[List[float]] = None,
        method: str = "COBYLA",
        maxiter: int = 1000,
    ) -> Dict[str, Any]:
        """Solve MaxCut via QAOA.

        Parameters
        ----------
        graph : edge list
        p_layers : int
        weights : list of float, optional
        method : str
        maxiter : int

        Returns
        -------
        dict
            ``{'energy', 'samples', 'best_bitstring', 'best_cut', 'qaoa_result'}``
        """
        qaoa = QAOA(self.n_nodes)
        result = qaoa.optimize(graph, p_layers, method, weights, maxiter)

        samples = QAOA.sample_solution(result["state"], n_samples=2048)
        best_bs = max(samples, key=samples.get)  # type: ignore[arg-type]
        best_cut = self.cut_value(best_bs, graph, weights)

        return {
            "energy": result["energy"],
            "samples": samples,
            "best_bitstring": best_bs,
            "best_cut": best_cut,
            "qaoa_result": result,
        }

    # ----------------------------------------------------------------
    # Visualisation
    # ----------------------------------------------------------------

    @staticmethod
    def visualize_graph(
        graph: List[Tuple[int, int]],
        n_nodes: int,
        partition: Optional[str] = None,
        title: str = "Graph",
        ax: Any = None,
    ) -> Any:
        """Plot the graph using matplotlib (spring layout).

        Parameters
        ----------
        graph : edge list
        n_nodes : int
        partition : str, optional
            Bitstring encoding the partition (colours nodes).
        title : str
        ax : matplotlib Axes, optional

        Returns
        -------
        matplotlib Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6))

        # Simple circular layout
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}

        # Draw edges
        for i, j in graph:
            xi, yi = pos[i]
            xj, yj = pos[j]
            cut_edge = (
                partition is not None and partition[i] != partition[j]
            )
            color = "red" if cut_edge else "gray"
            lw = 2.5 if cut_edge else 1.0
            ax.plot([xi, xj], [yi, yj], color=color, linewidth=lw, zorder=1)

        # Draw nodes
        for node in range(n_nodes):
            x, y = pos[node]
            if partition is not None:
                c = "#4C72B0" if partition[node] == "0" else "#DD8452"
            else:
                c = "#4C72B0"
            ax.scatter(x, y, s=500, c=c, edgecolors="black", linewidths=1.5, zorder=2)
            ax.text(x, y, str(node), ha="center", va="center",
                    fontsize=12, fontweight="bold", color="white", zorder=3)

        ax.set_title(title, fontsize=14)
        ax.set_aspect("equal")
        ax.axis("off")
        return ax

    def __repr__(self) -> str:
        return f"MaxCutSolver(n_nodes={self.n_nodes})"
