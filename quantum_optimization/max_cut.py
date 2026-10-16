"""MaxCut: split a graph's nodes in two to cut as many edges as possible.

Provides random graphs, exact brute-force solutions for benchmarking, a QAOA
solver and a simple plot of the resulting partition.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quantum_optimization.qaoa import QAOA


class MaxCutSolver:
    """MaxCut toolkit for graphs with ``n_nodes`` nodes.

    Parameters
    ----------
    n_nodes : int
        Number of graph nodes (one qubit each).
    """

    def __init__(self, n_nodes: int) -> None:
        self.n_nodes = n_nodes


    @staticmethod
    def random_graph(
        n_nodes: int,
        edge_prob: float = 0.5,
        seed: int | None = None,
    ) -> list[tuple[int, int]]:
        """Erdős–Rényi random graph.

        Parameters
        ----------
        n_nodes : int
            Number of nodes.
        edge_prob : float, default 0.5
            Probability of each edge.
        seed : int, optional
            Random seed.

        Returns
        -------
        list of tuple of int
            Edges ``(i, j)`` with ``i < j``.
        """
        rng = np.random.default_rng(seed)
        edges: list[tuple[int, int]] = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < edge_prob:
                    edges.append((i, j))
        return edges


    @staticmethod
    def graph_to_hamiltonian(
        graph: list[tuple[int, int]],
        n_nodes: int,
        weights: list[float] | None = None,
    ) -> NDArray[np.complex128]:
        """MaxCut cost Hamiltonian; see :meth:`QAOA.cost_hamiltonian`."""
        return QAOA.cost_hamiltonian(graph, n_nodes, weights)


    @staticmethod
    def cut_value(
        bitstring: str,
        graph: list[tuple[int, int]],
        weights: list[float] | None = None,
    ) -> float:
        """Total weight of edges cut by a partition.

        Parameters
        ----------
        bitstring : str
            Side (``"0"`` or ``"1"``) of every node.
        graph : list of tuple of int
            Edges.
        weights : list of float, optional
            Edge weights.

        Returns
        -------
        float
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
        graph: list[tuple[int, int]],
        n_nodes: int,
        weights: list[float] | None = None,
    ) -> dict[str, Any]:
        """Exact MaxCut by enumerating all ``2^n`` partitions.

        Parameters
        ----------
        graph : list of tuple of int
            Edges.
        n_nodes : int
            Number of nodes.
        weights : list of float, optional
            Edge weights.

        Returns
        -------
        dict
            ``max_cut``, an optimal ``bitstring`` and ``all_cuts`` per bitstring.
        """
        best_cut = -1.0
        best_bs = ""
        all_cuts: dict[str, float] = {}
        for idx in range(2 ** n_nodes):
            bs = format(idx, f"0{n_nodes}b")
            val = MaxCutSolver.cut_value(bs, graph, weights)
            all_cuts[bs] = val
            if val > best_cut:
                best_cut = val
                best_bs = bs
        return {"max_cut": best_cut, "bitstring": best_bs, "all_cuts": all_cuts}


    def qaoa_solve(
        self,
        graph: list[tuple[int, int]],
        p_layers: int = 1,
        weights: list[float] | None = None,
        method: str = "COBYLA",
        maxiter: int = 1000,
        n_restarts: int = 3,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Solve MaxCut with QAOA and sampling.

        Parameters
        ----------
        graph : list of tuple of int
            Edges.
        p_layers : int, default 1
            Circuit depth.
        weights : list of float, optional
            Edge weights.
        method : str, default "COBYLA"
            Optimizer.
        maxiter : int, default 1000
            Optimizer iteration limit.
        n_restarts : int, default 3
            Random restarts of the angle optimization.
        seed : int, optional
            Seed for the initial angles and the sampling.

        Returns
        -------
        dict
            ``energy`` (expected cut), ``samples``, the best sampled
            ``best_bitstring`` and ``best_cut``, the ``approximation_ratio``
            ``<C> / C_max`` (computed against brute force) and the raw
            ``qaoa_result``.
        """
        qaoa = QAOA(self.n_nodes, seed=seed)
        result = qaoa.optimize(graph, p_layers, method, weights, maxiter,
                               n_restarts=n_restarts)

        samples = QAOA.sample_solution(result["state"], n_samples=2048, seed=seed)
        best_bs = max(samples, key=lambda bs: self.cut_value(bs, graph, weights))
        best_cut = self.cut_value(best_bs, graph, weights)
        optimum = self.brute_force_solve(graph, self.n_nodes, weights)["max_cut"]

        return {
            "energy": result["energy"],
            "samples": samples,
            "best_bitstring": best_bs,
            "best_cut": best_cut,
            "approximation_ratio": result["energy"] / optimum if optimum > 0 else 1.0,
            "qaoa_result": result,
        }

    @staticmethod
    def visualize_graph(
        graph: list[tuple[int, int]],
        n_nodes: int,
        partition: str | None = None,
        title: str = "Graph",
        ax: Any = None,
    ) -> Any:
        """Draw the graph on a circle, highlighting cut edges.

        Parameters
        ----------
        graph : list of tuple of int
            Edges.
        n_nodes : int
            Number of nodes.
        partition : str, optional
            Bitstring partition to color nodes and cut edges.
        title : str, default "Graph"
            Axes title.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created when omitted.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6))

        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}

        for i, j in graph:
            xi, yi = pos[i]
            xj, yj = pos[j]
            cut_edge = (
                partition is not None and partition[i] != partition[j]
            )
            color = "red" if cut_edge else "gray"
            lw = 2.5 if cut_edge else 1.0
            ax.plot([xi, xj], [yi, yj], color=color, linewidth=lw, zorder=1)

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
