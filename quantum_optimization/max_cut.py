from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from quantum_optimization.qaoa import QAOA


class MaxCutSolver:
    def __init__(self, n_nodes: int) -> None:
        self.n_nodes = n_nodes


    @staticmethod
    def random_graph(
        n_nodes: int,
        edge_prob: float = 0.5,
        seed: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        rng = np.random.default_rng(seed)
        edges: List[Tuple[int, int]] = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < edge_prob:
                    edges.append((i, j))
        return edges


    @staticmethod
    def graph_to_hamiltonian(
        graph: List[Tuple[int, int]],
        n_nodes: int,
        weights: Optional[List[float]] = None,
    ) -> NDArray[np.complex128]:
        return QAOA.cost_hamiltonian(graph, n_nodes, weights)


    @staticmethod
    def cut_value(
        bitstring: str,
        graph: List[Tuple[int, int]],
        weights: Optional[List[float]] = None,
    ) -> float:
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


    def qaoa_solve(
        self,
        graph: List[Tuple[int, int]],
        p_layers: int = 1,
        weights: Optional[List[float]] = None,
        method: str = "COBYLA",
        maxiter: int = 1000,
    ) -> Dict[str, Any]:
        qaoa = QAOA(self.n_nodes)
        result = qaoa.optimize(graph, p_layers, method, weights, maxiter)

        samples = QAOA.sample_solution(result["state"], n_samples=2048)
        best_bs = max(samples, key=samples.get)
        best_cut = self.cut_value(best_bs, graph, weights)

        return {
            "energy": result["energy"],
            "samples": samples,
            "best_bitstring": best_bs,
            "best_cut": best_cut,
            "qaoa_result": result,
        }


    @staticmethod
    def visualize_graph(
        graph: List[Tuple[int, int]],
        n_nodes: int,
        partition: Optional[str] = None,
        title: str = "Graph",
        ax: Any = None,
    ) -> Any:
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
