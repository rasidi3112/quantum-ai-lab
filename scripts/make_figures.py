from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quantum_algorithms import GroverSearch
from quantum_chaos import KickedTop
from quantum_crypto import E91Protocol
from quantum_optimization import QAOA, MaxCutSolver
from quantum_rl import CartPoleEnv, HybridAgent
from quantum_simulation import IsingModel

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 11,
    "legend.frameon": False,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def kicked_top(out: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, k, title in zip(axes, (0.5, 3.0, 6.0), ("Regular  k = 0.5", "Mixed  k = 3", "Chaotic  k = 6")):
        top = KickedTop(j=30, k=k)
        state = top.evolve(top.coherent_state(np.pi / 2, np.pi / 4), 150)
        theta, phi, Q = top.husimi_q_fast(state, n_theta=90, n_phi=180)
        ax.pcolormesh(phi, theta, Q, shading="auto", cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel(r"azimuth $\phi$")
        ax.invert_yaxis()
        ax.grid(False)
    axes[0].set_ylabel(r"polar angle $\theta$")
    fig.suptitle("Quantum kicked top — Husimi Q after 150 kicks", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "kicked_top.png"))
    plt.close(fig)


def grover(out: str) -> None:
    n, target = 8, 173
    steps = np.arange(0, 26)
    probs = [GroverSearch().search(n, target, n_iterations=int(k)).probabilities[target]
             if k else 1 / 2 ** n for k in steps]
    best = GroverSearch.optimal_iterations(2 ** n)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(steps, probs, "o-", color=BLUE, lw=2, ms=5)
    ax.axvline(best, color=MUTED, ls=":", lw=1)
    ax.text(0.3, 0.92, f"optimum: {best} oracle calls\nP(target) = {probs[best]:.4f}", color=INK)
    ax.set_xlabel("Grover iterations")
    ax.set_ylabel("P(target)")
    ax.set_title("Grover search over 256 items")
    fig.savefig(os.path.join(out, "grover.png"))
    plt.close(fig)


def bell(out: str) -> None:
    e91 = E91Protocol(seed=0)
    pair = E91Protocol.generate_entangled_pairs(1)[0]
    angles = np.linspace(0, 2 * np.pi, 25)
    measured = []
    for theta in angles:
        outcomes = np.array([e91.measure_entangled(pair, 0.0, theta) for _ in range(400)])
        measured.append(np.mean((1 - 2 * outcomes[:, 0]) * (1 - 2 * outcomes[:, 1])))
    fine = np.linspace(0, 2 * np.pi, 300)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(fine, -np.cos(fine), color=BLUE, lw=2, label=r"quantum theory  $-\cos\theta$")
    ax.plot(angles, measured, "o", color=ORANGE, ms=7, mec="#fcfcfb", mew=2,
            label="simulated singlet, 400 shots")
    ax.set_xticks(np.arange(0, 2.01 * np.pi, np.pi / 2), ["0", "π/2", "π", "3π/2", "2π"])
    ax.set_xlabel("Analyzer angle difference θ")
    ax.set_ylabel("Correlation E(θ)")
    ax.set_title("Entangled-pair correlations behind E91")
    ax.set_ylim(-1.15, 1.6)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(out, "bell_correlations.png"))
    plt.close(fig)


def qaoa_landscape(out: str) -> None:
    graph = MaxCutSolver.random_graph(8, 0.5, seed=1)
    gammas = np.linspace(0, np.pi, 80)
    betas = np.linspace(0, np.pi / 2, 60)
    landscape = QAOA.energy_landscape(graph, 8, gammas, betas)
    optimum = MaxCutSolver.brute_force_solve(graph, 8)["max_cut"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    mesh = ax.pcolormesh(betas, gammas, landscape / optimum, shading="auto", cmap="Blues")
    fig.colorbar(mesh, ax=ax, label=r"approximation ratio $\langle C\rangle / C_{max}$")
    ax.set_xlabel(r"mixer angle $\beta$")
    ax.set_ylabel(r"cost angle $\gamma$")
    ax.set_title("QAOA (p = 1) landscape, MaxCut on 8 nodes")
    ax.grid(False)
    fig.savefig(os.path.join(out, "qaoa_landscape.png"))
    plt.close(fig)


def ising(out: str) -> None:
    n = 10
    h_values = np.linspace(0, 2, 41)
    corr = []
    for h in h_values:
        _, psi = IsingModel(n, 1.0, h).get_ground_state()
        corr.append(IsingModel.correlation_function(psi, 0, n // 2, n))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(h_values, corr, "o-", color=BLUE, lw=2, ms=4)
    ax.axvline(1.0, color=MUTED, ls=":", lw=1)
    ax.text(1.03, 0.9, "critical point h = J", color=MUTED)
    ax.set_xlabel("Transverse field h / J")
    ax.set_ylabel(r"$\langle Z_0 Z_{N/2}\rangle$")
    ax.set_title(f"Quantum phase transition, Ising chain N = {n}")
    fig.savefig(os.path.join(out, "ising_transition.png"))
    plt.close(fig)


def rl_curve(out: str) -> None:
    env = CartPoleEnv(seed=1)
    agent = HybridAgent(seed=1, input_scale=env.obs_scale)
    rewards = np.array(agent.train(env, episodes=100, lr_quantum=0.1, lr_classical=0.1,
                                   verbose=False).episode_rewards)
    smooth = np.convolve(rewards, np.ones(10) / 10, mode="valid")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(np.arange(1, len(rewards) + 1), rewards, color=BLUE, alpha=0.35, lw=1, label="episode")
    ax.plot(np.arange(10, len(rewards) + 1), smooth, color=BLUE, lw=2.5, label="10-episode mean")
    ax.axhline(200, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps balanced")
    ax.set_title("4-qubit policy learns CartPole")
    ax.set_ylim(0, 245)
    ax.text(100, 204, "max 200", color=MUTED, ha="right", va="bottom")
    ax.legend(loc="upper left", ncol=2)
    fig.savefig(os.path.join(out, "rl_cartpole.png"))
    plt.close(fig)


FIGURES = {
    "kicked_top": kicked_top,
    "grover": grover,
    "bell": bell,
    "qaoa": qaoa_landscape,
    "ising": ising,
    "rl": rl_curve,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the README figures")
    parser.add_argument("--out", default="docs/assets")
    parser.add_argument("only", nargs="*", help=f"Subset of figures: {', '.join(FIGURES)}")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in args.only or FIGURES:
        print(f"rendering {name}...")
        FIGURES[name](args.out)


if __name__ == "__main__":
    main()
