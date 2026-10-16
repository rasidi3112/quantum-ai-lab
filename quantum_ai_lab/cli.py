"""Command-line interface for Quantum AI Lab.

Examples
--------
.. code-block:: bash

    qailab tour              # 3-second tour of all eight modules
    qailab list              # list the full demos
    qailab demo crypto       # run one module's demo
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from collections.abc import Callable

import numpy as np

from quantum_ai_lab import __version__

DEMOS = {
    "algorithms": ("quantum_algorithms.examples.factoring_demo", "Shor, Grover, Deutsch–Jozsa, phase estimation"),
    "crypto": ("quantum_crypto.examples.secure_channel_demo", "BB84 vs eavesdropper, E91 Bell test, quantum OTP"),
    "ml": ("quantum_ml.examples.iris_classification", "Quantum kernel classifier and variational classifier on Iris"),
    "chaos": ("quantum_chaos.examples.chaos_visualization", "Kicked top, level statistics, OTOC scrambling"),
    "simulation": ("quantum_simulation.examples.phase_transition_demo", "Ising criticality, Trotter errors, Hubbard model"),
    "optimization": ("quantum_optimization.examples.portfolio_optimization", "QAOA with X and XY mixers, MaxCut"),
    "rl": ("quantum_rl.examples.cartpole_quantum", "Hybrid quantum agent learns CartPole"),
    "vision": ("quantum_vision.examples.bars_and_stripes", "Quanvolution and an end-to-end hybrid classifier"),
}


class Style:
    """Minimal ANSI styling helper that degrades to plain text.

    Parameters
    ----------
    enabled : bool
        Emit ANSI escape codes when ``True``; return text unchanged otherwise.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, text: str, code: str) -> str:
        """Wrap ``text`` in the ANSI SGR ``code`` when styling is enabled."""
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, text: str) -> str:
        """Return ``text`` in bold."""
        return self(text, "1")

    def dim(self, text: str) -> str:
        """Return ``text`` dimmed."""
        return self(text, "2")

    def cyan(self, text: str) -> str:
        """Return ``text`` in cyan."""
        return self(text, "36")

    def magenta(self, text: str) -> str:
        """Return ``text`` in magenta."""
        return self(text, "35")

    def green(self, text: str) -> str:
        """Return ``text`` in green."""
        return self(text, "32")


def _use_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _banner(style: Style) -> str:
    return "\n".join([
        "  " + style.bold(style.magenta("⚛  Quantum AI Lab")) + style.dim(f"  v{__version__}"),
        "  " + style.dim("8 quantum computing modules · pure NumPy/SciPy · no quantum hardware needed"),
        "  " + style.magenta("─" * 76),
    ])


def _tour_algorithms() -> list[str]:
    from quantum_algorithms import GroverSearch, ShorFactoring

    shor = ShorFactoring(seed=1).factor(21)
    grover = GroverSearch().search(10, 777)
    return [
        f"Shor:   21 = {min(shor.factors)} × {max(shor.factors)} "
        f"(period r = {shor.period_found} of a = {shor.a_used} mod 21)",
        f"Grover: found item 777 of 1024 in {grover.n_iterations} oracle calls, "
        f"P = {grover.found_probability:.4f}",
    ]


def _tour_crypto() -> list[str]:
    from quantum_crypto import BB84Protocol, E91Protocol

    _, clean = BB84Protocol(seed=1).run_protocol(1000)
    _, attacked = BB84Protocol(seed=1).run_protocol(1000, eve_present=True)
    _, e91 = E91Protocol(seed=2).run_protocol(3000)
    return [
        f"BB84:   QBER {clean.qber:.1%} → key of {clean.final_key_length} bits; "
        f"with Eve QBER {attacked.qber:.1%} → aborted",
        f"E91:    CHSH |S| = {abs(e91.chsh_value):.3f} > 2, Bell inequality violated "
        f"(ideal 2√2 ≈ 2.828)",
    ]


def _tour_ml() -> list[str]:
    from quantum_ml import QuantumKernel

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.8, 0.3, (12, 2)), rng.normal(2.3, 0.3, (12, 2))])
    y = np.array([-1] * 12 + [1] * 12)
    idx = rng.permutation(24)
    train, test = idx[:16], idx[16:]
    preds = QuantumKernel(n_qubits=2, n_layers=1).classify_svm(X[train], y[train], X[test], alpha=0.1)
    return [f"Kernel: ZZ feature-map quantum kernel classifier, test accuracy {np.mean(preds == y[test]):.0%}"]


def _tour_chaos() -> list[str]:
    from quantum_chaos import LevelSpacing

    rng = np.random.default_rng(0)
    A = rng.normal(size=(400, 400))
    goe = LevelSpacing.ratio_statistic(np.linalg.eigvalsh((A + A.T) / 2))
    poisson = LevelSpacing.ratio_statistic(rng.uniform(0, 1, 400))
    return [f"Levels: ⟨r⟩ = {goe['r_mean']:.3f} for GOE (chaotic, theory 0.531), "
            f"{poisson['r_mean']:.3f} for Poisson (theory 0.386)"]


def _tour_simulation() -> list[str]:
    from quantum_simulation import IsingModel

    n, h = 8, 1.0
    E0, _ = IsingModel(n, J=1.0, h=h).get_ground_state()
    k = np.pi * (2 * np.arange(n) + 1) / n
    exact = -np.sum(np.sqrt(1 + h ** 2 - 2 * h * np.cos(k)))
    return [f"Ising:  critical chain E₀ = {E0:.10f}, exact free-fermion "
            f"solution agrees to {abs(E0 - exact):.0e}"]


def _tour_optimization() -> list[str]:
    from quantum_optimization import MaxCutSolver

    graph = MaxCutSolver.random_graph(8, 0.5, seed=1)
    result = MaxCutSolver(8).qaoa_solve(graph, p_layers=2, seed=0)
    optimum = MaxCutSolver.brute_force_solve(graph, 8)["max_cut"]
    return [f"QAOA:   MaxCut on 8 nodes → cut {result['best_cut']:.0f}/{optimum:.0f} "
            f"(bitstring {result['best_bitstring']})"]


def _tour_rl() -> list[str]:
    from quantum_rl import CartPoleEnv, HybridAgent

    env = CartPoleEnv(seed=1)
    agent = HybridAgent(seed=1, input_scale=env.obs_scale)
    result = agent.train(env, episodes=40, lr_quantum=0.1, lr_classical=0.1, verbose=False)
    first, last = np.mean(result.episode_rewards[:10]), np.mean(result.episode_rewards[-10:])
    return [f"Agent:  16-parameter quantum policy, CartPole reward {first:.0f} → {last:.0f} "
            f"in 40 episodes"]


def _tour_vision() -> list[str]:
    from quantum_vision import QuantumEncoder, Quanvolution

    image = np.random.default_rng(0).uniform(0, 1, (8, 8))
    info = QuantumEncoder.encoding_info(image)
    quanv = Quanvolution(n_qubits=4, n_layers=1, seed=0)
    fmap = quanv.quanvolve(image, quanv.random_quantum_filter(seed=0))
    return [f"Encode: 8×8 image in {info['amplitude_encoding']['qubits']} qubits (amplitude), "
            f"{info['FRQI']['qubits']} (FRQI); quanvolution → {fmap.shape}"]


TOUR: list[tuple[str, str, Callable[[], list[str]]]] = [
    ("Algorithms", "quantum_algorithms", _tour_algorithms),
    ("Cryptography", "quantum_crypto", _tour_crypto),
    ("Machine learning", "quantum_ml", _tour_ml),
    ("Chaos", "quantum_chaos", _tour_chaos),
    ("Simulation", "quantum_simulation", _tour_simulation),
    ("Optimization", "quantum_optimization", _tour_optimization),
    ("Reinforcement learning", "quantum_rl", _tour_rl),
    ("Vision", "quantum_vision", _tour_vision),
]


def run_tour(style: Style) -> int:
    """Run a one-line showcase of every module and print the results.

    Parameters
    ----------
    style : Style
        Output styling.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    print(_banner(style))
    print()
    start = time.perf_counter()
    for i, (title, package, fn) in enumerate(TOUR, 1):
        t0 = time.perf_counter()
        lines = fn()
        elapsed = time.perf_counter() - t0
        print(f"  {style.cyan(f'[{i}/{len(TOUR)}]')} {style.bold(title)} "
              f"{style.dim(f'· {package} · {elapsed:.2f}s')}")
        for line in lines:
            print(f"        {line}")
        print()
    total = time.perf_counter() - start
    print(style.green(f"  ✓ Tour finished in {total:.1f}s on a classical CPU.") +
          style.dim("  Try `qailab demo <module>` for the full demos."))
    return 0


def run_demo(name: str, extra: list[str]) -> int:
    """Run the full demo of one module.

    Parameters
    ----------
    name : str
        Demo key from :data:`DEMOS`, e.g. ``"crypto"``.
    extra : list of str
        Additional command-line arguments forwarded to the demo.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    module_name, _ = DEMOS[name]
    module = importlib.import_module(module_name)
    sys.argv = [module_name, *extra]
    module.main()
    return 0


def list_demos(style: Style) -> int:
    """Print the available demos.

    Parameters
    ----------
    style : Style
        Output styling.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    print(_banner(style))
    print("\n  Demos (run with `qailab demo <name>`):\n")
    for name, (_, description) in DEMOS.items():
        print(f"    {style.cyan(f'{name:13s}')} {description}")
    print(f"\n  Or run {style.bold('qailab tour')} for a quick tour of everything.\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a ``qailab`` subcommand.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse instead of ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="qailab",
        description="Quantum AI Lab — quantum algorithms, cryptography, ML and more in pure NumPy.",
    )
    parser.add_argument("--version", action="version", version=f"quantum-ai-lab {__version__}")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("tour", help="Run a fast tour of all eight modules")
    sub.add_parser("list", help="List available demos")
    demo = sub.add_parser("demo", help="Run a full module demo")
    demo.add_argument("name", choices=sorted(DEMOS))

    args, extra = parser.parse_known_args(argv)
    style = Style(_use_color() and not args.no_color)

    if args.command == "tour":
        return run_tour(style)
    if args.command == "demo":
        return run_demo(args.name, extra)
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    return list_demos(style)


if __name__ == "__main__":
    raise SystemExit(main())
