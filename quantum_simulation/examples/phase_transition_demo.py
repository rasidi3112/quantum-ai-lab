from __future__ import annotations

import argparse
import os

import numpy as np

from quantum_simulation import HubbardModel, IsingModel, TrotterEvolution
from quantum_simulation.ising_model import identity, sigma_x, sigma_z, tensor_product


def run_ising(n_sites: int, save_dir: str | None) -> None:
    print("\n" + "-" * 60)
    print(f"  1. Transverse-field Ising chain (N={n_sites}) — quantum phase transition")
    print("-" * 60)
    h_values = np.linspace(0.0, 2.0, 21)
    correlations, gaps = [], []
    for h in h_values:
        H = IsingModel.build_hamiltonian(n_sites, 1.0, h, periodic=True)
        energies = IsingModel.energy_spectrum(H, n_states=3)
        _, psi0 = IsingModel.ground_state(H)
        correlations.append(IsingModel.correlation_function(psi0, 0, n_sites // 2, n_sites))
        gaps.append(energies[2] - energies[0])

    print("   h/J   ⟨Z₀ Z_N/2⟩   gap Δ   ")
    for h, c, g in zip(h_values[::2], correlations[::2], gaps[::2]):
        bar = "█" * int(round(20 * c))
        print(f"  {h:4.1f}   {c:8.4f}   {g:6.3f}   {bar}")
    print("  Long-range order melts and the gap is smallest near the critical point h = J.")

    if save_dir:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
        ax1.plot(h_values, correlations, "o-", color="#2a78d6", lw=2, ms=5)
        ax1.set_ylabel(r"$\langle Z_0 Z_{N/2}\rangle$")
        ax1.set_title("Long-range order")
        ax2.plot(h_values, gaps, "o-", color="#eb6834", lw=2, ms=5)
        ax2.set_ylabel("Energy gap Δ")
        ax2.set_title("Excitation gap")
        for ax in (ax1, ax2):
            ax.axvline(1.0, color="#8a8984", linestyle=":", lw=1)
            ax.set_xlabel("Transverse field h / J")
            ax.grid(alpha=0.3)
        fig.suptitle(f"Transverse-field Ising chain, N = {n_sites}")
        fig.tight_layout()
        path = os.path.join(save_dir, "ising_phase_transition.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved plot to {path}")


def run_trotter() -> None:
    print("\n" + "-" * 60)
    print("  2. Trotter–Suzuki product formulas — error vs step size")
    print("-" * 60)
    n = 4
    H_zz = sum(
        tensor_product(*[sigma_z if k in (i, (i + 1) % n) else identity for k in range(n)])
        for i in range(n)
    )
    H_x = sum(tensor_product(*[sigma_x if k == i else identity for k in range(n)]) for i in range(n))
    psi0 = np.zeros(2 ** n, dtype=np.complex128)
    psi0[0] = 1.0
    dt_values = [0.2, 0.1, 0.05]
    errors = TrotterEvolution.compare_orders(psi0, [-H_zz, -0.8 * H_x], 1.0, dt_values)
    print("    dt      1st order    2nd order    4th order")
    for k, dt in enumerate(dt_values):
        print(f"  {dt:5.2f}   {errors[1][k]:.3e}    {errors[2][k]:.3e}    {errors[4][k]:.3e}")
    print("  Halving dt shrinks the infidelity by ~4×, ~16× and ~256× respectively.")


def run_hubbard() -> None:
    print("\n" + "-" * 60)
    print("  3. Fermi–Hubbard dimer — interaction suppresses double occupancy")
    print("-" * 60)
    print("    U/t    E₀        ⟨n↑n↓⟩")
    for U in (0.0, 2.0, 4.0, 8.0, 16.0):
        model = HubbardModel(n_sites=2, t_hop=1.0, U=U)
        H = model.hamiltonian
        half_filling = np.array([bin(i).count("1") == 2 for i in range(model.dim)])
        H_sector = H[np.ix_(half_filling, half_filling)]
        energies, vectors = np.linalg.eigh(H_sector)
        psi0 = np.zeros(model.dim, dtype=np.complex128)
        psi0[half_filling] = vectors[:, 0]
        docc = model.double_occupancy(psi0).mean()
        print(f"  {U:5.1f}   {energies[0]:8.4f}   {docc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum many-body simulation demo")
    parser.add_argument("--save-dir", default=None, help="Directory to save plots")
    args = parser.parse_args()
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("  Quantum Simulation — Phase Transitions & Dynamics")
    print("=" * 60)
    run_ising(8, args.save_dir)
    run_trotter()
    run_hubbard()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
