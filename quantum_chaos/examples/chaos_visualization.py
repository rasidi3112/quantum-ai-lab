#!/usr/bin/env python3
"""
Quantum Chaos Visualization Demo
=================================

This script runs a complete suite of quantum chaos simulations using:
1. Level spacing statistics (Poisson vs. Wigner-Dyson distributions).
2. Quantum kicked top Husimi Q representation on phase space.
3. Out-of-Time-Order Correlators (OTOC) for Lyapunov exponent estimation.

It saves high-resolution plots illustrating the transition from integrable
to chaotic behavior.
"""

import os
import sys
import argparse
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt

from quantum_chaos.level_spacing import LevelSpacing
from quantum_chaos.kicked_top import KickedTop
from quantum_chaos.lyapunov import LyapunovEstimator


def run_level_spacing_demo(save_dir: str):
    print("\n=== Running Level Spacing Statistics Demo ===")
    ls = LevelSpacing()
    dim = 400

    print("Generating integrable spectrum (Poisson)...")
    np.random.seed(42)
    H_integrable = np.diag(np.random.uniform(0, 100, dim))
    analysis_int = ls.analyze(H_integrable)
    print(f"Poisson - Mean spacing: {analysis_int['mean_spacing']:.4f}")
    print(f"Poisson - Ratio statistic <r_tilde>: {analysis_int['ratio_stat']['r_mean']:.4f} ({analysis_int['ratio_stat']['diagnosis']})")
    print(f"Poisson - Brody parameter q: {analysis_int['brody_q']:.4f} ± {analysis_int['brody_q_err']:.4f}")

    print("Generating chaotic spectrum (GOE)...")
    H_goe = np.random.randn(dim, dim)
    H_goe = (H_goe + H_goe.T) / 2.0
    analysis_chaotic = ls.analyze(H_goe)
    print(f"GOE - Mean spacing: {analysis_chaotic['mean_spacing']:.4f}")
    print(f"GOE - Ratio statistic <r_tilde>: {analysis_chaotic['ratio_stat']['r_mean']:.4f} ({analysis_chaotic['ratio_stat']['diagnosis']})")
    print(f"GOE - Brody parameter q: {analysis_chaotic['brody_q']:.4f} ± {analysis_chaotic['brody_q_err']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    s_vals = np.linspace(0, 4, 200)

    ax = axes[0]
    spacings_int = analysis_int['spacings']
    ax.hist(spacings_int, bins=30, density=True, alpha=0.6, color="#4A90E2", edgecolor="white", label="Observed Spacings")
    ax.plot(s_vals, ls.poisson_distribution(s_vals), 'r--', lw=2.5, label="Poisson (Theory)")
    ax.plot(s_vals, ls.wigner_surmise(s_vals), 'k:', lw=2, label="GOE Wigner (Theory)")
    if not np.isnan(analysis_int['brody_q']):
        ax.plot(s_vals, ls.brody_distribution(s_vals, analysis_int['brody_q']), 'g-', lw=2, 
                label=f"Brody Fit (q={analysis_int['brody_q']:.2f})")
    ax.set_title("Integrable System: Poisson Statistics", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Spacing $s$", fontsize=12)
    ax.set_ylabel("Probability Density $P(s)$", fontsize=12)
    ax.set_xlim(0, 4)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")

    ax = axes[1]
    spacings_chaotic = analysis_chaotic['spacings']
    ax.hist(spacings_chaotic, bins=30, density=True, alpha=0.6, color="#E2844A", edgecolor="white", label="Observed Spacings")
    ax.plot(s_vals, ls.poisson_distribution(s_vals), 'r--', lw=2.5, label="Poisson (Theory)")
    ax.plot(s_vals, ls.wigner_surmise(s_vals), 'k:', lw=2, label="GOE Wigner (Theory)")
    if not np.isnan(analysis_chaotic['brody_q']):
        ax.plot(s_vals, ls.brody_distribution(s_vals, analysis_chaotic['brody_q']), 'g-', lw=2, 
                label=f"Brody Fit (q={analysis_chaotic['brody_q']:.2f})")
    ax.set_title("Chaotic System: GOE Wigner-Dyson Statistics", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Spacing $s$", fontsize=12)
    ax.set_xlim(0, 4)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    output_path = os.path.join(save_dir, "level_spacing.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved level spacing plot to: {output_path}")


def run_kicked_top_demo(save_dir: str):
    print("\n=== Running Quantum Kicked Top Phase Space Demo ===")
    j = 30.0
    n_kicks = 150
    theta_0, phi_0 = np.pi / 2, np.pi / 4

    k_vals = [0.5, 3.0, 6.0]
    k_names = ["Regular (k=0.5)", "Mixed (k=3.0)", "Fully Chaotic (k=6.0)"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, k in enumerate(k_vals):
        print(f"Simulating kicked top for k = {k}...")
        kt = KickedTop(j=j, k=k)
        
        state = kt.coherent_state(theta_0, phi_0)
        
        evolved_state = kt.evolve(state, n_kicks)
        
        print(f"Computing Husimi Q grid for k = {k}...")
        theta_grid, phi_grid, Q = kt.husimi_q_fast(evolved_state, n_theta=80, n_phi=160)
        
        ax = axes[idx]
        im = ax.pcolormesh(phi_grid, theta_grid, Q, shading='auto', cmap='inferno')
        ax.set_title(k_names[idx], fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel(r"Azimuthal angle $\phi$", fontsize=12)
        if idx == 0:
            ax.set_ylabel(r"Polar angle $\theta$", fontsize=12)
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(0, np.pi)
        ax.invert_yaxis()
        fig.colorbar(im, ax=ax, label="Husimi Q Density")

    plt.tight_layout()
    output_path = os.path.join(save_dir, "kicked_top_husimi.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Husimi Q plot to: {output_path}")


def run_otoc_demo(save_dir: str):
    print("\n=== Running OTOC and Lyapunov Exponent Demo ===")
    le = LyapunovEstimator()
    
    da, db = 8, 8
    dim = da * db
    
    np.random.seed(42)
    OA = np.random.randn(da, da)
    OA = (OA + OA.T) / 2.0
    OB = np.random.randn(db, db)
    OB = (OB + OB.T) / 2.0
    
    W = np.kron(OA, np.eye(db))
    V = np.kron(np.eye(da), OB)
    
    W /= np.linalg.norm(W)
    V /= np.linalg.norm(V)
    
    H_chaotic = np.random.randn(dim, dim)
    H_chaotic = (H_chaotic + H_chaotic.T) / 2.0
    
    H_integrable = np.diag(np.random.normal(0, 5.0, dim))
    
    times = np.linspace(0.0, 0.5, 50)
    
    print("Computing OTOC for Chaotic Hamiltonian...")
    otoc_chaotic = le.compute_otoc(H_chaotic, W, V, times)
    
    print("Computing OTOC for Integrable Hamiltonian...")
    otoc_integrable = le.compute_otoc(H_integrable, W, V, times)
    
    fit_res = le.estimate_lyapunov(times, otoc_chaotic, t_start=0.01, t_end=0.18)
    
    print("\nOTOC Fit Results (Chaotic):")
    print(f"Lyapunov exponent lambda: {fit_res['lyapunov_exponent']:.4f}")
    print(f"Fit R^2: {fit_res['r_squared']:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, otoc_chaotic, 'o-', color='#D0021B', lw=2, label='Chaotic (GOE)')
    plt.plot(times, otoc_integrable, 's-', color='#4A90E2', lw=2, label='Integrable (Diagonal)')
    
    if len(fit_res['fit_times']) > 0:
        plt.plot(fit_res['fit_times'], fit_res['fit_values'], 'k--', lw=2.5, 
                 label=rf"Fit: $A e^{{2\lambda t}}$ ($\lambda={fit_res['lyapunov_exponent']:.3f}$)")
        
    plt.title("Out-of-Time-Order Correlator (OTOC) Growth", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Time $t$", fontsize=12)
    plt.ylabel("OTOC $C(t)$", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11, loc="lower right")
    
    info_text = rf"Chaotic $\lambda \approx {fit_res['lyapunov_exponent']:.3f}$" + "\n" + rf"$R^2 = {fit_res['r_squared']:.4f}$"
    plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    output_path = os.path.join(save_dir, "otoc_growth.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved OTOC plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Quantum Chaos Simulation Demo")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save output plots")
    args = parser.parse_args()
    
    if args.save_dir is None:
        default_artifact_dir = "/Users/macbook/.gemini/antigravity-ide/brain/82e6466a-3fc5-4105-8414-356200946be4"
        if os.path.exists(default_artifact_dir):
            save_dir = default_artifact_dir
        else:
            save_dir = "./plots"
    else:
        save_dir = args.save_dir
        
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving all plots to: {os.path.abspath(save_dir)}")
    
    run_level_spacing_demo(save_dir)
    run_kicked_top_demo(save_dir)
    run_otoc_demo(save_dir)
    print("\nAll simulations completed successfully!")


if __name__ == "__main__":
    main()
