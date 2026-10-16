from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_banner import (  # noqa: E402
    ACCENT,
    BACKGROUND,
    CAMERA_AZIM,
    SOFT,
    WHITE,
    kicked_top_frames,
    shaded_colors,
)


def render(out: str, frame: int = 12) -> str:
    times, states, top = kicked_top_frames(n_kicks=5, steps_per_phase=4)
    theta = np.linspace(0, np.pi, 160)
    phi = np.linspace(0, 2 * np.pi, 320)
    T, P = np.meshgrid(theta, phi, indexing="ij")
    X, Y, Z = np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)
    Q = top.husimi_q(states[frame], theta, phi)

    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=BACKGROUND)
    fig.text(0.06, 0.66, "Quantum AI Lab", color=WHITE, fontsize=52, fontweight="bold")
    fig.text(0.062, 0.565, "Quantum computing × AI from first principles", color=SOFT, fontsize=19)
    fig.text(0.062, 0.51, "in pure NumPy", color=SOFT, fontsize=19)
    fig.text(0.062, 0.38, "Shor · Grover · BB84 · E91 · QAOA · VQE", color=ACCENT, fontsize=16)
    fig.text(0.062, 0.33, "Quantum ML · Quantum RL · Quantum Chaos", color=ACCENT, fontsize=16)
    fig.text(0.062, 0.16, "100+ physics-checked tests  ·  MIT licensed  ·  qailab tour",
             color=SOFT, fontsize=13)
    fig.text(0.062, 0.10, "github.com/rasidi3112/quantum-ai-lab", color=SOFT, fontsize=13)

    ax = fig.add_axes([0.6, 0.06, 0.38, 0.88], projection="3d", facecolor=BACKGROUND)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    azim, elev = CAMERA_AZIM + 30 * np.sin(2 * np.pi * frame / len(states)), 15
    ax.plot_surface(X, Y, Z, facecolors=shaded_colors(Q, np.stack([X, Y, Z]), azim, elev),
                    rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.view_init(elev=elev, azim=azim)
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-0.72, 0.72)

    path = os.path.join(out, "social_preview.png")
    fig.savefig(path, facecolor=BACKGROUND)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the 1280x640 GitHub social preview")
    parser.add_argument("--out", default="docs/assets")
    parser.add_argument("--frame", type=int, default=12)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = render(args.out, args.frame)
    print(f"wrote {path} ({os.path.getsize(path) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
