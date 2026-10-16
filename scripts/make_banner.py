from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.linalg import expm

from quantum_chaos import KickedTop

BACKGROUND = "#0d1117"
WHITE, SOFT, ACCENT = "#f0f6fc", "#9da7b3", "#58a6ff"
SPHERE_CMAP = LinearSegmentedColormap.from_list(
    "sphere", ["#132a4a", "#1f5fae", "#3987e5", "#9cc8ff", "#ffffff"]
)
CAMERA_AZIM = 90
MODULES = "Shor · Grover · BB84 · E91 · QAOA · VQE · Quantum ML · Quantum RL · Chaos"


def kicked_top_frames(n_kicks: int, steps_per_phase: int) -> tuple[list[float], list[np.ndarray], KickedTop]:
    top = KickedTop(j=20, k=6.0)
    precess = expm(-1j * top.p / steps_per_phase * top.Jy)
    kick = expm(-1j * top.k / (2.0 * top.j) / steps_per_phase * top.Jz @ top.Jz)
    psi = top.coherent_state(np.pi / 2.6, np.deg2rad(CAMERA_AZIM))
    times, states = [0.0], [psi]
    for _kick in range(n_kicks):
        for step_op in (precess, kick):
            for _ in range(steps_per_phase):
                psi = step_op @ psi
                times.append(times[-1] + 0.5 / steps_per_phase)
                states.append(psi)
    return times + times[-2:0:-1], states + states[-2:0:-1], top


def shaded_colors(Q: np.ndarray, normals: np.ndarray, azim: float, elev: float) -> np.ndarray:
    a, e = np.deg2rad(azim + 25), np.deg2rad(elev + 25)
    light = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    diffuse = np.clip(np.tensordot(normals, light, axes=([0], [0])), 0, 1)
    rgba = SPHERE_CMAP((Q / Q.max()) ** 0.85)
    rgba[..., :3] *= (0.6 + 0.4 * diffuse)[..., None]
    return rgba


def render(out: str, n_kicks: int = 5, steps_per_phase: int = 4, frame_ms: int = 75) -> str:
    times, states, top = kicked_top_frames(n_kicks, steps_per_phase)
    n_frames = len(states)
    theta = np.linspace(0, np.pi, 110)
    phi = np.linspace(0, 2 * np.pi, 220)
    T, P = np.meshgrid(theta, phi, indexing="ij")
    X, Y, Z = np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)
    normals = np.stack([X, Y, Z])

    fig = plt.figure(figsize=(12, 4), dpi=100, facecolor=BACKGROUND)
    fig.text(0.045, 0.66, "Quantum AI Lab", color=WHITE, fontsize=46, fontweight="bold")
    fig.text(0.047, 0.53, "Quantum computing × AI, built from first principles in pure NumPy",
             color=SOFT, fontsize=15)
    fig.text(0.047, 0.36, MODULES, color=ACCENT, fontsize=11.5)
    fig.text(0.047, 0.17, "8 modules  ·  100+ physics-checked tests  ·  no quantum hardware needed",
             color=SOFT, fontsize=11)
    ax = fig.add_axes([0.66, 0.0, 0.34, 1.0], projection="3d", facecolor=BACKGROUND)
    caption = fig.text(0.83, 0.05, "", color=SOFT, fontsize=10, ha="center")

    frames = []
    for f, (t, psi) in enumerate(zip(times, states)):
        Q = top.husimi_q(psi, theta, phi)
        ax.clear()
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 1))
        azim, elev = CAMERA_AZIM + 30 * np.sin(2 * np.pi * f / n_frames), 15
        ax.plot_surface(X, Y, Z, facecolors=shaded_colors(Q, normals, azim, elev),
                        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
        ax.view_init(elev=elev, azim=azim)
        lim = 0.72
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        direction = "time-reversed ◀" if f > n_frames // 2 else "chaotic kicks ▶"
        caption.set_text(f"quantum kicked top · {direction} · t = {t:4.1f}")
        fig.canvas.draw()
        frames.append(Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                                       fig.canvas.buffer_rgba()).convert("RGB"))
    plt.close(fig)

    sample = frames[:: max(1, n_frames // 12)]
    atlas = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for k, frame in enumerate(sample):
        atlas.paste(frame, (0, k * frame.height))
    palette = atlas.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    path = os.path.join(out, "banner.gif")
    quantized[0].save(path, save_all=True, append_images=quantized[1:], duration=frame_ms,
                      loop=0, optimize=True, disposal=1)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the animated README banner")
    parser.add_argument("--out", default="docs/assets")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = render(args.out)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
