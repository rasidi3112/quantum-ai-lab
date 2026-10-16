from __future__ import annotations

import argparse
import io
import os
from contextlib import redirect_stdout

import matplotlib
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from quantum_ai_lab.cli import main as qailab

FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
BACKGROUND, CHROME = "#0d1117", "#161b22"
COLORS = {
    "text": "#e6edf3", "dim": "#7d8590", "cyan": "#39c5cf",
    "magenta": "#d2a8ff", "green": "#3fb950", "prompt": "#58a6ff",
}
FONT_SIZE, LINE_HEIGHT, PAD_X, PAD_Y, CHROME_H = 15, 21, 22, 18, 34


class Fonts:
    def __init__(self) -> None:
        mono = os.path.join(FONT_DIR, "DejaVuSansMono.ttf")
        self.regular = ImageFont.truetype(mono, FONT_SIZE)
        self.bold = ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSansMono-Bold.ttf"), FONT_SIZE)
        self.fallback = ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSans.ttf"), FONT_SIZE)
        self.cmap = TTFont(mono).getBestCmap()
        self.cell = self.regular.getlength("M")

    def for_char(self, ch: str, bold: bool) -> ImageFont.FreeTypeFont:
        if ord(ch) not in self.cmap:
            return self.fallback
        return self.bold if bold else self.regular


def styled(line: str) -> list[tuple[str, str, bool]]:
    stripped = line.strip()
    if stripped.startswith("$"):
        return [("$ ", "prompt", True), (stripped[2:], "text", False)]
    if stripped.startswith("Quantum AI Lab"):
        head, _, version = line.partition("  v")
        return [(head, "magenta", True), (f"  v{version}", "dim", False)]
    if stripped.startswith("─") or stripped.startswith("8 quantum"):
        return [(line, "magenta" if stripped.startswith("─") else "dim", False)]
    if stripped.startswith("["):
        tag_end = line.index("]") + 1
        title, _, meta = line[tag_end:].partition(" · ")
        return [(line[:tag_end], "cyan", True), (title, "text", True), (f" · {meta}", "dim", False)]
    if stripped.startswith("Tour finished"):
        head, _, tail = line.partition("  Try")
        return [(head, "green", True), (f"  Try{tail}" if tail else "", "dim", False)]
    return [(line, "text", False)]


def draw_frame(lines: list[str], size: tuple[int, int], fonts: Fonts, cursor: bool) -> Image.Image:
    image = Image.new("RGB", size, BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, size[0], CHROME_H], fill=CHROME)
    for k, color in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        cx = 20 + 20 * k
        draw.ellipse([cx - 6, CHROME_H / 2 - 6, cx + 6, CHROME_H / 2 + 6], fill=color)
    draw.text((size[0] / 2, CHROME_H / 2), "qailab tour", fill=COLORS["dim"],
              font=fonts.regular, anchor="mm")

    y = CHROME_H + PAD_Y
    for line in lines:
        x = PAD_X
        for text, color, bold in styled(line):
            for ch in text:
                draw.text((x, y), ch, fill=COLORS[color], font=fonts.for_char(ch, bold))
                x += fonts.cell
        y += LINE_HEIGHT
    if cursor:
        x = PAD_X + fonts.cell * len(lines[-1]) if lines else PAD_X
        draw.rectangle([x, y - LINE_HEIGHT + 3, x + fonts.cell - 1, y - 3], fill=COLORS["text"])
    return image


def tour_lines() -> list[str]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        qailab(["--no-color", "tour"])
    return buffer.getvalue().rstrip("\n").split("\n")


def render(out: str) -> str:
    fonts = Fonts()
    output = tour_lines()
    command = "qailab tour"
    width = int(PAD_X * 2 + fonts.cell * (max(len(line) for line in output + [command]) + 2))
    height = CHROME_H + PAD_Y * 2 + LINE_HEIGHT * (len(output) + 1)
    size = (width, height)

    frames: list[tuple[Image.Image, int]] = [(draw_frame(["$ "], size, fonts, True), 600)]
    for k in range(1, len(command) + 1):
        frames.append((draw_frame([f"$ {command[:k]}"], size, fonts, True), 70))
    frames.append((draw_frame([f"$ {command}"], size, fonts, False), 400))

    shown = [f"$ {command}"]
    for line in output:
        shown.append(line)
        is_block_end = line.strip() == ""
        frames.append((draw_frame(shown, size, fonts, False), 260 if is_block_end else 40))
    frames.append((draw_frame(shown, size, fonts, False), 4500))

    palette = frames[-1][0].quantize(colors=200, method=Image.Quantize.MEDIANCUT)
    images = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame, _ in frames]
    path = os.path.join(out, "tour.gif")
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=[ms for _, ms in frames], loop=0, optimize=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the terminal recording of `qailab tour`")
    parser.add_argument("--out", default="docs/assets")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = render(args.out)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
