"""Render a scene's true state as a top-down camera image."""
import random

from PIL import Image, ImageDraw

W, H = 640, 480
COLORS = {"red": "#d62828", "blue": "#1d5fd1", "green": "#2a9d4a", "yellow": "#f2c80f",
          "purple": "#8338ec", "white": "#f5f5f5", "black": "#1f1f1f"}
SLOTS = [(30, 30), (235, 30), (440, 30)]
SIZE = (170, 150)


def _draw_object(d, name, x, y, r=22):
    color_name, shape = name.split("_")
    fill = COLORS[color_name]
    if shape == "block":
        d.rectangle([x - r, y - r, x + r, y + r], fill=fill, outline="#222", width=2)
    elif shape == "ball":
        d.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline="#222", width=2)
    else:
        d.polygon([(x - r, y - r), (x + r, y - r), (x + r - 8, y + r), (x - r + 8, y + r)],
                  fill=fill, outline="#222")


def render(scene, path=None):
    state = scene["state"]
    rng = random.Random(scene["render_seed"])
    image = Image.new("RGB", (W, H), "#c9a877")
    d = ImageDraw.Draw(image)

    names = list(state["receptacles"])
    slots = rng.sample(SLOTS, len(names))
    for name, (x, y) in zip(names, slots):
        r = state["receptacles"][name]
        w, h = SIZE
        if r["kind"] == "surface":
            d.rounded_rectangle([x, y, x + w, y + h], 14, fill="#9aa5b1", outline="#333", width=3)
        elif r["open"]:
            d.rectangle([x, y, x + w, y + h], fill="#3b3b3b", outline="#111", width=5)
            d.rectangle([x + 10, y + 10, x + w - 10, y + h - 10], fill="#7a6a58")
        else:
            d.rectangle([x, y, x + w, y + h], fill="#5b4632", outline="#111", width=5)
            for offset in range(10, w, 20):
                d.line([x + offset, y, x + offset, y + h], fill="#3f2f20", width=3)
            d.rectangle([x + w // 2 - 22, y + h // 2 - 8, x + w // 2 + 22, y + h // 2 + 8], fill="#c0c0c0")
        d.text((x + 6, y + h + 4), name.upper(), fill="#111")
        inside = [o for o, v in state["objects"].items() if v["on"] == name]
        for i, obj in enumerate(inside):
            _draw_object(d, obj, x + 45 + i * 45, y + 45 + (i % 2) * 55)

    grid = [(70 + c * 125, 280 + rw * 90) for rw in range(2) for c in range(5)]
    rng.shuffle(grid)
    on_table = [o for o, v in state["objects"].items() if v["on"] == "table"]
    for obj, (x, y) in zip(on_table, grid):
        _draw_object(d, obj, x + rng.randint(-14, 14), y + rng.randint(-10, 10))

    if path:
        image.save(path)
    return image
