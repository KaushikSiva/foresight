"""Generate meshes (OBJ with UVs) and textures for every object in the warehouse."""
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

GEN = Path(__file__).resolve().parent / "assets" / "gen"
FONT_PATHS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"]


def font(size):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _noise(size, scales=((4, 0.5), (16, 0.3), (64, 0.2)), seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size))
    for cells, weight in scales:
        coarse = Image.fromarray((rng.random((cells, cells)) * 255).astype("uint8"))
        out += np.asarray(coarse.resize((size, size), Image.BICUBIC), dtype=float) / 255 * weight
    return out


def _tint(noise, low, high):
    low, high = np.array(low, float), np.array(high, float)
    return Image.fromarray((low + (high - low) * np.clip(noise, 0, 1)[..., None]).astype("uint8"))


def _barcode(d, x, y, w, h, seed):
    rng = np.random.default_rng(seed)
    cursor = x
    while cursor < x + w:
        bar = int(rng.integers(1, 5))
        d.rectangle([cursor, y, cursor + bar, y + h], fill="#111")
        cursor += bar + int(rng.integers(1, 5))


def cardboard(seed=1, size=1024, label=True):
    base = _noise(size, seed=seed)
    grain = np.tile(np.random.default_rng(seed).random((size, 1)), (1, size))
    img = _tint(base * 0.75 + grain * 0.25, (150, 106, 66), (205, 160, 112))
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, size // 2 - 42, size, size // 2 + 42], fill=(196, 170, 120, 170))
    d.line([0, size // 2 - 42, size, size // 2 - 42], fill=(255, 245, 210, 110), width=2)
    if label:
        d.rectangle([90, 120, 430, 340], fill=(245, 245, 240, 255), outline=(40, 40, 40, 255), width=3)
        d.text((110, 135), "SHIP TO: DOCK 14", font=font(28), fill="#111")
        _barcode(d, 110, 190, 300, 90, seed)
        d.text((110, 290), f"PKG {seed:05d}", font=font(26), fill="#111")
    return img


def concrete(size=2048):
    n = _noise(size, ((6, 0.45), (32, 0.3), (180, 0.25)), seed=4)
    img = _tint(n, (92, 92, 90), (158, 156, 150))
    d = ImageDraw.Draw(img, "RGBA")
    rng = np.random.default_rng(5)
    for _ in range(26):
        x, y = rng.integers(0, size, 2)
        points = [(int(x), int(y))]
        for _ in range(9):
            x, y = x + rng.integers(-60, 60), y + rng.integers(-60, 60)
            points.append((int(x), int(y)))
        d.line(points, fill=(40, 40, 40, 70), width=1)
    for _ in range(40):
        x, y, r = rng.integers(0, size), rng.integers(0, size), rng.integers(20, 120)
        d.ellipse([x - r, y - r // 2, x + r, y + r // 2], fill=(30, 30, 30, 14))
    return img.filter(ImageFilter.GaussianBlur(0.5))


def bench_wood(size=1024):
    rng = np.random.default_rng(8)
    y = np.linspace(0, 1, size)[:, None]
    x = np.linspace(0, 1, size)[None, :]
    grain = np.sin((y * 90 + 6 * np.sin(x * 7 + y * 11)) * 3) * 0.5 + 0.5
    n = grain * 0.5 + _noise(size, seed=9) * 0.5 + rng.random((size, size)) * 0.05
    return _tint(n, (124, 88, 52), (196, 156, 106))


def pallet_wood(size=512):
    x = np.linspace(0, 1, size)[None, :]
    y = np.linspace(0, 1, size)[:, None]
    grain = np.sin((x * 60 + 3 * np.sin(y * 6)) * 3) * 0.5 + 0.5
    return _tint(grain * 0.5 + _noise(size, seed=11) * 0.5, (120, 94, 62), (182, 150, 108))


def _can_label(top, bottom, title, sub, seed, size=1024):
    img = Image.new("RGB", (size, size // 2), top)
    d = ImageDraw.Draw(img)
    d.rectangle([0, size // 4, size, size // 2], fill=bottom)
    for offset in (0, size // 2):
        d.text((offset + 60, 60), title, font=font(74), fill="white" if sum(_rgb(top)) < 420 else "#111")
        d.text((offset + 60, size // 4 + 30), sub, font=font(38), fill="white" if sum(_rgb(bottom)) < 420 else "#111")
        _barcode(d, offset + 330, size // 4 + 110, 130, 90, seed)
    return img


def _rgb(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def _box_art(base, title, sub, seed, accent="#ffffff", size=1024):
    img = Image.new("RGB", (size, size), base)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, size, 170], fill=accent)
    ink = "#111" if sum(_rgb(accent)) > 420 else "white"
    d.text((50, 45), title, font=font(84), fill=ink)
    d.text((50, 240), sub, font=font(52), fill="white" if sum(_rgb(base)) < 420 else "#111")
    d.ellipse([310, 410, 720, 820], outline=(255, 255, 255, 170), width=20)
    _barcode(d, 620, 860, 340, 120, seed)
    return img


def sku_textures():
    return {
        "ceramic_mug": _speckle((243, 240, 232), seed=21),
        "glass_jar": _can_label("#f3e7c8", "#b5651d", "ACACIA HONEY", "RAW 500 g", 22),
        "soup_can": _can_label("#c8201f", "#f7f3ea", "TOMATO SOUP", "CONDENSED 400 g", 23),
        "paint_can": _can_label("#d9dde0", "#1f4f8f", "PRO PAINT 1 L", "SATIN BLUE LATEX", 24),
        "detergent_jug": _box_art("#1e6fd9", "AQUAWASH", "LAUNDRY LIQUID 1.8 L", 25, "#f7c600"),
        "hardcover_book": _box_art("#233a2b", "THE ATLAS", "HARDCOVER EDITION", 26, "#c9a24b"),
        "headphones_box": _box_art("#17181c", "SONIQ H9", "WIRELESS HEADPHONES", 27, "#e6e7ea"),
        "phone_box": _box_art("#f4f4f4", "NOVA 12", "SMARTPHONE 256 GB", 28, "#3044c9"),
        "bearing_box": _box_art("#e4581d", "STEEL BEARINGS", "6204-2RS BOX OF 10", 29, "#222222"),
        "tissue_box": _box_art("#8fd0c7", "SOFTLEAF", "FACIAL TISSUES 2-PLY", 30, "#ffffff"),
    }


def _speckle(rgb, seed, size=512):
    n = np.random.default_rng(seed).random((size, size))
    arr = np.ones((size, size, 3)) * rgb
    arr[n > 0.992] *= 0.65
    return Image.fromarray(arr.astype("uint8"))


def tote_tag(name, size=256):
    img = Image.new("RGB", (size, size), "#f6f4ec")
    d = ImageDraw.Draw(img)
    d.rectangle([6, 6, size - 6, size - 6], outline="#111", width=8)
    d.text((size // 2, size // 2), name, font=font(130), fill="#111", anchor="mm")
    return img


def slot_mark(text, size=256):
    img = Image.new("RGB", (size, size), "#d7b892")
    ImageDraw.Draw(img).text((size // 2, size // 2), text, font=font(120), fill="#5a3d20", anchor="mm")
    return img


# ---- meshes ----

def _write_obj(path, vertices, uvs, normals, faces):
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for t in uvs:
            f.write(f"vt {t[0]:.5f} {t[1]:.5f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.5f} {n[1]:.5f} {n[2]:.5f}\n")
        for tri in faces:
            f.write("f " + " ".join(f"{i + 1}/{i + 1}/{i + 1}" for i in tri) + "\n")


def lathe(path, profile, segments=64, extra=None):
    """Revolve (r, z) points about z. UV u wraps around, v runs along the profile."""
    lengths = [0.0]
    for a, b in zip(profile, profile[1:]):
        lengths.append(lengths[-1] + math.dist(a, b))
    vertices, uvs, normals, faces = [], [], [], []
    rows = len(profile)
    for i, (r, z) in enumerate(profile):
        a, b = profile[max(0, i - 1)], profile[min(rows - 1, i + 1)]
        nr, nz = b[1] - a[1], -(b[0] - a[0])
        norm = math.hypot(nr, nz) or 1.0
        for s in range(segments + 1):
            t = 2 * math.pi * s / segments
            vertices.append((r * math.cos(t), r * math.sin(t), z))
            uvs.append((s / segments, lengths[i] / lengths[-1]))
            normals.append((nr / norm * math.cos(t), nr / norm * math.sin(t), nz / norm))
    width = segments + 1
    for i in range(rows - 1):
        for s in range(segments):
            p = i * width + s
            faces += [(p, p + 1, p + width), (p + 1, p + width + 1, p + width)]
    if extra:
        extra(vertices, uvs, normals, faces)
    _write_obj(path, vertices, uvs, normals, faces)


def _handle(radius, z, major=0.024, minor=0.0065, segs=28, tube=12):
    """A mug handle as a half torus on the +x side."""
    def add(vertices, uvs, normals, faces):
        start = len(vertices)
        for i in range(segs + 1):
            a = -math.pi / 2 + math.pi * i / segs
            cx, cz = radius - 0.002 + major * math.cos(a), z + major * 1.15 * math.sin(a)
            for j in range(tube + 1):
                b = 2 * math.pi * j / tube
                nx, ny, nz = math.cos(a) * math.cos(b), math.sin(b), math.sin(a) * math.cos(b)
                vertices.append((cx + minor * nx, minor * ny, cz + minor * nz))
                uvs.append((i / segs, j / tube))
                normals.append((nx, ny, nz))
        for i in range(segs):
            for j in range(tube):
                p = start + i * (tube + 1) + j
                faces += [(p, p + tube + 1, p + 1), (p + 1, p + tube + 1, p + tube + 2)]
    return add


def box_mesh(path, hx, hy, hz):
    """A box with each side face mapped to the whole texture, and bevel-free hard normals."""
    vertices, uvs, normals, faces = [], [], [], []
    sides = [
        ((1, 0, 0), (0, 1, 0), (0, 0, 1), hx, hy, hz), ((-1, 0, 0), (0, -1, 0), (0, 0, 1), hx, hy, hz),
        ((0, 1, 0), (-1, 0, 0), (0, 0, 1), hy, hx, hz), ((0, -1, 0), (1, 0, 0), (0, 0, 1), hy, hx, hz),
        ((0, 0, 1), (0, 1, 0), (-1, 0, 0), hz, hy, hx), ((0, 0, -1), (0, -1, 0), (-1, 0, 0), hz, hy, hx),
    ]
    for n, u, v, depth, half_u, half_v in sides:
        n, u, v = np.array(n), np.array(u), np.array(v)
        start = len(vertices)
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            vertices.append(tuple(n * depth + u * half_u * su + v * half_v * sv))
            uvs.append(((su + 1) / 2, (sv + 1) / 2))
            normals.append(tuple(n))
        faces += [(start, start + 1, start + 2), (start, start + 2, start + 3)]
    _write_obj(path, vertices, uvs, normals, faces)


def mug_profile(r, h):
    # Outer wall, lip, then inside surface down to the floor.
    return [(0, 0), (r * 0.82, 0), (r * 0.94, 0.004), (r, 0.012), (r, h - 0.003), (r - 0.0015, h), (r - 0.005, h),
            (r - 0.0065, h - 0.004), (r - 0.0065, 0.012), (r * 0.6, 0.008), (0, 0.007)]


def jar_profile(r, h):
    return [(0, 0), (r * 0.85, 0), (r, 0.008), (r, h * 0.72), (r * 0.92, h * 0.80), (r * 0.82, h * 0.84),
            (r * 0.86, h * 0.86), (r * 0.86, h - 0.002), (r * 0.8, h), (0, h)]


def can_profile(r, h):
    return [(0, 0.002), (r * 0.9, 0.001), (r, 0.005), (r, h - 0.005), (r * 0.97, h - 0.001), (r * 0.86, h - 0.003), (0, h - 0.004)]


def generate():
    from .catalog import CATALOG, TOTES

    GEN.mkdir(parents=True, exist_ok=True)
    for name, image in sku_textures().items():
        image.save(GEN / f"tex_{name}.png")
    for i in range(1, 5):
        cardboard(i).save(GEN / f"cardboard{i}.png")
    cardboard(9, label=False).save(GEN / "carton_plain.png")
    concrete().save(GEN / "concrete.png")
    bench_wood().save(GEN / "bench.png")
    pallet_wood().save(GEN / "pallet.png")
    for name in TOTES:
        tote_tag(name).save(GEN / f"tag_{name}.png")
    for text in ("S", "L"):
        tote_tag(text).save(GEN / f"tag_{text}.png")

    for name, sku in CATALOG.items():
        path = GEN / f"{name}.obj"
        if sku.shape == "box":
            box_mesh(path, *sku.size)
            continue
        r, h = sku.size
        if name == "ceramic_mug":
            lathe(path, mug_profile(r, h), extra=_handle(r * 0.98, h * 0.52))
        elif name == "glass_jar":
            lathe(path, jar_profile(r, h))
        else:
            lathe(path, can_profile(r, h))
    box_mesh(GEN / "unit_box.obj", 0.5, 0.5, 0.5)


if __name__ == "__main__":
    generate()
