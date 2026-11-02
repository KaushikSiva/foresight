"""Photoreal renderer: our scenes placed in Kaushik's MuJoCo packing station (warehouse/).

Reuses his bench, warehouse racks, lighting, Panda arm (tucked clear of the camera), materials
and mug mesh. Adds our containers (tray, bin, box, tote, cartons) with lids or taped flaps when
closed, label plates the overhead camera can read, and our items as colored 3D objects
(cup = mug, block = cube, ball = sphere), so colors and lookalikes stay part of the test.

Static frames only: nothing is simulated, every pose comes from the symbolic scene, so each
image still has exact ground truth.
"""
import hashlib
import io
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from .config import ROOT
from .render import RGB
from .schema import CAPACITY, Scene, Task

for _p in (ROOT, ROOT.parent):  # warehouse/ lives beside this package, or at the repo root
    if (_p / "warehouse").is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
try:
    from warehouse.robot import TUCKED  # noqa: E402
except ImportError:  # older warehouse/ without robot.py: the same arm pose, swung clear of the camera
    TUCKED = np.array([2.6, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
from warehouse.world import FLOOR_Z, PANDA_DIR, _assets, _box, _panda_include, _rack  # noqa: E402

WIDTH, HEIGHT = 1280, 720
LABELS = ROOT / "data" / "labels"
CACHE = ROOT / "data" / "images3d"

# Container slots along the far edge of the bench (world x up the image, -y to the right).
SLOT_CENTERS = [(0.66, 0.42), (0.66, 0.0), (0.66, -0.42)]
SLOT_OF = {"tray": 0, "bin": 1, "box": 2, "carton_a": 0, "carton_b": 1, "tote": 2}
# Inner size (x, y) and wall height per kind.
DIMS = {"tray": (0.22, 0.30, 0.025), "bin": (0.24, 0.32, 0.10), "box": (0.20, 0.26, 0.12),
        "tote": (0.24, 0.32, 0.10), "carton": (0.24, 0.30, 0.13)}
# Bench cells for loose items, clear of the tucked arm at the bottom center of the image.
CELLS = ([(x, y) for x in (0.38, 0.24) for y in (-0.66, -0.47, -0.28, -0.09, 0.10, 0.29, 0.48, 0.67)]
         + [(0.10, y) for y in (-0.70, -0.50, 0.50, 0.70)])
INNER = [(-0.05, 0.075), (-0.05, -0.075), (0.05, 0.075), (0.05, -0.075)]


def _label_texture(text: str) -> Path:
    LABELS.mkdir(parents=True, exist_ok=True)
    path = LABELS / f"{text.replace(' ', '_')}.png"
    if not path.exists():
        img = Image.new("RGB", (512, 128), (248, 248, 244))
        d = ImageDraw.Draw(img)
        for f in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            try:
                font = ImageFont.truetype(f, 84)
                break
            except OSError:
                font = ImageFont.load_default()
        d.rectangle([4, 4, 507, 123], outline=(20, 20, 20), width=6)
        d.text((256, 66), text, fill=(15, 15, 15), font=font, anchor="mm")
        img.save(path)
    return path


def _station() -> str:
    """Bench, floor, racks and lights from his station, without his fixed totes and cartons."""
    p = [_box("bench_top", (0.33, 0.0, -0.025), (0.62, 0.85, 0.025), "bench")]
    for sx in (-0.26, 0.92):
        for sy in (-0.8, 0.8):
            p.append(_box(f"bench_leg{sx}{sy}", (sx, sy, (FLOOR_Z - 0.05) / 2), (0.035, 0.035, (-FLOOR_Z - 0.05) / 2),
                          "painted_steel", collide=False))
    p.append(_box("bench_shelf", (0.33, 0.0, FLOOR_Z + 0.22), (0.58, 0.8, 0.012), "painted_steel", collide=False))
    p.append('<geom type="cylinder" pos="0 0 0.003" size="0.13 0.003" material="painted_steel"/>')
    p.append(f'<geom name="floor" type="plane" pos="0 0 {FLOOR_Z}" size="30 30 0.1" material="floor"/>')
    for name, y, seed in (("rackA", -1.5, 1), ("rackB", 1.5, 2), ("rackC", 4.4, 3), ("rackD", -4.4, 4)):
        p.append(_rack(name, 2.4, y, seed=seed))
    p.append(_box("safety", (1.35, 0, FLOOR_Z + 0.001), (0.05, 8.0, 0.001), rgba="0.95 0.75 0.05 1", collide=False))
    p.append(_box("wall", (3.4, 0, 3), (0.1, 12, 4), rgba="0.72 0.74 0.76 1", collide=False))
    for y in (-3.5, -1.2, 1.2, 3.5):
        p.append(_box(f"lamp{y}", (1.2, y, 5.6), (0.06, 0.8, 0.02), rgba="1 1 0.96 1", collide=False))
    return "".join(p)


def _container(cid: str, kind: str, is_open: bool) -> str:
    cx, cy = SLOT_CENTERS[SLOT_OF[cid]]
    ix, iy, h = DIMS[kind]
    t = 0.006
    look = {"tray": 'rgba="0.42 0.44 0.48 1"', "bin": 'material="tote"', "tote": 'material="tote"',
            "box": 'material="carton"', "carton": 'material="carton"'}[kind]
    mat = lambda: look.split('"')[1] if look.startswith("material") else None
    rgba = None if mat() else look.split('"')[1]
    p = [_box(f"{cid}_floor", (cx, cy, 0.003), (ix / 2 + t, iy / 2 + t, 0.003), mat(), rgba)]
    for lab, dx, dy, sx, sy in (("xp", ix / 2 + t, 0, t, iy / 2 + t), ("xn", -ix / 2 - t, 0, t, iy / 2 + t),
                                ("yp", 0, iy / 2 + t, ix / 2 + t, t), ("yn", 0, -iy / 2 - t, ix / 2 + t, t)):
        p.append(_box(f"{cid}_{lab}", (cx + dx, cy + dy, h / 2), (sx, sy, h / 2), mat(), rgba))
    cardboard = kind in ("box", "carton")
    if cardboard and is_open:  # flaps folded outward, as in his station
        c, s, f = 0.924, 0.383, 0.035
        p.append(_box(f"{cid}_fxp", (cx + ix / 2 + t + f * 0.92, cy, h + f * 0.38), (f, iy / 2 + t, 0.002), "carton", collide=False, quat=f"{c} 0 {s} 0"))
        p.append(_box(f"{cid}_fxn", (cx - ix / 2 - t - f * 0.92, cy, h + f * 0.38), (f, iy / 2 + t, 0.002), "carton", collide=False, quat=f"{c} 0 {-s} 0"))
        p.append(_box(f"{cid}_fyp", (cx, cy + iy / 2 + t + f * 0.92, h + f * 0.38), (ix / 2 + t, f, 0.002), "carton", collide=False, quat=f"{c} {-s} 0 0"))
        p.append(_box(f"{cid}_fyn", (cx, cy - iy / 2 - t - f * 0.92, h + f * 0.38), (ix / 2 + t, f, 0.002), "carton", collide=False, quat=f"{c} {s} 0 0"))
    elif cardboard:  # flaps folded shut with a strip of packing tape
        p.append(_box(f"{cid}_top", (cx, cy, h + 0.002), (ix / 2 + t, iy / 2 + t, 0.002), "carton", collide=False))
        p.append(_box(f"{cid}_seam", (cx, cy, h + 0.0045), (0.0015, iy / 2 + t, 0.0005), rgba="0.35 0.25 0.15 1", collide=False))
        p.append(_box(f"{cid}_tape", (cx, cy, h + 0.005), (0.024, iy / 2 + t + 0.004, 0.0006), rgba="0.82 0.66 0.40 0.9", collide=False))
    elif not is_open:  # plastic lid with two handles
        p.append(_box(f"{cid}_lid", (cx, cy, h + 0.006), (ix / 2 + t + 0.006, iy / 2 + t + 0.006, 0.006), rgba="0.24 0.26 0.29 1", collide=False))
        for dy in (-0.09, 0.09):
            p.append(_box(f"{cid}_handle{dy}", (cx, cy + dy, h + 0.016), (0.05, 0.012, 0.004), rgba="0.15 0.16 0.18 1", collide=False))
    # Label plate on the bench in front of the container, readable from the overhead camera.
    lx = cx - ix / 2 - t - 0.07
    p.append(f'<geom type="plane" size="0.11 0.0275 0.01" pos="{lx:.3f} {cy:.3f} 0.001" quat="0.7071 0 0 -0.7071" '
             f'material="label_{cid}" contype="0" conaffinity="0"/>')
    return "".join(p)


def _item(oid: str, color: str, kind: str, x: float, y: float, z: float, yaw: float) -> str:
    q = f"{np.cos(yaw / 2):.5f} 0 0 {np.sin(yaw / 2):.5f}"
    if kind == "cup":
        g = f'<geom type="mesh" mesh="ceramic_mug" material="c_{color}"/>'
    elif kind == "ball":
        g = f'<geom type="sphere" size="0.034" pos="0 0 0.034" material="c_{color}"/>'
    else:
        g = f'<geom type="box" size="0.031 0.031 0.031" pos="0 0 0.031" material="c_{color}"/>'
    return f'<body name="{oid}" pos="{x:.4f} {y:.4f} {z:.4f}" quat="{q}">{g}</body>'


def positions(task: Task, scene: Scene) -> Dict[str, Tuple[float, float, float]]:
    """Where each item sits. Loose items keep a stable cell per task; packed items fill container slots."""
    rng = random.Random(task.id)
    cells = CELLS[:]
    rng.shuffle(cells)
    order = [o.id for o in task.scene.objects] + [o.id for o in scene.objects if task.scene.obj(o.id) is None]
    home = {oid: cells[i % len(cells)] for i, oid in enumerate(order)}
    out = {}
    for o in scene.objects:
        if o.location == "table":
            x, y = home[o.id]
            out[o.id] = (x, y, 0.0)
        elif o.location != "gripper":
            c = scene.container(o.location)
            cx, cy = SLOT_CENTERS[SLOT_OF[c.id]]
            idx = [x.id for x in scene.contents(c.id)].index(o.id)
            dx, dy = INNER[idx % len(INNER)] if CAPACITY[c.kind] > 1 else (0.0, 0.0)
            out[o.id] = (cx + dx, cy + dy, 0.006)
    return out


def build_xml(task: Task, scene: Optional[Scene] = None) -> str:
    scene = scene or task.scene
    rng = random.Random(task.id + "yaw")
    labels = "".join(
        f'<texture name="t_label_{c.id}" type="2d" file="{_label_texture(c.id.replace("_", " ").upper())}"/>'
        f'<material name="label_{c.id}" texture="t_label_{c.id}" specular="0.05"/>' for c in scene.containers)
    colors = "".join(f'<material name="c_{n}" rgba="{r / 255:.3f} {g / 255:.3f} {b / 255:.3f} 1" specular="0.35" shininess="0.45"/>'
                     for n, (r, g, b) in RGB.items())
    pos = positions(task, scene)
    items = "".join(_item(o.id, o.color, o.kind, *pos[o.id], rng.uniform(-3.1, 3.1)) for o in scene.objects if o.id in pos)
    conts = "".join(_container(c.id, c.kind, c.is_open) for c in scene.containers)
    return f"""<mujoco model="foresight_station">
<compiler angle="radian" autolimits="true" meshdir="assets" texturedir="assets"/>
<size memory="40M"/>
<visual>
  <global offwidth="{WIDTH}" offheight="{HEIGHT}" fovy="35"/>
  <quality shadowsize="8192" offsamples="8"/>
  <headlight ambient="0.32 0.32 0.32" diffuse="0.18 0.18 0.18" specular="0.05 0.05 0.05"/>
  <map fogstart="4" fogend="14" znear="0.01" zfar="60" shadowclip="1.6" shadowscale="0.8"/>
  <rgba fog="0.7 0.72 0.74 1" haze="0.75 0.77 0.8 1"/>
</visual>
<include file="{_panda_include()}"/>
<asset>{_assets()}{labels}{colors}</asset>
<worldbody>
  <light pos="0.6 -0.5 2.4" dir="-0.05 0.15 -1" castshadow="true" diffuse="0.72 0.71 0.68" specular="0.3 0.3 0.3" cutoff="70" exponent="6"/>
  <light pos="-0.8 1.1 2.0" dir="0.5 -0.4 -1" castshadow="true" diffuse="0.3 0.31 0.34" specular="0.1 0.1 0.1" cutoff="80" exponent="2"/>
  <light pos="1.2 0 5.2" dir="0 0 -1" directional="true" castshadow="false" diffuse="0.26 0.26 0.25" specular="0 0 0"/>
  {_station()}{conts}{items}
  <camera name="overhead" pos="0.40 0.0 1.62" xyaxes="0 -1 0 1 0 0" fovy="50"/>
  <camera name="cinema" pos="-0.72 -1.14 0.66" xyaxes="0.84 -0.54 0 0.16 0.25 0.95" fovy="48"/>
</worldbody>
</mujoco>"""


def render(task: Task, scene: Optional[Scene] = None, camera: str = "overhead") -> Image.Image:
    xml = build_xml(task, scene)
    path = PANDA_DIR / f"_fs_{hashlib.sha1(xml.encode()).hexdigest()[:12]}.xml"
    path.write_text(xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(path))
    finally:
        path.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    for i, q in enumerate(TUCKED, start=1):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
        data.qpos[model.jnt_qposadr[j]] = q
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, HEIGHT, WIDTH) as r:
        r.update_scene(data, camera=camera)
        return Image.fromarray(r.render())


def jpeg_bytes(task: Task, scene: Optional[Scene] = None, camera: str = "overhead") -> bytes:
    """Initial overhead frames are cached on disk; after-states are rendered on demand."""
    cached = CACHE / f"{task.id}-{camera}.jpg"
    if scene is None and cached.exists():
        return cached.read_bytes()
    buf = io.BytesIO()
    render(task, scene, camera).save(buf, format="JPEG", quality=90)
    if scene is None:
        CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(buf.getvalue())
    return buf.getvalue()
