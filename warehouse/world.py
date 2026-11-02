"""Build the MuJoCo packing station for a scene: robot, bench, totes, cartons, warehouse and SKU units."""
import os
import re
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from .assets import GEN
from .catalog import CARTON_HEIGHT, CARTONS, CATALOG, TOTE_SIZE, TOTES, UNIT_OFFSETS_Y

PANDA_DIR = Path(__file__).resolve().parent / "assets" / "panda"
FLOOR_Z = -0.76
WIDTH, HEIGHT = 1280, 720
SKU_LOOK = {
    "ceramic_mug": 'specular="0.9" shininess="0.8" reflectance="0.04"',
    "glass_jar": 'specular="1" shininess="0.95" reflectance="0.05"',
    "soup_can": 'specular="0.8" shininess="0.7" reflectance="0.05"',
    "paint_can": 'specular="0.9" shininess="0.6" reflectance="0.06"',
}


def _panda_include():
    """The stock Panda without keyframes (scenes add free joints) or arm collisions (the arm cannot strike items)."""
    target = PANDA_DIR / "panda_station.xml"
    if not target.exists():
        xml = (PANDA_DIR / "panda.xml").read_text()
        xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml, flags=re.S)
        xml = xml.replace('<mujoco model="panda">', '<mujocoinclude>').replace("</mujoco>", "</mujocoinclude>")
        target.write_text(xml)
    return target.name


def _box(name, pos, size, material=None, rgba=None, collide=True, quat=None):
    look = f'material="{material}"' if material else f'rgba="{rgba}"'
    flags = 'contype="1" conaffinity="1"' if collide else 'contype="0" conaffinity="0"'
    q = f'quat="{quat}"' if quat else ""
    return (f'<geom name="{name}" type="box" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
            f'size="{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}" {look} {flags} {q} friction="0.8 0.02 0.001"/>')


def _rack(name, x, y, width=2.7, depth=1.0, levels=(0.1, 1.5, 2.9), seed=0):
    """Steel pallet racking with blue uprights, orange beams and stocked pallets."""
    rng = np.random.default_rng(seed)
    parts, z0 = [], FLOOR_Z
    height = 4.4
    for i, dx in enumerate((-width / 2, width / 2)):
        for j, dy in enumerate((-depth / 2, depth / 2)):
            parts.append(_box(f"{name}_up{i}{j}", (x + dx, y + dy, z0 + height / 2), (0.045, 0.045, height / 2), "steel_blue", collide=False))
    for k, level in enumerate(levels):
        for j, dy in enumerate((-depth / 2, depth / 2)):
            parts.append(_box(f"{name}_beam{k}{j}", (x, y + dy, z0 + level), (width / 2, 0.035, 0.06), "steel_orange", collide=False))
        parts.append(_box(f"{name}_deck{k}", (x, y, z0 + level + 0.06), (width / 2, depth / 2, 0.006), rgba="0.32 0.33 0.35 1", collide=False))
        for p in (-0.65, 0.65):
            px, base = x + p, z0 + level + 0.066
            parts.append(_box(f"{name}_pal{k}{p}", (px, y, base + 0.06), (0.55, 0.45, 0.06), "pallet", collide=False))
            stack_z = base + 0.12
            for tier in range(int(rng.integers(2, 4))):
                for bx in (-0.28, 0.28):
                    for by in (-0.22, 0.22):
                        if rng.random() < 0.08:
                            continue
                        h = 0.17 + rng.random() * 0.05
                        tex = f"cardboard{int(rng.integers(1, 5))}"
                        parts.append(_box(f"{name}_b{k}{p}{tier}{bx}{by}", (px + bx, y + by, stack_z + h),
                                          (0.26, 0.2, h), tex, collide=False))
                stack_z += 0.44
    return "".join(parts)


def _static_scene():
    parts = []
    # Bench: top surface at z=0.
    parts.append(_box("bench_top", (0.33, 0.0, -0.025), (0.62, 0.85, 0.025), "bench"))
    for sx in (-0.26, 0.92):
        for sy in (-0.8, 0.8):
            parts.append(_box(f"bench_leg{sx}{sy}", (sx, sy, (FLOOR_Z - 0.05) / 2), (0.035, 0.035, (-FLOOR_Z - 0.05) / 2), "painted_steel", collide=False))
    parts.append(_box("bench_shelf", (0.33, 0.0, FLOOR_Z + 0.22), (0.58, 0.8, 0.012), "painted_steel", collide=False))
    # Robot riser plate.
    parts.append(f'<geom type="cylinder" pos="0 0 0.003" size="0.13 0.003" material="painted_steel" contype="0" conaffinity="0"/>')

    tx, ty, wall = TOTE_SIZE
    t = 0.004
    for name, (x, y) in TOTES.items():
        parts.append(_box(f"{name}_floor", (x, y, 0.003), (tx / 2, ty / 2, 0.003), "tote"))
        for label, dx, dy, sx, sy in (("xp", tx / 2, 0, t, ty / 2), ("xn", -tx / 2, 0, t, ty / 2), ("yp", 0, ty / 2, tx / 2, t), ("yn", 0, -ty / 2, tx / 2, t)):
            parts.append(_box(f"{name}_{label}", (x + dx, y + dy, wall / 2), (sx, sy, wall / 2), "tote"))
        # Tag on the bench beside the tote, facing the overhead camera.
        parts.append(f'<geom type="mesh" mesh="unit_box" material="tag_{name}" contype="0" conaffinity="0" '
                     f'pos="{x:.3f} {y + ty / 2 + 0.003:.3f} 0.0015" size="1 1 1" />' if False else "")
        parts.append(f'<body pos="{x:.3f} {y:.3f} 0.0"><geom type="plane" size="0.028 0.028 0.01" pos="-0.062 0 0.0065" '
                     f'material="tag_{name}" contype="0" conaffinity="0"/></body>')

    for name, carton in CARTONS.items():
        cx, cy = carton["center"]
        ix, iy = carton["inner"]
        h, t = CARTON_HEIGHT, 0.004
        parts.append(_box(f"carton{name}_floor", (cx, cy, 0.002), (ix / 2 + t, iy / 2 + t, 0.002), "carton"))
        walls = (("xp", ix / 2 + t, 0, t, iy / 2 + t), ("xn", -ix / 2 - t, 0, t, iy / 2 + t), ("yp", 0, iy / 2 + t, ix / 2 + t, t), ("yn", 0, -iy / 2 - t, ix / 2 + t, t))
        for label, dx, dy, sx, sy in walls:
            parts.append(_box(f"carton{name}_{label}", (cx + dx, cy + dy, h / 2), (sx, sy, h / 2), "carton"))
        # Open flaps, angled outward. Visual only.
        c, s = 0.924, 0.383
        flap = 0.05
        parts.append(_box(f"carton{name}_fxp", (cx + ix / 2 + t + flap * 0.92, cy, h + flap * 0.38), (flap, iy / 2 + t, 0.002), "carton", collide=False, quat=f"{c} 0 {s} 0"))
        parts.append(_box(f"carton{name}_fxn", (cx - ix / 2 - t - flap * 0.92, cy, h + flap * 0.38), (flap, iy / 2 + t, 0.002), "carton", collide=False, quat=f"{c} 0 {-s} 0"))
        parts.append(_box(f"carton{name}_fyp", (cx, cy + iy / 2 + t + flap * 0.92, h + flap * 0.38), (ix / 2 + t, flap, 0.002), "carton", collide=False, quat=f"{c} {-s} 0 0"))
        parts.append(_box(f"carton{name}_fyn", (cx, cy - iy / 2 - t - flap * 0.92, h + flap * 0.38), (ix / 2 + t, flap, 0.002), "carton", collide=False, quat=f"{c} {s} 0 0"))
        parts.append(f'<geom type="plane" size="0.03 0.03 0.01" pos="{cx - ix / 2 - 0.045:.3f} {cy:.3f} 0.001" material="tag_{name}" contype="0" conaffinity="0"/>')

    parts.append(f'<geom name="floor" type="plane" pos="0 0 {FLOOR_Z}" size="30 30 0.1" material="floor" contype="1" conaffinity="1"/>')
    parts.append(_rack("rackA", 2.4, -1.5, seed=1))
    parts.append(_rack("rackB", 2.4, 1.5, seed=2))
    parts.append(_rack("rackC", 2.4, 4.4, seed=3))
    parts.append(_rack("rackD", 2.4, -4.4, seed=4))
    # Yellow floor safety line.
    parts.append(_box("safety", (1.35, 0, FLOOR_Z + 0.001), (0.05, 8.0, 0.001), rgba="0.95 0.75 0.05 1", collide=False))
    # Back wall and roof strip lights.
    parts.append(_box("wall", (3.4, 0, 3), (0.1, 12, 4), rgba="0.72 0.74 0.76 1", collide=False))
    for y in (-3.5, -1.2, 1.2, 3.5):
        parts.append(_box(f"lamp{y}", (1.2, y, 5.6), (0.06, 0.8, 0.02), rgba="1 1 0.96 1", collide=False))
    return "".join(parts)


def _assets():
    gen = str(GEN)
    parts = [
        '<texture name="sky" type="skybox" builtin="gradient" rgb1="0.82 0.84 0.86" rgb2="0.46 0.47 0.5" width="256" height="256"/>',
        f'<texture name="t_floor" type="2d" file="{gen}/concrete.png"/><material name="floor" texture="t_floor" texrepeat="6 6" texuniform="true" specular="0.18" shininess="0.35" reflectance="0.07"/>',
        f'<texture name="t_bench" type="cube" file="{gen}/bench.png"/><material name="bench" texture="t_bench" specular="0.3" shininess="0.3" reflectance="0.02"/>',
        f'<texture name="t_pallet" type="cube" file="{gen}/pallet.png"/><material name="pallet" texture="t_pallet" specular="0.05" shininess="0.1"/>',
        f'<texture name="t_carton" type="cube" file="{gen}/carton_plain.png"/><material name="carton" texture="t_carton" specular="0.08" shininess="0.15"/>',
        '<material name="steel_blue" rgba="0.08 0.2 0.5 1" specular="0.6" shininess="0.5" reflectance="0.05"/>',
        '<material name="steel_orange" rgba="0.92 0.42 0.05 1" specular="0.6" shininess="0.5" reflectance="0.05"/>',
        '<material name="painted_steel" rgba="0.2 0.22 0.25 1" specular="0.5" shininess="0.4"/>',
        '<material name="tote" rgba="0.1 0.28 0.5 1" specular="0.35" shininess="0.35"/>',
        f'<mesh name="unit_box" file="{gen}/unit_box.obj"/>',
    ]
    for i in range(1, 5):
        parts.append(f'<texture name="t_cardboard{i}" type="cube" file="{gen}/cardboard{i}.png"/><material name="cardboard{i}" texture="t_cardboard{i}" specular="0.08" shininess="0.15"/>')
    for name in list(TOTES) + list(CARTONS):
        parts.append(f'<texture name="t_tag_{name}" type="2d" file="{gen}/tag_{name}.png"/><material name="tag_{name}" texture="t_tag_{name}" specular="0.1"/>')
    for name in CATALOG:
        look = SKU_LOOK.get(name, 'specular="0.3" shininess="0.35" reflectance="0.01"')
        parts.append(f'<texture name="t_{name}" type="2d" file="{gen}/tex_{name}.png"/><material name="m_{name}" texture="t_{name}" {look}/><mesh name="{name}" file="{gen}/{name}.obj"/>')
    return "".join(parts)


def _unit(unit_id, sku_name, x, y, rng):
    sku = CATALOG[sku_name]
    yaw = rng.uniform(-0.12, 0.12)
    quat = f"{np.cos(yaw / 2):.5f} 0 0 {np.sin(yaw / 2):.5f}"
    if sku.shape == "lathe":
        r, h = sku.size
        collision = f'<geom type="cylinder" size="{r} {h / 2}" pos="0 0 {h / 2}" mass="{sku.kg}" group="3" friction="0.8 0.02 0.001" condim="4"/>'
        visual = f'<geom type="mesh" mesh="{sku_name}" material="m_{sku_name}" contype="0" conaffinity="0" mass="0"/>'
        z = 0.008
    else:
        hx, hy, hz = sku.size
        collision = f'<geom type="box" size="{hx} {hy} {hz}" pos="0 0 {hz}" mass="{sku.kg}" group="3" friction="0.8 0.02 0.001" condim="4"/>'
        visual = f'<geom type="mesh" mesh="{sku_name}" material="m_{sku_name}" pos="0 0 {hz}" contype="0" conaffinity="0" mass="0"/>'
        z = 0.008
    return (f'<body name="{unit_id}" pos="{x:.4f} {y:.4f} {z}" quat="{quat}"><freejoint name="{unit_id}_j"/>{collision}{visual}</body>')


def build_xml(totes, seed=0):
    """totes maps tote id to {"sku": name, "count": n}."""
    rng = np.random.default_rng(seed)
    units = []
    for tote, content in totes.items():
        if not content or not content["count"]:
            continue
        x, y = TOTES[tote]
        for i in range(content["count"]):
            units.append(_unit(f"{tote}_{i}", content["sku"], x + 0.015, y + UNIT_OFFSETS_Y[i], rng))
    return f"""<mujoco model="packing_station">
<compiler angle="radian" autolimits="true" meshdir="assets" texturedir="assets"/>
<option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
<size memory="40M"/>
<visual>
  <global offwidth="{WIDTH}" offheight="{HEIGHT}" fovy="35" elevation="-25" azimuth="150"/>
  <quality shadowsize="8192" offsamples="8"/>
  <headlight ambient="0.32 0.32 0.32" diffuse="0.18 0.18 0.18" specular="0.05 0.05 0.05"/>
  <map fogstart="4" fogend="14" znear="0.01" zfar="60" shadowclip="1.6" shadowscale="0.8"/>
  <rgba fog="0.7 0.72 0.74 1" haze="0.75 0.77 0.8 1"/>
</visual>
<include file="{_panda_include()}"/>
<asset>{_assets()}</asset>
<worldbody>
  <light name="key" pos="0.6 -0.5 2.4" dir="-0.05 0.15 -1" directional="false" castshadow="true" diffuse="0.72 0.71 0.68" specular="0.3 0.3 0.3" attenuation="1 0 0" cutoff="70" exponent="6"/>
  <light name="fill" pos="-0.8 1.1 2.0" dir="0.5 -0.4 -1" castshadow="true" diffuse="0.3 0.31 0.34" specular="0.1 0.1 0.1" cutoff="80" exponent="2"/>
  <light name="hall" pos="1.2 0 5.2" dir="0 0 -1" directional="true" castshadow="false" diffuse="0.26 0.26 0.25" specular="0 0 0"/>
  {_static_scene()}
  {''.join(units)}
  <camera name="overhead" pos="0.37 -0.02 1.62" xyaxes="0 -1 0 1 0 0" fovy="50"/>
  <camera name="cinema" pos="-0.72 -1.14 0.66" xyaxes="0.84 -0.54 0 0.16 0.25 0.95" fovy="48"/>
  <camera name="closeup" pos="0.92 0.72 0.42" xyaxes="-0.62 0.78 0 -0.42 -0.33 0.84" fovy="42"/>
</worldbody>
</mujoco>"""


def load_model(totes, seed=0):
    # Written beside panda.xml so the include and its mesh directory resolve.
    fd, path = tempfile.mkstemp(prefix="scene_", suffix=".xml", dir=PANDA_DIR)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(build_xml(totes, seed))
        return mujoco.MjModel.from_xml_path(path)
    finally:
        os.unlink(path)
