"""Synthetic task generator: 4 difficulties, deterministic from a seed.

Distractors use near-miss colors (red/orange/pink) and repeated kinds so that
perception is not trivial and perception errors show up in the score.
"""
import random
from typing import Dict, List, Optional, Tuple

from .schema import CAPACITY, TABLE, Container, Goal, Obj, Robot, Scene, Task

COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink"]
KINDS = ["block", "cup", "ball"]
DIFFICULTIES = ["easy", "multi_step", "blocked", "impossible", "packing"]

W, H = 640, 480
_SLOTS = [(420, 24, 196, 120), (420, 168, 196, 140), (420, 332, 196, 124)]
# Tabletop scenes use tray/bin/box; warehouse scenes use carton_a/carton_b/tote. They never mix.
CONTAINER_SLOTS = {"tray": _SLOTS[0], "bin": _SLOTS[1], "box": _SLOTS[2],
                   "carton_a": _SLOTS[0], "carton_b": _SLOTS[1], "tote": _SLOTS[2]}
TABLE_CELLS = [(30 + 92 * c, 30 + 106 * r) for r in range(4) for c in range(4)]

PHRASE = {"tray": "on the tray", "bin": "in the bin", "box": "in the box"}


def _name(oid: str) -> str:
    return oid.replace("_", " ")


class _Builder:
    def __init__(self, rng: random.Random, containers: Dict[str, bool]):
        self.rng = rng
        self.objects: List[Obj] = []
        self.containers = [Container(id=k, kind=k.split("_")[0], is_open=v) for k, v in containers.items()]
        self.used = set()

    def new_obj(self, location: str = TABLE, avoid: Optional[set] = None) -> Obj:
        while True:
            color, kind = self.rng.choice(COLORS), self.rng.choice(KINDS)
            oid = f"{color}_{kind}"
            if oid not in self.used and (not avoid or oid not in avoid):
                break
        self.used.add(oid)
        o = Obj(id=oid, color=color, kind=kind, location=location)
        self.objects.append(o)
        return o

    def fill(self, n_total: int) -> None:
        # Distractors: mostly on the table, sometimes in an open bin/box/tote with spare room.
        while len(self.objects) < n_total:
            loc = TABLE
            open_bins = [c for c in self.containers if c.kind in ("bin", "box", "tote") and c.is_open]
            if open_bins and self.rng.random() < 0.25:
                c = self.rng.choice(open_bins)
                if not any(o.location == c.id for o in self.objects):
                    loc = c.id
            self.new_obj(loc)

    def new_stock(self, k: int, want_cup: bool) -> List[Obj]:
        """k order items, some already in the tote. With want_cup, one of them is a fragile cup."""
        items = []
        for i in range(k):
            loc = "tote" if self.rng.random() < 0.4 else TABLE
            while True:
                o = self.new_obj(loc)
                if not (want_cup and i == 0) or o.kind == "cup":
                    break
                self.objects.remove(o)
                self.used.discard(o.id)
            items.append(o)
        return items

    def scene(self) -> Scene:
        return Scene(objects=self.objects, containers=self.containers, robot=Robot())

    def layout(self) -> Dict[str, Tuple[int, int, int, int]]:
        out = {c.id: CONTAINER_SLOTS[c.id] for c in self.containers}
        cells = self.rng.sample(TABLE_CELLS, k=len(TABLE_CELLS))
        for o in self.objects:
            if o.location == TABLE:
                x, y = cells.pop()
                out[o.id] = (x + self.rng.randint(0, 20), y + self.rng.randint(0, 24), 48, 48)
        return out


def _containers(rng: random.Random, must: List[str], closed: Optional[str] = None) -> Dict[str, bool]:
    kinds = set(must) | {k for k in CAPACITY if rng.random() < 0.5}
    return {k: (k != closed) for k in ["tray", "bin", "box"] if k in kinds}


def _easy(rng):
    target = rng.choice(["tray", "bin", "box"])
    b = _Builder(rng, _containers(rng, [target]))
    o = b.new_obj()
    b.fill(rng.randint(3, 5))
    _clear_tray(b, target, keep=set())
    return b, [Goal(obj=o.id, location=target)], f"Put the {_name(o.id)} {PHRASE[target]}."


def _multi(rng):
    if rng.random() < 0.5:
        target = rng.choice(["bin", "box"])
        b = _Builder(rng, _containers(rng, [target]))
        objs = [b.new_obj() for _ in range(2)]
        goals = [Goal(obj=o.id, location=target) for o in objs]
        text = f"Put the {_name(objs[0].id)} and the {_name(objs[1].id)} {PHRASE[target]}."
    else:
        other = rng.choice(["bin", "box"])
        b = _Builder(rng, _containers(rng, ["tray", other]))
        o1, o2 = b.new_obj(), b.new_obj()
        goals = [Goal(obj=o1.id, location="tray"), Goal(obj=o2.id, location=other)]
        text = f"Put the {_name(o1.id)} on the tray and the {_name(o2.id)} {PHRASE[other]}."
    b.fill(rng.randint(4, 6))
    for g in goals:
        _clear_tray(b, g.location, keep={x.obj for x in goals})
        _make_room(b, g.location, need=sum(x.location == g.location for x in goals))
    return b, goals, text


def _blocked(rng):
    kind = rng.choice(["closed", "occupied", "both"])
    if kind == "occupied":
        b = _Builder(rng, _containers(rng, ["tray"]))
        o = b.new_obj()
        b.new_obj(location="tray")
        b.fill(rng.randint(3, 5))
        return b, [Goal(obj=o.id, location="tray")], f"Put the {_name(o.id)} on the tray."
    target = rng.choice(["bin", "box"])
    if kind == "closed":
        b = _Builder(rng, _containers(rng, [target], closed=target))
        o = b.new_obj()
        b.fill(rng.randint(3, 5))
        return b, [Goal(obj=o.id, location=target)], f"Put the {_name(o.id)} {PHRASE[target]}."
    b = _Builder(rng, _containers(rng, ["tray", target], closed=target))
    o1, o2 = b.new_obj(), b.new_obj()
    b.new_obj(location="tray")
    b.fill(rng.randint(4, 6))
    goals = [Goal(obj=o1.id, location="tray"), Goal(obj=o2.id, location=target)]
    return b, goals, f"Put the {_name(o1.id)} on the tray and the {_name(o2.id)} {PHRASE[target]}."


def _impossible(rng):
    if rng.random() < 0.7:
        # Missing object, ideally a near-miss of something that IS there.
        target = rng.choice(["tray", "bin", "box"])
        b = _Builder(rng, _containers(rng, [target]))
        b.fill(rng.randint(3, 5))
        _clear_tray(b, target, keep=set())
        present = b.objects[0]
        alt_colors = [c for c in COLORS if f"{c}_{present.kind}" not in b.used]
        missing = f"{rng.choice(alt_colors)}_{present.kind}"
        return b, [], f"Put the {_name(missing)} {PHRASE[target]}."
    # Missing container.
    absent = rng.choice(["tray", "bin", "box"])
    kinds = {k: True for k in ["tray", "bin", "box"] if k != absent and rng.random() < 0.7} or \
            {next(k for k in ["tray", "bin", "box"] if k != absent): True}
    b = _Builder(rng, kinds)
    o = b.new_obj()
    b.fill(rng.randint(3, 5))
    return b, [], f"Put the {_name(o.id)} {PHRASE[absent]}."


def _clear_tray(b: _Builder, target: str, keep: set) -> None:
    if target == "tray":
        for o in b.objects:
            if o.location == "tray" and o.id not in keep:
                o.location = TABLE


def _make_room(b: _Builder, target: str, need: int) -> None:
    c = next(c for c in b.containers if c.id == target)
    inside = [o for o in b.objects if o.location == target]
    while len(inside) + need > CAPACITY[c.kind]:
        inside.pop().location = TABLE


def _list(objs) -> str:
    names = [_name(o) for o in objs]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _packing(rng):
    """Warehouse order packing: closed cartons, fragile cups, exact contents, seal at the end."""
    two = rng.random() < 0.5
    b = _Builder(rng, {"carton_a": False, **({"carton_b": False} if two else {}), "tote": True})
    order_a = b.new_stock(rng.choice([2, 3]), want_cup=rng.random() < 0.75)
    order_b = b.new_stock(rng.choice([1, 2]), want_cup=rng.random() < 0.5) if two else []
    while sum(o.location == "tote" for o in b.objects) > CAPACITY["tote"]:
        next(o for o in b.objects if o.location == "tote").location = TABLE
    b.fill(len(b.objects) + rng.randint(2, 3))
    while sum(o.location == "tote" for o in b.objects) > CAPACITY["tote"]:
        next(o for o in reversed(b.objects) if o.location == "tote").location = TABLE

    # The cup is listed first so a naive plan packs it first and then crushes it.
    ids_a, ids_b = [o.id for o in order_a], [o.id for o in order_b]
    out_of_stock = rng.random() < 0.2
    if out_of_stock:
        victim = rng.choice(order_a + order_b)
        alt = [c for c in COLORS if f"{c}_{victim.kind}" not in b.used]
        missing = f"{rng.choice(alt)}_{victim.kind}"
        ids_a = [missing if x == victim.id else x for x in ids_a]
        ids_b = [missing if x == victim.id else x for x in ids_b]

    if two:
        text = (f"Pack order A ({_list(ids_a)}) into carton A and order B ({_list(ids_b)}) into carton B. "
                f"Seal both cartons.")
    else:
        text = f"Pack order A ({_list(ids_a)}) into carton A, then seal it."
    if out_of_stock:
        return b, [], text, []
    goals = [Goal(obj=x, location="carton_a") for x in ids_a] + [Goal(obj=x, location="carton_b") for x in ids_b]
    return b, goals, text, ["carton_a"] + (["carton_b"] if two else [])


GEN = {"easy": _easy, "multi_step": _multi, "blocked": _blocked, "impossible": _impossible, "packing": _packing}


def generate(n_per_difficulty: int = 40, heldout_per_difficulty: Tuple[int, ...] = (10, 10, 10, 10, 10),
             seed: int = 7) -> List[Task]:
    rng = random.Random(seed)
    tasks: List[Task] = []
    for d, n_held in zip(DIFFICULTIES, heldout_per_difficulty):
        held = set(rng.sample(range(n_per_difficulty), n_held))
        for i in range(n_per_difficulty):
            out = GEN[d](rng)
            b, goals, text = out[:3]
            sealed = out[3] if len(out) > 3 else []
            tasks.append(Task(
                id=f"{d}-{i:03d}", split="heldout" if i in held else "tune", difficulty=d,
                instruction=text, scene=b.scene(), goals=goals, sealed=sealed, possible=bool(goals),
                layout=b.layout(),
            ))
    return tasks
