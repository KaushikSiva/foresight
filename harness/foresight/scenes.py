"""Synthetic task generator: 5 tiers, each mixing several trap mechanisms so no single recipe solves it.

Every scene has lookalike distractors (near-miss colors, same color with a different shape), so
perception is part of the test. Every possible task is proven solvable by the A* oracle at
generation time, and every impossible task is impossible by construction (a missing item or
container, or more items than the destination can ever hold).
"""
import random
from typing import Callable, Dict, List, Optional, Tuple

from .schema import CAPACITY, TABLE, Container, Goal, Obj, Robot, Scene, Task

COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink"]
KINDS = ["block", "cup", "ball"]
DIFFICULTIES = ["easy", "multi_step", "blocked", "impossible", "packing"]
# Colors a camera (or a model) can confuse.
LOOKALIKE = {"red": ["orange", "pink"], "orange": ["red", "yellow"], "yellow": ["orange"], "green": ["blue"],
             "blue": ["purple", "green"], "purple": ["blue", "pink"], "pink": ["red", "purple"]}

W, H = 640, 480
_SLOTS = [(420, 24, 196, 120), (420, 168, 196, 140), (420, 332, 196, 124)]
# Tabletop scenes use tray/bin/box; warehouse scenes use carton_a/carton_b/tote. They never mix.
CONTAINER_SLOTS = {"tray": _SLOTS[0], "bin": _SLOTS[1], "box": _SLOTS[2],
                   "carton_a": _SLOTS[0], "carton_b": _SLOTS[1], "tote": _SLOTS[2]}
TABLE_CELLS = [(30 + 92 * c, 30 + 106 * r) for r in range(4) for c in range(4)]
PHRASE = {"tray": "on the tray", "bin": "in the bin", "box": "in the box"}


def _name(oid: str) -> str:
    return oid.replace("_", " ")


def _list(ids: List[str]) -> str:
    names = [_name(i) for i in ids]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


class _Builder:
    def __init__(self, rng: random.Random, containers: Dict[str, bool]):
        self.rng = rng
        self.objects: List[Obj] = []
        self.containers = [Container(id=k, kind=k.split("_")[0], is_open=v) for k, v in containers.items()]
        self.used = set()
        self.traps: List[str] = []

    def free_id(self, color: Optional[str] = None, kind: Optional[str] = None) -> Optional[str]:
        options = [f"{c}_{k}" for c in ([color] if color else COLORS) for k in ([kind] if kind else KINDS)
                   if f"{c}_{k}" not in self.used]
        return self.rng.choice(options) if options else None

    def new_obj(self, location: str = TABLE, color: Optional[str] = None, kind: Optional[str] = None) -> Obj:
        oid = self.free_id(color, kind) or self.free_id(kind=kind) or self.free_id()
        self.used.add(oid)
        c, k = oid.split("_")
        o = Obj(id=oid, color=c, kind=k, location=location)
        self.objects.append(o)
        return o

    def lookalike(self, target: Obj, location: str = TABLE) -> Optional[Obj]:
        """A distractor one attribute away from target: a near-miss color, or the same color in another shape."""
        options = [(c, target.kind) for c in LOOKALIKE[target.color]] + [(target.color, k) for k in KINDS if k != target.kind]
        self.rng.shuffle(options)
        for c, k in options:
            if f"{c}_{k}" not in self.used:
                o = self.new_obj(location, c, k)
                if "lookalike" not in self.traps:
                    self.traps.append("lookalike")
                return o
        return None

    def absent_lookalike(self, target: Obj) -> str:
        """An id that looks like target but is NOT in the scene (missing / out-of-stock tasks)."""
        options = [f"{c}_{target.kind}" for c in LOOKALIKE[target.color]] + [f"{target.color}_{k}" for k in KINDS]
        options = [o for o in options if o not in self.used]
        oid = self.rng.choice(options) if options else self.free_id(kind=target.kind) or self.free_id()
        self.used.add(oid)  # reserved, so later distractors can never create it
        return oid

    def fill(self, n_total: int, spill: Tuple[str, ...] = ("bin", "box", "tote")) -> None:
        while len(self.objects) < n_total:
            loc = TABLE
            room = [c for c in self.containers if c.kind in spill and c.is_open and self.room(c.id) > 1]
            if room and self.rng.random() < 0.25:
                loc = self.rng.choice(room).id
            self.new_obj(loc)

    def loc(self, oid: str) -> str:
        return next(o.location for o in self.objects if o.id == oid)

    def room(self, cid: str) -> int:
        c = next(c for c in self.containers if c.id == cid)
        return CAPACITY[c.kind] - sum(o.location == cid for o in self.objects)

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


# A tier generator returns (builder, requested goals, instruction, sealed, possible).
Gen = Tuple[_Builder, List[Goal], str, List[str], bool]


def _tabletop(rng, must: List[str], closed: Tuple[str, ...] = ()) -> _Builder:
    kinds = set(must) | {k for k in ("tray", "bin", "box") if rng.random() < 0.5}
    return _Builder(rng, {k: k not in closed for k in ("tray", "bin", "box") if k in kinds})


def _distract(b: _Builder, targets: List[Obj], total: int) -> None:
    for t in targets:
        b.lookalike(t)
    b.fill(total, spill=("bin", "box"))


def _settle(b: _Builder, goals: List[Goal], keep_full: Tuple[str, ...] = ()) -> None:
    """Remove accidental blockers, keeping only the ones a variant created on purpose."""
    goal_objs = {g.obj for g in goals}
    for cid in sorted({g.location for g in goals} - {TABLE} - set(keep_full)):
        need = sum(1 for g in goals if g.location == cid and b.loc(g.obj) != cid)
        while b.room(cid) < need:
            next(o for o in reversed(b.objects) if o.location == cid and o.id not in goal_objs).location = TABLE


# ---------------- easy: one item, open destination, lookalikes nearby ----------------
def _easy(rng) -> Gen:
    target = rng.choice(["tray", "bin", "box"])
    b = _tabletop(rng, [target])
    o = b.new_obj()
    b.lookalike(o)
    b.lookalike(o)
    b.fill(rng.randint(5, 7), spill=("bin", "box"))
    goals = [Goal(obj=o.id, location=target)]
    _settle(b, goals)
    return b, goals, f"Put the {_name(o.id)} {PHRASE[target]}.", [], True


# ---------------- multi-step: several items whose order matters ----------------
def _multi(rng) -> Gen:
    v = rng.choice(["make_room", "swap", "three_items", "relocate"])
    keep: Tuple[str, ...] = ()
    if v == "make_room":
        # Two items for a bin that has room for one: something must come out first.
        b = _tabletop(rng, ["bin"])
        b.new_obj("bin"), b.new_obj("bin")
        objs = [b.new_obj(), b.new_obj()]
        goals = [Goal(obj=o.id, location="bin") for o in objs]
        text = f"Put the {_name(objs[0].id)} and the {_name(objs[1].id)} in the bin."
        keep = ("bin",)
    elif v == "swap":
        # The tray holds B. "A on the tray" is listed first but only works after B has moved.
        other = rng.choice(["bin", "box"])
        b = _tabletop(rng, ["tray", other])
        a, bb = b.new_obj(), b.new_obj("tray")
        objs = [a, bb]
        goals = [Goal(obj=a.id, location="tray"), Goal(obj=bb.id, location=other)]
        text = f"Put the {_name(a.id)} on the tray and the {_name(bb.id)} {PHRASE[other]}."
        keep = ("tray",)
    elif v == "three_items":
        # Y sits on the tray and belongs in the bin; Z for the tray is listed first.
        b = _tabletop(rng, ["tray", "bin"])
        x, y, z = b.new_obj(), b.new_obj("tray"), b.new_obj()
        objs = [x, y, z]
        goals = [Goal(obj=z.id, location="tray"), Goal(obj=x.id, location="bin"), Goal(obj=y.id, location="bin")]
        text = f"Put the {_name(z.id)} on the tray, and the {_name(x.id)} and the {_name(y.id)} in the bin."
        keep = ("tray",)
    else:
        # The source is full; A must leave it before C fits. C is listed first.
        src = rng.choice(["bin", "box"])
        b = _tabletop(rng, ["tray", src])
        a = b.new_obj(src)
        for _ in range(CAPACITY[src] - 1):
            b.new_obj(src)
        c = b.new_obj()
        objs = [a, c]
        goals = [Goal(obj=c.id, location=src), Goal(obj=a.id, location="tray")]
        text = f"Put the {_name(c.id)} in the {src}, and move the {_name(a.id)} from the {src} to the tray."
        keep = (src,)
    b.traps.append(v)
    _distract(b, objs, rng.randint(6, 8))
    _settle(b, goals, keep_full=keep)
    return b, goals, text, [], True


# ---------------- blocked: an obstacle the instruction does not mention ----------------
def _blocked(rng) -> Gen:
    v = rng.choices(["closed_lid", "occupied", "full", "two_obstacles"], weights=[2, 1, 2, 3])[0]
    sealed: List[str] = []
    keep: Tuple[str, ...] = ()
    if v == "closed_lid":
        target = rng.choice(["bin", "box"])
        b = _tabletop(rng, [target], closed=(target,))
        o = b.new_obj()
        objs, goals = [o], [Goal(obj=o.id, location=target)]
        text = f"Put the {_name(o.id)} {PHRASE[target]}."
        if rng.random() < 0.5:
            sealed = [target]
            b.traps.append("reclose")
            text = f"Put the {_name(o.id)} {PHRASE[target]} and close it again."
    elif v == "occupied":
        b = _tabletop(rng, ["tray"])
        o = b.new_obj()
        b.new_obj("tray")
        objs, goals, keep = [o], [Goal(obj=o.id, location="tray")], ("tray",)
        text = f"Put the {_name(o.id)} on the tray."
    elif v == "full":
        target = rng.choice(["bin", "box"])
        b = _tabletop(rng, [target])
        o = b.new_obj()
        for _ in range(CAPACITY[target]):
            b.new_obj(target)
        objs, goals, keep = [o], [Goal(obj=o.id, location=target)], (target,)
        text = f"Put the {_name(o.id)} {PHRASE[target]}."
    else:
        # A full open box AND a closed bin: two different obstacles in one task.
        b = _Builder(rng, {"tray": True, "bin": False, "box": True})
        o1, o2 = b.new_obj(), b.new_obj()
        for _ in range(CAPACITY["box"]):
            b.new_obj("box")
        objs = [o1, o2]
        goals = [Goal(obj=o1.id, location="box"), Goal(obj=o2.id, location="bin")]
        text = f"Put the {_name(o1.id)} in the box and the {_name(o2.id)} in the bin."
        keep = ("box",)
    b.traps.append(v)
    _distract(b, objs, rng.randint(6, 8))
    _settle(b, goals, keep_full=keep)
    return b, goals, text, sealed, True


# ---------------- impossible: the right answer is to refuse ----------------
def _impossible(rng) -> Gen:
    v = rng.choices(["missing_object", "missing_container", "over_capacity"], weights=[4, 2, 4])[0]
    if v == "missing_object":
        target = rng.choice(["tray", "bin", "box"])
        b = _tabletop(rng, [target])
        anchor = b.new_obj()
        b.lookalike(anchor)
        missing = b.absent_lookalike(anchor)
        req = [Goal(obj=missing, location=target)]
        text = f"Put the {_name(missing)} {PHRASE[target]}."
    elif v == "missing_container":
        absent = rng.choice(["tray", "bin", "box"])
        present = [k for k in ("tray", "bin", "box") if k != absent]
        b = _Builder(rng, {k: True for k in present if rng.random() < 0.7} or {present[0]: True})
        o = b.new_obj()
        b.lookalike(o)
        req = [Goal(obj=o.id, location=absent)]
        text = f"Put the {_name(o.id)} {PHRASE[absent]}."
    else:
        # More items than the destination can ever hold, even when empty.
        target = rng.choice(["tray", "box"])
        b = _tabletop(rng, [target])
        objs = [b.new_obj() for _ in range(CAPACITY[target] + 1)]
        b.lookalike(objs[0])
        req = [Goal(obj=o.id, location=target) for o in objs]
        text = f"Put the {', the '.join(_name(o.id) for o in objs[:-1])} and the {_name(objs[-1].id)} {PHRASE[target]}."
    b.fill(rng.randint(6, 8), spill=("bin",))
    b.traps.append(v)
    return b, req, text, [], False


# ---------------- packing: warehouse orders into shipping cartons ----------------
def _packing(rng) -> Gen:
    v = rng.choices(["basic", "repack_fragile", "wrong_item", "two_orders", "out_of_stock", "over_capacity"],
                    weights=[1, 3, 2, 3, 2, 1])[0]
    two = v == "two_orders"
    partial = v in ("repack_fragile", "wrong_item", "two_orders")
    b = _Builder(rng, {"carton_a": partial, **({"carton_b": False} if two else {}), "tote": True})

    def stock(kind: Optional[str] = None) -> Obj:
        return b.new_obj("tote" if rng.random() < 0.4 and b.room("tote") > 0 else TABLE, kind=kind)

    sturdy = lambda: rng.choice(["block", "ball"])
    if partial and v != "wrong_item":
        # A cup is already in open carton A and the order adds sturdy items: the cup must come out,
        # the sturdy items go in, then the cup goes back. No fixed packing order solves this.
        order_a = [b.new_obj("carton_a", kind="cup")] + [stock(sturdy()) for _ in range(rng.choice([1, 2]))]
        b.traps.append("repack_fragile")
    elif v == "wrong_item":
        b.new_obj("carton_a")  # a mis-pick that must be removed before sealing
        order_a = [stock("cup")] + [stock(sturdy()) for _ in range(rng.choice([1, 2]))]
    else:
        n = 4 if v == "over_capacity" else rng.choice([2, 3])
        order_a = [stock("cup")] + [stock(sturdy()) for _ in range(n - 1)]
    order_b = ([stock(sturdy())] + ([stock("cup")] if rng.random() < 0.6 else [])) if two else []

    for o in order_a + order_b:
        if b.lookalike(o, TABLE):
            break
    b.fill(len(b.objects) + rng.randint(1, 2), spill=("tote",))

    # Random listing order, so "the cup is listed first" is never a cue.
    ids_a = [o.id for o in order_a]
    ids_b = [o.id for o in order_b]
    rng.shuffle(ids_a)
    rng.shuffle(ids_b)
    possible = v not in ("out_of_stock", "over_capacity")
    if v == "out_of_stock":
        k = rng.randrange(len(ids_a))
        ids_a[k] = b.absent_lookalike(next(o for o in order_a if o.id == ids_a[k]))

    if partial and v != "wrong_item":
        a_clause = f"Carton A already holds part of order A. Finish order A ({_list(ids_a)})"
    elif v == "wrong_item":
        a_clause = f"Carton A must hold exactly order A ({_list(ids_a)}); fix it"
    else:
        a_clause = f"Pack order A ({_list(ids_a)}) into carton A"
    if two:
        text = f"{a_clause}. Pack order B ({_list(ids_b)}) into carton B. Seal both cartons."
    else:
        text = f"{a_clause}, then seal it."
    req = [Goal(obj=x, location="carton_a") for x in ids_a] + [Goal(obj=x, location="carton_b") for x in ids_b]
    b.traps.append(v)
    return b, req, text, ["carton_a"] + (["carton_b"] if two else []), possible


GEN: Dict[str, Callable] = {"easy": _easy, "multi_step": _multi, "blocked": _blocked, "impossible": _impossible,
                            "packing": _packing}


def _valid(task: Task) -> bool:
    from .solver import search
    ids = [o.id for o in task.scene.objects]
    if len(ids) != len(set(ids)) or len(ids) > len(TABLE_CELLS):
        return False
    if any(len(task.scene.contents(c.id)) > CAPACITY[c.kind] for c in task.scene.containers):
        return False
    if task.possible:
        return search(task.scene, task.goals, task.sealed) is not None
    return True


def generate(n_per_difficulty: int = 40, heldout_per_difficulty: Tuple[int, ...] = (10, 10, 10, 10, 10),
             seed: int = 7) -> List[Task]:
    rng = random.Random(seed)
    tasks: List[Task] = []
    for d, n_held in zip(DIFFICULTIES, heldout_per_difficulty):
        held = set(rng.sample(range(n_per_difficulty), n_held))
        i = 0
        while i < n_per_difficulty:
            b, req, text, sealed, possible = GEN[d](rng)
            task = Task(
                id=f"{d}-{i:03d}", split="heldout" if i in held else "tune", difficulty=d, instruction=text,
                scene=b.scene(), goals=req if possible else [], sealed=sealed if possible else [],
                requested=req, possible=possible, traps=sorted(set(b.traps)), layout=b.layout(),
            )
            if _valid(task):  # regenerate the rare inconsistent draw
                tasks.append(task)
                i += 1
    return tasks
