"""Seeded order generator, plus a solver that proves each order can be packed under the rules."""
import random

from .catalog import CARTONS, CATALOG, MAX_LAYERS, TOTES
from .rules import MAX_STACK_M, column_height, fillable

TIERS = ["easy", "mixed", "hard", "short"]
HELD_OUT_PER_TIER = [13, 13, 12, 12]
SCENES_PER_TIER = 50


def _can_place(column, sku_name, kg, carton):
    sku = CATALOG[sku_name]
    if len(column) >= MAX_LAYERS or kg + sku.kg > carton["max_kg"] + 1e-9:
        return False
    if column and (sku.heavy or any(CATALOG[s].fragile for s in column)):
        return False
    if sku.liquid and any(CATALOG[s].electronics for s in column):
        return False
    return column_height(column) + sku.height <= MAX_STACK_M + 1e-9


def _assign(units, carton):
    """Backtracking search. Heavy and sturdy items go first, so fragile ones land on top."""
    slots = list(carton["slots"])
    columns = {s: [] for s in slots}
    placements = []

    def search(i, kg):
        if i == len(units):
            return True
        tried = set()
        for slot in slots:
            key = tuple(columns[slot])
            if key in tried or not _can_place(columns[slot], units[i], kg, carton):
                continue
            tried.add(key)
            columns[slot].append(units[i])
            placements.append((units[i], slot, len(columns[slot])))
            if search(i + 1, kg + CATALOG[units[i]].kg):
                return True
            columns[slot].pop()
            placements.pop()
        return False

    return placements if search(0, 0.0) else None


def solve_order(order):
    """The cheapest carton that fits, with a placement for every unit."""
    units = sorted((s for s, q in order.items() for _ in range(q)),
                   key=lambda s: (CATALOG[s].fragile, -CATALOG[s].kg))
    for name in sorted(CARTONS, key=lambda c: CARTONS[c]["cost_usd"]):
        placements = _assign(units, CARTONS[name])
        if placements:
            return name, placements
    return None, None


def solve(scene):
    """Reference program. Used to validate the eval and build training data, never shown to a planner on eval scenes."""
    if not fillable(scene["totes"], scene["order"]):
        return 'hold("insufficient stock")'
    carton, placements = solve_order(scene["order"])
    tote_of = {c["sku"]: t for t, c in scene["totes"].items() if c}
    lines = [f'open_carton("{carton}")']
    for sku, slot, layer in placements:
        lines += [f'pick("{tote_of[sku]}")', f'place("{slot}", {layer})']
    lines.append("seal()")
    return "\n".join(lines)


def _totes(rng, must_have=()):
    names = list(must_have)
    others = [s for s in CATALOG if s not in names]
    rng.shuffle(others)
    names += others[:max(0, rng.randint(5, 6) - len(names))]
    rng.shuffle(names)
    tote_ids = rng.sample(sorted(TOTES), len(names))
    return {t: {"sku": s, "count": rng.randint(1, 3)} for t, s in zip(tote_ids, names)}


def _order_from(rng, totes, total):
    pool = [t for t in totes.values()]
    rng.shuffle(pool)
    order = {}
    left = total
    for tote in pool:
        if left <= 0:
            break
        take = min(left, rng.randint(1, tote["count"]))
        order[tote["sku"]] = take
        left -= take
    return order, total - left


def _flags(order):
    skus = [CATALOG[s] for s in order]
    return {"fragile": any(s.fragile for s in skus), "heavy": any(s.heavy for s in skus),
            "liquid": any(s.liquid for s in skus), "electronics": any(s.electronics for s in skus),
            "kg": sum(CATALOG[s].kg * q for s, q in order.items())}


def _easy(rng):
    while True:
        totes = _totes(rng)
        order, _ = _order_from(rng, totes, rng.randint(1, 2))
        carton, _ = solve_order(order)
        if carton == "S" and not _flags(order)["heavy"]:
            return totes, order


def _mixed(rng):
    while True:
        totes = _totes(rng)
        order, total = _order_from(rng, totes, rng.randint(3, 4))
        flags = _flags(order)
        if total >= 3 and solve_order(order)[0] and (flags["fragile"] or flags["heavy"]):
            return totes, order


def _hard(rng):
    while True:
        totes = _totes(rng)
        for tote in totes.values():
            tote["count"] = rng.randint(2, 3)
        order, total = _order_from(rng, totes, rng.randint(5, 7))
        flags = _flags(order)
        traps = sum([flags["fragile"], flags["heavy"], flags["liquid"] and flags["electronics"], flags["kg"] > 4.0])
        if total >= 5 and traps >= 2 and solve_order(order)[0]:
            return totes, order


def _short(rng):
    while True:
        totes = _totes(rng)
        order, _ = _order_from(rng, totes, rng.randint(2, 4))
        if rng.random() < 0.5:
            sku = rng.choice(sorted(order))
            stock = next(t["count"] for t in totes.values() if t["sku"] == sku)
            order[sku] = stock + rng.randint(1, 2)
        else:
            absent = [s for s in CATALOG if not any(t["sku"] == s for t in totes.values())]
            order[rng.choice(absent)] = 1
        if not fillable(totes, order):
            return totes, order


GENERATORS = {"easy": _easy, "mixed": _mixed, "hard": _hard, "short": _short}


def generate(seed=11):
    rng = random.Random(seed)
    scenes = []
    for tier, held_out in zip(TIERS, HELD_OUT_PER_TIER):
        held = set(rng.sample(range(SCENES_PER_TIER), held_out))
        for i in range(SCENES_PER_TIER):
            totes, order = GENERATORS[tier](rng)
            scene = {"id": f"{tier}-{i:02d}", "tier": tier, "split": "held_out" if i in held else "tuning",
                     "totes": totes, "order": order, "fillable": fillable(totes, order), "seed": rng.randrange(2**31)}
            scene["best_carton"] = solve_order(order)[0] if scene["fillable"] else None
            scenes.append(scene)
    return scenes


def order_text(order):
    return ", ".join(f"{qty} x {sku}" for sku, qty in sorted(order.items()))
