"""Build the extreme stress test: fixed before any model sees it, and packable orders are proven solvable in physics."""
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import imageio.v3 as iio

from warehouse.catalog import CATALOG, TOTES
from warehouse.orders import solve, solve_order
from warehouse.robot import perception_image
from warehouse.rules import fillable, parse_program, run
from warehouse.scoring import score

OUT = ROOT / "data" / "warehouse" / "extreme"
SEED = 20261102
PACKABLE, SHORT = 24, 6
LOOKALIKES = [("paint_can", "ceramic_mug"), ("tissue_box", "phone_box")]


def _flags(order):
    skus = [CATALOG[s] for s in order]
    return {"fragile": sum(q for s, q in order.items() if CATALOG[s].fragile),
            "heavy": sum(q for s, q in order.items() if CATALOG[s].heavy),
            "liquid": any(s.liquid for s in skus), "electronics": any(s.electronics for s in skus),
            "kg": sum(CATALOG[s].kg * q for s, q in order.items()), "units": sum(order.values())}


def _naive_breaks_rules(totes, order):
    """A listed-order, first-free-slot plan must break a rule, so the order cannot be packed without thinking."""
    tote_of = {c["sku"]: t for t, c in totes.items()}
    columns = {s: 0 for s in "ABCD"}
    lines = ['open_carton("L")']
    for sku, qty in sorted(order.items()):
        for _ in range(qty):
            slot = min(columns, key=lambda s: (columns[s], s))
            columns[slot] += 1
            lines += [f'pick("{tote_of[sku]}")', f'place("{slot}", {columns[slot]})']
    lines.append("seal()")
    result = run(totes, parse_program("\n".join(lines)))
    return bool(result.violations or result.halted)


def _candidate(rng):
    a, b = rng.choice(LOOKALIKES)
    names = [a, b]
    others = [s for s in CATALOG if s not in names]
    rng.shuffle(others)
    names += others[:4]
    rng.shuffle(names)
    totes = {t: {"sku": s, "count": rng.randint(2, 3)} for t, s in zip(rng.sample(sorted(TOTES), 6), names)}
    order, left = {}, rng.randint(7, 8)
    pool = list(names)
    rng.shuffle(pool)
    for sku in pool:
        if left <= 0:
            break
        stock = next(t["count"] for t in totes.values() if t["sku"] == sku)
        take = min(left, rng.randint(1, stock))
        order[sku] = take
        left -= take
    return totes, order, a, b


def _is_extreme(totes, order, a, b):
    flags = _flags(order)
    return (flags["units"] >= 7 and flags["fragile"] >= 2 and flags["heavy"] >= 2 and flags["liquid"] and flags["electronics"]
            and flags["kg"] >= 7.0 and (a in order or b in order))


def generate():
    rng = random.Random(SEED)
    scenes = []
    while sum(s["fillable"] for s in scenes) < PACKABLE * 2:  # spares, in case physics rejects some
        totes, order, a, b = _candidate(rng)
        if _is_extreme(totes, order, a, b) and solve_order(order)[0] and _naive_breaks_rules(totes, order):
            scenes.append({"id": f"extreme-{len(scenes):02d}", "tier": "extreme", "split": "extreme", "totes": totes, "order": order,
                           "fillable": True, "seed": rng.randrange(2**31), "best_carton": solve_order(order)[0]})
    shorts = []
    while len(shorts) < SHORT:
        totes, order, a, b = _candidate(rng)
        if _is_extreme(totes, order, a, b) and solve_order(order)[0]:
            sku = rng.choice(sorted(order))
            stock = next(t["count"] for t in totes.values() if t["sku"] == sku)
            order[sku] = stock + 1  # exactly one unit short, in an otherwise packable-looking order
            if not fillable(totes, order):
                shorts.append({"id": f"extreme-short-{len(shorts):02d}", "tier": "extreme_short", "split": "extreme", "totes": totes, "order": order,
                               "fillable": False, "seed": rng.randrange(2**31), "best_carton": None})
    return scenes, shorts


def _check(scene):
    return scene["id"], score(scene, solve(scene))["success"]


def _render(scene):
    iio.imwrite(OUT / "images" / f"{scene['id']}.jpg", perception_image(scene["totes"], scene["seed"]), quality=88)


def main():
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    packable, shorts = generate()
    with ProcessPoolExecutor(8) as pool:
        passed = dict(pool.map(_check, packable))
        kept = [s for s in packable if passed[s["id"]]][:PACKABLE]
        rejected = sum(not ok for ok in passed.values())
        scenes = kept + shorts
        list(pool.map(_render, scenes))
    with open(OUT / "scenes.jsonl", "w") as f:
        for scene in scenes:
            f.write(json.dumps(scene) + "\n")
    units = [sum(s["order"].values()) for s in kept]
    print(f"{len(kept)} packable + {len(shorts)} short; candidates failing solver physics, dropped: {rejected} of {len(packable)}; units per packable order {min(units)}-{max(units)}")
    print("cartons", {c: sum(s["best_carton"] == c for s in kept) for c in ("S", "L")})


if __name__ == "__main__":
    main()
