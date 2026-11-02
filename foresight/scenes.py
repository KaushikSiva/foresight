"""Seeded scene generator, plus a scripted solver that proves every scene is scored correctly."""
import random

from .sim import TABLE

OBJECTS = ["red_block", "blue_block", "green_block", "yellow_ball", "purple_ball", "white_cup", "black_cup"]
RECEPTACLES = {
    "tray": {"kind": "surface", "open": True, "capacity": 1},
    "bin": {"kind": "container", "open": True, "capacity": 3},
    "box": {"kind": "container", "open": True, "capacity": 2},
}
TIERS = ["easy", "multi_step", "blocked", "impossible"]
HELD_OUT_PER_TIER = [13, 13, 12, 12]
SCENES_PER_TIER = 50


def _say(name):
    return name.replace("_", " ")


def _instruction(goal):
    parts = [f"the {_say(o)} {'on' if t == 'tray' else 'in'} the {t}" for o, t in goal]
    return "Put " + " and ".join(parts) + "."


def _base(rng, closed=()):
    names = rng.sample(sorted(RECEPTACLES), 2) if not closed else sorted(
        set(closed) | set(rng.sample(sorted(RECEPTACLES), 2)))
    receptacles = {n: dict(RECEPTACLES[n]) for n in names}
    for n in closed:
        receptacles[n]["open"] = False
    objects = {o: {"on": TABLE} for o in rng.sample(OBJECTS, rng.randint(3, 5))}
    # Closed receptacles start empty so no object is hidden from the camera.
    for o in objects:
        spots = [n for n, r in receptacles.items() if r["kind"] == "container" and r["open"]]
        if spots and rng.random() < 0.15:
            target = rng.choice(spots)
            if not any(v["on"] == target for v in objects.values()):
                objects[o]["on"] = target
    return {"objects": objects, "receptacles": receptacles, "holding": None, "at": None}


def _free(state, target):
    used = sum(1 for o in state["objects"].values() if o["on"] == target)
    return state["receptacles"][target]["capacity"] - used


def _pick_goals(rng, state, count, exclude=()):
    goals, free = [], {t: _free(state, t) for t in state["receptacles"]}
    movable = [o for o in state["objects"] if o not in exclude]
    rng.shuffle(movable)
    for obj in movable:
        if len(goals) == count:
            break
        here = state["objects"][obj]["on"]
        targets = [t for t, n in free.items() if n > 0 and t != here]
        if targets:
            target = rng.choice(sorted(targets))
            free[target] -= 1
            goals.append((obj, target))
    return goals


def _easy(rng):
    while True:
        state = _base(rng)
        goals = _pick_goals(rng, state, 1)
        if goals:
            return state, goals, True


def _multi(rng):
    while True:
        state = _base(rng)
        goals = _pick_goals(rng, state, rng.choice([2, 2, 3]))
        if len(goals) >= 2:
            return state, goals, True


def _blocked(rng):
    while True:
        if rng.random() < 0.5:
            closed = rng.choice(["bin", "box"])
            state = _base(rng, closed=(closed,))
            movable = [o for o, v in state["objects"].items() if v["on"] != closed]
            if movable:
                return state, [(rng.choice(movable), closed)], True
        else:
            state = _base(rng)
            if "tray" not in state["receptacles"]:
                state["receptacles"]["tray"] = dict(RECEPTACLES["tray"])
            names = sorted(state["objects"])
            occupant, obj = rng.sample(names, 2)
            state["objects"][occupant]["on"] = "tray"
            if state["objects"][obj]["on"] != "tray":
                return state, [(obj, "tray")], True


def _impossible(rng):
    state = _base(rng)
    absent_objects = [o for o in OBJECTS if o not in state["objects"]]
    absent_targets = [t for t in RECEPTACLES if t not in state["receptacles"]]
    if absent_targets and rng.random() < 0.5:
        return state, [(rng.choice(sorted(state["objects"])), rng.choice(absent_targets))], False
    return state, [(rng.choice(absent_objects), rng.choice(sorted(state["receptacles"])))], False


GENERATORS = {"easy": _easy, "multi_step": _multi, "blocked": _blocked, "impossible": _impossible}


def generate(seed=7):
    rng = random.Random(seed)
    scenes = []
    for tier, held_out in zip(TIERS, HELD_OUT_PER_TIER):
        held = set(rng.sample(range(SCENES_PER_TIER), held_out))
        for i in range(SCENES_PER_TIER):
            state, goal, possible = GENERATORS[tier](rng)
            scenes.append({
                "id": f"{tier}-{i:02d}",
                "tier": tier,
                "split": "held_out" if i in held else "tuning",
                "instruction": _instruction(goal),
                "state": state,
                "goal": [list(g) for g in goal],
                "possible": possible,
                "render_seed": rng.randrange(2**31),
            })
    return scenes


def solve(scene):
    """Ground-truth solver. Used only to validate the eval, never by a model variant."""
    if not scene["possible"]:
        return 'refuse("requested item is not in the scene")'
    state = scene["state"]
    lines = []

    def carry(obj, target):
        lines.extend([f'move_to("{obj}")', f'pick("{obj}")', f'move_to("{target}")', f'place("{obj}", "{target}")'])

    for obj, target in scene["goal"]:
        receptacle = state["receptacles"][target]
        if not receptacle["open"]:
            lines.extend([f'move_to("{target}")', f'open("{target}")'])
        for other, value in state["objects"].items():
            if value["on"] == target and other != obj and receptacle["capacity"] == 1:
                carry(other, TABLE)
        carry(obj, target)
    return "\n".join(lines)
