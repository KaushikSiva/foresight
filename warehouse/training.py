"""Fine-tuning examples: camera image to tote contents, and tote contents plus order to a checked plan.

Training scenes come from fresh seeds, and anything matching an eval scene is dropped.
"""
import json
import random

from .orders import generate, solve
from .planner import check_plan, plan_prompt, replan_feedback
from .rules import apply_call, new_state, parse_call

FIRST_SEED = 5000
REPAIR_SHARE = 0.12


def fingerprint(scene):
    return json.dumps([scene["totes"], scene["order"]], sort_keys=True)


def correct_steps(scene):
    state = new_state(scene["totes"])
    steps = []
    for line in solve(scene).split("\n"):
        name, args = parse_call(line)
        apply_call(state, name, args)
        if name == "hold":
            return [{"call": line}]
        steps.append({"call": line, "holding": state["holding"], "carton_kg": state["kg"]})
    return steps


def _broken_plan(rng, steps):
    """A plausible wrong first draft, so the model learns to repair a failed plan."""
    broken = [dict(s) for s in steps]
    if len(broken) < 3:
        return None
    kind = rng.choice(["drop_seal", "wrong_kg", "swap_layer"])
    if kind == "drop_seal":
        broken = broken[:-1]
    elif kind == "wrong_kg":
        i = rng.randrange(len(broken))
        broken[i]["carton_kg"] = round(broken[i]["carton_kg"] + 0.5, 2)
    else:
        places = [s for s in broken if s["call"].startswith("place")]
        if not places:
            return None
        s = rng.choice(places)
        s["call"] = s["call"].replace(", 1)", ", 2)") if ", 1)" in s["call"] else s["call"].replace(", 2)", ", 1)")
    return broken


def plan_example(scene, rng):
    steps = correct_steps(scene)
    feedback = ""
    if scene["fillable"] and rng.random() < REPAIR_SHARE:
        broken = _broken_plan(rng, steps)
        errors = check_plan(scene["totes"], scene["order"], broken, physics=False)[0] if broken else []
        if errors:
            feedback = replan_feedback(errors, broken)
    return {"prompt": plan_prompt(scene["totes"], scene["order"], feedback), "answer": json.dumps({"steps": steps})}


def vision_answer(scene):
    return json.dumps({"totes": dict(sorted(scene["totes"].items()))})


def build_scenes(count):
    seen = {fingerprint(s) for s in generate()}
    scenes, seed = [], FIRST_SEED
    while len(scenes) < count:
        for scene in generate(seed):
            key = fingerprint(scene)
            if key not in seen:
                seen.add(key)
                scene["id"] = f"train{seed}-{scene['id']}"
                scenes.append(scene)
        seed += 1
    random.Random(FIRST_SEED).shuffle(scenes)
    return scenes[:count]
