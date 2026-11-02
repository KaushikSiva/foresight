"""Build fine-tuning examples from generated scenes. Eval scenes are never used."""
import json
import random

from .agent import check_plan, plan_prompt, replan_feedback
from .scenes import generate, solve
from .sim import execute, parse_program

REPLAN_SHARE = 0.15
FIRST_SEED = 1000


def fingerprint(scene):
    return json.dumps([scene["state"], scene["goal"]], sort_keys=True)


def _diff(before, after):
    """What a single step changed, in the format the planner is asked to imagine."""
    expect = {}
    if before["holding"] != after["holding"]:
        expect["holding"] = after["holding"]
    moved = {o: v["on"] for o, v in after["objects"].items() if v["on"] != before["objects"][o]["on"]}
    if moved:
        expect["on"] = moved
    opened = {r: v["open"] for r, v in after["receptacles"].items() if v["open"] != before["receptacles"][r]["open"]}
    if opened:
        expect["open"] = opened
    return expect


def correct_steps(scene):
    lines = solve(scene).split("\n")
    calls = [parse_program(line)[0] for line in lines]
    if calls[0][0] == "refuse":
        return [{"call": lines[0], "expect": {}}]
    trace = execute(scene["state"], calls).trace
    before = [scene["state"]] + trace[:-1]
    return [{"call": line, "expect": _diff(b, a)} for line, b, a in zip(lines, before, trace)]


def _broken(rng, steps):
    """A plausible wrong first draft, for teaching the model to repair a failed plan."""
    broken = [dict(step) for step in steps]
    moves = [i for i, step in enumerate(broken) if step["call"].startswith("move_to")]
    if moves and rng.random() < 0.6:
        del broken[rng.choice(moves)]
    else:
        changed = [i for i, step in enumerate(broken) if "holding" in step["expect"]]
        if not changed:
            return None
        i = rng.choice(changed)
        broken[i] = {"call": broken[i]["call"], "expect": {"holding": "wrong_object"}}
    return broken


def example(scene, rng):
    steps = correct_steps(scene)
    state, goal = scene["state"], scene["goal"]
    feedback = ""
    if scene["possible"] and rng.random() < REPLAN_SHARE:
        broken = _broken(rng, steps)
        errors = check_plan(state, goal, broken)[0] if broken else []
        if errors:
            feedback = replan_feedback(errors, broken)
    return {"messages": [
        {"role": "user", "content": plan_prompt(state, goal, feedback)},
        {"role": "assistant", "content": json.dumps({"steps": steps})},
    ]}


def build(train_size=1000, validation_size=100):
    """Unique scenes from fresh seeds, with anything matching an eval scene dropped."""
    seen = {fingerprint(s) for s in generate()}
    rng = random.Random(FIRST_SEED)
    scenes, seed = [], FIRST_SEED
    while len(scenes) < train_size + validation_size:
        for scene in generate(seed):
            key = fingerprint(scene)
            if key not in seen:
                seen.add(key)
                scenes.append(scene)
        seed += 1
    rng.shuffle(scenes)
    scenes = scenes[:train_size + validation_size]
    examples = [example(s, rng) for s in scenes]
    return examples[:train_size], examples[train_size:], scenes
