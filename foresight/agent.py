"""Perception plus the planner variants. None of them ever sees the scene's true state."""
import json
import os
import re

from .llm import PERCEPTION_MODEL, PLANNER_MODEL, image_part
from .scenes import OBJECTS, RECEPTACLES
from .sim import TABLE, ParseError, execute, goal_met, parse_program

MAX_REPLANS = 2

LEGEND = (
    "The image is a top-down camera view of a table. Squares are blocks, circles are balls, trapezoids are cups. "
    f"Object names: {', '.join(OBJECTS)}. Receptacles are labeled: tray (surface, holds 1), bin (holds 3), box (holds 2). "
    "A bin or box with dark planks and a metal handle is closed; one showing its interior is open."
)
SKILLS = (
    'Skills: move_to("x"), pick("obj"), place("obj", "target"), open("container"), close("container"), refuse("reason").\n'
    "Rules: move_to an object before pick. move_to the target before place, open or close. Hold one object at a time. "
    "The gripper must be empty to open or close. Open a closed container before placing into it. "
    'A full receptacle must be emptied first; place("obj", "table") is allowed. '
    "If a requested object or receptacle is not in the scene, the whole program must be one refuse call."
)
PERCEIVE_PROMPT = (
    "{legend}\nReport only what is visible, using the exact names above. Include only receptacles and objects that are in the image. "
    "Objects sit on table, tray, bin or box. Copy the goal from the instruction even if an item is missing from the image.\n"
    'Reply with JSON only: {{"objects": {{"name": {{"on": "table"}}}}, "receptacles": {{"name": {{"open": true}}}}, '
    '"goal": [["object", "target"]]}}.\nInstruction: {instruction}'
)
BASELINE_PROMPT = (
    "Write a robot program for this instruction, one skill call per line, with no explanation.\n"
    "Scene: {perceived}\n{skills}\nInstruction: {instruction}"
)
CONSTRAINED_PROMPT = (
    'Scene: {perceived}\n{skills}\nExample for "Put the red block in the closed bin":\n'
    '{{"program": ["move_to(\\"bin\\")", "open(\\"bin\\")", "move_to(\\"red_block\\")", "pick(\\"red_block\\")", '
    '"move_to(\\"bin\\")", "place(\\"red_block\\", \\"bin\\")"]}}\n'
    'Reply with JSON only: {{"program": [one skill call per string]}}.\nInstruction: {instruction}'
)
PLAN_PROMPT = (
    "Scene you perceived: {perceived}\nGoal: {goal}\n{skills}\n"
    "For each step, imagine the result in expect, listing only what that step changes. Use {{}} if nothing changes.\n"
    'Reply with JSON only, in exactly this format: {{"steps": ['
    '{{"call": "move_to(\\"bin\\")", "expect": {{}}}}, '
    '{{"call": "open(\\"bin\\")", "expect": {{"open": {{"bin": true}}}}}}, '
    '{{"call": "move_to(\\"cup\\")", "expect": {{}}}}, '
    '{{"call": "pick(\\"cup\\")", "expect": {{"holding": "cup"}}}}, '
    '{{"call": "move_to(\\"bin\\")", "expect": {{}}}}, '
    '{{"call": "place(\\"cup\\", \\"bin\\")", "expect": {{"holding": null, "on": {{"cup": "bin"}}}}}}'
    ']}}.{feedback}'
)


class ModelOutputError(ValueError):
    pass


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ModelOutputError("no JSON object in reply")
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        raise ModelOutputError(f"invalid JSON: {e.msg}") from e


def _strip_fences(text):
    return re.sub(r"^```\w*\n?|```\s*$", "", text.strip(), flags=re.M).strip()


def normalize_perception(raw):
    """Turn a perception reply into a sim state. Physical specs are public rules; contents come from the model."""
    if not isinstance(raw, dict):
        raise ModelOutputError("perception must be an object")
    receptacles = {}
    for name, value in (raw.get("receptacles") or {}).items():
        if name not in RECEPTACLES:
            raise ModelOutputError(f"unknown receptacle {name}")
        spec = dict(RECEPTACLES[name])
        spec["open"] = spec["kind"] == "surface" or bool((value or {}).get("open", True))
        receptacles[name] = spec
    objects = {}
    for name, value in (raw.get("objects") or {}).items():
        if name not in OBJECTS:
            raise ModelOutputError(f"unknown object {name}")
        location = (value or {}).get("on", TABLE)
        if location != TABLE and location not in receptacles:
            raise ModelOutputError(f"{name} is on unseen receptacle {location}")
        objects[name] = {"on": location}
    goal = raw.get("goal") or []
    if not goal or not all(isinstance(g, list) and len(g) == 2 for g in goal):
        raise ModelOutputError("goal must be a list of [object, target]")
    return {"objects": objects, "receptacles": receptacles, "holding": None, "at": None}, goal


def perceive(client, instruction, image, model=PERCEPTION_MODEL):
    """Read the image once. Every planner variant then shares this same output."""
    reply = client.chat(model, [{"role": "user", "content": [
        image_part(image),
        {"type": "text", "text": PERCEIVE_PROMPT.format(legend=LEGEND, instruction=instruction)},
    ]}], json_mode=True)
    call = {k: reply[k] for k in ("model", "prompt_tokens", "completion_tokens", "latency_s")}
    try:
        perceived, goal = normalize_perception(extract_json(reply["text"]))
        return {"perceived": perceived, "goal": goal, "error": None, "call": call}
    except ModelOutputError as e:
        return {"perceived": None, "goal": None, "error": str(e), "call": call}


def plan_prompt(perceived, goal, feedback=""):
    return PLAN_PROMPT.format(perceived=json.dumps(perceived), goal=json.dumps(goal), skills=SKILLS, feedback=feedback)


def replan_feedback(errors, steps):
    return ("\nYour previous plan failed these checks. Fix them:\n" + "\n".join(errors)
            + f"\nPrevious plan: {json.dumps(steps)}")


def check_plan(perceived, goal, steps):
    """Check imagined steps against the perceived scene. Returns (errors, predictions_right, predictions_total)."""
    calls = []
    try:
        for step in steps:
            parsed = parse_program(step["call"])
            if len(parsed) != 1:
                raise ParseError("each step must hold exactly one skill call")
            calls.extend(parsed)
    except (ParseError, KeyError, TypeError) as e:
        return [f"bad program: {e}"], 0, 0

    missing = [item for pair in goal for item in pair
               if item != TABLE and item not in perceived["objects"] and item not in perceived["receptacles"]]
    refused = any(name == "refuse" for name, _ in calls)
    if refused:
        if len(calls) > 1:
            return ["refuse must be the only step"], 0, 0
        return ([] if missing else ["refused, but every requested item is in the scene"]), 0, 0
    if missing:
        return [f"{', '.join(missing)} not in the scene; the program must be one refuse call"], 0, 0

    result = execute(perceived, calls)
    errors, right, total = [], 0, 0
    if result.violation:
        v = result.violation
        errors.append(f"step {v.step + 1} {calls[v.step][0]}{calls[v.step][1]} broke a rule: {v.message}")
    for i, state in enumerate(result.trace):
        expect = steps[i].get("expect") or {}
        checks = []
        if "holding" in expect:
            checks.append((f"holding {expect['holding']}", expect["holding"], state["holding"]))
        for obj, location in (expect.get("on") or {}).items():
            checks.append((f"{obj} on {location}", location, state["objects"].get(obj, {}).get("on")))
        for name, is_open in (expect.get("open") or {}).items():
            checks.append((f"{name} open={is_open}", is_open, state["receptacles"].get(name, {}).get("open")))
        for label, predicted, actual in checks:
            total += 1
            if predicted == actual:
                right += 1
            else:
                errors.append(f"step {i + 1} predicted {label}, but the simulation gives {actual}")
    if not result.violation and not goal_met(result.state, goal):
        errors.append("the program ends without reaching the goal with an empty gripper")
    return errors, right, total


class Run:
    def __init__(self, client):
        self.client = client
        self.calls = []

    def ask(self, model, prompt, json_mode=False):
        reply = self.client.chat(model, [{"role": "user", "content": prompt}], json_mode=json_mode)
        self.calls.append({k: reply[k] for k in ("model", "prompt_tokens", "completion_tokens", "latency_s")})
        return reply["text"]

    def record(self, **fields):
        return {"calls": self.calls, **fields}


def baseline(client, instruction, perception, model=None):
    run = Run(client)
    text = run.ask(model or PLANNER_MODEL, BASELINE_PROMPT.format(
        perceived=json.dumps(perception["perceived"]), skills=SKILLS, instruction=instruction))
    return run.record(status="planned", program=_strip_fences(text))


def constrained(client, instruction, perception, model=None):
    run = Run(client)
    text = run.ask(model or PLANNER_MODEL, CONSTRAINED_PROMPT.format(
        perceived=json.dumps(perception["perceived"]), skills=SKILLS, instruction=instruction), json_mode=True)
    try:
        return run.record(status="planned", program="\n".join(extract_json(text)["program"]))
    except (ModelOutputError, KeyError, TypeError) as e:
        return run.record(status="planned", program="", output_error=str(e))


def foresight(client, instruction, perception, model=None):
    """Imagine, check and replan. Returns a program only if it passes the check."""
    run = Run(client)
    perceived, goal = perception["perceived"], perception["goal"]
    history, feedback = [], ""
    for _ in range(MAX_REPLANS + 1):
        text = run.ask(model or PLANNER_MODEL, plan_prompt(perceived, goal, feedback), json_mode=True)
        try:
            steps = extract_json(text)["steps"]
            errors, right, total = check_plan(perceived, goal, steps)
        except (ModelOutputError, KeyError, TypeError, AttributeError):
            steps, errors, right, total = [], ['unreadable plan: use the exact format, such as "on": {"cup": "bin"} and "open": {"bin": true}'], 0, 0
        history.append({"steps": steps, "errors": errors, "predictions_right": right, "predictions_total": total})
        if not errors:
            return run.record(status="planned", program="\n".join(s["call"] for s in steps),
                              perceived=perceived, goal=goal, history=history)
        feedback = replan_feedback(errors, steps)
    return run.record(status="withheld", program=None, perceived=perceived, goal=goal, history=history)


def tuned(client, instruction, perception, model=None):
    """Foresight on the fine-tuned planner. Refuses to fall back silently to the base model."""
    model = model or os.environ.get("FORESIGHT_TUNED_MODEL")
    if not model:
        raise RuntimeError("FORESIGHT_TUNED_MODEL is not set")
    return foresight(client, instruction, perception, model=model)


VARIANTS = {"A_baseline": baseline, "B_constrained": constrained, "C_foresight": foresight, "D_tuned": tuned}
