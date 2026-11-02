"""Prompts and JSON schemas. The fine-tuning export uses these same builders, so
the fine-tuned model sees exactly the prompts it is evaluated with."""
import base64
import json
from typing import List

from . import config
from .scenes import COLORS, KINDS, generate
from .schema import CAPACITY, SKILLS, Plan, Scene, Task, step_dict
from .solver import solve


def refusal_reason(t: Task) -> str:
    """A specific reason for an impossible task, so examples and fine-tune targets never teach a vague refusal."""
    for g in t.requested:
        if t.scene.obj(g.obj) is None:
            return f"{g.obj} is not on the bench"
        if g.location != "table" and t.scene.container(g.location) is None:
            return f"there is no {g.location} here"
    loc = t.requested[0].location
    c = t.scene.container(loc)
    return f"the {loc} holds only {CAPACITY[c.kind]} items" if c else "the task cannot be done under the rules"

LOOK = {
    "3d": ("Kinds: block (cube), ball (sphere), cup (mug with a handle).",
           "A closed bin has a dark grey lid with handles; a closed box or carton has its flaps folded shut and taped. "
           "Each container's name is printed on the white plate in front of it."),
    "2d": ("Kinds: block (square), ball (shaded circle), cup (ring with a small handle).",
           "A closed bin, box or carton has a grey striped lid marked CLOSED."),
}[config.RENDER]

RULES = f"""You control a one-armed robot at a packing bench, seen from directly above. "table" means the bench top.
Places: "table", plus containers. A container's id is its printed label in lower case with spaces as underscores ("CARTON A" is "carton_a").
  tray: holds 1, no lid. bin: holds 3. box: holds 2. tote (warehouse stock): holds 4, no lid. carton (shipping): holds 3.
  Bins, boxes and cartons can be closed. {LOOK[1]} You cannot see inside a closed container.
Objects have ids "<color>_<kind>". Colors: {", ".join(COLORS)}. {LOOK[0]}
Skills. Each step is {{"skill": name, "args": [...]}}:
  move_to(x)          x is an object id, "table", or a container id
  pick(obj)           you must be at obj, or at the place holding it, and the gripper must be empty
  place(obj, target)  you must be holding obj and be at target
  open(c), close(c)   you must be at c and the gripper must be empty
  refuse(reason)      when the task names something that is not there, or the rules make it impossible
Rules: the gripper holds one object at a time. Nothing goes into or out of a closed container. A full container takes nothing more. Cups are fragile: putting anything that is not a cup into a carton that already holds a cup crushes it, so pack sturdy items first. A packed order must hold exactly its items and its carton must be sealed (closed) at the end. If any ordered item is out of stock, refuse the whole order. The first broken rule stops the robot.
You may move other items out of the way (for example onto the table) to open a lid, clear the tray or make room. A closed, full or occupied container is an obstacle to clear, not a reason to refuse. Refuse only when something named is not there or the rules make the task impossible, and say exactly which."""

_STEP = {"type": "object", "properties": {"skill": {"type": "string", "enum": list(SKILLS)},
                                          "args": {"type": "array", "items": {"type": "string"}}},
         "required": ["skill", "args"]}
PLAN_SCHEMA = {"type": "object", "properties": {"steps": {"type": "array", "items": _STEP}}, "required": ["steps"]}

_EXPECT = {"type": "object", "properties": {
    "holding": {"type": ["string", "null"]},
    "locations": {"type": "object", "additionalProperties": {"type": "string"}},
    "open": {"type": "object", "additionalProperties": {"type": "boolean"}}}, "required": ["holding"]}
FORESIGHT_SCHEMA = {"type": "object", "properties": {
    "goal": {"type": "array", "items": {"type": "object", "properties": {"obj": {"type": "string"}, "location": {"type": "string"}},
                                          "required": ["obj", "location"]}},
    "sealed": {"type": "array", "items": {"type": "string"}},
    "steps": {"type": "array", "items": {**_STEP, "properties": {**_STEP["properties"], "expect": _EXPECT}}}},
    "required": ["goal", "steps"]}

SCENE_SCHEMA = {"type": "object", "properties": {
    "objects": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"}, "color": {"type": "string", "enum": COLORS}, "kind": {"type": "string", "enum": KINDS},
        "location": {"type": "string"}}, "required": ["id", "color", "kind", "location"]}},
    "containers": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"}, "kind": {"type": "string", "enum": ["tray", "bin", "box", "tote", "carton"]},
        "is_open": {"type": "boolean"}},
        "required": ["id", "kind", "is_open"]}}},
    "required": ["objects", "containers"]}


def image_part(img: bytes) -> dict:
    mime = "image/jpeg" if img[:2] == b"\xff\xd8" else "image/png"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(img).decode()}}


def scene_json(scene: Scene) -> str:
    return json.dumps({"objects": [o.model_dump() for o in scene.objects],
                       "containers": [c.model_dump() for c in scene.containers]}, separators=(",", ":"))


def plan_json(plan: Plan, with_expect: bool, goal=None, sealed=(), with_goal=None) -> str:
    with_goal = with_expect if with_goal is None else with_goal
    steps = [step_dict(s) if with_expect else {"skill": s.skill, "args": s.args} for s in plan.steps]
    out = ({"goal": [g.model_dump() for g in goal or []], "sealed": list(sealed), "steps": steps}
           if with_goal else {"steps": steps})
    return json.dumps(out, separators=(",", ":"))


# Few-shot examples come from a separate seed, so they never overlap the eval set.
def _examples() -> List[Task]:
    pool = generate(n_per_difficulty=8, heldout_per_difficulty=(0, 0, 0, 0, 0), seed=12345)
    blocked = next((t for t in pool if t.difficulty == "blocked" and len(t.goals) == 2),
                   next(t for t in pool if t.difficulty == "blocked"))
    missing = next(t for t in pool if t.difficulty == "impossible")
    packing = next((t for t in pool if t.difficulty == "packing" and t.possible and
                    any(t.scene.obj(g.obj).kind == "cup" for g in t.goals) and len(t.goals) > 1),
                   next(t for t in pool if t.difficulty == "packing" and t.possible))
    return [blocked, packing, missing]


EXAMPLES = _examples()


def _shots(with_expect: bool, with_goal=None) -> str:
    parts = []
    for t in EXAMPLES:
        parts.append(f"Scene: {scene_json(t.scene)}\nTask: {t.instruction}\nOutput: "
                     + plan_json(solve(t, refusal_reason(t)), with_expect, t.goals, t.sealed, with_goal))
    return "\n\n".join(parts)


# --- Variant A: plain prompt, image in, free-form plan out -------------------------------
def direct_plain(png: bytes, instruction: str) -> list:
    return [{"role": "user", "content": [image_part(png), {"type": "text", "text":
            f"{RULES}\n\nTask: {instruction}\n\nReply with a JSON object {{\"steps\": [...]}} listing the steps."}]}]


# --- Variant B: structured output + few-shot ------------------------------------------------
def direct_constrained(png: bytes, instruction: str) -> list:
    return [
        {"role": "system", "content": RULES + "\n\nExamples (scenes written as JSON for brevity):\n\n" + _shots(False)},
        {"role": "user", "content": [image_part(png), {"type": "text", "text":
            f"Task: {instruction}\nOutput only the JSON plan."}]},
    ]


# --- Variant C: perceive, then imagine + plan ------------------------------------------------
def perceive(png: bytes) -> list:
    return [
        {"role": "system", "content": RULES},
        {"role": "user", "content": [image_part(png), {"type": "text", "text":
            "List everything in the image as JSON. For each visible object give id, color, kind and location "
            "(\"table\" or the container it sits in). For each container give id (its label in lower case), kind and is_open. "
            "Do not guess the contents of closed containers. Output only JSON."}]},
    ]


PLAN_INSTRUCTIONS = (
    "Write the plan as JSON. First state the goal as a list of {obj, location}, and in \"sealed\" list every "
    "container that must end closed (packed cartons, or a lid the task says to close). Then, for every step, "
    "imagine the world after it and write that in \"expect\": holding (the object in the gripper, or null), "
    "locations (objects that moved and where they are now) and open (containers whose lid changed). "
    "If the task names something that is not in the scene, the plan is a single refuse step with an empty goal."
)


def plan_from_scene(scene: Scene, instruction: str) -> list:
    return [
        {"role": "system", "content": RULES + "\n\n" + PLAN_INSTRUCTIONS + "\n\nExamples:\n\n" + _shots(True)},
        {"role": "user", "content": f"Scene: {scene_json(scene)}\nTask: {instruction}\nOutput:"},
    ]


# --- Variant F (ablation): the same check-and-replan loop, but no imagined states ----------
CHECK_ONLY_INSTRUCTIONS = (
    "Write the plan as JSON. First state the goal as a list of {obj, location}, and in \"sealed\" list every "
    "container that must end closed (packed cartons, or a lid the task says to close). Then list the steps. "
    "If the task names something that is not in the scene, the plan is a single refuse step with an empty goal."
)
CHECK_ONLY_SCHEMA = {**FORESIGHT_SCHEMA, "properties": {**FORESIGHT_SCHEMA["properties"],
                                                        "steps": {"type": "array", "items": _STEP}}}


def plan_check_only(scene: Scene, instruction: str) -> list:
    return [
        {"role": "system", "content": RULES + "\n\n" + CHECK_ONLY_INSTRUCTIONS + "\n\nExamples:\n\n"
         + _shots(False, with_goal=True)},
        {"role": "user", "content": f"Scene: {scene_json(scene)}\nTask: {instruction}\nOutput:"},
    ]


def fix_perception(issues: List[str]) -> dict:
    return {"role": "user", "content": "That scene JSON has problems:\n- " + "\n- ".join(issues)
            + "\nObject ids are <color>_<kind>, each listed once; containers go only in \"containers\". "
              "Output the corrected JSON only."}


def replan(errors: List[str]) -> dict:
    return {"role": "user", "content": "A checker ran your plan in a simulator of the scene above and found:\n- "
            + "\n- ".join(errors) + "\nWrite the full corrected plan. Output only JSON."}
