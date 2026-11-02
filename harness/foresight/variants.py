"""The five variants. Each returns one log record per task, graded in the hidden true scene.

A  Baseline     image -> plan, plain prompt
B  Constrained  image -> plan, JSON schema + few-shot
C  Foresight    image -> scene JSON -> plan with predicted states -> check -> replan (<= MAX_REPLANS)
D  Routed       Foresight on the small model (1 replan); escalate to the large model if the check still fails
E  Fine-tuned   Foresight on the fine-tuned model
"""
from typing import List, Optional, Tuple

from pydantic import ValidationError

from . import config, prompts
from .llm import Call, parse_json
from .render import image_bytes
from .schema import Goal, Plan, Scene, Step, Task, step_dict
from .sim import score
from .validator import check

VARIANTS = {"A": "Baseline", "B": "Constrained", "C": "Foresight", "D": "Routed", "E": "Fine-tuned"}


def models(mock: bool) -> dict:
    if mock:
        return {"main": "mock-main", "small": "mock-small", "large": "mock-main", "ft": "mock-ft"}
    m = {"main": config.MODEL, "small": config.MODEL_SMALL, "large": config.MODEL_LARGE, "ft": config.MODEL_FT}
    return m


def to_plan(obj) -> Tuple[Optional[Plan], List[Goal], List[str]]:
    """Returns (plan, declared goal, declared sealed containers)."""
    if isinstance(obj, list):
        obj = {"steps": obj}
    if not isinstance(obj, dict):
        return None, [], []
    try:
        plan = Plan.model_validate({"steps": obj.get("steps", [])})
        goal = [Goal.model_validate(g) for g in obj.get("goal") or []]
        sealed = [str(c) for c in obj.get("sealed") or []]
    except (ValidationError, TypeError):
        return None, [], []
    return (plan if plan.steps else None), goal, sealed


def to_scene(obj) -> Optional[Scene]:
    if not isinstance(obj, dict):
        return None
    try:
        return Scene.model_validate({"objects": obj.get("objects", []), "containers": obj.get("containers", [])})
    except ValidationError:
        return None


async def _direct(llm, model: str, task: Task, constrained: bool) -> dict:
    png = image_bytes(task)
    if constrained:
        call = await llm.chat(model, prompts.direct_constrained(png, task.instruction), "direct_constrained",
                              schema=prompts.PLAN_SCHEMA)
    else:
        call = await llm.chat(model, prompts.direct_plain(png, task.instruction), "direct_plain")
    plan, _, _ = to_plan(parse_json(call.text))
    return {"calls": [call], "attempts": [{"model": model, "text": call.text, "errors": []}], "final_plan": plan,
            "perceived": None, "passed_check": None}


async def foresight(llm, model: str, task: Task, max_replans: int, fail_closed: bool) -> dict:
    calls: List[Call] = []
    call = await llm.chat(model, prompts.perceive(image_bytes(task)), "perceive", schema=prompts.SCENE_SCHEMA)
    calls.append(call)
    scene = to_scene(parse_json(call.text))
    out = {"calls": calls, "attempts": [], "perceived": scene, "final_plan": None, "passed_check": False,
           "fail_closed": False}
    if scene is None:
        out["attempts"].append({"model": model, "text": call.text, "errors": ["perception output was not a valid scene"]})
        return out

    messages = prompts.plan_from_scene(scene, task.instruction)
    plan, errors = None, []
    for attempt in range(1 + max_replans):
        call = await llm.chat(model, messages, "plan" if attempt == 0 else "replan", schema=prompts.FORESIGHT_SCHEMA)
        calls.append(call)
        plan, goal, sealed = to_plan(parse_json(call.text))
        errors = ["the output was not a valid JSON plan"] if plan is None else check(scene, plan, goal, sealed)
        out["attempts"].append({"model": model, "text": call.text, "errors": errors})
        if not errors:
            break
        messages = messages + [{"role": "assistant", "content": call.text}, prompts.replan(errors)]

    out["passed_check"] = not errors
    out["final_plan"] = plan
    if errors and fail_closed:
        out["final_plan"] = Plan(steps=[Step(skill="refuse", args=["no plan passed the safety check"])])
        out["fail_closed"] = True
    return out


async def run_task(variant: str, task: Task, llm) -> dict:
    m = models(llm.mock)
    escalated = False
    if variant == "A":
        r = await _direct(llm, m["main"], task, constrained=False)
    elif variant == "B":
        r = await _direct(llm, m["main"], task, constrained=True)
    elif variant == "C":
        r = await foresight(llm, m["main"], task, config.MAX_REPLANS, config.FAIL_CLOSED)
    elif variant == "D":
        r = await foresight(llm, m["small"], task, 1, fail_closed=False)
        if not r["passed_check"]:
            escalated = True
            big = await foresight(llm, m["large"], task, config.MAX_REPLANS, config.FAIL_CLOSED)
            big["calls"] = r["calls"] + big["calls"]
            big["attempts"] = r["attempts"] + big["attempts"]
            r = big
    elif variant == "E":
        if not m["ft"]:
            raise RuntimeError("FORESIGHT_MODEL_FT is not set")
        r = await foresight(llm, m["ft"], task, config.MAX_REPLANS, config.FAIL_CLOSED)
    else:
        raise ValueError(variant)

    calls: List[Call] = r["calls"]
    plan: Optional[Plan] = r["final_plan"]
    return {
        "variant": variant, "task_id": task.id, "split": task.split, "difficulty": task.difficulty,
        "possible": task.possible, "instruction": task.instruction, "mock": llm.mock,
        "perceived": r["perceived"].model_dump() if r["perceived"] else None,
        "attempts": r["attempts"],
        "final_plan": {"steps": [step_dict(s) for s in plan.steps]} if plan else None,
        "passed_check": r["passed_check"], "fail_closed": r.get("fail_closed", False), "escalated": escalated,
        "score": score(task, plan),
        "calls": [c.model_dump(exclude={"text"}) for c in calls],
        "api_errors": [c.error for c in calls if c.error],
        "latency_s": round(sum(c.latency_s for c in calls), 3),  # calls run sequentially within a task
    }
