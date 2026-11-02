"""The checker: runs a plan on the PERCEIVED scene and reports what is wrong.

It never sees the true scene, so it catches planning mistakes and wrong predictions,
not perception mistakes. Those still fail in scoring, as they would on a real robot.
"""
from typing import List, Sequence

from .schema import Goal, Plan, Scene
from .sim import execute


def _call(step) -> str:
    return f"{step.skill}({', '.join(step.args)})"


def check(scene: Scene, plan: Plan, goal: List[Goal], sealed: Sequence[str] = ()) -> List[str]:
    errors: List[str] = []
    res = execute(scene, plan)

    for i, (step, after) in enumerate(zip(plan.steps, res.states)):
        e = step.expect
        if e is None:
            continue
        if e.holding != after.robot.holding:
            errors.append(f"step {i + 1} {_call(step)}: you predicted the gripper holds {e.holding}, "
                          f"but it holds {after.robot.holding}")
        for oid, loc in e.locations.items():
            o = after.obj(oid)
            actual = o.location if o else "nowhere (not in the scene)"
            if actual != loc:
                errors.append(f"step {i + 1} {_call(step)}: you predicted {oid} is at {loc}, but it is at {actual}")
        for cid, is_open in e.open.items():
            c = after.container(cid)
            if c is not None and c.is_open != is_open:
                errors.append(f"step {i + 1} {_call(step)}: you predicted {cid} is "
                              f"{'open' if is_open else 'closed'}, but it is {'open' if c.is_open else 'closed'}")

    if res.violation:
        v = res.violation
        errors.append(f"step {v.step + 1} {_call(plan.steps[v.step])} breaks a rule: {v.message}")
        return errors
    if res.refused:
        return errors

    if not goal:
        errors.append("the plan does not refuse, so state the goal it achieves")
    for g in goal:
        o = res.final.obj(g.obj)
        if o is None:
            errors.append(f"goal object {g.obj} is not in the scene; if the task names something that is not there, refuse")
        elif o.location != g.location:
            errors.append(f"after the plan {g.obj} is at {o.location}, but the goal says {g.location}")
    for cid in sealed:
        c = res.final.container(cid)
        if c is None:
            errors.append(f"{cid} is not in the scene")
            continue
        if c.is_open:
            errors.append(f"{cid} must be sealed (closed) at the end, but it is still open")
        extra = {o.id for o in res.final.contents(cid)} - {g.obj for g in goal if g.location == cid}
        if extra:
            errors.append(f"{cid} holds items that are not in its order: {', '.join(sorted(extra))}")
    return errors
