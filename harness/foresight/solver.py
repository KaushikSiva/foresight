"""Oracle planner.

On the TRUE scene it proves every task is solvable and produces fine-tuning
targets for the tuning split. It is never used by a scored variant (the mock
LLM reuses it, but mock runs are flagged and never reported).
"""
from typing import List, Sequence

from .schema import FRAGILE, TABLE, Expect, Goal, Plan, Scene, Step, Task
from .sim import RuleError, apply_step


def fill_expects(scene: Scene, steps: List[Step]) -> List[Step]:
    """Attach the true after-state to each step, stopping at the first rule break."""
    out = []
    for st in steps:
        if st.skill == "refuse":
            out.append(st)
            continue
        try:
            after = apply_step(scene, st)
        except RuleError:
            out.extend(steps[len(out):])
            break
        moved = {o.id: o.location for o in after.objects if scene.obj(o.id).location != o.location}
        toggled = {c.id: c.is_open for c in after.containers if scene.container(c.id).is_open != c.is_open}
        out.append(st.model_copy(update={"expect": Expect(holding=after.robot.holding, locations=moved, open=toggled)}))
        scene = after
    return out


def plan_steps(scene: Scene, goals: List[Goal], prereqs: bool = True, sealed: Sequence[str] = ()) -> List[Step]:
    """Pick-and-place steps for each goal, then close every sealed container once it is packed.

    prereqs=False is the naive plan: no lids opened, no tray cleared, items packed in the order
    the task lists them (so a fragile cup may go in first and be crushed)."""
    steps: List[Step] = []
    S = lambda skill, *a: steps.append(Step(skill=skill, args=list(a)))
    goal_objs = {g.obj for g in goals}
    if prereqs:  # group by destination; within a carton, sturdy items before fragile ones
        dest = list(dict.fromkeys(g.location for g in goals))
        kind = lambda oid: (scene.obj(oid).kind if scene.obj(oid) else "")
        goals = sorted(goals, key=lambda g: (dest.index(g.location), kind(g.obj) in FRAGILE))
    opened = set()
    for i, g in enumerate(goals):
        c = scene.container(g.location)
        if prereqs and c is not None and c.kind == "tray":
            for occ in scene.contents(c.id):
                if occ.id not in goal_objs:
                    S("move_to", occ.id); S("pick", occ.id); S("move_to", TABLE); S("place", occ.id, TABLE)
        if prereqs and c is not None and not c.is_open and c.id not in opened:
            S("move_to", c.id); S("open", c.id)
            opened.add(c.id)
        S("move_to", g.obj); S("pick", g.obj); S("move_to", g.location); S("place", g.obj, g.location)
        last_for_dest = all(h.location != g.location for h in goals[i + 1:])
        if g.location in sealed and last_for_dest:
            S("close", g.location)
    return steps


def solve(task: Task, reason: str = "") -> Plan:
    if not task.possible:
        return Plan(steps=[Step(skill="refuse", args=[reason or "the task refers to something that is not in the scene"])])
    return Plan(steps=fill_expects(task.scene, plan_steps(task.scene, task.goals, sealed=task.sealed)))
