"""Planners over the TRUE scene. None of them is ever used by a scored variant.

  search / solve   A* oracle: proves solvability, gives shortest plans for few-shot and fine-tuning
  template_plan    a fixed recipe, used by the audit to show no single recipe solves a tier
  naive_plan       the literal instruction, used by the audit to show tasks need preparation
The mock LLM reuses these, but mock runs are flagged and never reported.
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


# ------------------------------------------------------------------------------------------
# Search oracle. A* over macro actions (move_to + pick/place/open/close) using the simulator's
# own rules, so "solvable" means solvable under exactly the rules that score the model.
# ------------------------------------------------------------------------------------------
import heapq
from itertools import count
from typing import Optional


def _key(s: Scene):
    return (tuple(o.location for o in s.objects), tuple(c.is_open for c in s.containers), s.robot.holding)


def _done(s: Scene, goals: List[Goal], sealed: Sequence[str]) -> bool:
    if s.robot.holding is not None:
        return False
    if any((o := s.obj(g.obj)) is None or o.location != g.location for g in goals):
        return False
    for cid in sealed:
        c = s.container(cid)
        if c is None or c.is_open or {o.id for o in s.contents(cid)} != {g.obj for g in goals if g.location == cid}:
            return False
    return True


def _h(s: Scene, goals: List[Goal], sealed: Sequence[str]) -> int:
    n = 0
    for g in goals:
        o = s.obj(g.obj)
        if o is None:
            return 10 ** 6
        if o.location != g.location:
            n += 1 if s.robot.holding == g.obj else 2
    return n + sum(1 for cid in sealed if s.container(cid) and s.container(cid).is_open)


def search(scene: Scene, goals: List[Goal], sealed: Sequence[str] = (), budget: int = 60000) -> Optional[List[Step]]:
    """Shortest macro plan reaching the goals, or None if none exists within budget."""
    if any(scene.obj(g.obj) is None or (g.location != TABLE and scene.container(g.location) is None) for g in goals):
        return None
    targets = {g.location for g in goals} | set(sealed)
    movable = {g.obj for g in goals} | {o.id for o in scene.objects if o.location in targets}
    places = [TABLE] + [c.id for c in scene.containers]
    lids = [c.id for c in scene.containers if c.kind in ("bin", "box", "carton")]

    def macros(s: Scene):
        h = s.robot.holding
        if h is None:
            for oid in movable:
                yield [Step(skill="move_to", args=[oid]), Step(skill="pick", args=[oid])]
            for cid in lids:
                yield [Step(skill="move_to", args=[cid]), Step(skill="open" if not s.container(cid).is_open else "close", args=[cid])]
        else:
            for p in places:
                if s.obj(h).location != p:
                    yield [Step(skill="move_to", args=[p]), Step(skill="place", args=[h, p])]

    start = scene.model_copy(deep=True)
    start.robot.at = None
    tie = count()
    frontier = [(_h(start, goals, sealed), 0, next(tie), start, [])]
    seen = {_key(start): 0}
    while frontier and budget > 0:
        _, g, _, s, path = heapq.heappop(frontier)
        if _done(s, goals, sealed):
            return path
        budget -= 1
        for m in macros(s):
            try:
                t = s
                for st in m:
                    t = apply_step(t, st)
            except RuleError:
                continue
            k = _key(t)
            if seen.get(k, 1 << 30) <= g + 1:
                continue
            seen[k] = g + 1
            heapq.heappush(frontier, (g + 1 + _h(t, goals, sealed), g + 1, next(tie), t, path + m))
    return None


def solve(task: Task, reason: str = "") -> Plan:
    if not task.possible:
        return Plan(steps=[Step(skill="refuse", args=[reason or "the task cannot be done as asked"])])
    steps = search(task.scene, task.goals, task.sealed)
    if steps is None:
        raise ValueError(f"{task.id}: no plan found")
    return Plan(steps=fill_expects(task.scene, steps))


def template_plan(task: Task) -> Plan:
    """A fixed recipe: refuse if something named is missing, else open lids, clear the tray,
    pack sturdy before fragile, seal. Measures whether a tier can be solved without lookahead."""
    req = task.requested
    s = task.scene
    if any(s.obj(g.obj) is None or (g.location != TABLE and s.container(g.location) is None) for g in req):
        return Plan(steps=[Step(skill="refuse", args=["something named is not here"])])
    return Plan(steps=plan_steps(s, req, prereqs=True, sealed=task.sealed))


def naive_plan(task: Task) -> Plan:
    """Exactly what the words say, in the listed order, with no preparation."""
    req = task.requested
    s = task.scene
    if any(s.obj(g.obj) is None or (g.location != TABLE and s.container(g.location) is None) for g in req):
        return Plan(steps=[Step(skill="refuse", args=["something named is not here"])])
    return Plan(steps=plan_steps(s, req, prereqs=False, sealed=task.sealed))
