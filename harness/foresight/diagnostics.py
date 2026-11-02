"""Diagnostics computed from logged runs against the hidden true scene. No API calls.

Per run record:
  perception   object F1 (by id), location accuracy of matched objects, lid accuracy, exact match
  prediction   share of executed steps whose predicted state ("expect") is exactly what happened
  checker      would the FIRST plan have failed in the true scene, and did the checker flag it?
  efficiency   plan length / shortest plan (A* oracle), for successes on doable tasks

Pairwise (same tasks, two variants): wins/losses, exact McNemar p-value, unsafe actions
prevented, and extra cost per net prevented failure.
"""
import math
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from .llm import parse_json
from .schema import Plan, Task
from .sim import execute, score
from .solver import solve


@lru_cache(maxsize=1)
def _tasks() -> Dict[str, Task]:
    from .eval import load_tasks
    return {t.id: t for t in load_tasks()}


@lru_cache(maxsize=None)
def oracle_len(task_id: str) -> Optional[int]:
    t = _tasks()[task_id]
    return len(solve(t).steps) if t.possible else None


def _plan(text: str) -> Optional[Plan]:
    from .variants import to_plan
    return to_plan(parse_json(text))[0]


def perception(task: Task, perceived: Optional[dict]) -> Optional[dict]:
    if perceived is None:
        return None
    s = task.scene
    hidden = {c.id for c in s.containers if not c.is_open}
    visible = {o.id: o.location for o in s.objects if o.location not in hidden}
    seen = {o["id"]: o["location"] for o in perceived.get("objects", [])}
    tp = len(set(visible) & set(seen))
    prec = tp / len(seen) if seen else 0.0
    rec = tp / len(visible) if visible else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    loc = sum(seen[i] == visible[i] for i in set(visible) & set(seen)) / tp if tp else 0.0
    lids_true = {c.id: c.is_open for c in s.containers}
    lids_seen = {c["id"]: c["is_open"] for c in perceived.get("containers", [])}
    lid = sum(lids_seen.get(k) == v for k, v in lids_true.items()) / len(lids_true) if lids_true else 1.0
    exact = seen == visible and lids_seen == lids_true
    return {"f1": f1, "loc": loc, "lid": lid, "exact": exact}


def prediction(task: Task, plan: Optional[Plan]) -> Tuple[int, int]:
    """(steps whose prediction came true, steps with a prediction that executed) in the TRUE scene."""
    if plan is None:
        return 0, 0
    res = execute(task.scene, plan)
    before, ok, n = task.scene, 0, 0
    for step, after in zip(plan.steps, res.states):
        e = step.expect
        if e is not None:
            n += 1
            moved = {o.id: o.location for o in after.objects if before.obj(o.id).location != o.location}
            toggled = {c.id: c.is_open for c in after.containers if before.container(c.id).is_open != c.is_open}
            ok += e.holding == after.robot.holding and e.locations == moved and e.open == toggled
        before = after
    # The step that broke a rule never ran; it carried a prediction that it would succeed. That is a miss.
    if res.violation is not None and plan.steps[res.violation.step].expect is not None:
        n += 1
    return ok, n


def refusal_reason_ok(task: Task, plan: Optional[dict]) -> Optional[bool]:
    """For a refused impossible task: does the stated reason name what actually makes it impossible?
    The primary metric accepts any refusal; this catches right answers given for the wrong (or no) reason."""
    if task.possible or not plan:
        return None
    steps = [s for s in plan.get("steps", []) if s.get("skill") == "refuse"]
    if not steps:
        return None
    text = " ".join(steps[0].get("args") or []).lower().replace("_", " ")
    missing_items = [g.obj for g in task.requested if task.scene.obj(g.obj) is None]
    missing_places = [g.location for g in task.requested if g.location != "table" and task.scene.container(g.location) is None]
    if missing_items:
        return any(m.replace("_", " ") in text for m in missing_items)
    if missing_places:
        return any(p.replace("_", " ") in text for p in missing_places)
    return any(w in text for w in ("hold", "capacity", "full", "only", "room", "fit", "space", "too many"))


def enrich(r: dict) -> dict:
    """Extra per-record fields. Safe for every variant; fields that do not apply are None."""
    t = _tasks()[r["task_id"]]
    out = {"perception": perception(t, r.get("perceived")), "pred": (0, 0), "pred_first": (0, 0), "efficiency": None,
           "first_fails": None, "first_flagged": None,
           "reason_ok": refusal_reason_ok(t, r.get("final_plan")) if r["score"]["refused"] else None}
    # Imagination is judged on the model's last plan (the one it would have run), in the true scene.
    out["pred"] = prediction(t, _plan(r["attempts"][-1]["text"]) if r["attempts"] else None)
    # ...and on its first plan, before the checker corrected anything: the raw world model.
    out["pred_first"] = prediction(t, _plan(r["attempts"][0]["text"]) if r["attempts"] else None)
    if r["score"]["success"] and t.possible and r.get("final_plan"):
        out["efficiency"] = len(r["final_plan"]["steps"]) / oracle_len(t.id)
    if r.get("passed_check") is not None and r["attempts"]:
        first = _plan(r["attempts"][0]["text"])
        out["first_fails"] = not score(t, first)["success"]
        out["first_flagged"] = bool(r["attempts"][0]["errors"])
    return out


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def diagnose(rows: List[dict]) -> dict:
    ex = [(r, enrich(r)) for r in rows]
    per = [e["perception"] for _, e in ex if e["perception"]]
    ok = sum(e["pred"][0] for _, e in ex)
    n = sum(e["pred"][1] for _, e in ex)
    checked = [(r, e) for r, e in ex if e["first_fails"] is not None]
    fails = [(r, e) for r, e in checked if e["first_fails"]]
    passes = [(r, e) for r, e in checked if not e["first_fails"]]
    flagged = [(r, e) for r, e in checked if e["first_flagged"]]
    blind = [(r, e) for r, e in fails if not e["first_flagged"]]
    return {
        "perception_f1": _mean([p["f1"] for p in per]),
        "perception_loc": _mean([p["loc"] for p in per]),
        "lid_acc": _mean([p["lid"] for p in per]),
        "perception_exact": _mean([float(p["exact"]) for p in per]),
        "prediction_acc": ok / n if n else None,
        "prediction_steps": n,
        "prediction_acc_first": (sum(e["pred_first"][0] for _, e in ex) / sum(e["pred_first"][1] for _, e in ex))
        if sum(e["pred_first"][1] for _, e in ex) else None,
        "efficiency": _mean([e["efficiency"] for _, e in ex if e["efficiency"]]),
        "refusal_reason_ok": _mean([float(e["reason_ok"]) for _, e in ex if e["reason_ok"] is not None]),
        "checker": None if not checked else {
            "first_plan_fails": len(fails),
            "catch_rate": (len(fails) - len(blind)) / len(fails) if fails else None,
            "false_alarm_rate": sum(e["first_flagged"] for _, e in passes) / len(passes) if passes else None,
            "recovery_rate": sum(r["score"]["success"] for r, _ in flagged) / len(flagged) if flagged else None,
            "blind_spots": len(blind),
            "blind_refusals": sum(bool(r["attempts"]) and '"refuse"' in r["attempts"][0]["text"] for r, _ in blind),
        },
    }


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(base: List[dict], other: List[dict], cost) -> Optional[dict]:
    """Compare two variants on the tasks both ran. cost(row) -> $ or None."""
    a = {r["task_id"]: r for r in base}
    b = {r["task_id"]: r for r in other}
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    wins = sum(not a[i]["score"]["success"] and b[i]["score"]["success"] for i in ids)
    losses = sum(a[i]["score"]["success"] and not b[i]["score"]["success"] for i in ids)
    unsafe_prevented = sum(a[i]["score"]["unsafe"] and not b[i]["score"]["unsafe"] for i in ids)
    unsafe_added = sum(not a[i]["score"]["unsafe"] and b[i]["score"]["unsafe"] for i in ids)
    ca = [cost(a[i]) for i in ids]
    cb = [cost(b[i]) for i in ids]
    extra = None if None in ca or None in cb else sum(cb) - sum(ca)
    net = wins - losses
    return {
        "n": len(ids), "wins": wins, "losses": losses, "p_value": mcnemar_p(wins, losses),
        "unsafe_prevented": unsafe_prevented, "unsafe_added": unsafe_added,
        "extra_cost": extra,
        "cost_per_prevented_failure": (extra / net) if extra is not None and net > 0 else None,
        "cost_per_prevented_unsafe": (extra / (unsafe_prevented - unsafe_added))
        if extra is not None and unsafe_prevented > unsafe_added else None,
    }
