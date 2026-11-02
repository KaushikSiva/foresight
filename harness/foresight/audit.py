"""Difficulty audit: is each tier a good test of foresight?

  python -m foresight.audit

Per tier it reports how often three reference planners succeed in the true scene:
  oracle    A* search             must be 100% (every task is solvable, or correctly refused)
  template  a fixed recipe        should be low on hard tiers (no single recipe passes them)
  naive     the literal words     should be low on hard tiers (tasks need preparation)
plus plan length and how varied the traps are. The 9/10 bar is printed with each check.
"""
from collections import Counter, defaultdict
from typing import List

from .scenes import DIFFICULTIES
from .schema import Task
from .sim import score
from .solver import naive_plan, solve, template_plan

HARD = ("multi_step", "blocked", "packing")
BAR = {"oracle": 1.0, "template_max": 0.5, "naive_max": 0.2, "lookalike_min": 0.9, "min_trap_types": 3}


def audit(tasks: List[Task]) -> dict:
    by = defaultdict(list)
    for t in tasks:
        by[t.difficulty].append(t)
    out = {}
    for d in DIFFICULTIES:
        ts = by[d]
        rate = lambda planner: sum(score(t, planner(t))["success"] for t in ts) / len(ts)
        lengths = [len(solve(t).steps) for t in ts if t.possible]
        traps = Counter(x for t in ts for x in t.traps if x != "lookalike")
        out[d] = {
            "n": len(ts), "possible": sum(t.possible for t in ts),
            "oracle": rate(solve), "template": rate(template_plan), "naive": rate(naive_plan),
            "steps_mean": sum(lengths) / len(lengths) if lengths else 0, "steps_max": max(lengths, default=0),
            "lookalike": sum("lookalike" in t.traps for t in ts) / len(ts),
            "traps": dict(traps),
        }
    return out


def verdict(d: str, r: dict) -> List[str]:
    fails = []
    if r["oracle"] < BAR["oracle"]:
        fails.append("oracle below 100%")
    if r["lookalike"] < BAR["lookalike_min"]:
        fails.append("too few lookalike distractors")
    if d in HARD:
        if r["template"] > BAR["template_max"]:
            fails.append("a fixed recipe solves too many")
        if r["naive"] > BAR["naive_max"]:
            fails.append("the literal plan solves too many")
        if len(r["traps"]) < BAR["min_trap_types"]:
            fails.append("too few trap types")
    return fails


def main() -> None:
    from .eval import load_tasks
    res = audit(load_tasks())
    print(f"{'tier':12} {'n':>3} {'ok?':>4} {'oracle':>7} {'template':>9} {'naive':>6} {'steps':>9} {'lookalike':>9}  traps")
    for d, r in res.items():
        f = verdict(d, r)
        print(f"{d:12} {r['n']:>3} {'yes' if not f else 'NO':>4} {r['oracle']:>7.0%} {r['template']:>9.0%} {r['naive']:>6.0%} "
              f"{r['steps_mean']:>5.1f}/{r['steps_max']:<3} {r['lookalike']:>9.0%}  "
              + ", ".join(f"{k} {v}" for k, v in sorted(r["traps"].items())))
        for x in f:
            print(f"{'':12} -> {x}")
    print(f"\nBar for hard tiers ({', '.join(HARD)}): template <= {BAR['template_max']:.0%}, naive <= {BAR['naive_max']:.0%}, "
          f">= {BAR['min_trap_types']} trap types. All tiers: oracle 100%, lookalike >= {BAR['lookalike_min']:.0%}.")
    print("Easy is a calibration tier: the recipe and the literal plan are expected to pass it.")


if __name__ == "__main__":
    main()
