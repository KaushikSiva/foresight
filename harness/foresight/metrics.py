"""Aggregate run logs into the comparison table. Costs come from logged tokens x prices.json."""
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from . import config
from .diagnostics import diagnose, paired


def load_runs(runs_dir: Path = config.RUNS) -> List[dict]:
    rows = []
    for p in sorted(runs_dir.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append({**json.loads(line), "run": p.stem})
    return rows


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    i = (len(v) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def call_cost(call: dict, prices: dict) -> Optional[float]:
    p = prices.get(call["model"])
    if p is None:
        return None
    return (call["prompt_tokens"] * p[0] + call["completion_tokens"] * p[1]) / 1e6


def summarize(rows: List[dict]) -> dict:
    prices = config.prices()
    n = len(rows)
    ok = sum(r["score"]["success"] for r in rows)
    possible = [r for r in rows if r["possible"]]
    impossible = [r for r in rows if not r["possible"]]
    costs = [call_cost(c, prices) for r in rows for c in r["calls"]]
    total_cost = None if (not costs or any(c is None for c in costs)) else sum(costs)
    by_diff = defaultdict(list)
    for r in rows:
        by_diff[r["difficulty"]].append(r["score"]["success"])
    lo, hi = wilson(ok, n)
    lat = [r["latency_s"] for r in rows]
    return {
        "n": n,
        "success": ok / n if n else 0, "success_ci": [lo, hi],
        "invalid": sum(r["score"]["invalid"] for r in rows) / n if n else 0,
        "unsafe": sum(r["score"]["unsafe"] for r in rows) / n if n else 0,
        "invalid_step": sum(r["score"].get("invalid_step", False) for r in rows) / n if n else 0,
        "correct_refusal": (sum(r["score"]["refused"] for r in impossible) / len(impossible)) if impossible else None,
        "false_refusal": (sum(r["score"]["refused"] for r in possible) / len(possible)) if possible else None,
        "p50_latency_s": pct(lat, 0.5), "p95_latency_s": pct(lat, 0.95),
        "calls_per_task": sum(len(r["calls"]) for r in rows) / n if n else 0,
        "replan_rate": sum(sum(1 for c in r["calls"] if c["purpose"] == "replan") > 0 for r in rows) / n if n else 0,
        "escalation_rate": sum(r.get("escalated", False) for r in rows) / n if n else 0,
        "fail_closed_rate": sum(r.get("fail_closed", False) for r in rows) / n if n else 0,
        "api_error_tasks": sum(bool(r.get("api_errors")) for r in rows),
        "total_cost": total_cost,
        "cost_per_success": (total_cost / ok) if (total_cost is not None and ok) else None,
        "tokens": sum(c["prompt_tokens"] + c["completion_tokens"] for r in rows for c in r["calls"]),
        "by_difficulty": {d: sum(v) / len(v) for d, v in sorted(by_diff.items())},
        "mock": any(r["mock"] for r in rows),
        "manifest": sorted({r.get("manifest") for r in rows if r.get("manifest")}),
        **diagnose(rows),
    }


def table(rows: Optional[List[dict]] = None) -> Dict[str, Dict[str, dict]]:
    """{split: {variant: summary}}. The latest run file per (split, variant, mock) wins."""
    rows = load_runs() if rows is None else rows
    latest: Dict[tuple, str] = {}
    for r in rows:
        latest[(r["split"], r["variant"], r["mock"])] = r["run"]
    groups = defaultdict(list)
    for r in rows:
        if latest[(r["split"], r["variant"], r["mock"])] == r["run"]:
            groups[(r["split"], r["variant"] + ("*" if r["mock"] else ""))].append(r)
    out: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for (split, variant), rs in sorted(groups.items()):
        out[split][variant] = summarize(rs)
    return out


def _latest(rows: List[dict]) -> Dict[tuple, List[dict]]:
    latest: Dict[tuple, str] = {}
    for r in rows:
        latest[(r["split"], r["variant"], r["mock"])] = r["run"]
    groups = defaultdict(list)
    for r in rows:
        if latest[(r["split"], r["variant"], r["mock"])] == r["run"]:
            groups[(r["split"], r["variant"], r["mock"])].append(r)
    return groups


def compare(rows: Optional[List[dict]] = None, base: str = "A") -> Dict[str, Dict[str, dict]]:
    """{split: {variant: paired stats vs base}} on the tasks both ran. Real runs only, never mock."""
    rows = load_runs() if rows is None else rows
    groups = _latest([r for r in rows if not r["mock"]])
    prices = config.prices()
    cost = lambda r: (None if any(call_cost(c, prices) is None for c in r["calls"])
                      else sum(call_cost(c, prices) for c in r["calls"]))
    out: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for (split, v, _), rs in sorted(groups.items()):
        if v != base and (split, base, False) in groups:
            p = paired(groups[(split, base, False)], rs, cost)
            if p:
                out[split][v] = p
    return out


def fmt(t: Dict[str, Dict[str, dict]]) -> str:
    f = lambda x, p=True: "-" if x is None else (f"{100 * x:5.1f}%" if p else f"{x:6.2f}")
    money = lambda x: "-" if x is None else f"${x:.4f}"
    lines = []
    for split, vs in t.items():
        lines.append(f"\n== {split} ==   (* = mock model, pipeline test only)")
        lines.append(f"{'var':4} {'n':>4} {'success':>8} {'95% CI':>15} {'invalid':>8} {'badstep':>8} {'unsafe':>7} {'refuse✓':>8} "
                     f"{'refuse✗':>8} {'p50 s':>6} {'p95 s':>6} {'$/success':>10}")
        for v, s in vs.items():
            ci = f"{100 * s['success_ci'][0]:.0f}-{100 * s['success_ci'][1]:.0f}%"
            lines.append(f"{v:4} {s['n']:>4} {f(s['success']):>8} {ci:>15} {f(s['invalid']):>8} {f(s['invalid_step']):>8} {f(s['unsafe']):>7} "
                         f"{f(s['correct_refusal']):>8} {f(s['false_refusal']):>8} {f(s['p50_latency_s'], False):>6} "
                         f"{f(s['p95_latency_s'], False):>6} {money(s['cost_per_success']):>10}")
    return "\n".join(lines)


def fmt_diagnostics(t: Dict[str, Dict[str, dict]]) -> str:
    f = lambda x: "-" if x is None else f"{100 * x:5.1f}%"
    lines = []
    for split, vs in t.items():
        rows = [(v, s) for v, s in vs.items() if not s["mock"]]
        if not rows:
            continue
        lines.append(f"\n== {split}: diagnostics (true scene) ==")
        lines.append(f"{'var':4} {'percF1':>7} {'lids':>7} {'scene ok':>9} {'pred 1st':>8} {'pred end':>8} {'steps/opt':>9} {'reason ok':>9} "
                     f"{'1st fails':>9} {'caught':>7} {'falseAl':>8} {'recover':>8} {'blind':>6} {'blindRef':>8}")
        for v, s in rows:
            c = s["checker"] or {}
            eff = "-" if s["efficiency"] is None else f"{s['efficiency']:.2f}x"
            lines.append(f"{v:4} {f(s['perception_f1']):>7} {f(s['lid_acc']):>7} {f(s['perception_exact']):>9} "
                         f"{f(s['prediction_acc_first']):>8} {f(s['prediction_acc']):>8} {eff:>9} {f(s['refusal_reason_ok']):>9} {c.get('first_plan_fails', '-'):>9} {f(c.get('catch_rate')):>7} "
                         f"{f(c.get('false_alarm_rate')):>8} {f(c.get('recovery_rate')):>8} {c.get('blind_spots', '-'):>6} "
                         f"{c.get('blind_refusals', '-'):>8}")
    return "\n".join(lines)


def fmt_compare(c: Dict[str, Dict[str, dict]], base: str = "A") -> str:
    money = lambda x: "-" if x is None else f"${x:.4f}"
    lines = []
    for split, vs in c.items():
        lines.append(f"\n== {split}: paired vs {base} (same tasks) ==")
        lines.append(f"{'var':4} {'n':>4} {'wins':>5} {'losses':>7} {'McNemar p':>10} {'unsafe-':>8} {'unsafe+':>8} "
                     f"{'extra $':>9} {'$/prevented fail':>17} {'$/prevented unsafe':>19}")
        for v, p in vs.items():
            lines.append(f"{v:4} {p['n']:>4} {p['wins']:>5} {p['losses']:>7} {p['p_value']:>10.4f} {p['unsafe_prevented']:>8} "
                         f"{p['unsafe_added']:>8} {money(p['extra_cost']):>9} {money(p['cost_per_prevented_failure']):>17} "
                         f"{money(p['cost_per_prevented_unsafe']):>19}")
    return "\n".join(lines)


def report() -> str:
    rows = load_runs()
    return fmt(table(rows)) + "\n" + fmt_diagnostics(table(rows)) + "\n" + fmt_compare(compare(rows))


if __name__ == "__main__":
    print(report())
