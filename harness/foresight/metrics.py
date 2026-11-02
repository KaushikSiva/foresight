"""Aggregate run logs into the comparison table. Costs come from logged tokens x prices.json."""
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from . import config


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


def fmt(t: Dict[str, Dict[str, dict]]) -> str:
    f = lambda x, p=True: "-" if x is None else (f"{100 * x:5.1f}%" if p else f"{x:6.2f}")
    money = lambda x: "-" if x is None else f"${x:.4f}"
    lines = []
    for split, vs in t.items():
        lines.append(f"\n== {split} ==   (* = mock model, pipeline test only)")
        lines.append(f"{'var':4} {'n':>4} {'success':>8} {'95% CI':>15} {'invalid':>8} {'unsafe':>7} {'refuse✓':>8} "
                     f"{'refuse✗':>8} {'p50 s':>6} {'p95 s':>6} {'$/success':>10}")
        for v, s in vs.items():
            ci = f"{100 * s['success_ci'][0]:.0f}-{100 * s['success_ci'][1]:.0f}%"
            lines.append(f"{v:4} {s['n']:>4} {f(s['success']):>8} {ci:>15} {f(s['invalid']):>8} {f(s['unsafe']):>7} "
                         f"{f(s['correct_refusal']):>8} {f(s['false_refusal']):>8} {f(s['p50_latency_s'], False):>6} "
                         f"{f(s['p95_latency_s'], False):>6} {money(s['cost_per_success']):>10}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(fmt(table()))
