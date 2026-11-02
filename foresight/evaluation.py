"""Run a variant over a split, score against the true state, and summarize logged runs."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .agent import VARIANTS
from .sim import score

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data" / "images"
RUNS = ROOT / "runs"
PRICES = ROOT / "prices.json"  # {"model": [usd_per_million_input, usd_per_million_output]}


def load_scenes(split=None):
    with open(ROOT / "data" / "scenes.jsonl") as f:
        scenes = [json.loads(line) for line in f]
    return [s for s in scenes if split is None or s["split"] == split]


def run_scene(client, variant, scene):
    record = {"scene": scene["id"], "tier": scene["tier"], "variant": variant}
    try:
        output = VARIANTS[variant](client, scene["instruction"], IMAGES / f"{scene['id']}.png")
    except RuntimeError as e:
        return {**record, "status": "error", "error": str(e), "calls": [],
                "score": {"success": False, "unsafe": False, "invalid": False, "refused": False, "violation": None}}
    record.update(output)
    if record.get("program"):
        record["score"] = score(scene, record["program"])
    else:
        # A withheld plan never reaches the robot, and it does not count as a correct refusal.
        record["score"] = {"success": False, "unsafe": False, "invalid": False, "refused": False, "violation": None}
    return record


def run_split(client, variant, split, limit=None, workers=8):
    scenes = load_scenes(split)[:limit]
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda s: run_scene(client, variant, s), scenes))


def _percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def _cost(calls, prices):
    total = 0.0
    for call in calls:
        price = prices.get(call["model"])
        if price is None:
            return None
        total += (call["prompt_tokens"] * price[0] + call["completion_tokens"] * price[1]) / 1e6
    return total


def summarize(records):
    prices = json.loads(PRICES.read_text()) if PRICES.exists() else {}
    n = len(records)
    successes = sum(r["score"]["success"] for r in records)
    latencies = [sum(c["latency_s"] for c in r["calls"]) for r in records if r["calls"]]
    costs = [_cost(r["calls"], prices) for r in records]
    known_cost = all(c is not None for c in costs) and n > 0
    impossible = [r for r in records if r["tier"] == "impossible"]
    summary = {
        "n": n,
        "successes": successes,
        "success_rate": successes / n if n else None,
        "unsafe": sum(r["score"]["unsafe"] for r in records),
        "invalid": sum(r["score"]["invalid"] for r in records),
        "withheld": sum(r["status"] == "withheld" for r in records),
        "errors": sum(r["status"] == "error" for r in records),
        "correct_refusals": sum(r["score"]["success"] for r in impossible),
        "impossible_total": len(impossible),
        "escalated": sum(bool(r.get("escalated")) for r in records),
        "p50_latency_s": _percentile(latencies, 0.5),
        "p95_latency_s": _percentile(latencies, 0.95),
        "tokens": sum(c["prompt_tokens"] + c["completion_tokens"] for r in records for c in r["calls"]),
        "total_cost_usd": sum(costs) if known_cost else None,
        "cost_per_success_usd": sum(costs) / successes if known_cost and successes else None,
        "by_tier": {},
    }
    for tier in sorted({r["tier"] for r in records}):
        group = [r for r in records if r["tier"] == tier]
        summary["by_tier"][tier] = {"n": len(group), "successes": sum(r["score"]["success"] for r in group)}
    return summary


def load_run(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def latest_runs():
    """Newest logged full run per (split, variant). Partial debugging runs are not reported."""
    latest = {}
    for path in sorted(RUNS.glob("*.jsonl")):
        split, variant, _ = path.stem.split("__")
        latest[(split, variant)] = path
    return latest
