"""Run planners over orders and score every plan in physics. Results are logged, never hand-edited."""
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from .catalog import CARTONS
from .planner import VARIANTS
from .scoring import score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "warehouse"
IMAGES = DATA / "images"
PERCEPTION = DATA / "perception.jsonl"
RUNS = Path(os.environ.get("FORESIGHT_RUNS", ROOT / "runs" / "warehouse"))
PRICE_PER_M = {"glm-5p3-flash": (0.15, 0.50)}  # serverless input, output, from fireworks.ai/pricing
NO_SCORE = {"success": False, "outcome": "no_plan", "unsafe": False, "wasteful": False}


def load_scenes(split=None):
    with open(DATA / "scenes.jsonl") as f:
        scenes = [json.loads(line) for line in f]
    return [s for s in scenes if split is None or s["split"] == split]


def load_perception():
    with open(PERCEPTION) as f:
        return {r["scene"]: r for r in map(json.loads, f)}


def sample_scenes(scenes, per_tier):
    """A fixed, evenly spread subset, so every checkpoint is judged on the same orders."""
    picked = []
    for tier in sorted({s["tier"] for s in scenes}):
        group = [s for s in scenes if s["tier"] == tier]
        step = max(1, len(group) // per_tier)
        picked += group[::step][:per_tier]
    return picked


def _score_job(job):
    scene, program = job
    result = score(scene, program)
    report = result.pop("report", None)
    if report:
        result.update(carton=report["carton"], crushed=report["crushed"], spilled=report["spilled"],
                      rule_problems=[p["code"] for p in report["rule_problems"]])
    return result


def run_variant(client, model, variant_fn, scenes, perception, workers=5, pool=None):
    """Plan with threads (network), then score with processes (physics)."""
    def plan(scene):
        seen = perception[scene["id"]]
        record = {"scene": scene["id"], "tier": scene["tier"]}
        if not seen["totes"]:
            return {**record, "status": "perception_failed", "program": None, "calls": [], "error": seen["error"]}
        try:
            return {**record, **variant_fn(client, model, seen["totes"], scene["order"], scene["seed"])}
        except RuntimeError as e:
            return {**record, "status": "error", "program": None, "calls": [], "error": str(e)}

    with ThreadPoolExecutor(workers) as threads:
        records = list(threads.map(plan, scenes))
    jobs = [(s, r["program"]) for s, r in zip(scenes, records) if r.get("program")]
    own = pool is None
    pool = pool or ProcessPoolExecutor(8)
    try:
        results = iter(list(pool.map(_score_job, jobs)))
    finally:
        if own:
            pool.shutdown()
    for record in records:
        record["score"] = next(results) if record.get("program") else dict(NO_SCORE)
    return records


def summarize(records, scenes_by_id=None):
    n = len(records)
    successes = sum(r["score"]["success"] for r in records)
    latencies = sorted(sum(c["latency_s"] for c in r["calls"]) for r in records if r["calls"])
    tokens_in = sum(c["prompt_tokens"] for r in records for c in r["calls"])
    tokens_out = sum(c["completion_tokens"] for r in records for c in r["calls"])
    outcomes = {}
    for r in records:
        outcomes[r["score"]["outcome"]] = outcomes.get(r["score"]["outcome"], 0) + 1
    cartons = sum(CARTONS[r["score"]["carton"]]["cost_usd"] for r in records if r["score"].get("carton") and r["score"]["success"])
    summary = {
        "n": n, "successes": successes, "success_rate": successes / n if n else None,
        "unsafe": sum(r["score"]["unsafe"] for r in records),
        "crushed": sum(bool(r["score"].get("crushed")) for r in records),
        "wasteful": sum(r["score"]["wasteful"] for r in records),
        "withheld": sum(r["status"] == "withheld" for r in records),
        "errors": sum(r["status"] == "error" for r in records),
        "outcomes": outcomes,
        "p50_latency_s": latencies[len(latencies) // 2] if latencies else None,
        "tokens_in": tokens_in, "tokens_out": tokens_out,
        "carton_cost_usd": round(cartons, 2),
        "by_tier": {},
    }
    for tier in sorted({r["tier"] for r in records}):
        group = [r for r in records if r["tier"] == tier]
        summary["by_tier"][tier] = {"n": len(group), "successes": sum(r["score"]["success"] for r in group)}
    return summary


def price_of(summary, model_key="glm-5p3-flash"):
    rate_in, rate_out = PRICE_PER_M[model_key]
    return (summary["tokens_in"] * rate_in + summary["tokens_out"] * rate_out) / 1e6


__all__ = ["VARIANTS", "load_scenes", "load_perception", "sample_scenes", "run_variant", "summarize", "price_of"]
