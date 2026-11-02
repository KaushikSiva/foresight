"""Evaluate planner variants on the serverless base model. Scores come from physics; every run is logged."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from foresight.llm import FireworksClient
from warehouse.evaluation import RUNS, VARIANTS, load_perception, load_scenes, price_of, run_variant, sample_scenes, summarize

MODEL = "accounts/fireworks/models/glm-5p3-flash"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="+", choices=VARIANTS)
    parser.add_argument("--split", default="tuning", choices=["tuning", "held_out"])
    parser.add_argument("--per-tier", type=int, help="evaluate a fixed subset instead of the whole split")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    scenes = load_scenes(args.split)
    if args.per_tier:
        scenes = sample_scenes(scenes, args.per_tier)
    perception = load_perception()
    client = FireworksClient(timeout=240)
    for variant in args.variants:
        if args.split == "held_out" and not args.force and list(RUNS.glob(f"held_out__{variant}__*.jsonl")):
            sys.exit(f"{variant} already has a held-out run; pass --force only if that run was broken")
        records = run_variant(client, MODEL, VARIANTS[variant], scenes, perception)
        summary = summarize(records)
        path = RUNS / f"{args.split}__{variant}__{len(scenes)}__{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(variant, json.dumps({**summary, "api_cost_usd": round(price_of(summary), 4)}), "\nlogged", path, flush=True)


if __name__ == "__main__":
    main()
