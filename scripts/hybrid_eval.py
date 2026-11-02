"""Exploratory: the tuned model's logged camera readings, planned by reasoning base GLM behind Foresight.

Chosen after the extreme results were seen, so report it as exploratory rather than as a pre-planned test.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from foresight.llm import FireworksClient
from warehouse.evaluation import RUNS, VARIANTS, price_of, run_variant, summarize

MODEL = "accounts/fireworks/models/glm-5p3-flash"


def main():
    run_dir = Path(sys.argv[1])
    with open(ROOT / "data" / "warehouse" / "extreme" / "scenes.jsonl") as f:
        scenes = [json.loads(line) for line in f]
    with open(run_dir / "extreme_C_tuned_foresight.jsonl") as f:
        perception = {r["scene"]: {"totes": r["perceived"], "error": None, "call": None} for r in map(json.loads, f)}
    records = run_variant(FireworksClient(timeout=240), MODEL, VARIANTS["C_foresight"], scenes, perception)
    with open(RUNS / "exploratory_hybrid_extreme.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps({**record, "perceived": perception[record["scene"]]["totes"]}) + "\n")
    summary = summarize(records)
    print(json.dumps({**summary, "api_cost_usd": round(price_of(summary), 4)}))
    print("fails", [(r["scene"], r["score"]["outcome"]) for r in records if not r["score"]["success"]])


if __name__ == "__main__":
    main()
