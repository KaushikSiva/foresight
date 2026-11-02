"""Run the extreme stress test once on base GLM: plain vs. Foresight. Scores come from physics, and the run is logged."""
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from foresight.llm import FireworksClient
from warehouse.evaluation import RUNS, VARIANTS, price_of, run_variant, summarize
from warehouse.planner import perceive

MODEL = "accounts/fireworks/models/glm-5p3-flash"
DATA = ROOT / "data" / "warehouse" / "extreme"


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    if list(RUNS.glob("extreme__*.jsonl")) and "--force" not in sys.argv:
        sys.exit("the extreme test already ran; pass --force only if that run was broken")
    with open(DATA / "scenes.jsonl") as f:
        scenes = [json.loads(line) for line in f]
    client = FireworksClient(timeout=240)

    def read(scene):
        try:
            return scene["id"], perceive(client, MODEL, DATA / "images" / f"{scene['id']}.jpg")
        except RuntimeError as e:
            return scene["id"], {"totes": None, "error": str(e), "call": None}

    with ThreadPoolExecutor(5) as pool:
        perception = dict(pool.map(read, scenes))
    right = sum(perception[s["id"]]["totes"] == s["totes"] for s in scenes)
    print(f"images read right: {right}/{len(scenes)}", flush=True)
    results = {}
    for variant in ("A_baseline", "C_foresight"):
        records = run_variant(client, MODEL, VARIANTS[variant], scenes, perception)
        for record in records:
            record["perceived"] = perception[record["scene"]]["totes"]
        results[variant] = {r["scene"]: r for r in records}
        path = RUNS / f"extreme__{variant}__{len(scenes)}__{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        summary = summarize(records)
        print(variant, json.dumps({**summary, "api_cost_usd": round(price_of(summary), 4)}), "\nlogged", path, flush=True)

    plain, checked = results["A_baseline"], results["C_foresight"]
    wins = sum(checked[i]["score"]["success"] and not plain[i]["score"]["success"] for i in plain)
    losses = sum(plain[i]["score"]["success"] and not checked[i]["score"]["success"] for i in plain)
    n = wins + losses
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.0
    print(json.dumps({"foresight_wins": wins, "foresight_losses": losses, "mcnemar_p": round(p, 4),
                      "unsafe_prevented": sum(plain[i]["score"]["unsafe"] and not checked[i]["score"]["unsafe"] for i in plain),
                      "unsafe_added": sum(checked[i]["score"]["unsafe"] and not plain[i]["score"]["unsafe"] for i in plain)}), flush=True)


if __name__ == "__main__":
    main()
