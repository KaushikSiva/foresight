"""Run perception once per scene and cache it, so every variant plans from the same input."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from foresight.agent import perceive
from foresight.evaluation import IMAGES, PERCEPTION_CACHE, load_scenes, perception_correct
from foresight.llm import FireworksClient


def main():
    client = FireworksClient()
    scenes = load_scenes()

    def run(scene):
        try:
            result = perceive(client, scene["instruction"], IMAGES / f"{scene['id']}.png")
        except RuntimeError as e:
            result = {"perceived": None, "goal": None, "error": str(e), "call": None}
        return {"scene": scene["id"], **result}

    with ThreadPoolExecutor(8) as pool:
        records = list(pool.map(run, scenes))
    with open(PERCEPTION_CACHE, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    by_id = {r["scene"]: r for r in records}
    for split in ("tuning", "held_out"):
        group = [s for s in scenes if s["split"] == split]
        right = sum(perception_correct(s, by_id[s["id"]]) for s in group)
        failed = sum(by_id[s["id"]]["error"] is not None for s in group)
        print(f"{split}: {right}/{len(group)} scenes read exactly right, {failed} perception failures")
    usage = [r["call"] for r in records if r["call"]]
    print("tokens:", sum(c["prompt_tokens"] for c in usage), "in,", sum(c["completion_tokens"] for c in usage), "out")


if __name__ == "__main__":
    main()
