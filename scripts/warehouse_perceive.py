"""Read every camera image once with GLM 5.3 Flash and cache it, so every planner works from the same input."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from foresight.llm import FireworksClient
from warehouse.evaluation import IMAGES, PERCEPTION, load_scenes
from warehouse.planner import perceive

MODEL = "accounts/fireworks/models/glm-5p3-flash"


def main():
    client = FireworksClient(timeout=240)
    scenes = load_scenes()

    def run(scene):
        try:
            return {"scene": scene["id"], **perceive(client, MODEL, IMAGES / f"{scene['id']}.jpg")}
        except RuntimeError as e:
            return {"scene": scene["id"], "totes": None, "error": str(e), "call": None}

    with ThreadPoolExecutor(8) as pool:
        records = list(pool.map(run, scenes))
    with open(PERCEPTION, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    by_id = {r["scene"]: r for r in records}
    for split in ("tuning", "held_out"):
        group = [s for s in scenes if s["split"] == split]
        right = sum(by_id[s["id"]]["totes"] == s["totes"] for s in group)
        failed = sum(by_id[s["id"]]["totes"] is None for s in group)
        print(f"{split}: {right}/{len(group)} read exactly right, {failed} failures")
    usage = [r["call"] for r in records if r["call"]]
    print("tokens in/out:", sum(c["prompt_tokens"] for c in usage), sum(c["completion_tokens"] for c in usage))


if __name__ == "__main__":
    main()
