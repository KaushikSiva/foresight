"""Write data/scenes.jsonl and data/images/*.png."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from foresight.render import render
from foresight.scenes import generate, solve
from foresight.sim import score


def main():
    images = ROOT / "data" / "images"
    images.mkdir(parents=True, exist_ok=True)
    scenes = generate()
    with open(ROOT / "data" / "scenes.jsonl", "w") as f:
        for scene in scenes:
            assert score(scene, solve(scene))["success"], scene["id"]
            render(scene, images / f"{scene['id']}.png")
            f.write(json.dumps(scene) + "\n")
    print(len(scenes), "scenes", Counter((s["tier"], s["split"]) for s in scenes))


if __name__ == "__main__":
    main()
