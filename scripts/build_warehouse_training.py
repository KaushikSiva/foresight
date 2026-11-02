"""Render training images and write the vision and planning examples. No API calls, no cost."""
import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import imageio.v3 as iio

from warehouse.robot import perception_image
from warehouse.training import FIRST_SEED, build_scenes, plan_example, vision_answer

OUT = ROOT / "data" / "warehouse" / "train"


def render(scene):
    path = OUT / "images" / f"{scene['id']}.jpg"
    iio.imwrite(path, perception_image(scene["totes"], scene["seed"]), quality=88)
    return str(path.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision", type=int, default=300)
    parser.add_argument("--plans", type=int, default=300)
    args = parser.parse_args()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    scenes = build_scenes(args.vision + args.plans)
    vision_scenes, plan_scenes = scenes[:args.vision], scenes[args.vision:]
    with ProcessPoolExecutor(8) as pool:
        paths = list(pool.map(render, vision_scenes))
    with open(OUT / "vision.jsonl", "w") as f:
        for scene, path in zip(vision_scenes, paths):
            f.write(json.dumps({"image": path, "answer": vision_answer(scene)}) + "\n")
    rng = random.Random(FIRST_SEED + 1)
    with open(OUT / "plans.jsonl", "w") as f:
        for scene in plan_scenes:
            f.write(json.dumps(plan_example(scene, rng)) + "\n")
    print(len(paths), "vision examples,", len(plan_scenes), "plan examples")


if __name__ == "__main__":
    main()
