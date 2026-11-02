"""Write the warehouse eval: orders, camera images, and proof that the solver packs every order in physics."""
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import imageio.v3 as iio

from warehouse.orders import generate, solve
from warehouse.robot import perception_image
from warehouse.scoring import score

OUT = ROOT / "data" / "warehouse"


def check(scene):
    result = score(scene, solve(scene))
    return scene["id"], result["success"], result["outcome"], result.get("report", {}).get("ik_error_m")


def main():
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    scenes = generate()
    with open(OUT / "scenes.jsonl", "w") as f:
        for scene in scenes:
            f.write(json.dumps(scene) + "\n")
    for scene in scenes:
        iio.imwrite(OUT / "images" / f"{scene['id']}.jpg", perception_image(scene["totes"], scene["seed"]), quality=88)
    with ProcessPoolExecutor(8) as pool:
        results = list(pool.map(check, scenes))
    print("outcomes:", Counter(r[2] for r in results))
    failed = [r for r in results if not r[1]]
    print("solver failures in physics:", failed[:12])
    print("worst IK error (m):", max(r[3] or 0 for r in results))
    (OUT / "physics_validation.json").write_text(json.dumps(results))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
