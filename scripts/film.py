"""Film the reference plan for one order, or a deliberately bad plan that crushes a fragile item."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from warehouse.orders import solve
from warehouse.record import record
from warehouse.scoring import judge

CRUSH_SCENE = {
    "id": "crush-demo", "seed": 3, "fillable": True, "best_carton": "L",
    "totes": {"T1": {"sku": "ceramic_mug", "count": 1}, "T2": {"sku": "bearing_box", "count": 1}},
    "order": {"ceramic_mug": 1, "bearing_box": 1},
}
CRUSH_PROGRAM = 'open_carton("L")\npick("T1")\nplace("A", 1)\npick("T2")\nplace("A", 2)\nseal()'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", nargs="?", default="hard-00")
    parser.add_argument("--crush", action="store_true", help="film a bearing box stacked on a mug instead")
    args = parser.parse_args()

    out = ROOT / "recordings"
    out.mkdir(exist_ok=True)
    if args.crush:
        scene, program, title, path = CRUSH_SCENE, CRUSH_PROGRAM, "Bad plan: heavy box on a mug", out / "crush_demo.mp4"
    else:
        with open(ROOT / "data" / "warehouse" / "scenes.jsonl") as f:
            scene = next(s for s in map(json.loads, f) if s["id"] == args.scene)
        program, title, path = solve(scene), "Reference plan (solver)", out / f"{args.scene}_solver.mp4"

    def verdict(report):
        outcome = judge(scene, report)
        return outcome["outcome"].replace("_", " ").upper(), outcome["success"]

    report = record(scene, program, path, title, verdict)
    print(path, judge(scene, report), "crushed:", report["crushed"], "loads (N):", report["loads_n"])


if __name__ == "__main__":
    main()
