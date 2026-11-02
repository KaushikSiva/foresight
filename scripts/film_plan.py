"""Film one plan on one order. Runs as its own process, because MuJoCo rendering needs the main thread on macOS."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from warehouse.evaluation import load_scenes
from warehouse.record import record
from warehouse.scoring import judge

LABELS = {"packed": "PACKED CORRECTLY", "correct_hold": "CORRECTLY HELD: NOT ENOUGH STOCK", "wrong_hold": "WRONGLY HELD: STOCK WAS AVAILABLE",
          "shipped_short": "SHIPPED SHORT: STOCK WAS MISSING", "crushed": "ITEM CRUSHED", "spilled": "ITEM SPILLED", "wrong_items": "WRONG ITEMS PACKED",
          "rule_broken": "PACKING RULE BROKEN", "invalid": "INVALID PLAN", "not_sealed": "CARTON NOT SEALED"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("output")
    parser.add_argument("--title", required=True)
    parser.add_argument("--program", required=True, help="the plan, or an empty string if no plan was produced")
    args = parser.parse_args()
    scene = next(s for s in load_scenes() if s["id"] == args.scene)
    # With no plan the robot stays still, and the banner says so.
    program = args.program or 'hold("no plan")'

    def verdict(report):
        if not args.program:
            return "NO USABLE PLAN: ROBOT DID NOT MOVE", False
        outcome = judge(scene, report)
        return LABELS.get(outcome["outcome"], outcome["outcome"].upper()), outcome["success"]

    report = record(scene, program, args.output, args.title, verdict)
    print(json.dumps(judge(scene, report)))


if __name__ == "__main__":
    main()
