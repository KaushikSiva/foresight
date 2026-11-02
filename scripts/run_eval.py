"""Run one variant over a split. Held-out runs are one-shot unless --force is passed."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from foresight.agent import VARIANTS
from foresight.evaluation import RUNS, run_split, summarize
from foresight.llm import FireworksClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--split", default="tuning", choices=["tuning", "held_out"])
    parser.add_argument("--limit", type=int, help="debug on N scenes; the result is not logged")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    RUNS.mkdir(exist_ok=True)
    if args.split == "held_out" and not args.force and list(RUNS.glob(f"held_out__{args.variant}__*.jsonl")):
        sys.exit(f"{args.variant} already has a held-out run; pass --force only if the earlier run was broken")

    records = run_split(FireworksClient(), args.variant, args.split, args.limit)
    print(json.dumps(summarize(records), indent=2))
    if args.limit is None:
        path = RUNS / f"{args.split}__{args.variant}__{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print("logged", path)
    else:
        for record in records:
            print(record["scene"], record["status"], record["score"], repr(record.get("program")), record.get("error", ""))


if __name__ == "__main__":
    main()
