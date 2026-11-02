"""Write the fine-tuning train and validation files, and estimate their size."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from foresight.training import build


def main():
    train, validation, _ = build()
    out = ROOT / "data" / "finetune"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation)):
        with open(out / f"{name}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    chars = sum(len(m["content"]) for row in train for m in row["messages"])
    replans = sum("failed these checks" in row["messages"][0]["content"] for row in train)
    print(f"{len(train)} train, {len(validation)} validation, {replans} repair examples in train")
    print(f"about {chars // 4:,} training tokens per epoch (estimated at 4 characters per token)")


if __name__ == "__main__":
    main()
