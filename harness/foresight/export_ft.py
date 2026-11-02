"""Export fine-tuning data for the TUNING split only (never held-out), in chat "messages" JSONL.

  python -m foresight.export_ft

Writes data/ft/:
  perception.jsonl   image -> scene JSON          (vision; images inline as data URLs)
  planning.jsonl     scene + task -> goal + plan with predicted states
  replan.jsonl       a plan that skips prerequisites + checker errors -> corrected plan

The prompts are the same builders Variant C/E use, so train and eval prompts match exactly.
Check the exact JSONL format Fireworks expects for vision fine-tuning before uploading.
"""
import json

from . import prompts
from .eval import load_tasks
from .config import ROOT
from .render import png_bytes
from .solver import fill_expects, plan_steps, solve
from .schema import Plan
from .validator import check

OUT = ROOT / "data" / "ft"


def main() -> None:
    tasks = [t for t in load_tasks() if t.split == "tune"]
    OUT.mkdir(parents=True, exist_ok=True)
    n = {"perception": 0, "planning": 0, "replan": 0}
    with open(OUT / "perception.jsonl", "w") as fp, open(OUT / "planning.jsonl", "w") as fl, \
            open(OUT / "replan.jsonl", "w") as fr:
        for t in tasks:
            target_scene = prompts.scene_json(t.scene)
            fp.write(json.dumps({"messages": prompts.perceive(png_bytes(t)) +
                                 [{"role": "assistant", "content": target_scene}]}) + "\n")
            n["perception"] += 1

            msgs = prompts.plan_from_scene(t.scene, t.instruction)
            good = prompts.plan_json(solve(t), True, t.goals, t.sealed)
            fl.write(json.dumps({"messages": msgs + [{"role": "assistant", "content": good}]}) + "\n")
            n["planning"] += 1

            if t.possible and t.difficulty in ("blocked", "multi_step", "packing"):
                naive = Plan(steps=fill_expects(t.scene, plan_steps(t.scene, t.goals, prereqs=False, sealed=t.sealed)))
                errors = check(t.scene, naive, t.goals, t.sealed)
                if errors:
                    bad = prompts.plan_json(naive, True, t.goals, t.sealed)
                    fr.write(json.dumps({"messages": msgs + [{"role": "assistant", "content": bad},
                                                             prompts.replan(errors),
                                                             {"role": "assistant", "content": good}]}) + "\n")
                    n["replan"] += 1
    print(f"wrote {n} examples to {OUT.relative_to(ROOT)}/ (tuning split only)")


if __name__ == "__main__":
    main()
