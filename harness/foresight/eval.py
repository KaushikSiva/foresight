"""Eval harness.

  python -m foresight.eval gen                          # freeze the 200 tasks to data/tasks.json + images
  python -m foresight.eval run A B C --split tune       # writes runs/tune-A.jsonl, ...
  python -m foresight.eval run C --split tune --limit 10 --mock
  python -m foresight.eval run A B C D --split heldout --confirm-heldout   # once, at the end
  python -m foresight.eval report
"""
import argparse
import asyncio
import json
import sys
import time
from typing import List

from . import config
from .metrics import fmt, table
from .render import render
from .scenes import generate
from .schema import Task
from .variants import VARIANTS, run_task

TASKS = config.ROOT / "data" / "tasks.json"
IMAGES = config.ROOT / "data" / "images"


def load_tasks() -> List[Task]:
    if not TASKS.exists():
        sys.exit("data/tasks.json is missing: run `python -m foresight.eval gen` first")
    return [Task.model_validate(t) for t in json.loads(TASKS.read_text())]


def cmd_gen(args) -> None:
    if TASKS.exists() and not args.force:
        sys.exit("data/tasks.json already exists (the eval set is frozen). Use --force to regenerate.")
    tasks = generate(seed=args.seed)
    TASKS.parent.mkdir(parents=True, exist_ok=True)
    TASKS.write_text(json.dumps([t.model_dump() for t in tasks], indent=1))
    IMAGES.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        render(t).save(IMAGES / f"{t.id}.png")
    print(f"wrote {len(tasks)} tasks and images ({sum(t.split == 'heldout' for t in tasks)} held out)")


async def _run_variant(variant: str, tasks: List[Task], mock: bool) -> List[dict]:
    from .llm import Fireworks, Mock
    llm = None if mock else Fireworks()
    done, t0 = 0, time.time()

    async def one(task: Task) -> dict:
        nonlocal done
        rec = await run_task(variant, task, Mock(task) if mock else llm)
        done += 1
        if done % 10 == 0 or done == len(tasks):
            print(f"  {variant}: {done}/{len(tasks)}  ({time.time() - t0:.0f}s)", flush=True)
        return rec

    try:
        return await asyncio.gather(*(one(t) for t in tasks))
    finally:
        if llm:
            await llm.aclose()


def cmd_run(args) -> None:
    tasks = [t for t in load_tasks() if t.split == args.split]
    if args.difficulty:
        tasks = [t for t in tasks if t.difficulty == args.difficulty]
    if args.limit:
        tasks = tasks[: args.limit]
    config.RUNS.mkdir(exist_ok=True)
    for v in args.variants:
        if v not in VARIANTS:
            sys.exit(f"unknown variant {v}; choose from {list(VARIANTS)}")
        name = args.name or f"{args.split}-{v}{'-mock' if args.mock else ''}"
        out = config.RUNS / f"{name}.jsonl"
        if args.split == "heldout" and not args.mock:
            if not args.confirm_heldout:
                sys.exit("held-out runs are one-shot: pass --confirm-heldout")
            if out.exists() and not args.force:
                sys.exit(f"{out.name} already exists; the held-out set is run once. --force to overwrite.")
        print(f"running {v} ({VARIANTS[v]}) on {len(tasks)} {args.split} tasks" + (" [MOCK]" if args.mock else ""))
        recs = asyncio.run(_run_variant(v, tasks, args.mock))
        out.write_text("".join(json.dumps(r) + "\n" for r in recs))
        bad = sum(bool(r["api_errors"]) for r in recs)
        if bad:
            print(f"  warning: {bad} tasks had API errors (see api_errors in {out.name})")
    print(fmt(table()))


def main() -> None:
    p = argparse.ArgumentParser(prog="foresight.eval")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--force", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("variants", nargs="+")
    r.add_argument("--split", choices=["tune", "heldout"], default="tune")
    r.add_argument("--difficulty", choices=["easy", "multi_step", "blocked", "impossible", "packing"])
    r.add_argument("--limit", type=int)
    r.add_argument("--mock", action="store_true", help="offline mock model; numbers are not results")
    r.add_argument("--name")
    r.add_argument("--confirm-heldout", action="store_true")
    r.add_argument("--force", action="store_true")
    sub.add_parser("report")
    args = p.parse_args()
    if args.cmd == "gen":
        cmd_gen(args)
    elif args.cmd == "run":
        cmd_run(args)
    else:
        print(fmt(table()))


if __name__ == "__main__":
    main()
