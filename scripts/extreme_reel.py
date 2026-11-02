"""Film every order Foresight won in the extreme test, plain vs. Foresight, from the exact logged plans."""
import glob
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.make_reel import _card, _duration, _join, _side_by_side
from warehouse.orders import order_text

RUNS = ROOT / "runs" / "warehouse"
SCENES_FILE = ROOT / "data" / "warehouse" / "extreme" / "scenes.jsonl"
OUT = ROOT / "runs" / "warehouse" / "extreme_videos"


def latest(variant):
    path = sorted(glob.glob(str(RUNS / f"extreme__{variant}__*.jsonl")))[-1]
    with open(path) as f:
        return {r["scene"]: r for r in map(json.loads, f)}


def film(scene_id, program, title, path):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "film_plan.py"), scene_id, str(path), "--title", title,
                    "--program", program or "", "--scenes-file", str(SCENES_FILE)], check=True, capture_output=True, timeout=600)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plain, checked = latest("A_baseline"), latest("C_foresight")
    with open(SCENES_FILE) as f:
        scenes = {s["id"]: s for s in map(json.loads, f)}
    wins = [i for i in plain if checked[i]["score"]["success"] and not plain[i]["score"]["success"]]
    with ThreadPoolExecutor(4) as pool:
        jobs = []
        for scene_id in wins:
            jobs.append(pool.submit(film, scene_id, plain[scene_id].get("program"), "Plain GLM 5.3 Flash", OUT / f"{scene_id}_plain.mp4"))
            jobs.append(pool.submit(film, scene_id, checked[scene_id].get("program"), "GLM 5.3 Flash + Foresight check", OUT / f"{scene_id}_foresight.mp4"))
        for job in jobs:
            job.result()
    parts = []
    work = OUT / "reel_work"
    work.mkdir(exist_ok=True)
    for number, scene_id in enumerate(wins, 1):
        card, clip = work / f"card_{scene_id}.mp4", work / f"clip_{scene_id}.mp4"
        _card(card, [(f"Extreme order {number} of {len(wins)}", 56, "white"), (order_text(scenes[scene_id]["order"]), 30, "#c8ced8"),
                     (f"Left: plain GLM ({plain[scene_id]['score']['outcome'].replace('_', ' ')}). Right: GLM + Foresight (packed).", 30, "#ffc45c")])
        _side_by_side(OUT / f"{scene_id}_plain.mp4", OUT / f"{scene_id}_foresight.mp4", clip)
        parts += [card, clip]
    reel = OUT / "extreme_reel.mp4"
    _join(parts, reel, work)
    for path in work.iterdir():
        path.unlink()
    work.rmdir()
    print(reel, f"{_duration(reel):.1f}s", "wins", wins)


if __name__ == "__main__":
    main()
