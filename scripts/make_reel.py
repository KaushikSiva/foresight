"""Cut a comparison reel from logged videos only, with a title card before each order."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

from warehouse.assets import font
from warehouse.orders import order_text

W, H = 1920, 540
CARD_SECONDS = 3
ENCODE = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-r", "30"]


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def _duration(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                                capture_output=True, text=True, check=True).stdout)


def _card(path, lines):
    image = Image.new("RGB", (W, H), "#11141a")
    d = ImageDraw.Draw(image)
    y = H // 2 - 34 * len(lines)
    for text, size, color in lines:
        d.text((W // 2, y), text, font=font(size), fill=color, anchor="mm")
        y += size + 34
    png = path.with_suffix(".png")
    image.save(png)
    _ffmpeg("-loop", "1", "-t", str(CARD_SECONDS), "-i", str(png), "-vf", f"scale={W}:{H}", *ENCODE, str(path))
    png.unlink()


def _side_by_side(left, right, out):
    length = max(_duration(left), _duration(right))
    _ffmpeg("-i", str(left), "-i", str(right), "-filter_complex",
            f"[0:v]tpad=stop_mode=clone:stop_duration=90,scale={W // 2}:{H}[a];[1:v]tpad=stop_mode=clone:stop_duration=90,scale={W // 2}:{H}[b];[a][b]hstack=inputs=2[v]",
            "-map", "[v]", "-t", f"{length:.2f}", *ENCODE, str(out))


def _join(parts, out, work):
    listing = work / "reel_parts.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    _ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out))
    listing.unlink()


def training_reel(run):
    scenes = {}
    with open(ROOT / "data" / "warehouse" / "scenes.jsonl") as f:
        for scene in map(json.loads, f):
            scenes[scene["id"]] = scene
    videos = {}
    for path in sorted(run.glob("step*_*.mp4")):
        match = re.fullmatch(r"step(\d+)_(.+)\.mp4", path.name)
        videos.setdefault(match.group(2), []).append((int(match.group(1)), path))
    work = run / "reel_work"
    work.mkdir(exist_ok=True)
    parts = []
    for number, (scene_id, items) in enumerate(sorted(videos.items()), 1):
        items.sort()
        (first_step, first), (last_step, last) = items[0], items[-1]
        card, clip = work / f"card_{scene_id}.mp4", work / f"clip_{scene_id}.mp4"
        _card(card, [(f"Order {number} of {len(videos)}: {scene_id}", 56, "white"),
                     (order_text(scenes[scene_id]["order"]), 32, "#c8ced8"),
                     (f"Left: training step {first_step} (untrained, reasoning off). Right: training step {last_step}.", 30, "#ffc45c")])
        _side_by_side(first, last, clip)
        parts += [card, clip]
    out = run / "improvement_reel.mp4"
    _join(parts, out, work)
    print(out, f"{_duration(out):.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    training_reel(Path(parser.parse_args().run_dir))


if __name__ == "__main__":
    sys.exit(main())
