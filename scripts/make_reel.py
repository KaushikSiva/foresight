"""Cut the improvement reel: for each filmed order, its first and last checkpoint side by side, from logged videos only."""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run = Path(args.run_dir)
    by_scene = {}
    for path in sorted(run.glob("step*_*.mp4")):
        match = re.fullmatch(r"step(\d+)_(.+)\.mp4", path.name)
        by_scene.setdefault(match.group(2), []).append((int(match.group(1)), path))

    clips = []
    for scene, videos in sorted(by_scene.items()):
        videos.sort()
        (_, first), (_, last) = videos[0], videos[-1]
        clip = run / f"compare_{scene}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(first), "-i", str(last), "-filter_complex",
                        "[0:v]tpad=stop_mode=clone:stop_duration=60[a];[1:v]tpad=stop_mode=clone:stop_duration=60[b];[a][b]hstack=inputs=2,scale=1920:-2[v]",
                        "-map", "[v]", "-shortest", "-t", str(_longest(first, last)), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(clip)], check=True)
        clips.append(clip)
    listing = run / "reel_clips.txt"
    listing.write_text("".join(f"file '{c.name}'\n" for c in clips))
    reel = run / "improvement_reel.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(reel)], check=True)
    print(reel)


def _longest(a, b):
    def duration(path):
        return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                                    capture_output=True, text=True, check=True).stdout)
    return max(duration(a), duration(b))


if __name__ == "__main__":
    sys.exit(main())
