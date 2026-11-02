"""Film a packing run in MuJoCo as an MP4, with a caption strip and a carton close-up inset."""
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .assets import font
from .orders import order_text
from .rules import ParseError, parse_program
from .robot import execute
from .world import HEIGHT, WIDTH

FPS = 30
INSET = (384, 216)


def _strip(frame, title, order, caption, verdict=None, verdict_good=True):
    image = Image.fromarray(frame)
    d = ImageDraw.Draw(image, "RGBA")
    d.rectangle([0, HEIGHT - 86, WIDTH, HEIGHT], fill=(10, 12, 16, 205))
    d.text((24, HEIGHT - 78), title, font=font(26), fill="white")
    d.text((24, HEIGHT - 44), f"Order: {order}"[:112], font=font(19), fill=(205, 210, 218))
    d.text((WIDTH - 24, HEIGHT - 76), caption, font=font(22), fill=(255, 196, 92), anchor="ra")
    if verdict:
        color = (34, 150, 84, 235) if verdict_good else (196, 44, 44, 235)
        d.rectangle([0, 0, WIDTH, 96], fill=color)
        d.text((WIDTH // 2, 48), verdict, font=font(38), fill="white", anchor="mm")
    return np.asarray(image)


def _reset_scene(option):
    option.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = True


class Recorder:
    def __init__(self, path, title, order):
        self.title, self.order = title, order
        self.process = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)
        self.renderer = self.inset = None
        self.count = 0
        self.last = None

    def frame(self, station):
        self.count += 1
        if self.count % 2:  # sampling every other sim frame plays back at twice real time
            return
        if self.renderer is None:
            self.renderer = mujoco.Renderer(station.model, HEIGHT, WIDTH)
            self.inset = mujoco.Renderer(station.model, INSET[1], INSET[0])
        self.renderer.update_scene(station.data, camera="cinema")
        main = self.renderer.render().copy()
        self.inset.update_scene(station.data, camera="closeup")
        small = self.inset.render()
        main[16:16 + INSET[1], WIDTH - INSET[0] - 16:WIDTH - 16] = small
        self.last = main
        self._write(_strip(main, self.title, self.order, station.caption))

    def _write(self, frame):
        self.process.stdin.write(frame.tobytes())

    def finish(self, verdict, good):
        if self.last is not None:
            end = _strip(self.last, self.title, self.order, "", verdict, good)
            for _ in range(int(FPS * 2.2)):
                self._write(end)
        self.process.stdin.close()
        self.process.wait()
        for renderer in (self.renderer, self.inset):
            if renderer:
                renderer.close()


def record(scene, program, path, title, verdict_fn):
    """Run program on the true scene while filming. verdict_fn(report) returns (text, good)."""
    recorder = Recorder(path, title, order_text(scene["order"]))
    calls = parse_program(program)
    report = execute(scene["totes"], calls, scene["seed"], on_frame=recorder.frame)
    text, good = verdict_fn(report)
    recorder.finish(text, good)
    return report
