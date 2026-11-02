"""Top-down Pillow renderer. Images come only from the true scene, so every image has exact ground truth.

Container names are printed (like labels on a real bench); objects are not labelled,
so the model has to recognise color and shape. Closed lids hide their contents.
"""
import io
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from .scenes import H, W
from .schema import Scene, Task

RGB = {
    "red": (214, 40, 40), "orange": (245, 135, 30), "yellow": (238, 206, 40), "green": (46, 158, 72),
    "blue": (38, 88, 206), "purple": (128, 60, 178), "pink": (240, 128, 176),
}
TABLE_RGB = (222, 204, 170)
FILL = {"tray": (196, 200, 206), "bin": (90, 98, 110), "box": (166, 120, 78), "tote": (52, 110, 150),
        "carton": (204, 168, 116)}
SIZE = 44


def _font(size: int):
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_obj(d: ImageDraw.ImageDraw, kind: str, color: str, x: int, y: int) -> None:
    c, edge = RGB[color], (40, 34, 28)
    if kind == "block":
        d.rectangle([x, y, x + SIZE, y + SIZE], fill=c, outline=edge, width=2)
    elif kind == "ball":
        d.ellipse([x, y, x + SIZE, y + SIZE], fill=c, outline=edge, width=2)
        d.ellipse([x + 9, y + 8, x + 19, y + 18], fill=tuple(min(255, v + 70) for v in c))
    else:  # cup: rim ring seen from above, plus a handle
        d.rectangle([x + SIZE - 4, y + SIZE // 2 - 6, x + SIZE + 8, y + SIZE // 2 + 6], fill=c, outline=edge, width=2)
        d.ellipse([x, y, x + SIZE, y + SIZE], fill=c, outline=edge, width=2)
        d.ellipse([x + 10, y + 10, x + SIZE - 10, y + SIZE - 10], fill=(250, 246, 238), outline=edge, width=1)


def render(task: Task, scene: Optional[Scene] = None) -> Image.Image:
    scene = scene or task.scene
    img = Image.new("RGB", (W, H), TABLE_RGB)
    d = ImageDraw.Draw(img)
    label = _font(16)

    for c in scene.containers:
        x, y, w, h = task.layout[c.id]
        d.rounded_rectangle([x, y + 20, x + w, y + h], radius=8, fill=FILL[c.kind], outline=(40, 34, 28), width=3)
        d.text((x + 4, y), c.id.replace("_", " ").upper(), fill=(40, 34, 28), font=label)
        if not c.is_open:
            lid = [x + 6, y + 26, x + w - 6, y + h - 6]
            d.rectangle(lid, fill=(120, 124, 132))
            for ly in range(lid[1] + 8, lid[3], 12):
                d.line([lid[0], ly, lid[2], ly], fill=(98, 102, 110), width=3)
            d.text((x + w // 2 - 30, y + 20 + (h - 20) // 2 - 9), "CLOSED", fill=(255, 255, 255), font=label)
            continue
        items = scene.contents(c.id)
        rows = 1 if len(items) <= 3 else 2
        top = y + 20 + (h - 20 - rows * SIZE - (rows - 1) * 8) // 2
        for i, o in enumerate(items):
            _draw_obj(d, o.kind, o.color, x + 14 + (i % 3) * (SIZE + 14), top + (i // 3) * (SIZE + 8))

    spare = 0  # objects that started elsewhere and were moved to the table (demo after-states)
    for o in scene.objects:
        if o.location != "table":
            continue
        if o.id in task.layout:
            x, y, _, _ = task.layout[o.id]
        else:
            x, y, spare = 30 + spare * (SIZE + 16), H - SIZE - 12, spare + 1
        _draw_obj(d, o.kind, o.color, x, y)
    return img


def image_bytes(task: Task, scene: Optional[Scene] = None) -> bytes:
    """The camera image the model sees: the 3D station (JPEG) or the flat view (PNG), per config.RENDER."""
    from . import config
    if config.RENDER == "3d":
        from .render3d import jpeg_bytes
        return jpeg_bytes(task, scene)
    return png_bytes(task, scene)


def png_bytes(task: Task, scene: Optional[Scene] = None) -> bytes:
    buf = io.BytesIO()
    render(task, scene).save(buf, format="PNG")
    return buf.getvalue()
