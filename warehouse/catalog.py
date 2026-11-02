"""SKU catalog and packing-station geometry. Mass, size and handling flags are public rules given to every planner."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Sku:
    name: str
    kg: float
    shape: str  # "lathe" or "box"
    size: tuple  # lathe: (radius, height). box: (x, y, z) half extents
    fragile: bool = False
    liquid: bool = False
    electronics: bool = False

    @property
    def height(self):
        return self.size[1] if self.shape == "lathe" else self.size[2] * 2

    @property
    def heavy(self):
        return self.kg >= HEAVY_KG


HEAVY_KG = 1.5
CATALOG = {s.name: s for s in [
    Sku("ceramic_mug", 0.35, "lathe", (0.037, 0.090), fragile=True),
    Sku("glass_jar", 0.55, "lathe", (0.036, 0.105), fragile=True),
    Sku("soup_can", 0.42, "lathe", (0.034, 0.100)),
    Sku("paint_can", 2.40, "lathe", (0.037, 0.095), liquid=True),
    Sku("detergent_jug", 1.80, "box", (0.036, 0.055, 0.052), liquid=True),
    Sku("hardcover_book", 0.90, "box", (0.036, 0.055, 0.020)),
    Sku("headphones_box", 0.50, "box", (0.036, 0.060, 0.045), electronics=True),
    Sku("phone_box", 0.30, "box", (0.034, 0.060, 0.020), electronics=True),
    Sku("bearing_box", 3.20, "box", (0.036, 0.045, 0.035)),
    Sku("tissue_box", 0.15, "box", (0.036, 0.055, 0.045)),
]}

# Totes sit to the robot's right, cartons to its left. The robot base is the origin; the bench top is z=0.
TOTE_SIZE = (0.18, 0.28, 0.035)  # outer x, y, wall height
TOTES = {
    "T1": (0.27, -0.15), "T2": (0.455, -0.15), "T3": (0.64, -0.15),
    "T4": (0.085, -0.44), "T5": (0.27, -0.44), "T6": (0.455, -0.44),
}
UNIT_OFFSETS_Y = [-0.085, 0.085, 0.0]

CARTON_HEIGHT = 0.13
CARTONS = {
    # Inner x, y sizes and slot centers relative to the carton center.
    "S": {"center": (0.17, 0.42), "inner": (0.17, 0.34), "max_kg": 5.0, "cost_usd": 0.60,
          "slots": {"A": (0.0, -0.085), "B": (0.0, 0.085)}},
    "L": {"center": (0.50, 0.27), "inner": (0.34, 0.34), "max_kg": 12.0, "cost_usd": 1.45,
          "slots": {"A": (-0.085, -0.085), "B": (0.085, -0.085), "C": (-0.085, 0.085), "D": (0.085, 0.085)}},
}
MAX_LAYERS = 2
# Sustained load, in newtons, that crushes a fragile item in the physics simulation.
FRAGILE_LIMIT_N = 2.0
