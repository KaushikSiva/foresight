"""Model IDs and prices. Everything comes from the environment (see .env.example)."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
CACHE = ROOT / "data" / "cache"

# Minimal .env support; real environment variables win.
if (ROOT / ".env").exists():
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


FIREWORKS_URL = _env("FIREWORKS_URL", "https://api.fireworks.ai/inference/v1/chat/completions")

# A, B, C use MODEL. D starts on SMALL and escalates to LARGE. E is Foresight on the fine-tuned model.
MODEL = _env("FORESIGHT_MODEL")
MODEL_SMALL = _env("FORESIGHT_MODEL_SMALL") or MODEL
MODEL_LARGE = _env("FORESIGHT_MODEL_LARGE") or MODEL
MODEL_FT = _env("FORESIGHT_MODEL_FT")

CONCURRENCY = int(_env("FORESIGHT_CONCURRENCY", "8"))
# "3d" renders in the MuJoCo packing station (needs mujoco + warehouse/), "2d" is the flat Pillow view.
RENDER = _env("FORESIGHT_RENDER", "3d" if ((ROOT / "warehouse").is_dir() or (ROOT.parent / "warehouse").is_dir()) else "2d")
MAX_REPLANS = int(_env("FORESIGHT_MAX_REPLANS", "2"))
# For thinking models (e.g. GLM-5.3, which cannot turn thinking off). Same value for every variant. "" = omit.
REASONING_EFFORT = _env("FORESIGHT_REASONING_EFFORT", "low")
MAX_TOKENS = int(_env("FORESIGHT_MAX_TOKENS", "8000"))
# If the plan still fails the check after all replans, refuse instead of executing it.
FAIL_CLOSED = _env("FORESIGHT_FAIL_CLOSED", "1") == "1"


def prices() -> dict:
    """$ per 1M tokens, {model_id: [input, output]}. Costs are computed from logged tokens at
    report time, so prices can be corrected later without re-running anything."""
    p = ROOT / "prices.json"
    return json.loads(p.read_text()) if p.exists() else {}
