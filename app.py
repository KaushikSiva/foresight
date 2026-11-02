"""Demo server: live plans on a scene, plus results read only from logged eval runs."""
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from foresight.agent import VARIANTS
from foresight.evaluation import IMAGES, latest_runs, load_run, load_scenes, run_scene, summarize
from foresight.llm import FireworksClient
from foresight.sim import execute, parse_program

ROOT = Path(__file__).resolve().parent
SCENES = {s["id"]: s for s in load_scenes()}

app = FastAPI(title="Foresight")
app.mount("/images", StaticFiles(directory=IMAGES), name="images")


class PlanRequest(BaseModel):
    scene_id: str
    variant: str = "C_foresight"


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/scenes")
def scenes():
    return [{k: s[k] for k in ("id", "tier", "split", "instruction")} for s in SCENES.values()]


def _with_truth_run(scene, record):
    """Attach the step-by-step run on the true state, for display after planning has finished."""
    steps = []
    if record.get("program") and record["score"].get("parsed", True):
        calls = parse_program(record["program"])
        result = execute(scene["state"], calls)
        steps = [{"call": f"{name}({', '.join(repr(a) for a in args)})",
                  "ok": not (result.violation and result.violation.step == i),
                  "reached": not result.violation or i <= result.violation.step}
                 for i, (name, args) in enumerate(calls)]
        record["violation_message"] = result.violation.message if result.violation else None
    record["truth_steps"] = steps
    return record


@app.post("/api/plan")
def plan(request: PlanRequest):
    scene = SCENES.get(request.scene_id)
    if scene is None or request.variant not in VARIANTS:
        raise HTTPException(404, "unknown scene or variant")
    if not os.environ.get("FIREWORKS_API_KEY"):
        raise HTTPException(503, "FIREWORKS_API_KEY is not set, so live planning is off")
    return {"source": "live", **_with_truth_run(scene, run_scene(FireworksClient(), request.variant, scene))}


@app.get("/api/recorded/{scene_id}/{variant}")
def recorded(scene_id: str, variant: str):
    """Replay a logged result, so the demo survives bad venue wifi. The UI labels it as recorded."""
    scene = SCENES.get(scene_id)
    if scene is None:
        raise HTTPException(404, "unknown scene")
    for (_, logged_variant), path in latest_runs().items():
        if logged_variant == variant:
            for record in load_run(path):
                if record["scene"] == scene_id:
                    return {"source": "recorded", "run": path.name, **_with_truth_run(scene, record)}
    raise HTTPException(404, "no logged run for that scene and variant")


@app.get("/api/results")
def results():
    return [{"split": split, "variant": variant, "run": path.name, **summarize(load_run(path))}
            for (split, variant), path in sorted(latest_runs().items())]
