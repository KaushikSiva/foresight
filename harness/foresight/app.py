"""API + dashboard.

  uvicorn foresight.app:app --reload        # http://localhost:8000

Results come only from runs/*.jsonl. Live demo runs are written to data/live/
so they can never leak into the results table.
"""
import base64
import json
import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from . import config
from .eval import load_tasks
from .metrics import load_runs, table
from .render import png_bytes
from .schema import Plan
from .sim import execute
from .variants import VARIANTS, run_task

app = FastAPI(title="Foresight")
TASKS = {t.id: t for t in load_tasks()}
LIVE = config.ROOT / "data" / "live"
_llm = None


def _b64(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _with_after_image(rec: dict) -> dict:
    """Attach the TRUE scene after executing the final plan, as the robot would leave it."""
    task = TASKS[rec["task_id"]]
    if rec.get("final_plan"):
        res = execute(task.scene, Plan.model_validate(rec["final_plan"]))
        rec["after_image"] = _b64(png_bytes(task, res.final))
    return rec


@app.get("/")
def index():
    return FileResponse(config.ROOT / "static" / "index.html")


@app.get("/api/meta")
def meta():
    return {"variants": VARIANTS, "live_available": bool(os.environ.get("FIREWORKS_API_KEY")) and bool(config.MODEL),
            "models": {"main": config.MODEL, "small": config.MODEL_SMALL, "large": config.MODEL_LARGE,
                       "ft": config.MODEL_FT}}


@app.get("/api/results")
def results():
    return table()


@app.get("/api/tasks")
def tasks(split: Optional[str] = None):
    return [{"id": t.id, "split": t.split, "difficulty": t.difficulty, "instruction": t.instruction}
            for t in TASKS.values() if split is None or t.split == split]


@app.get("/api/tasks/{task_id}/image.png")
def image(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(404)
    return Response(png_bytes(TASKS[task_id]), media_type="image/png")


@app.get("/api/logged/{variant}/{task_id}")
def logged(variant: str, task_id: str):
    """The latest logged eval record for this task and variant (the demo's reliable path).
    Real runs win; a mock record is returned only when no real one exists, and it is flagged."""
    rows = [r for r in load_runs() if r["variant"] == variant and r["task_id"] == task_id]
    hits = [r for r in rows if not r["mock"]] or rows
    if not hits:
        raise HTTPException(404, "no logged run for this task and variant")
    return _with_after_image(hits[-1])


class RunReq(BaseModel):
    task_id: str
    variant: str
    mock: bool = False


@app.post("/api/run")
async def run(req: RunReq):
    global _llm
    if req.task_id not in TASKS or req.variant not in VARIANTS:
        raise HTTPException(404)
    task = TASKS[req.task_id]
    if req.mock:
        from .llm import Mock
        llm = Mock(task)
    else:
        if not os.environ.get("FIREWORKS_API_KEY") or not config.MODEL:
            raise HTTPException(400, "FIREWORKS_API_KEY and FORESIGHT_MODEL must be set for live runs")
        from .llm import Fireworks
        _llm = _llm or Fireworks()
        llm = _llm
    t0 = time.time()
    rec = await run_task(req.variant, task, llm)
    rec["wall_s"] = round(time.time() - t0, 2)
    LIVE.mkdir(parents=True, exist_ok=True)
    with open(LIVE / f"{time.strftime('%Y%m%d')}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return _with_after_image(rec)
