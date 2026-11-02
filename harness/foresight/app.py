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
from .metrics import _latest, compare, load_runs, table
from .render import image_bytes
from .schema import Plan
from .sim import execute
from .variants import VARIANTS, run_task

app = FastAPI(title="Foresight")
TASKS = {t.id: t for t in load_tasks()}
LIVE = config.ROOT / "data" / "live"
_llm = None


def _b64(img: bytes) -> str:
    mime = "image/jpeg" if img[:2] == b"\xff\xd8" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(img).decode()


def _with_after_image(rec: dict) -> dict:
    """Attach the TRUE scene after executing the final plan, as the robot would leave it."""
    task = TASKS[rec["task_id"]]
    if rec.get("final_plan"):
        res = execute(task.scene, Plan.model_validate(rec["final_plan"]))
        rec["after_image"] = _b64(image_bytes(task, res.final))
    return rec


@app.get("/")
def index():
    return FileResponse(config.ROOT / "static" / "index.html")


@app.get("/api/meta")
def meta():
    return {"variants": VARIANTS, "render": config.RENDER, "live_available": bool(os.environ.get("FIREWORKS_API_KEY")) and bool(config.MODEL),
            "models": {"main": config.MODEL, "small": config.MODEL_SMALL, "large": config.MODEL_LARGE,
                       "ft": config.MODEL_FT}}


@app.get("/api/results")
def results():
    return table()


@app.get("/api/compare")
def paired_results():
    """Each variant against the baseline on the same tasks: wins, losses, McNemar p, unsafe prevented, cost."""
    return compare()


@app.get("/api/tasks")
def tasks(split: Optional[str] = None):
    return [{"id": t.id, "split": t.split, "difficulty": t.difficulty, "instruction": t.instruction, "traps": t.traps}
            for t in TASKS.values() if split is None or t.split == split]


@app.get("/api/tasks/{task_id}/image.png")
def image(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(404)
    img = image_bytes(TASKS[task_id])
    return Response(img, media_type="image/jpeg" if img[:2] == b"\xff\xd8" else "image/png")


@app.get("/api/tasks/{task_id}/cinema.jpg")
def cinema(task_id: str):
    """A three-quarter view of the same scene for the demo. The model only ever sees the overhead camera."""
    if task_id not in TASKS or config.RENDER != "3d":
        raise HTTPException(404)
    from .render3d import jpeg_bytes
    return Response(jpeg_bytes(TASKS[task_id], camera="cinema"), media_type="image/jpeg")


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


# ------------------------------------------------------------------------------------------
# Manual review: step through tasks and graded runs, record verdicts in data/review.json.
# ------------------------------------------------------------------------------------------
REVIEW = config.ROOT / "data" / "review.json"


def _verdicts() -> dict:
    return json.loads(REVIEW.read_text()) if REVIEW.exists() else {}


def _runs_by_task(split: str) -> dict:
    """{task_id: {variant: record}} from the latest real (non-mock) run of each variant."""
    out: dict = {}
    for (sp, v, mock), rows in _latest(load_runs()).items():
        if sp != split or mock:
            continue
        for r in rows:
            out.setdefault(r["task_id"], {})[v] = r
    return out


def _outcome(r: dict) -> str:
    s = r["score"]
    if s["success"]:
        return "refused_ok" if s["refused"] else "success"
    if s.get("unsafe"):
        return "unsafe"
    if s.get("invalid_step"):
        return "bad_step"
    if s.get("invalid"):
        return "invalid"
    return "false_refusal" if s["refused"] else "goal_missed"


@app.get("/review")
def review_page():
    return FileResponse(config.ROOT / "static" / "review.html")


@app.get("/api/review/tasks")
def review_tasks(split: str = "tune"):
    runs, verdicts = _runs_by_task(split), _verdicts()
    return [{"id": t.id, "difficulty": t.difficulty, "traps": t.traps, "instruction": t.instruction, "possible": t.possible,
             "outcomes": {v: _outcome(r) for v, r in sorted(runs.get(t.id, {}).items())},
             "verdict": verdicts.get(t.id)} for t in TASKS.values() if t.split == split]


@app.get("/api/review/task/{task_id}")
def review_task(task_id: str):
    from .solver import solve
    if task_id not in TASKS:
        raise HTTPException(404)
    t = TASKS[task_id]
    runs = _runs_by_task(t.split).get(task_id, {})
    return {
        "task": t.model_dump(exclude={"layout"}),
        "oracle": [f"{s.skill}({', '.join(s.args)})" for s in solve(t).steps],
        "runs": {v: {**r, "outcome": _outcome(r)} for v, r in sorted(runs.items())},
        "verdict": _verdicts().get(task_id),
    }


@app.get("/api/review/after/{variant}/{task_id}")
def review_after(variant: str, task_id: str):
    """The true scene after this variant's final plan, rendered on demand."""
    r = _runs_by_task(TASKS[task_id].split).get(task_id, {}).get(variant)
    if not r or not r.get("final_plan"):
        raise HTTPException(404)
    img = image_bytes(TASKS[task_id], execute(TASKS[task_id].scene, Plan.model_validate(r["final_plan"])).final)
    return Response(img, media_type="image/jpeg" if img[:2] == b"\xff\xd8" else "image/png")


class Verdict(BaseModel):
    task: Optional[str] = None  # "ok" | "problem"
    grades: dict = {}  # variant -> "agree" | "disagree"
    note: str = ""


@app.post("/api/review/verdict/{task_id}")
def save_verdict(task_id: str, v: Verdict):
    if task_id not in TASKS:
        raise HTTPException(404)
    data = _verdicts()
    data[task_id] = {**v.model_dump(), "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    tmp = REVIEW.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    tmp.replace(REVIEW)
    return data[task_id]
