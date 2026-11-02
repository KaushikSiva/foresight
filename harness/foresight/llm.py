"""Fireworks client (OpenAI-compatible) with a disk cache, plus a mock for offline pipeline tests.

The cache stores the original latency, so re-running from cache reports real latency and cost.
"""
import asyncio
import hashlib
import json
import os
import random
import re
import time
from typing import List, Optional

import httpx
from pydantic import BaseModel

from . import config
from .schema import TABLE, Goal, Plan, Scene, Task, step_dict
from .solver import fill_expects, plan_steps, search


class Call(BaseModel):
    model: str
    purpose: str
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False
    error: Optional[str] = None


class Fireworks:
    mock = False

    def __init__(self, concurrency: int = config.CONCURRENCY, use_cache: bool = True):
        key = os.environ.get("FIREWORKS_API_KEY")
        if not key:
            raise RuntimeError("FIREWORKS_API_KEY is not set")
        self.client = httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"}, timeout=120)
        self.sem = asyncio.Semaphore(concurrency)
        self.use_cache = use_cache
        config.CACHE.mkdir(parents=True, exist_ok=True)

    async def chat(self, model: str, messages: list, purpose: str, schema: Optional[dict] = None,
                   max_tokens: int = 2000) -> Call:
        body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
        if schema is not None:
            body["response_format"] = {"type": "json_object", "schema": schema}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        path = config.CACHE / f"{key}.json"
        if self.use_cache and path.exists():
            return Call(**{**json.loads(path.read_text()), "purpose": purpose, "cached": True})

        async with self.sem:
            t0, last = time.perf_counter(), None
            for attempt in range(5):
                try:
                    r = await self.client.post(config.FIREWORKS_URL, json=body)
                except httpx.HTTPError as e:
                    last = repr(e)
                else:
                    if r.status_code == 200:
                        break
                    last = f"HTTP {r.status_code}: {r.text[:300]}"
                    if r.status_code not in (429, 500, 502, 503, 504):
                        return Call(model=model, purpose=purpose, error=last, latency_s=time.perf_counter() - t0)
                await asyncio.sleep(min(2 ** attempt, 20) + random.random())
            else:
                return Call(model=model, purpose=purpose, error=last, latency_s=time.perf_counter() - t0)
            latency = time.perf_counter() - t0

        data = r.json()
        usage = data.get("usage") or {}
        call = Call(model=model, purpose=purpose, text=data["choices"][0]["message"].get("content") or "",
                    prompt_tokens=usage.get("prompt_tokens", 0), completion_tokens=usage.get("completion_tokens", 0),
                    latency_s=latency)
        path.write_text(call.model_dump_json(exclude={"purpose", "cached"}))
        return call

    async def aclose(self):
        await self.client.aclose()


def parse_json(text: str):
    """Lenient JSON extraction: strips code fences and takes the first {...} or [...] block."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return None
    start = min(starts)
    end = text.rfind("}" if text[start] == "{" else "]")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# ------------------------------------------------------------------------------------------
# Mock model: for testing the pipeline without an API key. It knows what the instruction asks
# (task.requested), looks at a noisy copy of the true scene, and makes mistakes at a set rate. Its numbers
# are made up by construction, so every mock run is flagged and the dashboard labels it.
# ------------------------------------------------------------------------------------------
NEAR = {"red": "orange", "orange": "red", "pink": "red", "yellow": "orange", "green": "blue", "blue": "purple",
        "purple": "blue"}


class Mock:
    mock = True
    SLOPPY = {"direct_plain": 0.55, "direct_constrained": 0.4, "plan": 0.4, "replan": 0.05}

    def __init__(self, task: Task, perception_error: float = 0.08):
        self.task, self.perception_error = task, perception_error
        self.n = 0

    def _rng(self, *salt) -> random.Random:
        return random.Random("|".join([self.task.id, *map(str, salt)]))

    def _perceive(self, model: str) -> Scene:
        s = self.task.scene.model_copy(deep=True)
        rng = self._rng("perceive", model)
        if s.objects and rng.random() < self.perception_error:
            o = rng.choice(s.objects)
            new = NEAR[o.color]
            if f"{new}_{o.kind}" not in {x.id for x in s.objects}:
                o.color, o.id = new, f"{new}_{o.kind}"
        for c in s.containers:  # closed contents are invisible
            if not c.is_open:
                s.objects = [o for o in s.objects if o.location != c.id]
        return s

    def _plan(self, scene: Scene, sloppy: bool, with_expect: bool) -> dict:
        goals, sealed = self.task.requested, list(self.task.sealed) or [
            c for c in dict.fromkeys(g.location for g in self.task.requested) if c.startswith("carton")]
        missing = [g for g in goals if scene.obj(g.obj) is None or
                   (g.location != TABLE and scene.container(g.location) is None)]
        if missing and not sloppy:
            return {"goal": [], "sealed": [], "steps": [{"skill": "refuse", "args": [f"{missing[0].obj} or {missing[0].location} is not here"]}]}
        if missing:  # sloppy: grab a lookalike instead of refusing
            fixed = []
            for g in goals:
                o = scene.obj(g.obj) or next((x for x in scene.objects if x.kind == g.obj.split("_")[1]), None)
                fixed.append(Goal(obj=o.id if o else g.obj, location=g.location if scene.container(g.location) else TABLE))
            goals = fixed
        if sloppy:
            steps = plan_steps(scene, goals, prereqs=False, sealed=sealed)
        else:
            steps = search(scene, goals, sealed)
            if steps is None:
                return {"goal": [], "sealed": [], "steps": [{"skill": "refuse", "args": ["no safe plan can do this"]}]}
        if with_expect:
            steps = fill_expects(scene, steps)
        plan = Plan(steps=steps)
        return {"goal": [g.model_dump() for g in goals], "sealed": sealed, "steps": [step_dict(s) for s in plan.steps]}

    async def chat(self, model: str, messages: list, purpose: str, schema=None, max_tokens: int = 2000) -> Call:
        self.n += 1
        small_penalty = 0.15 if "small" in model else 0.0
        if purpose == "perceive":
            s = self._perceive(model)
            text = json.dumps({"objects": [o.model_dump() for o in s.objects],
                               "containers": [c.model_dump() for c in s.containers]})
        else:
            scene = self._perceive(model)
            sloppy = self._rng(purpose, model, self.n).random() < self.SLOPPY[purpose] + small_penalty
            text = json.dumps(self._plan(scene, sloppy, with_expect=purpose in ("plan", "replan")))
        prompt_chars = sum(len(json.dumps(m)) for m in messages)
        return Call(model=model, purpose=purpose, text=text, prompt_tokens=prompt_chars // 4,
                    completion_tokens=len(text) // 4, latency_s=0.35 + len(text) / 4 / 60)

    async def aclose(self):
        pass
