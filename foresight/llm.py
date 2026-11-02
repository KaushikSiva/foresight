"""Fireworks chat client (OpenAI-compatible) using only the standard library."""
import base64
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://api.fireworks.ai/inference/v1"
# Gemma 3 12B is text-only on Fireworks, so a serverless vision model reads the image.
PERCEPTION_MODEL = os.environ.get("FORESIGHT_PERCEPTION_MODEL", "accounts/fireworks/models/glm-5p3-flash")
PLANNER_MODEL = os.environ.get("FORESIGHT_PLANNER_MODEL", "accounts/fireworks/models/gemma-3-12b-it")
RETRYABLE = {408, 429, 500, 502, 503, 504}


def image_part(path):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


class FireworksClient:
    def __init__(self, api_key=None, timeout=120):
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        if not self.api_key:
            raise RuntimeError("FIREWORKS_API_KEY is not set")
        self.timeout = timeout

    def chat(self, model, messages, json_mode=False, max_tokens=1500):
        body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            f"{BASE_URL}/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        start = time.monotonic()
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                break
            except urllib.error.HTTPError as e:
                if e.code not in RETRYABLE or attempt == 3:
                    raise RuntimeError(f"Fireworks HTTP {e.code}: {e.read()[:300].decode(errors='replace')}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
                if attempt == 3:
                    raise RuntimeError(f"Fireworks request failed: {e}") from e
            time.sleep(2 ** attempt)
        usage = payload.get("usage", {})
        return {
            "text": payload["choices"][0]["message"].get("content") or "",
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "latency_s": time.monotonic() - start,
        }
