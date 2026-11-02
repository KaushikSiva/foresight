"""Find out what this Fireworks key can really run: Gemma availability, then one live image call per model."""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from foresight.llm import LARGE_MODEL, SMALL_MODEL, FireworksClient, image_part


def main():
    client = FireworksClient()
    request = urllib.request.Request(
        "https://api.fireworks.ai/v1/accounts/fireworks/models?filter=supports_serverless%3Dtrue&pageSize=200",
        headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            names = [m["name"] for m in json.load(response).get("models", [])]
        print("serverless models:", len(names))
        print("serverless gemma:", [n for n in names if "gemma" in n.lower()] or "none")
    except urllib.error.URLError as e:
        print("model list failed:", e)

    prompt = [{"role": "user", "content": [image_part(ROOT / "data/images/blocked-01.png"),
              {"type": "text", "text": 'List the objects you see and whether the bin is open. Reply with JSON only: {"objects": [], "bin_open": true}'}]}]
    for model in (SMALL_MODEL, LARGE_MODEL):
        try:
            reply = client.chat(model, prompt, json_mode=True, max_tokens=300)
            print(f"OK {model} {reply['latency_s']:.1f}s tokens={reply['prompt_tokens']}+{reply['completion_tokens']}\n{reply['text']}")
        except RuntimeError as e:
            print(f"FAILED {model}: {e}")


if __name__ == "__main__":
    main()
