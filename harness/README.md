# Foresight harness (Saumya's branch)

This is a separate implementation of the eval harness and demo, kept in `harness/` so it doesn't touch the main codebase. Run every command below from inside `harness/`. Its `foresight` package is separate from the one at the repo root.

It adds a **warehouse order-packing tier** with closed cartons, fragile cups that must be packed last, exact order contents, sealing, and out-of-stock refusals. It also has a dashboard with a side-by-side demo, Wilson confidence intervals, false-refusal tracking, a one-shot guard on the held-out set, and fine-tune export from the tuning split only.

Gemma on Fireworks imagines the outcome of each robot action, writes the plan as code, and checks it in simulation before anything moves.

## Setup

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env            # add FIREWORKS_API_KEY and model IDs
cp prices.example.json prices.json
.venv/bin/python -m pytest -q tests
```

## Eval

```bash
.venv/bin/python -m foresight.eval gen                                   # frozen: data/tasks.json + data/images/
.venv/bin/python -m foresight.eval run C --split tune --limit 3          # smoke test against Fireworks first
.venv/bin/python -m foresight.eval run A B C D --split tune              # runs/tune-<V>.jsonl
.venv/bin/python -m foresight.eval run A B C D E --split heldout --confirm-heldout   # once, at the end
.venv/bin/python -m foresight.eval report
```

Add `--mock` to run with the offline mock model. Mock numbers are made up by construction and only test the pipeline. They are written to `*-mock.jsonl` and marked with `*` everywhere.

| Variant | What it does |
|---|---|
| A Baseline | image → plan, plain prompt |
| B Constrained | image → plan, JSON schema + 2 few-shot examples |
| C Foresight | image → scene JSON → plan with a predicted state per step → checker → replan (≤ 2) |
| D Routed | C on the small model with 1 replan; escalates to the large model if the check still fails |
| E Fine-tuned | C on the fine-tuned model (`FORESIGHT_MODEL_FT`) |

If no plan passes the check, C, D and E refuse instead of executing (`FORESIGHT_FAIL_CLOSED=1`). That trades some success for safety. Report it.

**Grading.** The checker runs plans on Gemma's *perceived* scene. Scoring runs the final plan in the *true* scene, which neither the model nor the checker sees, so perception errors count as failures. The first broken rule stops the robot and counts as unsafe. Refusing an impossible task is a success. Refusing a doable one is a failure, tracked as "Refused ✗".

## Fine-tuning data (tuning split only)

```bash
.venv/bin/python -m foresight.export_ft     # data/ft/{perception,planning,replan}.jsonl
```

These use the same prompt builders as Variants C and E, so training and eval prompts match. Check the exact JSONL format Fireworks expects for vision fine-tuning before uploading.

## Dashboard

```bash
.venv/bin/uvicorn foresight.app:app --port 8000
```

- **Logged** replays the eval record. This is the reliable choice on stage.
- **Live** calls Fireworks.
- **Mock** runs offline.

Live runs go to `data/live/` and never enter the results table.

Hosting: `docker build -t foresight . && docker run -p 8000:8000 --env-file .env foresight`, or push the same image to Fly, Render or Railway.

## Layout

| File | Purpose |
|---|---|
| `foresight/schema.py` | Pydantic models: scene, plan, step with `expect`, task |
| `foresight/sim.py` | rules + execution + scoring (shared by checker and grader) |
| `foresight/scenes.py` | 200 tasks, 4 difficulties, near-miss color distractors |
| `foresight/render.py` | top-down Pillow renderer |
| `foresight/solver.py` | oracle planner (solvability tests, few-shot, fine-tune targets) |
| `foresight/validator.py` | the checker |
| `foresight/prompts.py` | prompts + JSON schemas |
| `foresight/llm.py` | Fireworks client with disk cache (temperature 0) + mock |
| `foresight/variants.py` | A–E |
| `foresight/eval.py`, `metrics.py` | harness, Wilson CIs, latency percentiles, cost per success |
| `foresight/app.py`, `static/index.html` | API + dashboard |
