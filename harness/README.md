# Foresight harness (Saumya's branch)

This is a separate implementation of the eval harness and demo, kept in `harness/` so it doesn't touch the main codebase. Run every command below from inside `harness/`. Its `foresight` package is separate from the one at the repo root. For 3D rendering it uses Kaushik's `warehouse/` from the repo root.

What it adds:
- **200 tasks across 5 tiers**, including warehouse order packing. Every task is proven solvable by A* search, or is impossible by construction.
- **A difficulty audit**, showing that no fixed recipe passes the hard tiers.
- **Photoreal frames** from the MuJoCo packing station.
- **A dashboard** with a side-by-side demo and a robot-camera / 3/4-view toggle.
- **Eval safeguards:** Wilson confidence intervals, false-refusal tracking, and a held-out set that can only be run once.
- **Fine-tune export** from the tuning split only.

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
.venv/bin/python -m foresight.eval gen                                   # frozen tasks + 3D renders (~4 min)
.venv/bin/python -m foresight.audit                                      # is each tier a good test? (see below)
.venv/bin/python -m foresight.eval run C --ids packing-001 packing-002 packing-006 --name smoke   # smoke test first
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

## Tasks and the difficulty audit

There are 200 tasks: 40 per tier across easy, multi-step, blocked, impossible and packing, split 150 tuning / 50 held out. Each tier mixes several traps, and every scene has a lookalike distractor.

- **Solvable means solvable.** An A* search over the simulator's own rules proves each doable task can be done. Impossible tasks are impossible by construction (missing item or container, or more items than the destination holds).
- **The audit** (`python -m foresight.audit`) checks two reference planners against each tier:
  - a fixed recipe: open lids, clear the tray, pack cups last, seal
  - the literal instruction, done in the listed order

  On multi-step, blocked and packing, the fixed recipe must pass at most 50% and the literal plan at most 20%. Every tier meets this. A test enforces it.

| Tier | Traps |
|---|---|
| Easy | calibration: lookalikes only |
| Multi-step | make room, swap, three items, relocate |
| Blocked | closed lid, reclose, occupied tray, full container, two obstacles |
| Impossible | missing item (a lookalike is present), missing container, over capacity |
| Packing | repack fragile (cup out, sturdy in, cup back), wrong item in carton, two orders, out of stock, over capacity |

## Rendering

By default, scenes render in Kaushik's MuJoCo packing station (`warehouse/`): his bench, lighting, racks and Panda arm, with our containers and colored items.
- **Robot camera:** the overhead view. It's the only one the model sees.
- **3/4 view:** for the demo only.

Frames are cached in `data/images3d/`. Set `FORESIGHT_RENDER=2d` to use the flat Pillow view instead.

## Fine-tuning data (tuning split only)

```bash
.venv/bin/python -m foresight.export_ft     # data/ft/{perception,planning,replan}.jsonl (needs the 3D renders)
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
| `foresight/scenes.py` | 200 tasks, 5 tiers of traps, lookalike distractors |
| `foresight/render.py` | flat Pillow renderer + image dispatch |
| `foresight/render3d.py` | scenes in the MuJoCo station (uses `warehouse/`) |
| `foresight/audit.py` | per-tier difficulty audit |
| `foresight/solver.py` | A* oracle, fixed-recipe and literal baselines |
| `foresight/validator.py` | the checker |
| `foresight/prompts.py` | prompts + JSON schemas |
| `foresight/llm.py` | Fireworks client with disk cache (temperature 0) + mock |
| `foresight/variants.py` | A–E |
| `foresight/eval.py`, `metrics.py` | harness, Wilson CIs, latency percentiles, cost per success |
| `foresight/app.py`, `static/index.html` | API + dashboard |
