# Foresight: warehouse order packing

A simulated Franka Panda packs warehouse orders in MuJoCo. The goal is to fine-tune GLM 5.3 Flash on Fireworks to plan each order, check the plan before anything moves, and record the improvement over time in Weights & Biases.

**Status: work in progress**

| Part | State |
|---|---|
| MuJoCo station, objects, arm control, crush and spill physics | Working |
| 200 orders (150 tuning, 50 held out) | Working: a reference solver packs all of them in physics |
| Video recording | Working |
| GLM planner on warehouse orders | Not built yet |
| Fine-tuning and W&B tracking | Not built yet |

There are no model results yet. The `foresight/` package, `app.py` and `static/` are an earlier tabletop prototype and are being replaced.

## Run it

Needs Python 3.11+ and `ffmpeg` (`brew install ffmpeg`). Developed on an Apple M4 Mac.

```bash
git clone https://github.com/KaushikSiva/foresight.git
cd foresight
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest tests/test_warehouse.py                 # physics and rules tests
python scripts/film.py hard-00                 # writes recordings/hard-00_solver.mp4
python scripts/film.py mixed-03 --crush        # films a bad plan that crushes a fragile item
python scripts/build_warehouse_eval.py         # rebuilds images and re-checks all 200 orders, a few minutes
```

No API keys are needed for any of the above.

## API keys

Model and tracking work needs keys. Put them in a `.env` file in the repo root; git ignores it. Never commit keys or paste them into code.

```bash
export FIREWORKS_API_KEY=...
export WANDB_API_KEY=...
```

Then run `source .env`. Ask Kaushik for the keys, or use your own accounts.

**Cost warning:** Fireworks fine-tuning and on-demand deployments bill real money. `scripts/ops.py` requires `--yes` before starting either.

## The task

The robot gets a camera image and an order, such as `2 x soup_can, 1 x glass_jar`. A plan is a short program over five skills:

| Skill | Meaning |
|---|---|
| `open_carton("S")` | Open carton S ($0.60, holds 5 kg, 2 slots) or L ($1.45, holds 12 kg, 4 slots) |
| `pick("T3")` | Pick one unit from a tote, T1 to T6 |
| `place("A", 1)` | Place into a slot at layer 1 (floor) or 2 (on top) |
| `seal()` | Seal the carton |
| `hold("reason")` | Do not ship, because stock cannot fill the order |

Packing rules: nothing on fragile items, heavy items (1.5 kg or more) on the floor only, no liquid above electronics, stay within the weight limit, and keep stacks to 17 cm or less.

Scoring runs the plan in physics on the true scene. Crushes come from measured contact forces, and spills from where items end up.

## Simulation limits

- It renders with MuJoCo's rasterizer, so it looks good but is not photoreal.
- The arm does not collide with objects, and a held item is rigidly attached to the hand from grasp to release.
- Fragile items count as crushed above a fixed 2 N sustained load, a simplification rather than a measured material limit.

## Layout

```
warehouse/catalog.py     SKUs, tote and carton geometry
warehouse/assets.py      generated meshes and textures
warehouse/world.py       MuJoCo scene
warehouse/robot.py       arm control, physical execution, crush and spill checks
warehouse/rules.py       skill parser and packing rules
warehouse/orders.py      order generator and reference solver
warehouse/scoring.py     physics-based scoring
warehouse/record.py      MP4 recording
scripts/film.py          film one order
scripts/autocommit.sh    commits and pushes every 5 minutes, with a secret scan
```

The Panda model comes from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) under Apache 2.0, in `warehouse/assets/panda/`.
