<div align="center">

# Foresight

### See the outcome before the arm moves.

Foresight checks a robot's plan in physics simulation before anything moves, then blocks the plans that would break something.

<img src="docs/hero.gif" width="720" alt="A simulated Franka Panda arm packing a warehouse order in MuJoCo">

**[▶ Watch the 2-minute demo](docs/foresight-demo.mp4)** · **[Training runs on W&B](https://wandb.ai/kaushik-siva88/foresight-warehouse)** · **[Quick start](#quick-start)**

</div>

---

## Why this exists

Language models are good at writing robot plans, but nothing checks those plans before the arm moves. The first sign of a bad plan can be a crushed product.

So we added a step in between. The model writes a plan and predicts each result, a harness checks it against the rules and runs it in physics, and only a plan that passes reaches the arm.

**Without a check, our fine-tuned model produced an unsafe plan on 16 of 30 hard orders (11 in another training run). With Foresight: 0.**

<img src="docs/before_after.png" alt="The same order and model side by side: plain GLM breaks a packing rule, and GLM with Foresight packs correctly">

*The same model on the same order. These plans are real logged outputs, run in MuJoCo.*

## Results

We built 30 extreme warehouse orders before any model saw them: 7–8 items mixing heavy, fragile, liquid and electronics, with lookalike products. Every plan was scored by running it in physics.

| Setup | Correct | Images read right | Unsafe |
|---|---:|---:|---:|
| Fine-tuned GLM alone, no check | 10/30 | 30/30 | 16 |
| Fine-tuned GLM + Foresight | 12/30 | 30/30 | **0** |
| Plain base GLM | 20/30 | 22/30 | 2 |
| Base GLM + Foresight | 25/30 | 22/30 | 1 |
| **Fine-tuned vision + reasoning planner + Foresight** | **30/30** | **30/30** | **0** |

<img src="docs/results.png" alt="Bar chart of the five setups, ending with the combined system at 30 out of 30">

**What we learned:**

- **The check helps:** base GLM went from 20/30 to 25/30, with 5 wins and 0 losses (p = 0.06, so promising rather than proven).
- **Fine-tuning fixed the camera:** on 50 held-out orders, correct image reads went from 32/50 to 50/50.
- **Fine-tuning alone wasn't enough:** on those same orders, success barely moved (43 to 44), and on the extreme orders it fell to 12/30, most likely because the tuned model plans without reasoning.
- **Splitting the jobs worked best:** tuned vision for reading the camera and a reasoning model for planning.

> **Read this before quoting 30/30.** We chose that combination after seeing these results, on these same 30 orders, using logged camera reads. It's a strong lead that still needs a fresh test as one live system.

## How it works

<img src="docs/architecture.png" alt="Architecture: camera to fine-tuned GLM to reasoning GLM to the Foresight harness, then the arm moves or the plan is withheld">

1. **See:** a vision model reads the overhead camera and reports what's in each tote.
2. **Plan and predict:** a reasoning model writes the program and predicts each step's result.
3. **Check:** the harness tests the program against the packing rules.
4. **Dry run:** a passing plan runs in MuJoCo, which measures crushes and spills.
5. **Retry or withhold:** failures go back with the exact errors, up to twice. A plan that still fails never reaches the arm.

Planners only ever see what the camera read. Final scoring runs in physics on the true scene, so a misread can't hide.

A plan is a short, parsed (never executed) program:

```python
open_carton("L")
pick("T4")          # bearing_box, 3.2 kg, goes on the carton floor
place("A", 1)
pick("T2")          # ceramic_mug, fragile, nothing may rest on it
place("B", 1)
seal()
```

Rules include: nothing rests on fragile items, heavy items stay on the floor, no liquid above electronics, stay within weight and stack-height limits, and use the cheapest carton that fits. If stock can't fill the order, the only correct plan is `hold("reason")`.

### The crush is measured, not scripted

MuJoCo reports contact forces, and a fragile item counts as crushed above a sustained 2 N. In our test, a 3.2 kg box on a mug registered 31.4 N.

<img src="docs/crush.png" width="640" alt="A real logged plan from the fine-tuned model ending in a crushed item">

### What the camera sees

<img src="docs/overhead.jpg" width="640" alt="Overhead camera view of the packing bench, totes and cartons">

## Fine-tuning

We trained a LoRA on **GLM 5.3 Flash** with Fireworks Serverless Training: 600 simulated examples (300 camera images, 300 packing plans) over 75 steps, for about $9 per run. No training scene matches an eval order.

At each checkpoint, the same orders are read, planned, scored in physics and filmed, so you can watch the improvement directly.

<img src="docs/training.png" alt="Training loss curve and camera read accuracy rising from 22 to 40 out of 40 across checkpoints">

<img src="docs/wandb.png" alt="Weights and Biases run page showing the videos logged at each checkpoint">

**[See every run, curve and checkpoint video on W&B →](https://wandb.ai/kaushik-siva88/foresight-warehouse)**

*Step 0 ran with reasoning off, so it looks worse than the real base model. Compare tuned results against the base model numbers above, not against step 0.*

## Quick start

Needs Python 3.11+ and `ffmpeg`. Developed on an Apple M4 Mac. No API keys are needed for these commands:

```bash
git clone https://github.com/KaushikSiva/foresight.git
cd foresight
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest tests/test_warehouse.py            # physics and rules tests
python scripts/film.py hard-00            # film a correct pack
python scripts/film.py --crush            # film a plan that crushes a mug
python scripts/build_warehouse_eval.py    # rebuild and re-check all 200 orders
```

Videos land in `recordings/`.

### With API keys

Put your keys in a `.env` file, which git ignores:

```bash
export FIREWORKS_API_KEY=...
export WANDB_API_KEY=...
```

```bash
source .env
python scripts/warehouse_perceive.py                # read every camera image, costs cents
python scripts/warehouse_eval.py A_baseline C_foresight --per-tier 10   # plain vs. Foresight
python scripts/build_warehouse_training.py          # build the training data, free
python scripts/train_glm.py --max-usd 12 --skip-held-out --output-model my-model   # about $9
```

`train_glm.py` bills real money and stops at `--max-usd`.

## Limits

This was built in a day at a hackathon, so please read these before relying on the numbers:

- **Simulation only:** nothing has run on a real robot, and all data is synthetic: 10 products, one bench, one camera.
- **Simplified physics:** the arm doesn't collide with objects, held items attach rigidly to the hand, and the crush limit is a fixed force rather than a measured material property.
- **Small samples:** 30 and 50 orders are enough to spot big gaps, not small ones.
- **Not airtight:** on one held-out order, a plan passed the dry run and then crushed a jar in final scoring, because small placement differences changed the outcome.
- **Slower:** the check makes planning about 3× slower, a 31 s median on extreme orders.
- **Not deployed:** the tuned model is saved on Fireworks, but serving it needs a paid on-demand GPU, and we haven't run the full system live.

**Next up:** test the combined system on new orders, then deploy it end to end.

## Repo guide

```
warehouse/      MuJoCo station, arm control, rules, orders, physics scoring, planners
scripts/        eval, training, filming and reel tools
data/warehouse/ 200 eval orders, 30 extreme orders, training examples
runs/           logged plans, scores and checkpoint videos from real runs
harness/        Saumya's separate eval harness, with stats and a dashboard
docs/           README images and the demo video
foresight/      early tabletop prototype
```

Saumya's [`harness/`](harness/README.md) tests the same idea on a broader 50-task set, with confidence intervals and paired statistics: plain GLM 36/50 vs. checked 50/50, with unsafe plans falling from 10 to 0. Its checker and grader share the same rules, so read its caveats alongside the result.

## Built with

[Fireworks AI](https://fireworks.ai) · [GLM 5.3 Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) · [MuJoCo](https://mujoco.org) · [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Franka Panda, Apache 2.0) · [Weights & Biases](https://wandb.ai) · [Claude Code](https://claude.com/claude-code)

## Team

**Saumya · Kaushik · Shalin**, built at the Fireworks Deploy Buildathon, San Francisco, November 2026.

If this is useful to you, a ⭐ helps others find it.
