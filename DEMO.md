# Foresight demo script (3 minutes)

**One-liner:** Foresight checks every robot plan in simulation before anything moves, and fine-tuning on simulated data teaches the model to read the station's camera.

Say "the Foresight harness" and "the fine-tuned model" separately. They are different things.

## Roles

- **Speaker A (Kaushik):** problem, live video, close.
- **Speaker B (teammate):** results and tradeoffs.
- Whoever isn't speaking runs the laptop.

## Open before you start

1. `runs/warehouse/extreme_videos/extreme_reel.mp4`, the main video
2. `runs/training/run-20261102-135436/improvement_reel.mp4`, the training video
3. [W&B project](https://wandb.ai/kaushik-siva88/foresight-warehouse), logged in
4. This file, as a backup for the numbers

## Script

### 0:00 to 0:25: The problem (A)

> A warehouse robot gets an order and has to pack it: heavy items down, nothing on fragile items, no liquid above electronics, inside the weight limit. One bad plan means crushed goods or a useless carton. Models are good at writing plans, but nothing checks them before the arm moves. So we built that check.

### 0:25 to 1:05: Watch it (A)

Play `extreme_reel.mp4` from the second title card, one order only.

> These are the hardest orders we could build: seven or eight items, mixing heavy, fragile, liquid and electronics, with lookalike products. Same model on both sides. On the left it plans alone. On the right it goes through Foresight: it predicts each step, checks the rules, runs a MuJoCo physics dry run, and retries if anything fails. Left, a rule breaks. Right, it's packed.

### 1:05 to 1:45: The numbers (B)

> We fixed these orders before seeing any results and ran them once, scoring every plan in physics.

| 30 extreme orders | Plain GLM | GLM + Foresight |
|---|---|---|
| Correct | 20/30 | **25/30** |
| When it read the image right | 17/22 | **22/22** |
| Median latency | 10 s | 31 s |
| API cost for the whole test | $0.03 | $0.08 |

> Foresight won 5 orders and lost none. That's p = 0.06, so promising but not proof. When the model read the camera correctly, it packed every order. The price is about three times the latency and under a cent per extra correct order.

> On a broader task set of 50, unsafe plans dropped from 10 to 0, and success rose from 36 to 50 out of 50.

### 1:45 to 2:30: Fine-tuning (B)

> Every remaining failure came from misreading the camera, so that's what we tuned. We trained GLM 5.3 Flash on 600 simulated examples for about $9, tracking it in Weights & Biases.

Show the W&B curve, then a few seconds of `improvement_reel.mp4`.

| 50 held-out orders, with Foresight | Base GLM | Fine-tuned GLM |
|---|---|---|
| Images read correctly | 32/50 | **50/50** |
| Correct overall | 43/50 | 44/50 |
| Reply tokens per order | 2,355 | **312** |
| Hard orders correct | **9/12** | 6/12 |

> Vision became perfect on this set and tokens dropped sharply. But overall success barely moved, and hard orders got worse, because we also turned reasoning off. That's the tradeoff, and we'd rather show it than hide it.

**Extreme test on the retrained model:** RESULT PENDING. Fill this in before presenting. If it isn't ready, don't mention it.

### 2:30 to 3:00: Close (A)

> So here's what we'd ship: fine-tuned vision to read the station, and a reasoning planner behind the Foresight check. A failed plan never reaches the robot; it gets withheld. That's safety you can measure, at under a cent per extra correct order.

The combined system in the first sentence is **untested**. Keep saying "what we'd ship next" unless it has been run.

## Rules for presenting

- **Don't say "statistically significant"** about the extreme test. Say "5 wins, 0 losses".
- **Don't say the model went from 3/40 to 38/40.** Step 0 ran with reasoning off, which makes it look far worse than the real base model. On those 40 tuning orders, base with Foresight scored 34/40.
- **Don't say "87% fewer tokens" without context.** Base with low reasoning used about 600 tokens per order on tuning orders, so the fair saving is closer to 40 to 50%.
- **Don't call the 50/50 task set warehouse-only.** It mixes tabletop and packing tasks, and its checker and scorer share the same rules.
- **Say "simulated."** Nothing has run on real hardware or with real warehouse data.
- **Say the extreme reel shows wins only.** Always give the 25/30 total alongside it.

## Questions to prepare for

**Is this production ready?**
No. The base model runs on Fireworks serverless. A tuned model needs a paid on-demand GPU, and we haven't deployed one. Check whether the retrained model has been saved before you answer.

**How realistic is the simulation?**
It uses MuJoCo physics and a real Franka Panda arm model, rendered with MuJoCo's standard renderer, so it's not photoreal. The arm doesn't collide with objects, held items attach rigidly to the hand, and a fragile item counts as crushed above a fixed 2 N load.

**Can the check still miss something?**
Yes. On one held-out order, a plan passed the dry run and then crushed a jar in final scoring, at 2.82 N against the 2 N limit. Small placement differences changed the outcome. Also, the check can't fix a misread scene.

**Why not Gemma?**
Gemma 4 with vision needs 4 B200 GPUs on Fireworks, about $52/hour, and Gemma 3 is listed as text-only. GLM 5.3 Flash is serverless and handles images.

**Why did hard orders get worse after tuning?**
Most likely because the tuned model answers without reasoning, though we didn't isolate that. Perception was correct on all 6 failures, and those plans never passed the check, so the robot didn't move.

**What did it cost?**
About $9.44 for the first training run. The retrain has a $12 cap. Each eval run costs cents.

## If something breaks

- **Video won't play:** use the same files in the GitHub repo, or the copies on Kaushik's Desktop.
- **W&B won't load:** skip the curve and use the tables above.
- **Over time:** cut the fine-tuning table down to one line: "Vision went from 32/50 to 50/50, overall success barely moved."

## Prep checklist

- [ ] Both speakers have rehearsed twice with a timer
- [ ] Both videos play from a local copy, with no wifi
- [ ] Retrained model result filled in, or that line removed
- [ ] Repo access works for judges, if they need it (it's private)
- [ ] Both people are registered, since solo entries aren't accepted
- [ ] Submit before 4:30 PM, with a 3:30 PM target
