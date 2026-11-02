# Foresight demo script (2 minutes)

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

## Script, word for word

**[Kaushik, 0:00]**

Warehouse robots pack orders under hard rules: heavy items go down, nothing goes on fragile items, and cartons stay under the weight limit. Models can write these plans, but nothing checks them before the arm moves. So we built that check, and we call it Foresight.

**[Kaushik, 0:15. Play `extreme_reel.mp4` from the second title card, one order only.]**

These are the hardest orders we could build: seven or eight items, mixing heavy, fragile, liquid and electronics. Both sides use the same model. On the left, it plans alone. On the right, Foresight predicts each step, checks the rules, and runs a physics dry run before anything moves. On the left, a rule breaks. On the right, it's packed correctly.

**[Teammate, 0:45. Show this table.]**

| 30 extreme orders | Plain GLM | GLM + Foresight |
|---|---|---|
| Correct | 20/30 | **25/30** |
| When it read the image right | 17/22 | **22/22** |
| Median latency | 10 s | 31 s |

We fixed thirty extreme orders in advance, ran them once, and scored every plan in physics. The plain model got twenty right. With Foresight, it got twenty-five, with five wins and no losses. When it read the camera correctly, it got all twenty-two. That's promising, not proof, and it costs three times the latency, at under a cent per extra correct order.

**[Teammate, 1:10. Show the W&B curve, then this table.]**

| 50 held-out orders, with Foresight | Base GLM | Fine-tuned GLM |
|---|---|---|
| Images read correctly | 32/50 | **50/50** |
| Correct overall | 43/50 | 44/50 |
| Hard orders correct | **9/12** | 6/12 |

Every remaining failure came from misreading the camera, so that's what we fine-tuned, using six hundred simulated examples for about nine dollars. On fifty new orders, correct image reads went from thirty-two to all fifty. But overall success barely moved, forty-three to forty-four, and hard orders got worse, because we also turned reasoning off. That's the real tradeoff.

**[Kaushik, 1:45]**

Next, we'd ship fine-tuned vision with a reasoning planner, behind the Foresight check. A plan that fails never reaches the robot. That's safety you can measure, for under a cent per extra correct order. Thank you.

**Retrained model on the extreme test:** result pending. Add a line only if it's ready and worth mentioning.

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
- **Over time:** replace the fine-tuning table with one line: "Vision went from 32/50 to 50/50, overall success barely moved."

## Prep checklist

- [ ] Both speakers have rehearsed twice with a timer
- [ ] Both videos play from a local copy, with no wifi
- [ ] Retrained model result filled in, or that line removed
- [ ] Repo access works for judges, if they need it (it's private)
- [ ] Both people are registered, since solo entries aren't accepted
- [ ] Submit before 4:30 PM, with a 3:30 PM target
