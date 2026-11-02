# Foresight

Gemma on Fireworks imagines the outcome of each robot action, writes the plan as code, and checks it in simulation before anything moves.

This is a symbolic world action model inspired by [AW-0](https://github.com/ZhuoyangLiu2005/agentic-wam). It is not AW-0: it predicts scene state rather than video, and it outputs skill calls rather than joint actions.

**Status:** in progress. No model results have been recorded yet.

## How it works

1. **Perceive:** Gemma reads the camera image and reports the scene as JSON.
2. **Imagine and plan:** it writes a program over the skill API, plus the state it expects after each step.
3. **Check:** a validator runs the program on the *perceived* scene and flags broken rules or wrong predictions.
4. **Replan:** failed checks go back to Gemma, up to 2 retries. A plan that never passes is withheld.
5. **Score:** the final program runs against the true scene, which no variant ever sees.

## Eval

200 synthetic tabletop scenes (150 tuning, 50 held out), split evenly across easy, multi-step, blocked and impossible tasks.

| Variant | What changes |
|---|---|
| A: Baseline | Stock Gemma, plain prompt |
| B: Constrained | Structured output and an example |
| C: Foresight | Imagine, check and replan |
| D: Routed | Small Gemma, escalating to a larger model when the check fails |

## Run it

```bash
pip install -r requirements.txt
python scripts/generate_eval.py
pytest tests

export FIREWORKS_API_KEY=...
export FORESIGHT_SMALL_MODEL=accounts/<account>/deployments/<id>
export FORESIGHT_LARGE_MODEL=accounts/<account>/deployments/<id>
python scripts/smoke.py
python scripts/run_eval.py A_baseline
uvicorn app:app
```

Gemma 4 is not serverless on Fireworks, so it needs an on-demand deployment, which bills for GPU time.
