"""Fine-tune GLM 5.3 Flash on Fireworks Serverless Training, tracked in W&B.

At every checkpoint the same orders are perceived and planned by that checkpoint, every plan is scored in
MuJoCo physics, and the same orders are filmed. A hard spend cap stops the run.
"""
import argparse
import base64
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import transformers
import wandb
from fireworks.training import sdk
from fireworks.training.sdk import FireworksClient, FiretitanServiceClient

from warehouse.evaluation import load_scenes, run_variant, sample_scenes, summarize
from warehouse.planner import PERCEIVE_PROMPT, SKU_LOOKS, baseline, extract_json, foresight, normalize_perception, ModelOutputError

BASE_MODEL = "accounts/fireworks/models/glm-5p3-flash"
TOKENIZER = "zai-org/GLM-5.3-Flash"
SERVERLESS_URL = "https://api.fireworks.ai/training/v1/serverless"
# Serverless Training rates for GLM 5.3 Flash, USD per million tokens, from fireworks.ai/pricing.
RATE_PREFILL, RATE_SAMPLE, RATE_TRAIN = 2.96, 7.41, 8.89
IMAGE_TOKENS = 1196  # the trainer's token count for a 1280x720 image
SYSTEM = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>"
ANSWER_OPEN = "<|assistant|><think></think>"
END = "<|endoftext|>"
STOP = ["<|user|>", "<|observation|>", END]
RUNS = ROOT / "runs" / "training"
FILM_POOL = ThreadPoolExecutor(2)


class Spend:
    def __init__(self, cap):
        self.cap, self.train, self.prefill, self.sample = cap, 0, 0, 0

    @property
    def usd(self):
        return (self.train * RATE_TRAIN + self.prefill * RATE_PREFILL + self.sample * RATE_SAMPLE) / 1e6

    def over(self):
        return self.usd >= self.cap


class Texts:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.placeholder = tokenizer.convert_tokens_to_ids("<|image|>")
        self.vision_text = PERCEIVE_PROMPT.format(looks="\n".join(f"{k}: {v}" for k, v in SKU_LOOKS.items()))

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def text_prompt(self, prompt):
        return self.encode(SYSTEM + prompt + ANSWER_OPEN)

    def image_parts(self, text):
        return self.encode(SYSTEM + "<|begin_of_image|>"), self.encode("<|end_of_image|>" + text + ANSWER_OPEN)

    def text_datum(self, prompt, answer):
        before, after = self.text_prompt(prompt), self.encode(answer + END)
        tokens = before + after
        return sdk.Datum(model_input=sdk.ModelInput.from_ints(tokens[:-1]),
                         loss_fn_inputs={"target_tokens": tokens[1:], "weights": [0.0] * (len(before) - 1) + [1.0] * len(after)}), len(tokens)

    def image_datum(self, image_bytes, answer):
        pre, post = self.image_parts(self.vision_text)
        after = self.encode(answer + END)
        model_input = sdk.ModelInput(chunks=[
            sdk.EncodedTextChunk(tokens=pre),
            sdk.types.ImageChunk(data=image_bytes, format="jpeg", expected_tokens=IMAGE_TOKENS),
            sdk.EncodedTextChunk(tokens=post + after[:-1]),
        ])
        prefix = len(pre) + IMAGE_TOKENS + len(post)
        targets = [0] * (prefix - 1) + after
        weights = [0.0] * (prefix - 1) + [1.0] * len(after)
        return sdk.Datum(model_input=model_input, loss_fn_inputs={"target_tokens": targets, "weights": weights}), prefix + len(after)


class SnapshotClient:
    """Lets the planner code call a training snapshot the same way it calls the chat API."""

    def __init__(self, sampler, texts, spend):
        self.sampler, self.texts, self.spend = sampler, texts, spend

    def chat(self, model, messages, json_mode=False, max_tokens=1500):
        content = messages[0]["content"]
        params = sdk.SamplingParams(max_tokens=min(max_tokens, 1800), temperature=0.0, stop=STOP)
        start = time.monotonic()
        try:
            if isinstance(content, str):
                prompt = self.texts.text_prompt(content)
                count = len(prompt)
                future = self.sampler.sample(prompt=sdk.ModelInput.from_ints(prompt), num_samples=1, sampling_params=params)
            else:
                pre, post = self.texts.image_parts(content[1]["text"])
                count = len(pre) + IMAGE_TOKENS + len(post)
                future = self.sampler.sample(prompt=sdk.ModelInput.from_ints(pre + [self.texts.placeholder] + post), num_samples=1,
                                             sampling_params=params, images=[content[0]["image_url"]["url"]])
            tokens = list(future.result().sequences[0].tokens)
        except Exception as e:
            raise RuntimeError(f"sampling failed: {e}") from e
        self.spend.prefill += count
        self.spend.sample += len(tokens)
        return {"text": self.texts.tok.decode(tokens, skip_special_tokens=True), "model": "snapshot",
                "prompt_tokens": count, "completion_tokens": len(tokens), "latency_s": time.monotonic() - start}


def perceive_with(client, texts, scenes, image_dir=ROOT / "data" / "warehouse" / "images"):
    def run(scene):
        path = Path(image_dir) / f"{scene['id']}.jpg"
        url = "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": texts.vision_text}]}]
        try:
            reply = client.chat("snapshot", messages, max_tokens=400)
            totes = normalize_perception(extract_json(reply["text"]))
            return scene["id"], {"totes": totes, "error": None, "call": None}
        except (ModelOutputError, RuntimeError) as e:
            return scene["id"], {"totes": None, "error": str(e)[:200], "call": None}

    with ThreadPoolExecutor(5) as pool:
        return dict(pool.map(run, scenes))


def film(scene_id, program, step, title, out_dir):
    path = out_dir / f"step{step:04d}_{scene_id}.mp4"
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "film_plan.py"), scene_id, str(path), "--title", title, "--program", program or ""],
                            capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print("film failed", scene_id, result.stderr[-400:], flush=True)
        return None
    return path


def evaluate(step, training_client, service, texts, scenes, spend, pool, film_ids, out_dir, log, filmer):
    started = time.monotonic()
    snapshot = training_client.save_weights_for_sampler(f"s{step:04d}").result().path
    client = SnapshotClient(service.create_sampling_client(model_path=snapshot, tokenizer=texts.tok), texts, spend)
    perception = perceive_with(client, texts, scenes)
    read_right = sum(perception[s["id"]]["totes"] == s["totes"] for s in scenes)
    records = run_variant(client, "snapshot", foresight, scenes, perception, pool=pool)
    summary = summarize(records)
    summary.update(step=step, perception_right=read_right, snapshot=snapshot, spend_usd=round(spend.usd, 3), eval_minutes=round((time.monotonic() - started) / 60, 1))
    with open(out_dir / f"step{step:04d}.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps({**record, "perceived": perception[record["scene"]]["totes"]}) + "\n")
    log.write(json.dumps({"eval": summary}) + "\n")
    log.flush()
    print("EVAL", json.dumps({k: summary[k] for k in ("step", "successes", "n", "perception_right", "unsafe", "withheld", "errors", "outcomes", "tokens_out", "spend_usd", "eval_minutes", "by_tier")}), flush=True)

    metrics = {"eval/success_rate": summary["success_rate"], "eval/perception_rate": read_right / len(scenes), "eval/unsafe": summary["unsafe"],
               "eval/withheld": summary["withheld"], "eval/reply_tokens_mean": summary["tokens_out"] / len(scenes), "spend_usd": spend.usd}
    metrics.update({f"tier/{tier}": group["successes"] / group["n"] for tier, group in summary["by_tier"].items()})
    table = wandb.Table(columns=["scene", "tier", "outcome", "status", "perceived_right", "plan"])
    by_id = {s["id"]: s for s in scenes}
    for r in records:
        table.add_data(r["scene"], r["tier"], r["score"]["outcome"], r["status"], perception[r["scene"]]["totes"] == by_id[r["scene"]]["totes"], r.get("program") or "")
    metrics["eval/plans"] = table
    plan_of = {r["scene"]: r.get("program") for r in records}
    for scene_id in film_ids:
        filmer.append((scene_id, step, FILM_POOL.submit(film, scene_id, plan_of[scene_id], step, f"GLM 5.3 Flash, training step {step}", out_dir)))
    wandb.log(metrics, step=step)
    return summary


def held_out(best, service, texts, spend, pool, out_dir, log):
    """The one-time test: the checkpoint chosen on tuning orders, on orders it has never been judged on."""
    scenes = load_scenes("held_out")
    client = SnapshotClient(service.create_sampling_client(model_path=best["snapshot"], tokenizer=texts.tok), texts, spend)
    perception = perceive_with(client, texts, scenes)
    read_right = sum(perception[s["id"]]["totes"] == s["totes"] for s in scenes)
    records = run_variant(client, "snapshot", foresight, scenes, perception, pool=pool)
    summary = summarize(records)
    summary.update(step=best["step"], perception_right=read_right, split="held_out", spend_usd=round(spend.usd, 3))
    with open(out_dir / "held_out.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps({**record, "perceived": perception[record["scene"]]["totes"]}) + "\n")
    log.write(json.dumps({"held_out": summary}) + "\n")
    log.flush()
    print("HELD_OUT", json.dumps({k: summary[k] for k in ("step", "successes", "n", "perception_right", "unsafe", "withheld", "errors", "outcomes", "tokens_out", "p50_latency_s", "spend_usd", "by_tier")}), flush=True)
    return summary


def save_model(training_client, service, output_id):
    """Keep the tuned weights as a Fireworks model. Runs before anything can close the session."""
    training_client.save_weights_for_sampler("final").result()
    fw = FireworksClient(api_key=os.environ["FIREWORKS_API_KEY"])
    rows = [r for r in fw.list_training_session_checkpoints(service.training_session_name)
            if r.get("promotable") and "final" in (r.get("checkpointName") or r.get("name") or "")]
    if not rows:
        raise RuntimeError("no promotable final checkpoint found")
    target = max(rows, key=lambda r: r.get("createTime", ""))
    fw.promote_session_checkpoint(name=target["name"], output_model_id=output_id, base_model=BASE_MODEL)
    print("MODEL SAVED", output_id, "from", target["name"], flush=True)


def extreme(training_client, service, texts, spend, pool, out_dir, log, limit=None):
    """The fine-tuned model on the extreme orders, with and without the Foresight harness."""
    data = ROOT / "data" / "warehouse" / "extreme"
    with open(data / "scenes.jsonl") as f:
        scenes = [json.loads(line) for line in f]
    scenes = scenes[:limit] if limit else scenes
    snapshot = training_client.save_weights_for_sampler("extreme").result().path
    client = SnapshotClient(service.create_sampling_client(model_path=snapshot, tokenizer=texts.tok), texts, spend)
    perception = perceive_with(client, texts, scenes, data / "images")
    right = sum(perception[s["id"]]["totes"] == s["totes"] for s in scenes)
    for name, variant in (("A_tuned_plain", baseline), ("C_tuned_foresight", foresight)):
        records = run_variant(client, "snapshot", variant, scenes, perception, pool=pool)
        summary = summarize(records)
        summary.update(variant=name, perception_right=right, spend_usd=round(spend.usd, 3))
        with open(out_dir / f"extreme_{name}.jsonl", "w") as f:
            for record in records:
                f.write(json.dumps({**record, "perceived": perception[record["scene"]]["totes"]}) + "\n")
        log.write(json.dumps({"extreme": summary}) + "\n")
        log.flush()
        print("EXTREME", json.dumps({k: summary[k] for k in ("variant", "successes", "n", "perception_right", "unsafe", "withheld", "errors", "outcomes", "tokens_out", "p50_latency_s", "spend_usd", "by_tier")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="a tiny paid run to prove the loop: 4 examples, 2 eval orders")
    parser.add_argument("--max-usd", type=float, default=15.0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--checkpoints", default="0,25,50,75")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--per-tier", type=int, default=10)
    parser.add_argument("--output-model", default="foresight-glm-5p3-flash-warehouse")
    parser.add_argument("--keep-session", action="store_true", help="never delete the training session or its checkpoints")
    parser.add_argument("--skip-held-out", action="store_true", help="the held-out test is one-shot and has already run")
    parser.add_argument("--film", default="", help="comma-separated scene ids to film at every checkpoint")
    args = parser.parse_args()

    data = ROOT / "data" / "warehouse" / "train"
    tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    texts = Texts(tok)
    datums, tokens_each = [], []
    with open(data / "vision.jsonl") as f:
        vision = [json.loads(line) for line in f]
    with open(data / "plans.jsonl") as f:
        plans = [json.loads(line) for line in f]
    if args.smoke:
        vision, plans = vision[:2], plans[:2]
    for row in vision:
        datum, size = texts.image_datum((ROOT / row["image"]).read_bytes(), row["answer"])
        datums.append(datum)
        tokens_each.append(size)
    for row in plans:
        datum, size = texts.text_datum(row["prompt"], row["answer"])
        datums.append(datum)
        tokens_each.append(size)
    estimate = sum(tokens_each) * RATE_TRAIN / 1e6
    steps_total = len(datums) // args.batch + (1 if len(datums) % args.batch else 0)
    print(f"{len(vision)} vision + {len(plans)} plan examples, {sum(tokens_each):,} tokens, training estimate ${estimate:.2f}, {steps_total} steps, cap ${args.max_usd}", flush=True)
    if estimate >= args.max_usd:
        sys.exit("training alone would exceed the spend cap")

    scenes = sample_scenes(load_scenes("tuning"), 1 if args.smoke else args.per_tier)
    if args.smoke:
        scenes = scenes[:2]
    film_ids = [x for x in args.film.split(",") if x]
    checkpoints = [steps_total] if args.smoke else sorted({int(x) for x in args.checkpoints.split(",") if int(x) <= steps_total} | {steps_total})

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = RUNS / (("smoke-" if args.smoke else "run-") + stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "metrics.jsonl", "a")
    wandb.init(project="foresight-warehouse", name=out_dir.name, mode="online", config={**vars(args), "base_model": BASE_MODEL, "examples": len(datums), "eval_orders": [s["id"] for s in scenes]})

    spend = Spend(args.max_usd)
    service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"], base_url=SERVERLESS_URL)
    pool = ProcessPoolExecutor(8)
    results, filmer = [], []
    keep_session = False
    try:
        training_client = service.create_lora_training_client(BASE_MODEL, rank=args.rank)
        print("session", service.training_session_id, flush=True)
        if 0 in checkpoints:
            results.append(evaluate(0, training_client, service, texts, scenes, spend, pool, film_ids, out_dir, log, filmer))
        order = list(range(len(datums)))
        random.Random(7).shuffle(order)
        adam = sdk.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, grad_clip_norm=1.0)
        for step in range(1, steps_total + 1):
            if spend.over():
                print(f"STOPPED at step {step}: spend ${spend.usd:.2f} reached the cap", flush=True)
                break
            batch = order[(step - 1) * args.batch: step * args.batch]
            fb = training_client.forward_backward([datums[i] for i in batch], "cross_entropy").result()
            training_client.optim_step(adam).result()
            spend.train += sum(tokens_each[i] for i in batch)
            metrics = dict(fb.metrics or {})
            loss = metrics.get("loss:sum", 0.0) / max(float(metrics.get("response_tokens") or 1.0), 1.0)
            print(json.dumps({"step": step, "loss": round(loss, 4), "spend_usd": round(spend.usd, 3), "t": time.strftime("%H:%M:%S")}), flush=True)
            log.write(json.dumps({"step": step, "loss": loss, "spend_usd": spend.usd}) + "\n")
            log.flush()
            if step in checkpoints:
                results.append(evaluate(step, training_client, service, texts, scenes, spend, pool, film_ids, out_dir, log, filmer))
            wandb.log({"train/loss": loss, "spend_usd": spend.usd}, step=step)
        keep_session = True  # stays alive until the model is safely saved
        save_model(training_client, service, args.output_model)
        keep_session = False
        extreme(training_client, service, texts, spend, pool, out_dir, log, limit=2 if args.smoke else None)
        best = max(results, key=lambda r: (r["successes"], r["step"]))
        final = None
        if not args.smoke and not args.skip_held_out:
            final = held_out(best, service, texts, spend, pool, out_dir, log)
            wandb.summary.update({"held_out_success_rate": final["success_rate"], "held_out_unsafe": final["unsafe"]})
        for scene_id, step, future in filmer:
            path = future.result()
            if path:
                wandb.log({f"video/{scene_id}/step{step:04d}": wandb.Video(str(path), format="mp4")})
        (out_dir / "summary.json").write_text(json.dumps({"best": best, "held_out": final, "checkpoints": results, "spend_usd": spend.usd, "session": service.training_session_id,
                                                          "wandb_url": wandb.run.url}, indent=2))
        wandb.summary.update({"best_step": best["step"], "best_success_rate": best["success_rate"], "total_spend_usd": spend.usd})
        print("DONE best step", best["step"], f"{best['successes']}/{best['n']}", "spend", round(spend.usd, 2), "wandb", wandb.run.url, "dir", out_dir, flush=True)
    finally:
        pool.shutdown()
        log.close()
        wandb.finish()
        if keep_session or args.keep_session:
            print("SESSION KEPT:", service.training_session_id, flush=True)
        else:
            try:
                service.delete_training_session()
                print("session deleted", flush=True)
            except Exception as e:
                print("session delete failed:", e, flush=True)
        service.close()


if __name__ == "__main__":
    main()
