"""Perception and planner variants. No variant ever sees the true scene; scoring happens separately in physics."""
import base64
import json
import re
from pathlib import Path

from .catalog import CARTONS, CATALOG, HEAVY_KG, TOTES
from .orders import order_text
from .robot import execute
from .rules import MAX_STACK_M, ParseError, apply_call, fillable, new_state, packed_counts, parse_call

MAX_REPLANS = 2
# How each product looks to the overhead camera, which cannot see side labels.
SKU_LOOKS = {
    "ceramic_mug": "open white cup: a ring with a hollow inside and a small handle on one side",
    "glass_jar": "round cream-yellow lid with a brown rim around it",
    "soup_can": "solid round red top",
    "paint_can": "solid round pale grey lid with no hollow and no handle",
    "detergent_jug": "rectangle, bright blue with a yellow band across one end",
    "hardcover_book": "rectangle, dark green with a gold band across one end",
    "headphones_box": "rectangle, black with a pale band across one end",
    "phone_box": "rectangle, white with a blue band across one end",
    "bearing_box": "rectangle, orange with a dark band across one end",
    "tissue_box": "rectangle, pale teal with a white band across one end",
}


def catalog_table():
    rows = []
    for sku in CATALOG.values():
        flags = [f for f, on in (("fragile", sku.fragile), ("liquid", sku.liquid), ("electronics", sku.electronics), ("heavy", sku.heavy)) if on]
        rows.append(f"{sku.name}: {sku.kg} kg, {sku.height:.3f} m tall{', ' + ', '.join(flags) if flags else ''}")
    return "\n".join(rows)


RULES = (
    'Skills, one per step: open_carton("S" or "L"), pick("T1".."T6"), place("slot", layer), seal(), hold("reason").\n'
    f'Carton S costs ${CARTONS["S"]["cost_usd"]}, holds {CARTONS["S"]["max_kg"]} kg, slots A,B. '
    f'Carton L costs ${CARTONS["L"]["cost_usd"]}, holds {CARTONS["L"]["max_kg"]} kg, slots A,B,C,D.\n'
    "Each slot stacks at most 2 layers: layer 1 is the carton floor, layer 2 sits on layer 1.\n"
    "Rules: open exactly one carton first. Pick one unit, then place it, before picking again. "
    "Nothing may rest on a fragile item. "
    f"Heavy items ({HEAVY_KG} kg or more) go on layer 1 only. No liquid above electronics. "
    f"Stay within the carton weight limit. A slot's stack may be at most {MAX_STACK_M} m tall. "
    "Pack exactly the ordered units, use the cheapest carton that fits, and finish with seal().\n"
    'If stock cannot fill the whole order, the entire program must be one hold("reason") step.'
)
PERCEIVE_PROMPT = (
    "This is an overhead camera view of a warehouse packing bench. Six blue totes may hold products; each has a tag T1 to T6 beside it. "
    "Totes may be empty. Product appearance:\n{looks}\n"
    "For each non-empty tote report the product name and exact unit count. Do not list empty totes.\n"
    'Reply with JSON only: {{"totes": {{"T1": {{"sku": "soup_can", "count": 2}}}}}}'
)
BASELINE_PROMPT = (
    "Plan how a robot packs this order. Write the program, one skill call per line, with no explanation.\n"
    "Catalog:\n{catalog}\n{rules}\nTotes: {totes}\nOrder: {order}"
)
CONSTRAINED_PROMPT = (
    "Plan how a robot packs this order.\nCatalog:\n{catalog}\n{rules}\nTotes: {totes}\nOrder: {order}\n"
    'Reply with JSON only: {{"program": ["open_carton(\\"S\\")", "pick(\\"T1\\")", "place(\\"A\\", 1)", "seal()"]}}'
)
PLAN_PROMPT = (
    "Plan how a robot packs this order.\nCatalog:\n{catalog}\n{rules}\nTotes: {totes}\nOrder: {order}\n"
    "For each step, imagine the result: what the gripper holds, and the carton's total kg afterward.\n"
    'Reply with JSON only: {{"steps": [{{"call": "open_carton(\\"S\\")", "holding": null, "carton_kg": 0.0}}, '
    '{{"call": "pick(\\"T1\\")", "holding": "soup_can", "carton_kg": 0.0}}, '
    '{{"call": "place(\\"A\\", 1)", "holding": null, "carton_kg": 0.42}}, '
    '{{"call": "seal()", "holding": null, "carton_kg": 0.42}}]}}{feedback}'
)


class ModelOutputError(ValueError):
    pass


def extract_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ModelOutputError("no JSON object in reply")
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        raise ModelOutputError(f"invalid JSON: {e.msg}") from e


def strip_reply(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return re.sub(r"^```\w*\n?|```\s*$", "", text.strip(), flags=re.M).strip()


def normalize_perception(raw):
    if not isinstance(raw, dict) or not isinstance(raw.get("totes"), dict):
        raise ModelOutputError("perception must have a totes object")
    totes = {}
    for tote, value in raw["totes"].items():
        if tote not in TOTES:
            raise ModelOutputError(f"unknown tote {tote}")
        if not isinstance(value, dict) or value.get("sku") not in CATALOG:
            raise ModelOutputError(f"unknown product in {tote}")
        count = value.get("count")
        if not isinstance(count, int) or not 1 <= count <= 3:
            raise ModelOutputError(f"bad count in {tote}")
        totes[tote] = {"sku": value["sku"], "count": count}
    return totes


def perceive(client, model, image_path):
    data = base64.b64encode(Path(image_path).read_bytes()).decode()
    looks = "\n".join(f"{k}: {v}" for k, v in SKU_LOOKS.items())
    reply = client.chat(model, [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
        {"type": "text", "text": PERCEIVE_PROMPT.format(looks=looks)},
    ]}], json_mode=True, max_tokens=6000)
    call = {k: reply[k] for k in ("model", "prompt_tokens", "completion_tokens", "latency_s")}
    try:
        return {"totes": normalize_perception(extract_json(reply["text"])), "error": None, "call": call}
    except ModelOutputError as e:
        return {"totes": None, "error": str(e), "call": call}


def plan_prompt(totes, order, feedback=""):
    return PLAN_PROMPT.format(catalog=catalog_table(), rules=RULES, totes=json.dumps(totes, sort_keys=True),
                              order=order_text(order), feedback=feedback)


def replan_feedback(errors, steps):
    return ("\nYour previous plan failed these checks. Fix them:\n" + "\n".join(errors)
            + f"\nPrevious plan: {json.dumps(steps)}")


def check_plan(totes, order, steps, seed=0, physics=True):
    """Check a plan against the perceived scene. Returns (errors, predictions_right, predictions_total)."""
    try:
        calls = [parse_call(step["call"]) for step in steps]
    except (ParseError, KeyError, TypeError) as e:
        return [f"bad program: {e}"], 0, 0
    if not calls:
        return ["empty program"], 0, 0

    can_fill = fillable(totes, order)
    if any(name == "hold" for name, _ in calls):
        if len(calls) > 1:
            return ["hold must be the only step"], 0, 0
        return ([] if not can_fill else ["held, but stock can fill the order"]), 0, 0
    if not can_fill:
        return ["stock cannot fill this order; the program must be one hold step"], 0, 0

    state = new_state(totes)
    errors, right, total = [], 0, 0
    for i, (step, (name, args)) in enumerate(zip(steps, calls)):
        fatal, problems = apply_call(state, name, args)
        errors += [f"step {i + 1} {step['call']}: {message}" for _, message in problems]
        if fatal:
            return errors, right, total
        for key, actual in (("holding", state["holding"]), ("carton_kg", state["kg"])):
            if key not in step:
                continue
            predicted = step[key]
            total += 1
            same = abs(predicted - actual) < 0.011 if key == "carton_kg" and isinstance(predicted, (int, float)) else predicted == actual
            if same:
                right += 1
            else:
                errors.append(f"step {i + 1} predicted {key}={predicted}, but the result is {actual}")
    if packed_counts(state) != order:
        errors.append(f"packed {packed_counts(state)}, but the order is {order}")
    if not state["sealed"]:
        errors.append("the program does not end with seal()")
    if not errors and physics:
        report = execute(totes, calls, seed)
        for unit in report["crushed"]:
            errors.append(f"physics dry run: {unit} was crushed")
        for unit in report["spilled"]:
            errors.append(f"physics dry run: {unit} spilled out of the carton")
    return errors, right, total


class Run:
    def __init__(self, client, model, max_tokens):
        self.client, self.model, self.max_tokens = client, model, max_tokens
        self.calls = []

    def ask(self, prompt, json_mode=False):
        reply = self.client.chat(self.model, [{"role": "user", "content": prompt}], json_mode=json_mode, max_tokens=self.max_tokens)
        self.calls.append({k: reply[k] for k in ("model", "prompt_tokens", "completion_tokens", "latency_s")})
        return reply["text"]

    def record(self, **fields):
        return {"calls": self.calls, **fields}


def baseline(client, model, totes, order, seed=0, max_tokens=6000):
    run = Run(client, model, max_tokens)
    text = run.ask(BASELINE_PROMPT.format(catalog=catalog_table(), rules=RULES, totes=json.dumps(totes, sort_keys=True), order=order_text(order)))
    return run.record(status="planned", program=strip_reply(text))


def constrained(client, model, totes, order, seed=0, max_tokens=6000):
    run = Run(client, model, max_tokens)
    text = run.ask(CONSTRAINED_PROMPT.format(catalog=catalog_table(), rules=RULES, totes=json.dumps(totes, sort_keys=True), order=order_text(order)), json_mode=True)
    try:
        return run.record(status="planned", program="\n".join(extract_json(text)["program"]))
    except (ModelOutputError, KeyError, TypeError) as e:
        return run.record(status="planned", program="", output_error=str(e))


def foresight(client, model, totes, order, seed=0, max_tokens=6000, physics=True):
    """Imagine, check and replan. Returns a program only if it passes the check; otherwise withholds."""
    run = Run(client, model, max_tokens)
    history, feedback = [], ""
    for _ in range(MAX_REPLANS + 1):
        text = run.ask(plan_prompt(totes, order, feedback), json_mode=True)
        try:
            steps = extract_json(text)["steps"]
            errors, right, total = check_plan(totes, order, steps, seed, physics)
        except (ModelOutputError, KeyError, TypeError, AttributeError, ValueError):
            steps, errors, right, total = [], ['unreadable plan: reply with the exact JSON format'], 0, 0
        history.append({"steps": steps, "errors": errors, "predictions_right": right, "predictions_total": total})
        if not errors:
            return run.record(status="planned", program="\n".join(s["call"] for s in steps), history=history)
        feedback = replan_feedback(errors, steps)
    return run.record(status="withheld", program=None, history=history)


VARIANTS = {"A_baseline": baseline, "B_constrained": constrained, "C_foresight": foresight}
