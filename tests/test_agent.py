"""Pipeline tests with a scripted fake model. Nothing here is a measure of real model quality."""
import json

from foresight.agent import baseline, check_plan, constrained, foresight, normalize_perception
from foresight.evaluation import load_scenes, perception_correct, run_scene, summarize

GOOD_STEPS = [
    {"call": 'move_to("bin")', "expect": {}},
    {"call": 'open("bin")', "expect": {"open": {"bin": True}}},
    {"call": 'move_to("cup")', "expect": {}},
    {"call": 'pick("cup")', "expect": {"holding": "cup"}},
    {"call": 'move_to("bin")', "expect": {}},
    {"call": 'place("cup", "bin")', "expect": {"holding": None, "on": {"cup": "bin"}}},
]
PERCEPTION = {"objects": {"white_cup": {"on": "table"}}, "receptacles": {"bin": {"open": False}}, "goal": [["white_cup", "bin"]]}


def steps_for(obj="white_cup"):
    return [{"call": s["call"].replace("cup", obj), "expect": json.loads(json.dumps(s["expect"]).replace("cup", obj))} for s in GOOD_STEPS]


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.models = []

    def chat(self, model, messages, json_mode=False, max_tokens=1500):
        self.models.append(model)
        return {"text": self.replies.pop(0), "model": model, "prompt_tokens": 10, "completion_tokens": 5, "latency_s": 0.1}


def perceived():
    return normalize_perception(PERCEPTION)


def test_check_accepts_a_correct_plan_and_predictions():
    state, goal = perceived()
    errors, right, total = check_plan(state, goal, steps_for())
    assert errors == [] and right == total == 4


def test_check_catches_skipping_the_closed_lid():
    state, goal = perceived()
    errors, _, _ = check_plan(state, goal, steps_for()[2:])
    assert "closed" in errors[0]


def test_check_catches_a_wrong_imagined_outcome():
    state, goal = perceived()
    steps = steps_for()
    steps[3]["expect"] = {"holding": None}
    errors, right, total = check_plan(state, goal, steps)
    assert any("predicted" in e for e in errors) and right == total - 1


def test_check_requires_refusal_for_missing_items_and_only_then():
    state, _ = perceived()
    refuse = [{"call": 'refuse("gone")'}]
    assert check_plan(state, [["red_block", "bin"]], refuse)[0] == []
    assert check_plan(state, [["white_cup", "bin"]], refuse)[0]
    assert check_plan(state, [["red_block", "bin"]], steps_for())[0]


def test_check_never_executes_model_code():
    state, goal = perceived()
    errors, _, _ = check_plan(state, goal, [{"call": "__import__('os').system('touch pwned')"}])
    assert errors and "bad program" in errors[0]


def perception_record():
    state, goal = perceived()
    return {"perceived": state, "goal": goal, "error": None, "call": {"model": "vision", "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0.1}}


def test_foresight_replans_after_a_failed_check():
    client = FakeClient([json.dumps({"steps": steps_for()[2:]}), json.dumps({"steps": steps_for()})])
    output = foresight(client, "Put the white cup in the bin.", perception_record())
    assert output["status"] == "planned"
    assert len(output["history"]) == 2 and output["history"][0]["errors"] and not output["history"][1]["errors"]


def test_foresight_withholds_a_plan_that_never_passes():
    bad = json.dumps({"steps": steps_for()[2:]})
    output = foresight(FakeClient([bad, bad, bad]), "Put the white cup in the bin.", perception_record())
    assert output["status"] == "withheld" and output["program"] is None


def test_foresight_recovers_from_shorthand_expect_format():
    shorthand = [{"call": 'move_to("bin")', "expect": {"on": "bin"}}]
    client = FakeClient([json.dumps({"steps": shorthand}), json.dumps({"steps": steps_for()})])
    output = foresight(client, "Put the white cup in the bin.", perception_record())
    assert output["status"] == "planned" and "unreadable" in output["history"][0]["errors"][0]


def test_withheld_plan_is_not_counted_as_a_correct_refusal():
    scene = next(s for s in load_scenes() if s["tier"] == "impossible")
    record = run_scene(FakeClient(["not json"] * 3), "C_foresight", scene, perception_record())
    assert record["status"] == "withheld" and record["score"]["success"] is False


def test_failed_perception_fails_the_scene_without_calling_the_planner():
    client = FakeClient([])
    failed = {"perceived": None, "goal": None, "error": "bad", "call": None}
    record = run_scene(client, "A_baseline", load_scenes()[0], failed)
    assert record["status"] == "perception_failed" and record["score"]["success"] is False and client.models == []


def test_tuned_variant_never_falls_back_to_the_base_model(monkeypatch):
    monkeypatch.delenv("FORESIGHT_TUNED_MODEL", raising=False)
    client = FakeClient([])
    record = run_scene(client, "D_tuned", load_scenes()[0], perception_record())
    assert record["status"] == "error" and client.models == []


def test_tuned_variant_uses_the_tuned_model(monkeypatch):
    monkeypatch.setenv("FORESIGHT_TUNED_MODEL", "tuned")
    client = FakeClient([json.dumps({"steps": steps_for()})])
    run_scene(client, "D_tuned", load_scenes()[0], perception_record())
    assert client.models == ["tuned"]


def test_perception_correct_compares_contents_goal_and_lids():
    scene = load_scenes()[0]
    good = {"perceived": {"objects": scene["state"]["objects"], "receptacles": scene["state"]["receptacles"]}, "goal": scene["goal"]}
    assert perception_correct(scene, good)
    assert not perception_correct(scene, {**good, "goal": [["x", "y"]]})
    assert not perception_correct(scene, {"perceived": None, "goal": None})


def test_baseline_and_constrained_parse_replies():
    assert baseline(FakeClient(['```\nmove_to("x")\n```']), "x", perception_record())["program"] == 'move_to("x")'
    assert constrained(FakeClient(['{"program": ["move_to(\\"x\\")"]}']), "x", perception_record())["program"] == 'move_to("x")'
    assert constrained(FakeClient(["nope"]), "x", perception_record())["program"] == ""


def test_api_failures_are_reported_as_errors():
    class Down:
        def chat(self, *a, **k):
            raise RuntimeError("Fireworks HTTP 503")

    record = run_scene(Down(), "A_baseline", load_scenes()[0], perception_record())
    assert record["status"] == "error"
    assert summarize([record])["errors"] == 1


def test_summary_leaves_cost_empty_without_prices():
    record = run_scene(FakeClient(['move_to("table")']), "A_baseline", load_scenes()[0], perception_record())
    assert summarize([record])["cost_per_success_usd"] is None
