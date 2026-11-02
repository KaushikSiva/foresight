import json

from foresight.agent import check_plan
from foresight.scenes import generate
from foresight.training import build, correct_steps, fingerprint


def test_training_data_is_valid_and_never_contains_eval_scenes():
    train, validation, scenes = build(train_size=300, validation_size=30)
    assert len(train) == 300 and len(validation) == 30
    eval_keys = {fingerprint(s) for s in generate()}
    assert not eval_keys & {fingerprint(s) for s in scenes}
    assert len({fingerprint(s) for s in scenes}) == 330
    for scene, row in zip(scenes, train + validation):
        steps = json.loads(row["messages"][1]["content"])["steps"]
        assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
        assert check_plan(scene["state"], scene["goal"], steps)[0] == []


def test_every_prediction_in_a_training_answer_is_right():
    scene = next(s for s in generate() if s["tier"] == "blocked" and s["possible"])
    _, right, total = check_plan(scene["state"], scene["goal"], correct_steps(scene))
    assert total > 0 and right == total


def test_training_set_includes_repair_examples():
    train, _, _ = build(train_size=300, validation_size=30)
    assert sum("failed these checks" in r["messages"][0]["content"] for r in train) > 10
