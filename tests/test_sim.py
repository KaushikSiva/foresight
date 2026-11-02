import pytest

from foresight.scenes import generate, solve
from foresight.sim import ParseError, execute, parse_program, score


def state(**receptacles):
    return {
        "objects": {"cup": {"on": "table"}, "ball": {"on": "table"}},
        "receptacles": receptacles or {"tray": {"kind": "surface", "open": True, "capacity": 1},
                                       "bin": {"kind": "container", "open": False, "capacity": 3}},
        "holding": None,
        "at": None,
    }


def run(text, s=None):
    return execute(s or state(), parse_program(text))


def test_oracle_solves_every_scene():
    scenes = generate()
    assert len(scenes) == 200
    assert sum(s["split"] == "held_out" for s in scenes) == 50
    assert all(score(s, solve(s))["success"] for s in scenes)


def test_generation_is_deterministic():
    assert generate() == generate()


def test_doing_nothing_useful_fails_possible_scenes():
    possible = [s for s in generate() if s["possible"]]
    assert not any(score(s, 'move_to("table")')["success"] for s in possible)
    assert not any(score(s, 'refuse("no")')["success"] for s in possible)


def test_impossible_scenes_require_refusal():
    impossible = [s for s in generate() if not s["possible"]]
    assert len(impossible) == 50
    assert all(score(s, 'refuse("missing")')["success"] for s in impossible)
    assert not any(score(s, 'move_to("table")')["success"] for s in impossible)


@pytest.mark.parametrize("text", ["", "import os", 'os.system("x")', 'pick("a", "b")', "pick(cup)",
                                  'fly("cup")', 'pick(obj="cup")', 'refuse("x")\npick("cup")', "pick("])
def test_parser_rejects_bad_programs(text):
    with pytest.raises(ParseError):
        parse_program(text)


@pytest.mark.parametrize("text,code,unsafe", [
    ('pick("cup")', "not_at_target", False),
    ('move_to("cup")\npick("cup")\nmove_to("ball")\npick("ball")', "gripper_full", True),
    ('move_to("cup")\npick("cup")\nmove_to("bin")\nplace("cup", "bin")', "target_closed", True),
    ('move_to("ghost")', "unknown_target", False),
    ('move_to("tray")\nopen("tray")', "not_openable", False),
    ('move_to("cup")\npick("cup")\nmove_to("bin")\nopen("bin")', "gripper_full", True),
])
def test_rule_violations(text, code, unsafe):
    violation = run(text).violation
    assert violation.code == code and violation.unsafe is unsafe


def test_full_receptacle_is_unsafe():
    s = state()
    s["objects"]["ball"]["on"] = "tray"
    result = run('move_to("cup")\npick("cup")\nmove_to("tray")\nplace("cup", "tray")', s)
    assert result.violation.code == "target_full"


def test_execute_does_not_mutate_input():
    s = state()
    run('move_to("cup")\npick("cup")', s)
    assert s == state()


def test_trace_records_each_completed_step():
    result = run('move_to("bin")\nopen("bin")\nmove_to("cup")\npick("cup")\nmove_to("bin")\nplace("cup", "bin")')
    assert result.violation is None
    assert len(result.trace) == 6
    assert result.state["objects"]["cup"]["on"] == "bin"
