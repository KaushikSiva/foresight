import json
from pathlib import Path

import pytest

from warehouse.catalog import CATALOG
from warehouse.orders import generate, solve, solve_order
from warehouse.rules import ParseError, parse_call, parse_program, run
from warehouse.scoring import judge, score

TOTES = {"T1": {"sku": "ceramic_mug", "count": 1}, "T2": {"sku": "bearing_box", "count": 1}}
SCENE = {"seed": 3, "fillable": True, "best_carton": "S", "totes": TOTES, "order": {"ceramic_mug": 1, "bearing_box": 1}}


def test_eval_split_and_determinism():
    scenes = generate()
    assert len(scenes) == 200 and sum(s["split"] == "held_out" for s in scenes) == 50
    assert scenes == generate()


def test_committed_scenes_match_the_generator():
    with open(Path(__file__).parent.parent / "data" / "warehouse" / "scenes.jsonl") as f:
        assert [json.loads(line) for line in f] == generate()


def test_solver_plans_follow_the_rules():
    for scene in generate():
        result = run(scene["totes"], parse_program(solve(scene)))
        assert not result.violations and not result.halted, scene["id"]


def test_solver_picks_the_cheapest_carton_that_fits():
    assert solve_order({"soup_can": 1})[0] == "S"
    assert solve_order({"bearing_box": 2, "paint_can": 1})[0] == "L"


@pytest.mark.parametrize("text", ["", "import os", "__import__('os').system('x')", 'pick("T1", "T2")', "pick(T1)", 'fly("T1")', 'pick(tote="T1")'])
def test_parser_rejects_bad_programs(text):
    with pytest.raises(ParseError):
        parse_program(text) if text else parse_call(text)


@pytest.mark.parametrize("program,code", [
    ('open_carton("S")\npick("T1")\nplace("A", 1)\npick("T2")\nplace("A", 2)', "on_fragile"),
    ('open_carton("S")\npick("T2")\nplace("A", 1)\npick("T1")\nplace("B", 1)', None),
])
def test_stacking_rules(program, code):
    result = run(TOTES, parse_program(program))
    assert (code in [v.code for v in result.violations]) if code else not result.violations


def test_impossible_steps_halt():
    assert run(TOTES, parse_program('pick("T6")')).halted
    assert run(TOTES, parse_program('place("A", 1)')).halted


def test_physics_crushes_a_mug_under_a_heavy_box():
    result = score(SCENE, 'open_carton("L")\npick("T1")\nplace("A", 1)\npick("T2")\nplace("A", 2)\nseal()')
    assert result["outcome"] == "crushed" and result["unsafe"]
    assert result["report"]["loads_n"]["T1_0"] > 20


def test_physics_packs_the_same_items_correctly():
    result = score(SCENE, 'open_carton("S")\npick("T2")\nplace("A", 1)\npick("T1")\nplace("B", 1)\nseal()')
    assert result["outcome"] == "packed" and result["success"] and not result["wasteful"]


def test_bigger_carton_than_needed_is_flagged_as_wasteful():
    result = score(SCENE, 'open_carton("L")\npick("T2")\nplace("A", 1)\npick("T1")\nplace("B", 1)\nseal()')
    assert result["success"] and result["wasteful"]


def test_shipping_a_short_order_fails_and_holding_passes():
    short = {**SCENE, "fillable": False, "best_carton": None, "order": {"glass_jar": 1}}
    assert score(short, 'hold("no stock")')["outcome"] == "correct_hold"
    assert score(short, 'open_carton("S")\npick("T1")\nplace("A", 1)\nseal()')["outcome"] == "shipped_short"


def test_catalog_fits_the_gripper():
    for sku in CATALOG.values():
        width = sku.size[0] * 2
        assert width <= 0.08, sku.name
