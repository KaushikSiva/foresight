import pytest

from functools import lru_cache

from foresight.scenes import generate as _generate
from foresight.schema import Container, Obj, Plan, Robot, Scene, Step
from foresight.sim import RuleError, apply_step, execute, score
from foresight.solver import solve


@lru_cache(maxsize=1)
def _tasks():
    return tuple(_generate())


def generate():
    return list(_tasks())


def scene(**kw):
    base = dict(
        objects=[Obj(id="red_block", color="red", kind="block", location="table"),
                 Obj(id="blue_cup", color="blue", kind="cup", location="table")],
        containers=[Container(id="tray", kind="tray"), Container(id="bin", kind="bin", is_open=False)],
        robot=Robot(),
    )
    base.update(kw)
    return Scene(**base)


def S(skill, *args):
    return Step(skill=skill, args=list(args))


def run(sc, *steps):
    for st in steps:
        sc = apply_step(sc, st)
    return sc


def err(sc, *steps):
    with pytest.raises(RuleError) as e:
        run(sc, *steps)
    return e.value.code


def test_pick_and_place():
    sc = run(scene(), S("move_to", "red_block"), S("pick", "red_block"), S("move_to", "tray"), S("place", "red_block", "tray"))
    assert sc.obj("red_block").location == "tray" and sc.robot.holding is None


def test_must_move_first():
    assert err(scene(), S("pick", "red_block")) == "not_at"


def test_one_object_at_a_time():
    assert err(scene(), S("move_to", "red_block"), S("pick", "red_block"), S("move_to", "blue_cup"), S("pick", "blue_cup")) == "hand_full"


def test_no_place_into_closed():
    assert err(scene(), S("move_to", "red_block"), S("pick", "red_block"), S("move_to", "bin"), S("place", "red_block", "bin")) == "container_closed"


def test_open_needs_empty_hand():
    assert err(scene(), S("move_to", "red_block"), S("pick", "red_block"), S("move_to", "bin"), S("open", "bin")) == "hand_full"


def test_tray_capacity():
    sc = run(scene(), S("move_to", "red_block"), S("pick", "red_block"), S("move_to", "tray"), S("place", "red_block", "tray"))
    assert err(sc, S("move_to", "blue_cup"), S("pick", "blue_cup"), S("move_to", "tray"), S("place", "blue_cup", "tray")) == "target_full"


def test_unknown_object_and_tray_not_openable():
    assert err(scene(), S("move_to", "green_ball")) == "unknown_entity"
    assert err(scene(), S("move_to", "tray"), S("open", "tray")) == "not_openable"


def test_bad_arity():
    assert err(scene(), S("place", "red_block")) == "bad_args"


def test_violation_stops_execution():
    res = execute(scene(), Plan(steps=[S("pick", "red_block"), S("move_to", "tray")]))
    assert res.violation.step == 0 and res.steps_run == 0


def test_dataset_shape_and_oracle():
    tasks = generate()
    assert len(tasks) == 200 and len({t.id for t in tasks}) == 200
    assert {t.difficulty for t in tasks} == {"easy", "multi_step", "blocked", "impossible", "packing"}
    assert sum(t.split == "heldout" for t in tasks) == 50
    for t in tasks:
        ids = [o.id for o in t.scene.objects]
        assert len(ids) == len(set(ids)), t.id
        r = score(t, solve(t))
        assert r["success"] and not r["unsafe"], (t.id, r)


def test_refusing_a_possible_task_fails():
    t = next(t for t in generate() if t.possible)
    r = score(t, Plan(steps=[S("refuse", "nope")]))
    assert not r["success"] and r["refused"]


def test_generation_is_deterministic():
    assert [t.model_dump() for t in _generate()[:20]] == [t.model_dump() for t in generate()[:20]]


def test_foresight_plan_roundtrip_keeps_null_holding():
    import json
    from foresight import prompts
    from foresight.variants import to_plan
    t = next(t for t in generate() if t.difficulty == "blocked")
    plan, goal, sealed = to_plan(json.loads(prompts.plan_json(solve(t), True, t.goals, t.sealed)))
    from foresight.validator import check
    assert plan is not None and goal == t.goals and check(t.scene, plan, goal, sealed) == []


def test_checker_catches_skipped_prerequisite():
    from foresight.solver import plan_steps
    from foresight.validator import check
    t = next(t for t in generate() if "closed_lid" in t.traps)
    errs = check(t.scene, Plan(steps=plan_steps(t.scene, t.goals, prereqs=False)), t.goals)
    assert any("breaks a rule" in e for e in errs)


def carton_scene():
    return Scene(objects=[Obj(id="red_cup", color="red", kind="cup", location="table"),
                          Obj(id="blue_block", color="blue", kind="block", location="table")],
                 containers=[Container(id="carton_a", kind="carton", is_open=True)])


def test_fragile_cup_is_crushed():
    sc = run(carton_scene(), S("move_to", "red_cup"), S("pick", "red_cup"), S("move_to", "carton_a"), S("place", "red_cup", "carton_a"))
    assert err(sc, S("move_to", "blue_block"), S("pick", "blue_block"), S("move_to", "carton_a"), S("place", "blue_block", "carton_a")) == "crushed"


def test_sturdy_first_then_cup_is_fine():
    sc = run(carton_scene(), S("move_to", "blue_block"), S("pick", "blue_block"), S("move_to", "carton_a"), S("place", "blue_block", "carton_a"),
             S("move_to", "red_cup"), S("pick", "red_cup"), S("move_to", "carton_a"), S("place", "red_cup", "carton_a"))
    assert len(sc.contents("carton_a")) == 2


def test_packing_needs_seal_and_exact_contents():
    t = next(t for t in generate() if t.difficulty == "packing" and t.possible)
    plan = solve(t)
    assert score(t, plan)["success"]
    unsealed = Plan(steps=[s for s in plan.steps if s.skill != "close"])
    assert not score(t, unsealed)["success"]


def test_checker_catches_packing_mistakes():
    from foresight.solver import plan_steps
    from foresight.validator import check
    for t in generate():
        if t.difficulty == "packing" and t.possible:
            naive = Plan(steps=plan_steps(t.scene, t.goals, prereqs=False, sealed=t.sealed))
            assert check(t.scene, naive, t.goals, t.sealed), t.id


def test_impossible_tasks_have_no_plan():
    from foresight.solver import search
    for t in generate():
        if not t.possible:
            assert search(t.scene, t.requested, ["carton_a"] if "carton_a" in {g.location for g in t.requested} else []) is None, t.id


def test_audit_bar_holds():
    from foresight.audit import audit, verdict
    for d, r in audit(generate()).items():
        assert verdict(d, r) == [], (d, verdict(d, r))


def test_repack_fragile_needs_cup_out_and_back():
    t = next(t for t in generate() if "repack_fragile" in t.traps and t.possible)
    calls = [(s.skill, tuple(s.args)) for s in solve(t).steps]
    cup = next(o.id for o in t.scene.objects if o.location == "carton_a" and o.kind == "cup")
    assert calls.count(("pick", (cup,))) == 2  # out, then back in


def test_3d_render_is_a_full_frame():
    from foresight import config
    if config.RENDER != "3d":
        return
    from foresight.render3d import render
    t = next(t for t in generate() if t.difficulty == "packing")
    assert render(t).size == (1280, 720)


def test_pick_from_container_after_moving_to_it():
    sc = scene(objects=[Obj(id="red_block", color="red", kind="block", location="tray")])
    sc = run(sc, S("move_to", "tray"), S("pick", "red_block"))
    assert sc.robot.holding == "red_block"


def test_unsafe_is_physical_and_slips_are_invalid_steps():
    t = next(t for t in generate() if "closed_lid" in t.traps)
    target = t.goals[0].location
    forced = Plan(steps=[S("move_to", t.goals[0].obj), S("pick", t.goals[0].obj), S("move_to", target),
                         S("place", t.goals[0].obj, target)])
    r = score(t, forced)
    assert r["unsafe"] and not r["invalid_step"] and r["violation"]["code"] == "container_closed"
    slip = Plan(steps=[S("pick", t.goals[0].obj)])
    r = score(t, slip)
    assert r["invalid_step"] and not r["unsafe"] and not r["success"]


def test_prompts_are_identical_across_processes():
    """Few-shot examples come from the A* oracle; its tie-breaks must not depend on hash randomization."""
    import subprocess
    import sys
    code = "import hashlib; from foresight import prompts; print(hashlib.sha256((prompts._shots(True) + prompts._shots(False)).encode()).hexdigest())"
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**__import__("os").environ, "PYTHONHASHSEED": str(seed)}).stdout
            for seed in (1, 2, 3)}
    assert len(outs) == 1, outs


def test_check_only_variant_has_no_predictions():
    import json
    from foresight import prompts
    t = next(t for t in generate() if t.difficulty == "packing")
    assert "expect" not in json.dumps(prompts.plan_check_only(t.scene, t.instruction))
    assert "expect" not in json.dumps(prompts.CHECK_ONLY_SCHEMA)


def test_diagnostics_on_oracle_plans():
    from foresight.diagnostics import mcnemar_p, perception, prediction
    t = next(t for t in generate() if "repack_fragile" in t.traps)
    ok, n = prediction(t, solve(t))
    assert n > 0 and ok == n
    seen = {"objects": [o.model_dump() for o in t.scene.objects], "containers": [c.model_dump() for c in t.scene.containers]}
    assert perception(t, seen)["exact"]
    assert mcnemar_p(0, 0) == 1.0 and mcnemar_p(20, 2) < 0.001


def test_pick_after_moving_to_the_table():
    sc = run(scene(), S("move_to", "table"), S("pick", "red_block"))
    assert sc.robot.holding == "red_block"


def test_perception_validation_and_sanitizing():
    from foresight.variants import sanitize_scene, scene_issues
    raw = {"objects": [{"id": "bin", "color": "blue", "kind": "block", "location": "table"},
                       {"id": "tray", "color": "red", "kind": "cup", "location": "table"},
                       {"id": "red_cup", "color": "red", "kind": "cup", "location": "table"},
                       {"id": "red_cup", "color": "red", "kind": "cup", "location": "nowhere"}],
           "containers": [{"id": "bin", "kind": "bin", "is_open": True}]}
    assert len(scene_issues(raw)) == 3
    clean = sanitize_scene(raw)
    assert [o["id"] for o in clean["objects"]] == ["red_cup"] and scene_issues(clean) == []


def test_refusal_reason_metric():
    from foresight.diagnostics import refusal_reason_ok
    t = next(t for t in generate() if "missing_object" in t.traps)
    missing = next(g.obj for g in t.requested if t.scene.obj(g.obj) is None)
    assert refusal_reason_ok(t, {"steps": [{"skill": "refuse", "args": [f"{missing} is not here"]}]})
    assert not refusal_reason_ok(t, {"steps": [{"skill": "refuse", "args": ["the task cannot be done as asked"]}]})


def test_few_shot_refusal_is_specific():
    from foresight import prompts
    assert "cannot be done as asked" not in prompts._shots(True) + prompts._shots(False)


def test_3d_colours_render_on_their_named_hue():
    from foresight import config
    if config.RENDER != "3d":
        return
    import colorsys
    from foresight import render3d
    from foresight.render import RGB
    from foresight.schema import Robot, Task as T_
    objs = [Obj(id=f"{c}_block", color=c, kind="block", location="table") for c in RGB]
    t = T_(id="swatch", split="tune", difficulty="easy", instruction="", scene=Scene(objects=objs, containers=[], robot=Robot()),
           goals=[], possible=False)
    img, pos = render3d.render(t), render3d.positions(t, t.scene)
    hues = {}
    for c in RGB:
        x, y, _ = pos[f"{c}_block"]
        p = img.getpixel((int(640 - y * 476.5), int(360 - (x - 0.40) * 476.5)))
        hues[c] = colorsys.rgb_to_hsv(*[q / 255 for q in p])[0] * 360
        want = colorsys.rgb_to_hsv(*[q / 255 for q in RGB[c]])[0] * 360
        assert min(abs(hues[c] - want), 360 - abs(hues[c] - want)) < 15, (c, hues[c], want)
    assert abs(hues["orange"] - hues["yellow"]) > 20
