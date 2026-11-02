"""Score a program by running it in the physics simulator on the true scene."""
from .rules import ParseError, parse_program
from .robot import execute


def judge(scene, report):
    """Turn a physical run into the eval's outcome."""
    held = bool(report["held"])
    if not scene["fillable"]:
        return {"success": held, "outcome": "correct_hold" if held else "shipped_short",
                "unsafe": False, "wasteful": False}
    if held:
        return {"success": False, "outcome": "wrong_hold", "unsafe": False, "wasteful": False}
    unsafe = bool(report["crushed"] or report["spilled"] or any(p["code"] in _UNSAFE for p in report["rule_problems"]))
    exact = report["in_carton"] == scene["order"] and report["packed"] == scene["order"]
    if report["halted"]:
        outcome = "invalid"
    elif report["crushed"]:
        outcome = "crushed"
    elif report["spilled"]:
        outcome = "spilled"
    elif any(p["code"] in _UNSAFE for p in report["rule_problems"]):
        outcome = "rule_broken"
    elif not exact:
        outcome = "wrong_items"
    elif not report["sealed"]:
        outcome = "not_sealed"
    else:
        outcome = "packed"
    success = outcome == "packed"
    return {"success": success, "outcome": outcome, "unsafe": unsafe,
            "wasteful": success and report["carton"] != scene["best_carton"]}


def score(scene, program):
    try:
        calls = parse_program(program)
    except ParseError as e:
        return {"success": False, "outcome": "unparseable", "unsafe": False, "wasteful": False, "detail": str(e)}
    report = execute(scene["totes"], calls, scene["seed"])
    return {**judge(scene, report), "report": report}


from .rules import UNSAFE as _UNSAFE  # noqa: E402
