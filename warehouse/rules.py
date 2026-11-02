"""Packing skills and rules. The checker runs these on the perceived scene; the scorer on the true scene."""
import ast
import copy
from dataclasses import dataclass, field

from .catalog import CARTONS, CATALOG, MAX_LAYERS

MAX_STACK_M = 0.17
ARITY = {"open_carton": 1, "pick": 1, "place": 2, "seal": 0, "hold": 1}
# Breaking one of these risks damaged goods or an unshippable carton.
UNSAFE = {"over_weight", "on_fragile", "heavy_on_top", "liquid_over_electronics", "stack_too_tall"}


class ParseError(ValueError):
    pass


def parse_call(text):
    """Parse one skill call into (name, args) without ever executing model output."""
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as e:
        raise ParseError(f"syntax error in {text!r}") from e
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id not in ARITY:
        raise ParseError(f"not a known skill call: {text!r}")
    if node.keywords or len(node.args) != ARITY[node.func.id]:
        raise ParseError(f"{node.func.id} takes {ARITY[node.func.id]} positional argument(s)")
    args = []
    for arg in node.args:
        if not isinstance(arg, ast.Constant) or not isinstance(arg.value, (str, int)):
            raise ParseError(f"{node.func.id} arguments must be plain values")
        args.append(arg.value)
    return node.func.id, tuple(args)


def parse_program(text):
    calls = [parse_call(line) for line in text.strip().splitlines() if line.strip()]
    if not calls:
        raise ParseError("empty program")
    return calls


def new_state(totes):
    return {"totes": copy.deepcopy(totes), "carton": None, "columns": {}, "holding": None, "kg": 0.0, "sealed": False, "held": False}


@dataclass
class Violation:
    step: int
    code: str
    message: str

    @property
    def unsafe(self):
        return self.code in UNSAFE


@dataclass
class Result:
    state: dict
    trace: list = field(default_factory=list)
    violations: list = field(default_factory=list)
    halted: bool = False


def column_height(column):
    return sum(CATALOG[s].height for s in column)


def _policy_check(state, slot, layer, sku):
    """Rules a placement can break while still being physically possible."""
    column = state["columns"].get(slot, [])
    carton = CARTONS[state["carton"]]
    found = []
    if any(CATALOG[s].fragile for s in column):
        found.append(("on_fragile", f"{sku} placed on fragile {column[-1]} in slot {slot}"))
    if column and CATALOG[sku].heavy:
        found.append(("heavy_on_top", f"heavy {sku} must go on the carton floor"))
    if CATALOG[sku].liquid and any(CATALOG[s].electronics for s in column):
        found.append(("liquid_over_electronics", f"liquid {sku} above electronics in slot {slot}"))
    if state["kg"] + CATALOG[sku].kg > carton["max_kg"] + 1e-9:
        found.append(("over_weight", f"carton {state['carton']} would hold {state['kg'] + CATALOG[sku].kg:.2f} kg, limit {carton['max_kg']}"))
    if column_height(column) + CATALOG[sku].height > MAX_STACK_M + 1e-9:
        found.append(("stack_too_tall", f"slot {slot} would be {column_height(column) + CATALOG[sku].height:.3f} m tall, limit {MAX_STACK_M}"))
    return found


def apply_call(state, name, args):
    """Mutate state. Returns (fatal, [(code, message)]). A fatal problem means the step cannot physically happen."""
    if name == "open_carton":
        if args[0] not in CARTONS:
            return True, [("unknown_carton", f"no carton size {args[0]}")]
        if state["carton"]:
            return True, [("carton_already_open", f"carton {state['carton']} is already open")]
        state["carton"] = args[0]
    elif name == "pick":
        tote = state["totes"].get(args[0])
        if state["holding"]:
            return True, [("gripper_full", f"already holding {state['holding']}")]
        if not tote or tote["count"] < 1:
            return True, [("empty_tote", f"no units left in {args[0]}")]
        tote["count"] -= 1
        state["holding"] = tote["sku"]
    elif name == "place":
        slot, layer = args
        if not state["holding"]:
            return True, [("not_holding", "gripper is empty")]
        if not state["carton"]:
            return True, [("no_carton", "open a carton before placing")]
        if slot not in CARTONS[state["carton"]]["slots"]:
            return True, [("unknown_slot", f"carton {state['carton']} has no slot {slot}")]
        column = state["columns"].get(slot, [])
        if layer != len(column) + 1 or layer > MAX_LAYERS:
            return True, [("wrong_layer", f"slot {slot} holds {len(column)} item(s), so layer {layer} is not possible")]
        problems = _policy_check(state, slot, layer, state["holding"])
        state["columns"].setdefault(slot, []).append(state["holding"])
        state["kg"] = round(state["kg"] + CATALOG[state["holding"]].kg, 3)
        state["holding"] = None
        return False, problems
    elif name == "seal":
        if state["holding"] or not state["carton"]:
            return True, [("cannot_seal", "seal needs an open carton and an empty gripper")]
        state["sealed"] = True
    elif name == "hold":
        state["held"] = True
    return False, []


def run(totes, calls):
    """Run calls from a fresh scene, stopping only at a physically impossible step."""
    state = new_state(totes)
    result = Result(state=state)
    for i, (name, args) in enumerate(calls):
        fatal, problems = apply_call(state, name, args)
        result.violations += [Violation(i, c, m) for c, m in problems]
        if fatal:
            result.halted = True
            return result
        result.trace.append(copy.deepcopy(state))
    return result


def packed_counts(state):
    counts = {}
    for column in state["columns"].values():
        for sku in column:
            counts[sku] = counts.get(sku, 0) + 1
    return counts


def fillable(totes, order):
    """Whether stock covers the order."""
    stock = {}
    for tote in totes.values():
        if tote and tote.get("count"):
            stock[tote["sku"]] = stock.get(tote["sku"], 0) + tote["count"]
    return all(stock.get(sku, 0) >= qty for sku, qty in order.items())
