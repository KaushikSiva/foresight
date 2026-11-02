"""Data models shared by the simulator, the LLM layer and the eval.

The same Scene model is used for the hidden true scene and for the scene
Gemma perceives from the image, so the validator and the scoring simulator
run identical code on different inputs.
"""
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

SKILLS = ("move_to", "pick", "place", "open", "close", "refuse")
SKILL_ARITY = {"move_to": 1, "pick": 1, "place": 2, "open": 1, "close": 1, "refuse": 1}

ContainerKind = Literal["tray", "bin", "box", "tote", "carton"]
Difficulty = Literal["easy", "multi_step", "blocked", "impossible", "packing"]

TABLE = "table"
GRIPPER = "gripper"
CAPACITY = {"tray": 1, "bin": 3, "box": 2, "tote": 4, "carton": 3}
OPENABLE = ("bin", "box", "carton")
FRAGILE = ("cup",)


class Obj(BaseModel):
    id: str  # canonical "<color>_<kind>", e.g. "red_block"
    color: str
    kind: str
    location: str  # "table", a container id, or "gripper"


class Container(BaseModel):
    id: str  # its printed label, lower case, spaces as underscores: "bin", "carton_a"
    kind: ContainerKind
    is_open: bool = True


class Robot(BaseModel):
    at: Optional[str] = None
    holding: Optional[str] = None


class Scene(BaseModel):
    objects: List[Obj] = Field(default_factory=list)
    containers: List[Container] = Field(default_factory=list)
    robot: Robot = Field(default_factory=Robot)

    def obj(self, oid: str) -> Optional[Obj]:
        return next((o for o in self.objects if o.id == oid), None)

    def container(self, cid: str) -> Optional[Container]:
        return next((c for c in self.containers if c.id == cid), None)

    def contents(self, cid: str) -> List[Obj]:
        return [o for o in self.objects if o.location == cid]


class Expect(BaseModel):
    """What the planner predicts the world looks like after a step."""

    holding: Optional[str] = None  # null = predicts an empty gripper
    locations: Dict[str, str] = Field(default_factory=dict)
    open: Dict[str, bool] = Field(default_factory=dict)


class Step(BaseModel):
    skill: Literal["move_to", "pick", "place", "open", "close", "refuse"]
    args: List[str]
    expect: Optional[Expect] = None


def step_dict(s: "Step") -> dict:
    """JSON form of a step. Keeps holding=null inside expect (exclude_none would drop it)."""
    d = {"skill": s.skill, "args": s.args}
    if s.expect is not None:
        d["expect"] = s.expect.model_dump()
    return d


class Plan(BaseModel):
    steps: List[Step]

    @property
    def refused(self) -> bool:
        return any(s.skill == "refuse" for s in self.steps)


class Goal(BaseModel):
    obj: str
    location: str


class Task(BaseModel):
    id: str
    split: Literal["tune", "heldout"]
    difficulty: Difficulty
    instruction: str
    scene: Scene  # ground truth: never shown to the model or the validator
    goals: List[Goal]  # empty when the task is impossible
    # Containers that must end closed and hold exactly their goal objects (packed orders).
    sealed: List[str] = Field(default_factory=list)
    # What the instruction literally asks for, even when it cannot be done (used by baselines and audit).
    requested: List[Goal] = Field(default_factory=list)
    # The difficulty mechanisms in this task, e.g. "closed_lid", "repack_fragile", "lookalike".
    traps: List[str] = Field(default_factory=list)
    possible: bool
    # Pixel layout for rendering only: entity id -> (x, y, w, h)
    layout: Dict[str, Tuple[int, int, int, int]] = Field(default_factory=dict)
