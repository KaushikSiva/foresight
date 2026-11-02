"""Run a packing program on the simulated Panda, with crush and spill detection from the physics."""
import mujoco
import numpy as np

from .catalog import CARTON_HEIGHT, CARTONS, CATALOG, FRAGILE_LIMIT_N
from .rules import apply_call, new_state, packed_counts
from .world import load_model

READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
TUCKED = np.array([2.6, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
TCP_OFFSET = np.array([0.0, 0.0, 0.1034])
TRAVEL_Z = 0.34
CONTROL_STEPS = 10  # 20 ms control period at a 2 ms physics step
GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0


def _top_down_quat():
    """Hand pointing straight down, with the fingers closing along world x."""
    down, yaw, out = np.array([0.0, 1.0, 0.0, 0.0]), np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]), np.zeros(4)
    mujoco.mju_mulQuat(out, yaw, down)
    return out


class Station:
    def __init__(self, totes, seed=0, on_frame=None):
        self.model = load_model(totes, seed)
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        m = self.model
        joints = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
        self.qadr = np.array([m.jnt_qposadr[j] for j in joints])
        self.dadr = np.array([m.jnt_dofadr[j] for j in joints])
        self.hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.finger_q = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"finger_joint{i}")] for i in (1, 2)]
        self.quat = _top_down_quat()
        self.totes = {t: dict(c) for t, c in totes.items() if c}
        self.remaining = {t: list(range(c["count"])) for t, c in self.totes.items()}
        self.held = None
        self.held_rel = None
        self.placed = []  # (unit name, sku)
        self.on_frame = on_frame
        self.caption = ""
        self.max_ik_error = 0.0
        self.unit_bodies = {}
        for tote, content in self.totes.items():
            for i in range(content["count"]):
                self.unit_bodies[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{tote}_{i}")] = (f"{tote}_{i}", content["sku"])
        self.loads = {}
        self.data.qpos[self.qadr] = READY
        self.data.qpos[self.finger_q] = 0.04
        self.data.ctrl[:7] = READY
        self.data.ctrl[7] = GRIPPER_OPEN
        mujoco.mj_forward(m, self.data)
        self._settle(0.6)

    # ---- low level ----

    def _follow(self):
        if self.held is None:
            return
        d = self.data
        hand_pos, hand_quat = d.xpos[self.hand], d.xquat[self.hand]
        pos, quat = np.zeros(3), np.zeros(4)
        mujoco.mju_rotVecQuat(pos, self.held_rel[0], hand_quat)
        mujoco.mju_mulQuat(quat, hand_quat, self.held_rel[1])
        joint = self.model.body_jntadr[self.held]
        qadr, dadr = self.model.jnt_qposadr[joint], self.model.jnt_dofadr[joint]
        d.qpos[qadr:qadr + 3] = hand_pos + pos
        d.qpos[qadr + 3:qadr + 7] = quat
        d.qvel[dadr:dadr + 6] = 0

    def _step(self, n=CONTROL_STEPS):
        for _ in range(n):
            self._follow()
            mujoco.mj_step(self.model, self.data)
            self._frame_tick()

    def _frame_tick(self):
        if self.on_frame and self.data.time >= getattr(self, "_next_frame", 0.0):
            self._next_frame = self.data.time + 1 / 30
            self.on_frame(self)

    def _settle(self, seconds):
        for _ in range(int(seconds / (self.model.opt.timestep * CONTROL_STEPS))):
            self._step()

    def _ik(self, q, target):
        """Damped least squares from q toward the TCP target, keeping the hand pointing down."""
        m, d = self.model, self.ik_data
        d.qpos[self.qadr] = q
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        error = np.zeros(6)
        for _ in range(12):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            tcp = d.xpos[self.hand] + d.xmat[self.hand].reshape(3, 3) @ TCP_OFFSET
            error[:3] = target - tcp
            conj, delta = np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(conj, d.xquat[self.hand])
            mujoco.mju_mulQuat(delta, self.quat, conj)
            mujoco.mju_quat2Vel(error[3:], delta, 1.0)
            if np.linalg.norm(error) < 5e-4:
                break
            mujoco.mj_jac(m, d, jacp, jacr, tcp, self.hand)
            jac = np.vstack((jacp[:, self.dadr], jacr[:, self.dadr]))
            step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
            null = (np.eye(7) - np.linalg.pinv(jac) @ jac) @ (READY - d.qpos[self.qadr]) * 0.05
            d.qpos[self.qadr] += np.clip(step + null, -0.2, 0.2)
        self.max_ik_error = max(self.max_ik_error, float(np.linalg.norm(error[:3])))
        return d.qpos[self.qadr].copy()

    def _tcp(self):
        d = self.data
        return d.xpos[self.hand] + d.xmat[self.hand].reshape(3, 3) @ TCP_OFFSET

    def move(self, target, speed=0.45):
        start, target = self._tcp(), np.array(target, float)
        duration = max(0.25, np.linalg.norm(target - start) / speed)
        ticks = int(duration / (self.model.opt.timestep * CONTROL_STEPS))
        q = self.data.qpos[self.qadr].copy()
        for i in range(1, ticks + 1):
            s = i / ticks
            s = s * s * (3 - 2 * s)
            q = self._ik(q, start + (target - start) * s)
            self.data.ctrl[:7] = q
            self._step()
        self._step(100)

    def gripper(self, value, seconds=0.35):
        self.data.ctrl[7] = value
        self._settle(seconds)

    # ---- skills ----

    def pick(self, tote):
        index = self.remaining[tote].pop()
        name = f"{tote}_{index}"
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        sku = CATALOG[self.totes[tote]["sku"]]
        pos = self.data.xpos[body].copy()
        top = pos[2] + sku.height
        grip_z = max(top - 0.03, pos[2] + 0.012)
        self.caption = f"pick {sku.name} from {tote}"
        self.move([pos[0], pos[1], TRAVEL_Z])
        self.move([pos[0], pos[1], grip_z], speed=0.25)
        self.gripper(GRIPPER_CLOSED)
        self.held = body
        d = self.data
        conj, rel_quat, rel_pos = np.zeros(4), np.zeros(4), np.zeros(3)
        mujoco.mju_negQuat(conj, d.xquat[self.hand])
        mujoco.mju_mulQuat(rel_quat, conj, d.xquat[body])
        mujoco.mju_rotVecQuat(rel_pos, d.xpos[body] - d.xpos[self.hand], conj)
        self.held_rel = (rel_pos, rel_quat)
        self.move([pos[0], pos[1], TRAVEL_Z], speed=0.3)
        return name, sku.name

    def place(self, carton, slot, column):
        """Lower the unit onto the top of its column and release it."""
        c = CARTONS[carton]
        x = c["center"][0] + c["slots"][slot][0]
        y = c["center"][1] + c["slots"][slot][1]
        sku = self.unit_bodies[self.held][1]
        self.caption = f"place {sku} in {carton}-{slot}"
        stack = sum(CATALOG[s].height for s in column)
        body_z = self.data.xpos[self.held][2]
        target_bottom = 0.004 + stack + 0.006
        self.move([x, y, TRAVEL_Z])
        tcp_z = self._tcp()[2]
        self.move([x, y, tcp_z - (body_z - target_bottom)], speed=0.2)
        self.placed.append(self.unit_bodies[self.held])
        self.held = None
        self.gripper(GRIPPER_OPEN)
        self.move([x, y, TRAVEL_Z], speed=0.3)
        self._measure_loads()

    def _measure_loads(self, seconds=0.5):
        """Average the load pressing down on each packed unit from units resting on it."""
        m, d = self.model, self.data
        ticks = int(seconds / (m.opt.timestep * CONTROL_STEPS))
        force = np.zeros(6)
        totals = {}
        for _ in range(ticks):
            self._step()
            for i in range(d.ncon):
                contact = d.contact[i]
                a, b = m.geom_bodyid[contact.geom1], m.geom_bodyid[contact.geom2]
                if a not in self.unit_bodies or b not in self.unit_bodies:
                    continue
                mujoco.mj_contactForce(m, d, i, force)
                lower, upper = (a, b) if d.xpos[a][2] < d.xpos[b][2] else (b, a)
                totals[lower] = totals.get(lower, 0.0) + abs(force[0])
        for body, total in totals.items():
            self.loads[body] = max(self.loads.get(body, 0.0), total / ticks)

    def report(self, carton):
        """The physical outcome: what is inside the carton, what spilled, what was crushed."""
        result = {"in_carton": {}, "spilled": [], "crushed": [], "loads_n": {}, "ik_error_m": round(self.max_ik_error, 5)}
        c = CARTONS.get(carton) if carton else None
        for body, (unit, sku) in self.unit_bodies.items():
            pos = self.data.xpos[body]
            placed = (unit, sku) in self.placed
            inside = bool(c) and abs(pos[0] - c["center"][0]) < c["inner"][0] / 2 + 0.01 \
                and abs(pos[1] - c["center"][1]) < c["inner"][1] / 2 + 0.01 and 0 <= pos[2] < CARTON_HEIGHT + 0.12
            if inside:
                result["in_carton"][sku] = result["in_carton"].get(sku, 0) + 1
            elif placed:
                result["spilled"].append(unit)
            load = self.loads.get(body, 0.0)
            result["loads_n"][unit] = round(load, 2)
            if placed and CATALOG[sku].fragile and load > FRAGILE_LIMIT_N:
                result["crushed"].append(unit)
        result["loads_n"] = {k: v for k, v in result["loads_n"].items() if v > 0}
        return result


def execute(scene_totes, calls, seed=0, on_frame=None):
    """Run calls physically and by the rules. Physically impossible steps stop the run."""
    station = Station(scene_totes, seed, on_frame)
    state = new_state(scene_totes)
    problems, halted = [], None
    for i, (name, args) in enumerate(calls):
        column = list(state["columns"].get(args[0], [])) if name == "place" else None
        fatal, found = apply_call(state, name, args)
        problems += [{"step": i, "code": c, "message": m} for c, m in found]
        if fatal:
            halted = {"step": i, "code": found[0][0], "message": found[0][1]}
            break
        if name == "pick":
            station.pick(args[0])
        elif name == "place":
            station.place(state["carton"], args[0], column)
        elif name == "seal":
            station.caption = "seal carton"
            station._settle(0.5)
    if on_frame:
        station.move([0.3, 0.0, 0.42])
        station._settle(0.5)
    report = station.report(state["carton"])
    report.update(carton=state["carton"], rule_problems=problems, halted=halted, sealed=state["sealed"],
                  held=state["held"], kg=state["kg"], packed=packed_counts(state))
    return report


def perception_image(totes, seed=0, width=1280, height=720):
    """What the overhead camera sees, with the arm swung clear."""
    from .world import load_model as _load
    model = _load(totes, seed)
    data = mujoco.MjData(model)
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
    for j, q in zip(joints, TUCKED):
        data.qpos[model.jnt_qposadr[j]] = q
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height, width)
    renderer.update_scene(data, camera="overhead")
    return renderer.render()
