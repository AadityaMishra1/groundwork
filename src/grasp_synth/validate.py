"""Physics validation of a pre-grasp: approach -> close -> lift -> shake -> verdict.

The wrist free joint is driven kinematically (pose written per control tick);
the fingers are dynamic, force-limited actuators. The object has only a free
joint — whether the grasp holds is decided purely by contact friction. Trials
where the hand squeezes harder than SQUEEZE_BUDGET are rejected as
non-physical, since a kinematic wrist could otherwise brace the object
against the ground with unbounded force.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from .scene import (
    FINGER_GROUPS, FLEXION_KEYS, THUMB_OPPOSITION_KEY,
    CylinderSpec, build_scene, object_rest_pose,
)

APPROACH_S, SETTLE_S, CLOSE_S, SQUEEZE_S, LIFT_S, SHAKE_S, HOLD_S = 0.5, 0.2, 0.8, 0.3, 1.2, 0.5, 0.4
LIFT_HEIGHT = 0.35
SHAKE_AMP, SHAKE_HZ = 0.02, 4.0
BACKOFF = 0.08                # m; spawn this far along palm +y, approach in
CARRY_TILT = 0.6              # rad; supinate the palm toward "up" while lifting
GRIP_N = 12.0                 # N; force-servoed close freezes at this grip force
IN_PLACE_GRIP_N = 45.0        # N; firmer: loose cages let cylinders roll out
CONTACT_STOP_N = 8.0          # N; palm-on-object force that halts the approach (safety)
SQUEEZE_BUDGET = 150.0        # N; airborne only — catches solver blowups, not firm grips
MAX_START_PENETRATION = 0.006


@dataclass
class TrialResult:
    success: bool
    reason: str
    lift_height: float
    fingers_in_contact: int
    thumb_in_contact: bool
    peak_force: float = 0.0


def _geom_owner(model: mujoco.MjModel, geom_id: int) -> str:
    return model.body(model.geom_bodyid[geom_id]).name


PALM_REGION = ("palm", "metacarpal", "knuckle")


def _hand_object_force(model, data, region: tuple[str, ...] | None = None) -> float:
    force = np.zeros(6)
    total = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = _geom_owner(model, c.geom1)
        b2 = _geom_owner(model, c.geom2)
        if "object" not in (b1, b2):
            continue
        partner = b2 if b1 == "object" else b1
        if not partner.startswith("H_"):
            continue
        if region is not None and not any(r in partner for r in region):
            continue
        mujoco.mj_contactForce(model, data, i, force)
        total += abs(force[0])
    return total


def _contact_summary(model, data, obj_geom_id):
    groups, other = set(), False
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if obj_geom_id not in (g1, g2):
            continue
        partner = _geom_owner(model, g2 if g1 == obj_geom_id else g1)
        if partner.startswith("H_"):
            hit = [f for f in FINGER_GROUPS if f in partner]
            if hit:
                groups.add(hit[0])
            elif "palm" in partner or "metacarpal" in partner or "knuckle" in partner:
                groups.add("palm")
            else:
                other = True  # forearm cradling doesn't count
        else:
            other = True      # ground or anything else
    return groups, other


def run_trial(obj: CylinderSpec, palm_pos, palm_quat, closure: float,
              model: mujoco.MjModel | None = None,
              render_path: str | None = None,
              in_place: bool = False) -> TrialResult:
    """in_place=True: skip the scripted approach and close on the object with
    gravity disabled (grasp-STATE synthesis a la DexGraspNet), then restore
    gravity for the lift/shake verdict. The verdict itself is always honest
    physics — friction-only, force-limited fingers. Scripted approaches are
    scaffolding; reaching the grasp state is the RL policy's job."""
    model = model if model is not None else build_scene(obj)
    data = mujoco.MjData(model)

    frames = []
    renderer = cam = None
    if render_path:
        renderer = mujoco.Renderer(model, 320, 320)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation = 0.6, -25

    obj_pos, obj_quat = object_rest_pose(obj)
    qadr = model.jnt_qposadr[model.body("object").jntadr[0]]
    data.qpos[qadr:qadr + 3] = obj_pos
    data.qpos[qadr + 3:qadr + 7] = obj_quat

    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(palm_quat, dtype=float))
    away = R.reshape(3, 3)[:, 1]  # palm +y: opposite the grasp side
    grasp_wrist, wrist_quat = palm_to_wrist(model, palm_pos, palm_quat)
    start_wrist = grasp_wrist + BACKOFF * away

    wjnt = model.body("H_rh_forearm").jntadr[0]
    wadr = model.jnt_qposadr[wjnt]
    wdof = model.jnt_dofadr[wjnt]

    # carry orientation: palm normal tilted toward "up" so gravity presses
    # the object into the palm during transport (human supination reflex)
    n = -R.reshape(3, 3)[:, 1]
    axis = np.cross(n, [0.0, 0.0, 1.0])
    if np.linalg.norm(axis) > 1e-6:
        axis /= np.linalg.norm(axis)
        q_sup = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q_sup, axis, CARRY_TILT)
        carry_palm_quat = np.zeros(4)
        mujoco.mju_mulQuat(carry_palm_quat, q_sup, np.asarray(palm_quat, dtype=float))
        _, carry_wrist_quat = palm_to_wrist(model, palm_pos, carry_palm_quat)
    else:
        carry_wrist_quat = wrist_quat

    def nlerp(q0, q1, t):
        q = (1 - t) * np.asarray(q0) + t * np.asarray(q1) * np.sign(np.dot(q0, q1))
        return q / np.linalg.norm(q)

    prev_base = [None]
    quat_now = [np.asarray(wrist_quat, dtype=float)]

    def pin_base(pos, dt=None):
        data.qpos[wadr:wadr + 3] = pos
        data.qpos[wadr + 3:wadr + 7] = quat_now[0]
        data.qvel[wdof:wdof + 6] = 0.0
        if dt and prev_base[0] is not None:
            # carry the true base velocity so contact friction sees the
            # hand's real motion instead of a teleporting static surface
            data.qvel[wdof:wdof + 3] = (np.asarray(pos) - prev_base[0]) / dt
        prev_base[0] = np.asarray(pos, dtype=float).copy()

    pin_base(start_wrist)
    mujoco.mj_forward(model, data)

    obj_geom_id = model.geom("object_geom").id
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist < -MAX_START_PENETRATION:
            n1 = _geom_owner(model, c.geom1)
            n2 = _geom_owner(model, c.geom2)
            if n1.startswith("H_") or n2.startswith("H_"):
                return TrialResult(False, "invalid_start_penetration", 0.0, 0, False)

    nsub = max(1, round(1 / model.opt.timestep / 100))  # control at ~100 Hz
    peak_force = [0.0]
    base_now = [start_wrist.copy()]

    def run(seconds, base_fn=None, ctrl_fn=None):
        steps = int(seconds / (model.opt.timestep * nsub))
        for k in range(steps):
            a = k / max(1, steps - 1)
            if base_fn is not None:
                base_now[0] = base_fn(a)
            if ctrl_fn is not None:
                ctrl_fn(a)
            pin_base(base_now[0], dt=model.opt.timestep * nsub)
            for _ in range(nsub):
                mujoco.mj_step(model, data)
            peak_force[0] = max(peak_force[0], _hand_object_force(model, data))
            if renderer is not None and k % 10 == 0:
                cam.lookat = data.body("H_rh_palm").xpos
                renderer.update_scene(data, cam)
                frames.append(renderer.render().copy())

    lo = model.actuator_ctrlrange[:, 0]
    hi = model.actuator_ctrlrange[:, 1]
    open_ctrl = np.zeros(model.nu)
    closed_ctrl = open_ctrl.copy()
    for i in range(model.nu):
        name = model.actuator(i).name
        if any(k in name for k in FLEXION_KEYS):
            closed_ctrl[i] = lo[i] + closure * (hi[i] - lo[i])
        elif THUMB_OPPOSITION_KEY in name:
            closed_ctrl[i] = hi[i]
    # pre-shape into a light "C" during approach; fingertip grazes then slide
    # along the object instead of halting the wrist — only palm-region
    # contact stops the approach
    data.ctrl[:] = open_ctrl  # straight fingers: they clear the object's path into the palm
    touched = [False]

    def approach(a):
        if not touched[0] and _hand_object_force(
                model, data, region=PALM_REGION) > CONTACT_STOP_N:
            touched[0] = True
        if touched[0]:
            return base_now[0]
        return start_wrist + a * (grasp_wrist - start_wrist)

    if in_place:
        base_now[0] = grasp_wrist.copy()
        pin_base(grasp_wrist)
        gravity = model.opt.gravity[2]
        model.opt.gravity[2] = 0.0
        try:
            run(SETTLE_S)
        finally:
            pass
    else:
        run(APPROACH_S, base_fn=approach)
        run(SETTLE_S)

    # force-servoed close (how real grippers work): ramp the position targets
    # toward full closure but freeze the ramp once grip force is reached —
    # more closure past that point whips the fingertips through the envelope
    # and ejects the object
    grip = [0.0]
    ojnt = model.body("object").jntadr[0]
    odof = model.jnt_dofadr[ojnt]

    def close(a):
        f = _hand_object_force(model, data)
        target_n = IN_PLACE_GRIP_N if in_place else GRIP_N
        if f < target_n:
            grip[0] = min(1.0, max(grip[0], a))
        data.ctrl[:] = open_ctrl + grip[0] * (closed_ctrl - open_ctrl)
        if in_place:
            # object pinned during synthesis (scaffolding only) — it is
            # fully free in every verdict phase
            data.qpos[qadr:qadr + 3] = obj_pos
            data.qpos[qadr + 3:qadr + 7] = obj_quat
            data.qvel[odof:odof + 6] = 0.0

    run(CLOSE_S, ctrl_fn=close)

    if in_place:
        model.opt.gravity[2] = gravity  # verdict phases run under real gravity
        run(SETTLE_S)  # object released: let the grasp take the load

    run(SQUEEZE_S)  # let contacts seat under sustained finger force

    # squeeze budget applies to airborne phases only: bracing the object
    # against the ground during approach/close is legitimate
    peak_force[0] = 0.0

    lift_from = base_now[0].copy()

    def lift(a):
        quat_now[0] = nlerp(wrist_quat, carry_wrist_quat, min(1.0, a * 1.5))
        return lift_from + np.array([0, 0, LIFT_HEIGHT * a])

    run(LIFT_S, base_fn=lift)

    top = base_now[0].copy()

    def shake(a):
        t = a * SHAKE_S
        return top + SHAKE_AMP * np.array([
            np.sin(2 * np.pi * SHAKE_HZ * t),
            np.sin(2 * np.pi * SHAKE_HZ * t * 0.7),
            0.5 * np.sin(2 * np.pi * SHAKE_HZ * t * 1.3),
        ])
    run(SHAKE_S, base_fn=shake)
    run(HOLD_S, base_fn=lambda a: top)

    if renderer is not None and frames:
        import PIL.Image
        n = len(frames)
        pick = [frames[i] for i in range(0, n, max(1, n // 12))][:12]
        w = h = 320
        sheet = PIL.Image.new("RGB", (w * 4, h * 3))
        for i, f in enumerate(pick):
            sheet.paste(PIL.Image.fromarray(f), ((i % 4) * w, (i // 4) * h))
        sheet.save(render_path)
        renderer.close()

    obj_z = float(data.qpos[qadr + 2])
    groups, other_contact = _contact_summary(model, data, obj_geom_id)
    fingers = [g for g in groups if g in FINGER_GROUPS]
    ok_height = obj_z > 0.20
    ok_contacts = not other_contact
    ok_fingers = len(fingers) >= 3 and "th" in fingers
    ok_force = peak_force[0] <= SQUEEZE_BUDGET
    success = ok_height and ok_contacts and ok_fingers and ok_force
    reason = ("ok" if success
              else "over_squeeze" if not ok_force
              else "dropped" if not ok_height
              else "non_hand_contact" if not ok_contacts
              else "insufficient_fingers")
    return TrialResult(success, reason, obj_z, len(fingers), "th" in fingers,
                       peak_force[0])


def palm_to_wrist(model, palm_pos, palm_quat):
    """Convert a desired PALM world pose to the wrist free-joint (pos, quat).

    The palm body frame is both offset AND rotated relative to the forearm
    in the MJCF — correcting only the position silently rotates every grasp.
    """
    off, R_fp = _palm_xform(model)
    Rp = np.zeros(9)
    mujoco.mju_quat2Mat(Rp, np.asarray(palm_quat, dtype=float))
    Rf = Rp.reshape(3, 3) @ R_fp.T
    pos = np.asarray(palm_pos) - Rf @ off
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, Rf.flatten())
    return pos, quat


_PALM_XFORM_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _palm_xform(model) -> tuple[np.ndarray, np.ndarray]:
    """(palm origin in forearm frame, forearm->palm rotation) at qpos=0."""
    key = id(model)
    if key not in _PALM_XFORM_CACHE:
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        fore = d.body("H_rh_forearm")
        palm = d.body("H_rh_palm")
        Rf = fore.xmat.reshape(3, 3)
        Rp = palm.xmat.reshape(3, 3)
        _PALM_XFORM_CACHE[key] = (Rf.T @ (palm.xpos - fore.xpos), Rf.T @ Rp)
    return _PALM_XFORM_CACHE[key]
