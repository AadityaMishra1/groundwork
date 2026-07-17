"""Floating-hand grasp scene: five-finger hand on a mocap-welded free base,
one cylinder object resting on the ground.

The object has ONLY a free joint — no constraints ever touch it. The wrist is
driven kinematically through the mocap weld; that is legitimate for grasp
*synthesis* (the humanoid arm replaces it downstream). Whether a grasp holds
is decided purely by contact friction.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SHADOW_XML = REPO / "assets/menagerie/shadow_hand/right_hand.xml"

# Palm frame (measured empirically, see PLAN.md Phase 2):
#   +z: finger extension direction
#   -y: grasp side (fingertips curl toward -y)
#   +x: thumb side
PALM_BODY = "H_rh_palm"
WRIST_BODY = "H_rh_forearm"
FINGER_GROUPS = ("ff", "mf", "rf", "lf", "th")  # substrings of body names

# actuator name substrings -> closure behavior
FLEXION_KEYS = ("FFJ3", "FFJ0", "MFJ3", "MFJ0", "RFJ3", "RFJ0",
                "LFJ3", "LFJ0", "THJ2", "THJ1")
THUMB_OPPOSITION_KEY = "THJ4"


@dataclass
class CylinderSpec:
    radius: float = 0.035      # m
    height: float = 0.16       # m
    mass: float = 0.3          # kg
    friction: float = 0.8      # sliding friction, both surfaces
    lying: bool = False        # standing upright vs lying on its side
    yaw: float = 0.0           # world yaw of the cylinder axis when lying

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "CylinderSpec":
        return cls(
            radius=rng.uniform(0.025, 0.045),
            height=rng.uniform(0.10, 0.25),
            mass=rng.uniform(0.1, 0.8),
            friction=rng.uniform(0.4, 1.2),
            lying=bool(rng.random() < 0.5),
            yaw=rng.uniform(0, 2 * np.pi),
        )


def build_scene(obj: CylinderSpec, hand_xml: Path = SHADOW_XML) -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    spec.worldbody.add_geom(
        name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
        friction=[obj.friction, 0.005, 0.0001],
    )
    spec.worldbody.add_light(pos=[0, 0, 2.0])

    # object: free joint only, resting pose set in reset()
    body = spec.worldbody.add_body(name="object", pos=[0, 0, obj.height / 2])
    body.add_freejoint()
    # condim=6 (torsional + rolling friction) and priority=1 as in the
    # Menagerie's tuned manipulation scene — without these, squeezed light
    # objects skate and get ejected.
    body.add_geom(
        name="object_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[obj.radius, obj.height / 2, 0],
        mass=obj.mass, friction=[obj.friction, 0.01, 0.003],
        condim=6, priority=1,
        rgba=[0.8, 0.3, 0.2, 1],
    )

    # hand: attached under a free joint whose pose is written kinematically
    # each control tick (a mocap-weld base proved dynamically unstable with
    # this model — the hand thrashed meters from its target). Fingers remain
    # fully dynamic and force-limited; grasp verdicts come from finger
    # friction alone, and trials that exceed a squeeze-force budget are
    # rejected in validate.py.
    hand = mujoco.MjSpec.from_file(str(hand_xml))
    frame = spec.worldbody.add_frame(pos=[0, 0, 0.5])
    attached = frame.attach_body(hand.body("rh_forearm"), "H_", "")
    attached.add_freejoint()
    return spec.compile()


def grasp_clearance(model: mujoco.MjModel) -> float:
    """How far the open hand protrudes toward the grasp side (palm -y),
    measured from the palm origin over all finger/palm geoms at qpos=0.
    Object surfaces must pass below this line or the approach plows them."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    palm = data.body(PALM_BODY)
    Rp = palm.xmat.reshape(3, 3)
    v = -Rp[:, 1]  # grasp-side direction in world frame
    min_y = 0.0
    for g in range(model.ngeom):
        body = model.body(model.geom_bodyid[g]).name
        if not body.startswith("H_") or "forearm" in body or "wrist" in body:
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        size = model.geom_size[g]
        vloc = np.abs(data.geom_xmat[g].reshape(3, 3).T @ v)
        gt = model.geom_type[g]
        if gt == mujoco.mjtGeom.mjGEOM_SPHERE:
            ext = size[0]
        elif gt in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            ext = size[0] + size[1] * vloc[2]
        else:  # box, or mesh via its aabb half-sizes
            ext = float(size @ vloc)
        rel_y = (Rp.T @ (data.geom_xpos[g] - palm.xpos))[1]
        min_y = min(min_y, rel_y - ext)
    return -min_y


def object_rest_pose(obj: CylinderSpec) -> tuple[np.ndarray, np.ndarray]:
    """(pos, quat) of the cylinder resting on the ground at the origin."""
    if obj.lying:
        # axis horizontal along yaw direction; MuJoCo cylinder axis is local z
        pos = np.array([0.0, 0.0, obj.radius])
        # rotate local z into horizontal: 90 deg about world x, then yaw about z
        q_tilt = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
        q_yaw = np.array([np.cos(obj.yaw / 2), 0, 0, np.sin(obj.yaw / 2)])
        quat = _quat_mul(q_yaw, q_tilt)
    else:
        pos = np.array([0.0, 0.0, obj.height / 2])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
    return pos, quat


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
