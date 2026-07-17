"""Pre-grasp pose sampling for cylinders on the ground.

A pre-grasp is a wrist (palm) pose + closure fraction. The sampler only has
to be *plausible* — the physics filter in validate.py is the arbiter. Frames:
palm -y faces the object (grasp side), palm +z is finger extension.
"""

from dataclasses import dataclass, asdict

import numpy as np

from .scene import CylinderSpec, object_rest_pose


@dataclass
class PreGrasp:
    palm_pos: np.ndarray      # world, meters
    palm_quat: np.ndarray     # world, wxyz
    closure: float            # 0..1 fraction of flexion actuator range
    kind: str                 # "side" | "top" | "lying_top"

    def to_dict(self):
        d = asdict(self)
        d["palm_pos"] = self.palm_pos.tolist()
        d["palm_quat"] = self.palm_quat.tolist()
        return d


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w < 1e-6:
        # fall back through largest diagonal element
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(1e-12, 1 + R[i, i] - R[j, j] - R[k, k])) * 2
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
        return q / np.linalg.norm(q)
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _frame(grasp_dir: np.ndarray, finger_dir: np.ndarray) -> np.ndarray:
    """Rotation with palm -y = grasp_dir, palm +z = finger_dir (orthogonalized)."""
    y = -grasp_dir / np.linalg.norm(grasp_dir)
    z = finger_dir - np.dot(finger_dir, y) * y
    z = z / np.linalg.norm(z)
    x = np.cross(y, z)
    return _rot_to_quat(np.column_stack([x, y, z]))


def sample_pregrasp(rng: np.random.Generator, obj: CylinderSpec,
                    clearance: float = 0.023) -> PreGrasp:
    # `clearance` is the measured protrusion of the open hand toward the
    # grasp side (scene.grasp_clearance): the object surface must pass under
    # it or the approach plows the object over. Fingertips curl to ~0.06
    # below the palm plane, so the graspable band is clearance..0.06.
    pos, _ = object_rest_pose(obj)
    standoff = clearance + rng.uniform(0.002, 0.012)  # palm origin to object surface
    palm_depth = rng.uniform(0.08, 0.12)        # object center along finger axis
    closure = rng.uniform(0.65, 1.0)

    if obj.lying:
        # grasp from above, fingers straddling the cylinder axis
        axis = np.array([np.cos(obj.yaw), np.sin(obj.yaw), 0])
        along = rng.uniform(-0.3, 0.3) * obj.height
        target = pos + axis * along
        grasp_dir = np.array([0, 0, -1.0])
        perp = np.array([-axis[1], axis[0], 0])
        if rng.random() < 0.5:
            perp = -perp
        # tilt fingers slightly past vertical so they hook under the far side
        tilt = rng.uniform(-0.2, 0.35)
        finger_dir = perp * np.cos(tilt) + np.array([0, 0, -np.sin(tilt)])
        palm_pos = (target - grasp_dir * (obj.radius + standoff)
                    - finger_dir * palm_depth)
        return PreGrasp(palm_pos, _frame(grasp_dir, finger_dir), closure, "lying_top")

    theta = rng.uniform(0, 2 * np.pi)
    radial = np.array([np.cos(theta), np.sin(theta), 0])

    if obj.height >= 0.13 and rng.random() < 0.7:
        # side power grasp on a standing cylinder
        grasp_h = rng.uniform(0.30, 0.55) * obj.height  # low: less tipping torque
        target = np.array([0, 0, grasp_h])
        grasp_dir = -radial
        # fingers wrap horizontally around the axis; small downward pitch ok
        tangent = np.array([-radial[1], radial[0], 0])
        if rng.random() < 0.5:
            tangent = -tangent
        # near-horizontal fingers: pitched-down grasps close over the rim
        # instead of wrapping the barrel
        pitch = rng.uniform(-0.1, 0.1)
        finger_dir = tangent * np.cos(pitch) + np.array([0, 0, -np.sin(pitch)])
        palm_pos = (target - grasp_dir * (obj.radius + standoff)
                    - finger_dir * palm_depth)
        return PreGrasp(palm_pos, _frame(grasp_dir, finger_dir), closure, "side")

    # top grasp: palm faces down over the rim, fingers across the diameter
    grasp_dir = np.array([0, 0, -1.0])
    finger_dir = radial
    target = np.array([0, 0, obj.height])
    palm_pos = (target - grasp_dir * standoff
                - finger_dir * (palm_depth - obj.radius))
    return PreGrasp(palm_pos, _frame(grasp_dir, finger_dir), closure, "top")
