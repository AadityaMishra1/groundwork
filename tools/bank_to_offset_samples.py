"""Convert banked engine-verified in-air grasp states (Isaac harvest) into
palm-frame object-offset samples for tools/apply_grasp_offset.py.

Source: refs/held_states.npz — the pod's bank_run5.pt lift/hold families plus
chain_bank_grasp5.pt held (criteria from harvest_traj.py / harvest_success.py:
engine grasp >=3 finger groups + thumb via filtered contact sensors AND
obj_z > 0.22; hold/held additionally hold_counter >= 25). These are the
m2d policy's actual grasps in Isaac — real Inspire-hand holds. Used because
the current lead checkpoint grasps at the floor but does not lift (measured:
0 in-air frames in 6000 probe steps at obj_z>0.15; training telemetry qual=0
succ=0), so it cannot supply in-air frames at any threshold.

The palm body (right_wrist_yaw_link) kinematics are identical in the MuJoCo
and Isaac models through the wrist, so its world pose is recovered by MuJoCo
FK of the snapshot's own root + 29 body joints; the object pose is stored in
the same env frame.

    .venv/bin/python -u tools/bank_to_offset_samples.py
"""
import os
import sys

import numpy as np
import mujoco

TOOLS = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.join(os.path.dirname(TOOLS), "refs")
sys.path.insert(0, TOOLS)
import wbik as W  # noqa: E402
from apply_grasp_offset import quat_mul  # noqa: E402


def main():
    d = np.load(os.path.join(REFS, "held_states.npz"), allow_pickle=False)
    names = [str(n) for n in d["joint_names"]]
    cols = [names.index(n) for n in W.BODY_JOINTS]
    n = len(d["jp"])
    P, Q = np.zeros((n, 3)), np.zeros((n, 4))
    D, M = W.D, W.M
    for i in range(n):
        D.qpos[:] = 0.0
        D.qpos[0:3] = d["root"][i, 0:3]
        D.qpos[3:7] = d["root"][i, 3:7]
        D.qpos[W.JADR] = d["jp"][i, cols]
        for a, v in W.HAND_Q.items():
            D.qpos[a] = v
        mujoco.mj_kinematics(M, D)
        pp, pq = D.xpos[W.PALM].copy(), D.xquat[W.PALM].copy()
        qc = pq * np.array([1, -1, -1, -1])
        P[i] = quat_mul(quat_mul(qc, np.concatenate([[0.0],
                        d["obj"][i, 0:3] - pp])), pq)[1:]
        qr = quat_mul(qc, d["obj"][i, 3:7])
        Q[i] = -qr if qr[0] < 0 else qr
    out = os.path.join(REFS, "grasp_offset_samples.npz")
    np.savez(out, p_rel=P, q_rel=Q, obj_z=d["obj"][:, 2],
             env=np.arange(n), ep=np.zeros(n, np.int64),
             family=d["family"], min_z=0.22)
    for f in sorted(set(d["family"])):
        m = d["family"] == f
        med = np.median(P[m], axis=0)
        print(f"{f:8s} n={int(m.sum()):4d} objz p50={np.median(d['obj'][m, 2]):.3f} "
              f"p_rel med=({med[0]:+.4f},{med[1]:+.4f},{med[2]:+.4f})")
    print(f"saved {n} samples -> {out}")


if __name__ == "__main__":
    main()
