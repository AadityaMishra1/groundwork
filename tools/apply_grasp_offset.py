"""Rewrite the gc reference bank's ALOFT object placements from the MEASURED
in-grasp palm<->object offset (Isaac physics, Inspire hand).

Why: refs/ref_traj_bank_gc.npz places the aloft (phase>2) object at a
MuJoCo-derived constant world offset from the palm. The Isaac Inspire hand
holds the cylinder somewhere else, so aloft REF_TRACK restores show engine
grasp 0.000 at t=1 and the object free-falls — the starts are inert.
pod/g1_grasp/probe_grasp_offset.py measures where the lead policy's
engine-verified holds ACTUALLY put the object relative to the palm body
(right_wrist_yaw_link — a body whose kinematics are identical in the MuJoCo
and Isaac models up to the wrist, so a palm-frame offset transfers).

This tool:
  1. aggregates the harvest samples (median; if the p25/p75 spread exceeds
     3 cm on any axis it 2-means-clusters and takes the dominant mode);
  2. recomputes every aloft frame's object pose as
         obj = palm_pose (MuJoCo FK of the bank's own joints+root)  ∘  T_meas
     with a floor clamp (obj center z >= 0.076 — early P3 frames start at
     floor rest and must not penetrate);
  3. re-finite-differences the object velocities INSIDE the aloft segment
     (frames before the P2/P3 boundary are byte-untouched);
  4. writes refs/ref_traj_bank_gc2.npz — every other key (joint_pos,
     joint_targets, tau, kp_eff, root, phase, ...) copied verbatim.

    .venv/bin/python -u tools/apply_grasp_offset.py \
        --samples refs/grasp_offset_samples.npz
"""
import argparse
import os
import sys

import numpy as np
import mujoco

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
REFS = os.path.join(ROOT, "refs")
sys.path.insert(0, TOOLS)
import wbik as W  # noqa: E402

FLOOR_Z = 0.076          # cylinder half-height + 1 mm


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def quat_rot(q, v):
    return quat_mul(quat_mul(q, np.concatenate([[0.0], v])),
                    q * np.array([1, -1, -1, -1]))[1:]


def aggregate(path, min_z, families=""):
    d = np.load(path, allow_pickle=False)
    z = d["obj_z"]
    if families and "family" in d.files:
        fams = [f.strip() for f in families.split(",") if f.strip()]
        m = np.isin(d["family"], fams)
        note = f"families={fams} (stable carries; all engine-verified holds)"
    else:
        m = z > min_z
        note = f"band obj_z>{min_z}"
        if m.sum() < 500:
            min_z = 0.15
            m = z > min_z
            note = f"RELAXED to obj_z>{min_z} (in-air frames rare)"
    p = d["p_rel"][m]
    q = d["q_rel"][m]
    eps = len({(int(a), int(b)) for a, b in zip(d["env"][m], d["ep"][m])})
    p25, p75 = np.percentile(p, 25, axis=0), np.percentile(p, 75, axis=0)
    spread = p75 - p25
    med = np.median(p, axis=0)
    print(f"AGGREGATE {note}: n={int(m.sum())} episodes={eps}")
    print(f"  median p_rel = ({med[0]:+.4f}, {med[1]:+.4f}, {med[2]:+.4f}) m")
    print(f"  p25          = ({p25[0]:+.4f}, {p25[1]:+.4f}, {p25[2]:+.4f})")
    print(f"  p75          = ({p75[0]:+.4f}, {p75[1]:+.4f}, {p75[2]:+.4f})")
    print(f"  spread       = ({spread[0]:.4f}, {spread[1]:.4f}, {spread[2]:.4f})")
    pick = np.ones(len(p), dtype=bool)
    if (spread > 0.03).any():
        # bimodal suspicion: 2-means on p_rel, report clusters, take dominant
        rng = np.random.default_rng(0)
        c = p[rng.choice(len(p), 2, replace=False)]
        for _ in range(50):
            a = ((p[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
            cn = np.array([p[a == k].mean(0) if (a == k).any() else c[k]
                           for k in range(2)])
            if np.allclose(cn, c, atol=1e-6):
                break
            c = cn
        sizes = [(a == k).sum() for k in range(2)]
        print(f"  SPREAD>3cm — clusters: "
              f"A n={sizes[0]} at ({c[0][0]:+.4f},{c[0][1]:+.4f},{c[0][2]:+.4f}) | "
              f"B n={sizes[1]} at ({c[1][0]:+.4f},{c[1][1]:+.4f},{c[1][2]:+.4f})")
        k = int(np.argmax(sizes))
        pick = a == k
        med = np.median(p[pick], axis=0)
        print(f"  dominant cluster {'AB'[k]} ({sizes[k]}/{len(p)}) "
              f"median = ({med[0]:+.4f},{med[1]:+.4f},{med[2]:+.4f})")
    # The object is a CYLINDER: its yaw about its own axis is arbitrary
    # (random spawn yaw + axial symmetry), so a mean quaternion over samples
    # is meaningless. The physical content of q_rel is the cylinder AXIS
    # direction in the palm frame — take its spherical median and build the
    # minimal rotation that maps the object z-axis onto it.
    ez = np.array([0.0, 0.0, 1.0])
    axes = np.stack([quat_rot(qi, ez) for qi in q[pick]])
    a_med = axes.mean(axis=0)
    a_med = a_med / np.linalg.norm(a_med)
    ang = np.degrees(np.arccos(np.clip(axes @ a_med, -1, 1)))
    cr = np.cross(ez, a_med)
    s = np.linalg.norm(cr)
    th = np.arctan2(s, float(ez @ a_med))
    qm = (np.array([1.0, 0.0, 0.0, 0.0]) if s < 1e-9 else
          np.concatenate([[np.cos(th / 2)], cr / s * np.sin(th / 2)]))
    print(f"  axis in palm frame = ({a_med[0]:+.4f},{a_med[1]:+.4f},"
          f"{a_med[2]:+.4f})  tilt spread p50/p90 = "
          f"{np.percentile(ang, 50):.1f}/{np.percentile(ang, 90):.1f} deg")
    print(f"  q_rel (min rot z->axis) = ({qm[0]:+.4f},{qm[1]:+.4f},"
          f"{qm[2]:+.4f},{qm[3]:+.4f})")
    return med, qm, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=os.path.join(REFS, "grasp_offset_samples.npz"))
    ap.add_argument("--bank", default=os.path.join(REFS, "ref_traj_bank_gc.npz"))
    ap.add_argument("--out", default=os.path.join(REFS, "ref_traj_bank_gc2.npz"))
    ap.add_argument("--min_z", type=float, default=0.30)
    ap.add_argument("--families", default="",
                    help="comma list: restrict to these sample families "
                         "(e.g. 'held,hold' = stable carries)")
    args = ap.parse_args()

    p_med, q_med, note = aggregate(args.samples, args.min_z, args.families)

    b = dict(np.load(args.bank, allow_pickle=False))
    names = [str(n) for n in b["joint_names"]]
    assert names == W.BODY_JOINTS, "bank joint order != wbik BODY_JOINTS"
    N, T, J = b["joint_pos"].shape
    obj = b["obj"].astype(np.float64).copy()
    dt = 1.0 / float(b["fps"])

    D, M = W.D, W.M
    deltas = []
    attach_fs = []
    n_floor = 0
    for r in range(N):
        aloft = np.where(b["phase"][r] > 2)[0]
        pos = np.zeros((len(aloft), 3))
        quat = np.zeros((len(aloft), 4))
        # the attached pose per frame, and the first frame it clears the
        # floor (a tilted cylinder's support height, not the upright one —
        # a 40 deg tilt rests ~1.1 cm higher than FLOOR_Z)
        att = False
        f_att = None
        rest = obj[r, aloft[0] - 1].copy()      # floor-rest state from P2
        for k, f in enumerate(aloft):
            D.qpos[:] = 0.0
            D.qpos[0:3] = b["root"][r, f, 0:3]
            D.qpos[3:7] = b["root"][r, f, 3:7]
            D.qpos[W.JADR] = b["joint_pos"][r, f]
            for a, v in W.HAND_Q.items():
                D.qpos[a] = v
            mujoco.mj_kinematics(M, D)
            pq = D.xquat[W.PALM].copy()
            pp = D.xpos[W.PALM].copy()
            po = pp + quat_rot(pq, p_med)
            qo = quat_mul(pq, q_med)
            qo = qo / np.linalg.norm(qo)
            deltas.append(np.linalg.norm(po - obj[r, f, 0:3]))
            az = abs(float(quat_rot(qo, np.array([0.0, 0.0, 1.0]))[2]))
            h_sup = 0.075 * az + 0.045 * np.sqrt(max(0.0, 1 - az * az)) + 1e-3
            if not att and po[2] >= h_sup:
                att = True
                f_att = f
            if att:
                pos[k] = po
                quat[k] = qo
            else:
                # object physically still at floor rest — the hand has not
                # lifted it high enough for the measured grip to clear the
                # floor; restoring the attached pose here would wedge a
                # tilted cylinder into the ground (depenetration kick)
                pos[k] = rest[0:3]
                quat[k] = rest[3:7]
                n_floor += 1
        assert f_att is not None, f"ref {r}: palm never clears support height"
        attach_fs.append(f_att)
        obj[r, aloft, 0:3] = pos
        obj[r, aloft, 3:7] = quat
        # velocities: zero on the floor-rest frames, finite difference
        # inside the attached segment
        ka = int(np.where(aloft == f_att)[0][0])
        obj[r, aloft, 7:13] = 0.0
        seg = pos[ka:]
        v = np.zeros((len(seg), 3))
        if len(seg) > 2:
            v[1:-1] = (seg[2:] - seg[:-2]) / (2 * dt)
            v[0] = (seg[1] - seg[0]) / dt
            v[-1] = (seg[-1] - seg[-2]) / dt
        obj[r, aloft[ka:], 7:10] = v
        qseg = quat[ka:]
        wv = np.zeros((len(qseg), 3))
        for k in range(1, len(qseg) - 1):
            dq = quat_mul(qseg[k + 1], qseg[k - 1] * np.array([1, -1, -1, -1]))
            if dq[0] < 0:
                dq = -dq
            s = np.linalg.norm(dq[1:])
            if s > 1e-12:
                wv[k] = dq[1:] / s * (2 * np.arctan2(s, dq[0])) / (2 * dt)
        if len(qseg) > 2:
            wv[0], wv[-1] = wv[1], wv[-2]
        obj[r, aloft[ka:], 10:13] = wv

    deltas = np.array(deltas)
    out = dict(b)
    out["obj"] = obj.astype(np.float32)
    # BANK STIFFNESS LOCK (2026-07-29): stamp the canonical `leg_stiff`
    # scalar so the env-side guard (pod/g1_grasp/bank_guard.py) can
    # hard-assert the replay stiffness at load. Propagate from the input
    # bank (legacy spelling: leg_stiff_mult, written by make_ref_traj.py);
    # if the input carries neither, stamp the documented x4 assumption and
    # say so loudly — an unstamped output would defeat the lock.
    if "leg_stiff" not in out:
        if "leg_stiff_mult" in out:
            out["leg_stiff"] = np.float64(float(out["leg_stiff_mult"]))
        else:
            print("WARNING: input bank carries no leg_stiff(_mult) — "
                  "stamping the documented default 4.0 (make_ref_traj.py "
                  "KP_EFF table)")
            out["leg_stiff"] = np.float64(4.0)
    print(f"leg_stiff stamped: {float(out['leg_stiff']):g}")
    np.savez_compressed(args.out, **out)
    na = int((b["phase"][0] > 2).sum())
    print(f"placement shift vs gc bank (aloft frames): "
          f"p50={np.percentile(deltas, 50)*100:.1f} cm "
          f"p90={np.percentile(deltas, 90)*100:.1f} cm "
          f"max={deltas.max()*100:.1f} cm")
    print(f"attach frame f* p25/50/75: {np.percentile(attach_fs, 25):.0f}/"
          f"{np.percentile(attach_fs, 50):.0f}/{np.percentile(attach_fs, 75):.0f}"
          f"  floor-rest aloft frames: {n_floor}/{N * na}")
    print(f"objP4 z range: {obj[:, -1, 2].min():.3f}-{obj[:, -1, 2].max():.3f}")
    print(f"measured offset used: p_rel=({p_med[0]:+.4f},{p_med[1]:+.4f},"
          f"{p_med[2]:+.4f}) q_rel=({q_med[0]:+.4f},{q_med[1]:+.4f},"
          f"{q_med[2]:+.4f},{q_med[3]:+.4f})  [{note}]")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
