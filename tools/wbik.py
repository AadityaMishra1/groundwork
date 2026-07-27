"""Whole-body inverse kinematics / reachability for the G1, on the Mac.

Runs on the local MuJoCo Menagerie model — no pod, no GPU, no Isaac. Answers
kinematic questions by CONSTRAINED OPTIMIZATION rather than by sampling.

This exists because `pod/g1_grasp/probe_kneel.py` established the reach limit
that `docs/STATUS_2026-07-25.md` calls "the one real remaining technical
problem" using 200 uniform random samples in a 20-D joint space, measuring the
finger CENTROID, with the pelvis bolted and the legs frozen in a symmetric
seiza fold. Uniform sampling in 20-D cannot locate a workspace boundary. See
`docs/REDIRECT_2026-07-25.md` for what the optimizer found instead.

Optimizes the full 35-DoF whole-body pose — free base (6) + 12 leg + 3 waist +
14 arm — subject to:
  * supporting foot contact spheres planted on the ground plane
  * whole-body CoM inside the support polygon, shrunk 30% for margin
  * arm/hand links clear of both legs, the pelvis and the torso
    (analytic capsule keep-out: the "do the thighs block the arm" test)
  * joint limits, by projection each iteration

Every query is a FEASIBILITY problem ("can the palm be at THIS point under all
of the above?"), so the residual goes to zero at a solution and the constraint
weights are meaningful. Final poses are re-validated with full MuJoCo mesh
collision + contact detection (`report`/`valid`).

Use as a library (`solve`, `report`, `valid`, `STANCES`) or run directly for
the reach-floor and pregrasp-feasibility sweeps.

NOT ANSWERED HERE: static torque feasibility. An `mj_inverse` probe was tried
and is INVALID — `qfrc_inverse` reads 0 at the standing keyframe, and on poses
riding their joint limits it returns joint-limit constraint forces rather than
gravity load, which reads as a dramatic (and false) "over torque limit". A real
answer needs a contact-force QP with friction cones, or a settle test with
servo gains stiff enough not to sag. Do not reintroduce the mj_inverse version.
"""
import os
import numpy as np
import mujoco

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "assets/menagerie/unitree_g1")
M = mujoco.MjModel.from_xml_path(os.path.join(BASE, "scene_with_hands.xml"))
D = mujoco.MjData(M)
bid = lambda n: mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, n)
PALM, TIP = bid("right_wrist_yaw_link"), bid("right_hand_middle_1_link")
FLOOR_G = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_GEOM, "floor")

BODY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
JADR, JLO, JHI = [], [], []
for n in BODY_JOINTS:
    j = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_JOINT, n)
    JADR.append(M.jnt_qposadr[j]); JLO.append(M.jnt_range[j][0]); JHI.append(M.jnt_range[j][1])
JADR, JLO, JHI = np.array(JADR), np.array(JLO), np.array(JHI)
NJ = len(JADR)
IJ = {n: 6 + i for i, n in enumerate(BODY_JOINTS)}
WP = IJ["waist_pitch_joint"]

HAND_Q = {}
for n in ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
          "right_hand_middle_0_joint", "right_hand_middle_1_joint",
          "right_hand_index_0_joint", "right_hand_index_1_joint"]:
    j = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_JOINT, n)
    HAND_Q[M.jnt_qposadr[j]] = M.jnt_range[j][0] + 0.35 * (M.jnt_range[j][1] - M.jnt_range[j][0])

FOOT_SPH = {}
for s in ("left", "right"):
    b = bid(f"{s}_ankle_roll_link")
    FOOT_SPH[s] = sorted([g for g in range(M.ngeom)
                          if M.geom_bodyid[g] == b and M.geom_type[g] == 2],
                         key=lambda g: M.geom_pos[g][0])
SOLE_Z = 0.0031          # measured: sphere-centre height of a planted foot
LO = np.concatenate([[-1.5, -1.5, 0.05], [-1.0, -1.5, -1.0], JLO])
HI = np.concatenate([[1.5, 1.5, 1.10], [1.0, 1.5, 1.0], JHI])

LIMB_SEGS = [("left_hip_pitch_link", "left_knee_link", 0.085),
             ("left_knee_link", "left_ankle_pitch_link", 0.075),
             ("right_hip_pitch_link", "right_knee_link", 0.085),
             ("right_knee_link", "right_ankle_pitch_link", 0.075)]
SEG_IDS = [(bid(a), bid(b), c) for a, b, c in LIMB_SEGS]
ARM_IDS = [bid(a) for a in ["right_elbow_link", "right_wrist_roll_link",
                            "right_wrist_yaw_link", "right_hand_middle_1_link"]]
PELVIS, TORSO = bid("pelvis"), bid("torso_link")
KNEE_L, KNEE_R = bid("left_knee_link"), bid("right_knee_link")
# contacts present in the asset at rest (thumb vs its own wrist shell) — not
# a pose defect, must not disqualify a solution
BASELINE_SELF = 0.0010


def rpy2quat(r, p, y):
    cr, sr, cp, sp, cy, sy = (np.cos(r/2), np.sin(r/2), np.cos(p/2),
                              np.sin(p/2), np.cos(y/2), np.sin(y/2))
    return np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                     cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])


def fk(x, full=False):
    D.qpos[:] = 0.0
    D.qpos[0:3] = x[0:3]
    D.qpos[3:7] = rpy2quat(*x[3:6])
    D.qpos[JADR] = x[6:6+NJ]
    for a, v in HAND_Q.items():
        D.qpos[a] = v
    if full:
        mujoco.mj_forward(M, D)
    else:
        mujoco.mj_kinematics(M, D); mujoco.mj_comPos(M, D)


def hull(pts):
    pts = sorted(map(tuple, pts))
    if len(pts) < 3:
        return np.array(pts)
    def half(ps):
        h = []
        for p in ps:
            while len(h) >= 2 and ((h[-1][0]-h[-2][0])*(p[1]-h[-2][1])
                                   - (h[-1][1]-h[-2][1])*(p[0]-h[-2][0])) <= 0:
                h.pop()
            h.append(p)
        return h
    return np.array(half(pts)[:-1] + half(pts[::-1])[:-1])


def dist_outside(pt, poly):
    if len(poly) < 3:
        return float(np.linalg.norm(pt - poly.mean(axis=0)))
    inside, best, n = True, 1e9, len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i+1) % n]
        e = b - a
        if e[0]*(pt[1]-a[1]) - e[1]*(pt[0]-a[0]) < 0:
            inside = False
        t = np.clip(np.dot(pt-a, e)/max(np.dot(e, e), 1e-12), 0, 1)
        best = min(best, float(np.linalg.norm(pt - (a + t*e))))
    return 0.0 if inside else best


def seg_dist(p, a, b):
    e = b - a
    t = np.clip(np.dot(p-a, e)/max(np.dot(e, e), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t*e)))


STANCES = {
    "A_squat_upright":   {"sup": [("left", "all"), ("right", "all")], "pin": True},
    "B_squat_freetorso": {"sup": [("left", "all"), ("right", "all")], "pin": False},
    "C_lunge_R_toe":     {"sup": [("left", "all"), ("right", "toe")], "pin": False},
    "D_lunge_L_toe":     {"sup": [("left", "toe"), ("right", "all")], "pin": False},
}
W_T, W_F, W_C, W_K, W_R = 3.0, 10.0, 10.0, 10.0, 0.005


def support(stance):
    pts, allowed = [], set()
    for side, which in STANCES[stance]["sup"]:
        sph = FOOT_SPH[side]; allowed.update(sph)
        for g in (sph if which == "all" else sph[2:]):
            pts.append(D.geom_xpos[g][:2].copy())
    return pts, allowed


def residual(x, stance, target):
    fk(x)
    st = STANCES[stance]
    res = []
    for side, which in st["sup"]:
        sph = FOOT_SPH[side]
        for g in (sph if which == "all" else sph[2:]):
            res.append(W_F * (D.geom_xpos[g][2] - SOLE_Z))
        if which == "toe":
            for g in sph[:2]:
                res.append(W_F * min(0.0, D.geom_xpos[g][2] - SOLE_Z))
    pts, _ = support(stance)
    poly = hull(pts); c = poly.mean(axis=0)
    res.append(W_C * dist_outside(D.subtree_com[1][:2], c + (poly - c) * 0.70))
    ko = 0.0
    for ai in ARM_IDS:
        p = D.xpos[ai]
        for a, b, clr in SEG_IDS:
            ko += max(0.0, clr - seg_dist(p, D.xpos[a], D.xpos[b]))
        ko += max(0.0, 0.13 - float(np.linalg.norm(p - D.xpos[PELVIS])))
        ko += max(0.0, 0.16 - float(np.linalg.norm(p[:2] - D.xpos[TORSO][:2])))
    res.append(W_K * ko)
    fl = 0.0
    for ai in (ARM_IDS[0], ARM_IDS[1], PELVIS, TORSO, KNEE_L, KNEE_R):
        fl += max(0.0, 0.045 - D.xpos[ai][2])
    res.append(W_K * fl)
    res += list(W_T * (D.xpos[PALM] - target))
    res.append(W_R * float(np.linalg.norm(x[6:6+NJ])))
    if st["pin"]:
        res.append(4.0 * x[WP]); res.append(4.0 * x[4])
    return np.array(res)


def solve(stance, target, x0, iters=70):
    x = x0.copy(); r = residual(x, stance, target); lam = 1e-2
    n = len(x)
    for _ in range(iters):
        J = np.zeros((len(r), n))
        for k in range(n):
            xp = x.copy(); xp[k] = np.clip(xp[k] + 1e-5, LO[k], HI[k])
            dk = xp[k] - x[k]
            if abs(dk) < 1e-12:
                xp[k] = np.clip(x[k] - 1e-5, LO[k], HI[k]); dk = xp[k] - x[k]
                if abs(dk) < 1e-12:
                    continue
            J[:, k] = (residual(xp, stance, target) - r) / dk
        for _ls in range(10):
            try:
                dx = -np.linalg.solve(J.T @ J + lam*np.eye(n), J.T @ r)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            xn = np.clip(x + dx, LO, HI)
            rn = residual(xn, stance, target)
            if rn @ rn < r @ r:
                x, r, lam = xn, rn, max(lam*0.5, 1e-9); break
            lam *= 4
        else:
            break
        if r @ r < 1e-8:
            break
    return x, r


def report(x, stance):
    fk(x, full=True)
    pts, allowed = support(stance)
    poly = hull(pts); c = poly.mean(axis=0)
    pen_self = pen_floor = 0.0
    for i in range(D.ncon):
        ct = D.contact[i]
        if ct.dist >= 0:
            continue
        g1, g2 = ct.geom1, ct.geom2
        if g1 == FLOOR_G or g2 == FLOOR_G:
            g = g2 if g1 == FLOOR_G else g1
            if g not in allowed:
                pen_floor = max(pen_floor, -ct.dist)
        else:
            pen_self = max(pen_self, -ct.dist)
    st = STANCES[stance]
    ferr = max(abs(D.geom_xpos[g][2] - SOLE_Z) for s, w in st["sup"]
               for g in (FOOT_SPH[s] if w == "all" else FOOT_SPH[s][2:]))
    return dict(palm=D.xpos[PALM].copy(), tip=D.xpos[TIP].copy(), pelvis_z=x[2],
                torso_pitch=x[4]+x[WP], base_pitch=x[4], waist_pitch=x[WP],
                com_out=dist_outside(D.subtree_com[1][:2], c + (poly-c)*0.70),
                pen_self=max(0.0, pen_self - BASELINE_SELF), pen_floor=pen_floor,
                ferr=ferr, kneeL=x[IJ["left_knee_joint"]], kneeR=x[IJ["right_knee_joint"]],
                hipR=x[IJ["right_hip_pitch_joint"]], base=x[0:3].copy())


REJ = {}
def valid(r):
    bad = []
    if r["com_out"] >= 2e-3: bad.append("com")
    if r["pen_self"] >= 0.008: bad.append("self")
    if r["pen_floor"] >= 0.008: bad.append("floor")
    if r["ferr"] >= 0.010: bad.append("foot")
    if bad:
        REJ["+".join(bad)] = REJ.get("+".join(bad), 0) + 1
    return not bad


def seeds(rng, n):
    out = []
    for _ in range(n):
        x = np.zeros(6 + NJ)
        x[2] = rng.uniform(0.30, 0.72); x[4] = rng.uniform(-0.05, 0.7)
        t = rng.uniform(0.15, 1.0)
        for s in ("left", "right"):
            tt = t if s == "left" else t * rng.uniform(0.4, 1.0)
            x[IJ[f"{s}_hip_pitch_joint"]] = -0.2 - tt*1.8
            x[IJ[f"{s}_knee_joint"]] = 0.42 + tt*2.1
            x[IJ[f"{s}_ankle_pitch_joint"]] = -0.23 - tt*0.5
        x[WP] = rng.uniform(0.0, 0.4)
        x[IJ["right_shoulder_pitch_joint"]] = rng.uniform(-0.4, 0.8)
        x[IJ["right_shoulder_roll_joint"]] = rng.uniform(-0.7, 0.1)
        x[IJ["right_elbow_joint"]] = rng.uniform(0.1, 1.4)
        x[IJ["left_elbow_joint"]] = 0.87
        out.append(np.clip(x, LO, HI))
    return out




# ---------------------------------------------------------------------------
# pregrasp target model
# ---------------------------------------------------------------------------
# The palm frame (right_wrist_yaw_link) cannot descend below ~0.16 m: that is
# simply the hand's length below the wrist, and it is NOT a function of pelvis
# height. So a pregrasp target must NOT be the cylinder's axis point at
# mid-barrel (that asks the palm to be at h/2 <= 0.125, which is unreachable
# and was the bug in this file's first sweep). The hand grasps a floor cylinder
# from the SIDE: palm beside the barrel, fingers wrapping down around it.
PALM_FLOOR = 0.16


def pregrasp_target(r, th, h, offset=0.06):
    """Palm pose for grasping a standing cylinder of height h at (r, th)."""
    ax = np.array([r * np.cos(th), r * np.sin(th)])
    n = ax / max(np.linalg.norm(ax), 1e-9)
    return np.array([ax[0] + offset * n[0], ax[1] + offset * n[1],
                     max(PALM_FLOOR, 0.55 * h)])


def seeds(rng, n):
    out = []
    for _ in range(n):
        x = np.zeros(6 + NJ)
        x[2] = rng.uniform(0.30, 0.72)
        x[4] = rng.uniform(-0.05, 0.7)
        t = rng.uniform(0.15, 1.0)
        for s in ("left", "right"):
            tt = t if s == "left" else t * rng.uniform(0.4, 1.0)
            x[IJ[f"{s}_hip_pitch_joint"]] = -0.2 - tt * 1.8
            x[IJ[f"{s}_knee_joint"]] = 0.42 + tt * 2.1
            x[IJ[f"{s}_ankle_pitch_joint"]] = -0.23 - tt * 0.5
        x[WP] = rng.uniform(0.0, 0.4)
        x[IJ["right_shoulder_pitch_joint"]] = rng.uniform(-0.4, 0.8)
        x[IJ["right_shoulder_roll_joint"]] = rng.uniform(-0.7, 0.1)
        x[IJ["right_elbow_joint"]] = rng.uniform(0.1, 1.4)
        x[IJ["left_elbow_joint"]] = 0.87
        out.append(np.clip(x, LO, HI))
    return out


def best_pose(stance, tgt, S, tol=0.04, iters=80):
    """Lowest-error valid pose reaching tgt, over a seed set."""
    best = None
    for x0 in S:
        x, _ = solve(stance, tgt, x0, iters=iters)
        rp = report(x, stance)
        if not valid(rp):
            continue
        e = float(np.linalg.norm(rp["palm"] - tgt))
        if e < tol and (best is None or e < best[0]):
            best = (e, x, rp)
    return best


def main():
    rng = np.random.default_rng(0)
    S = seeds(rng, 6)

    print("=" * 84)
    print("REACH FLOOR — lowest palm / fingertip height, by stance family")
    print("=" * 84)
    ZS = [0.30, 0.26, 0.22, 0.18, 0.16, 0.15]
    GRID = [(r, th) for th in (-0.9, -0.5, -0.2) for r in (0.25, 0.35, 0.45)]
    for st in STANCES:
        rec = None
        for (r0, th0) in GRID:
            for x0 in S:
                warm = x0
                for z in ZS:
                    tgt = np.array([r0 * np.cos(th0), r0 * np.sin(th0), z])
                    x, _ = solve(st, tgt, warm)
                    rp = report(x, st)
                    if float(np.linalg.norm(rp["palm"] - tgt)) < 0.045 and valid(rp):
                        warm = x
                        if rec is None or rp["palm"][2] < rec[0]:
                            rec = (rp["palm"][2], rp, r0, th0)
                    else:
                        break
        if rec is None:
            print(f"  {st:20s} NO VALID POSE")
            continue
        z, rp, r0, th0 = rec
        print(f"  {st:20s} palm_z={z:.3f} tip_z={rp['tip'][2]:.3f} "
              f"pelvis={rp['pelvis_z']:.3f} torso={np.degrees(rp['torso_pitch']):+6.1f}deg "
              f"kneeL/R={rp['kneeL']:.2f}/{rp['kneeR']:.2f} "
              f"@r={r0:.2f},th={np.degrees(th0):+.0f}")

    print("\n" + "=" * 84)
    print("PREGRASP FEASIBILITY — palm beside the barrel, fingers wrapping down")
    print("=" * 84)
    for h in (0.10, 0.15, 0.25):
        print(f"\n  cylinder height {h * 100:.0f} cm")
        for st in STANCES:
            hits = []
            for th in (-0.7, -0.45, -0.2):
                for r in (0.30, 0.40, 0.50):
                    b = best_pose(st, pregrasp_target(r, th, h), S)
                    if b:
                        hits.append((b[0], r, th, b[2]))
            if not hits:
                print(f"    {st:20s} no valid pose")
                continue
            hits.sort(key=lambda t: t[0])
            e, r, th, rp = hits[0]
            pz = [g[3]["pelvis_z"] for g in hits]
            print(f"    {st:20s} reachable {len(hits)}/9 poses | best err={e:.3f}m "
                  f"@r={r:.2f},th={np.degrees(th):+.0f} | pelvis {min(pz):.2f}-{max(pz):.2f} "
                  f"| torso={np.degrees(rp['torso_pitch']):+.0f}deg tip_z={rp['tip'][2]:.3f}")


if __name__ == "__main__":
    main()
