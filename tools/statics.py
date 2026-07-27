"""Static feasibility of a whole-body pose: is there ANY valid torque assignment
that holds it, given the contacts?

Replaces the invalid `mj_inverse` probe (see tools/wbik.py header). That one read
joint-limit constraint forces and called them gravity load, and reported a
dramatic, entirely false "0/8 poses over torque limit".

This one solves the equilibrium problem directly and is gain-independent.

At a static pose (qvel = qacc = 0) MuJoCo's convention gives

    qfrc_bias = tau_applied + sum_i J_i^T lambda_i

with qfrc_bias = gravity. No actuator can act on the 6 free-base DoF, so the
base rows constrain the contact forces alone:

    (sum_i J_i^T lambda_i)[0:6] = qfrc_bias[0:6]              (6 equations)

That is underdetermined in lambda (3 per contact point), so we pick the lambda
that minimises actuator effort relative to each joint's capacity,

    minimise || W (qfrc_bias[6:] - (sum J^T lambda)[6:]) ||,  W = diag(1/tau_max)

by equality-constrained least squares (particular solution + nullspace), then
check the answer against the two things that make it physical:

  * friction cones:  lambda . n >= 0  and  |lambda_t| <= mu (lambda . n)
  * joint torque:    |tau_j| <= G1 datasheet limit

Only the SUPPORTING feet are allowed to carry load. Any other body touching the
floor is reported and excluded from the support set, which makes the test
strictly conservative.

Two controls run first, because a probe that reports a dramatic result on the
real question must first reproduce the known answer on a trivial one:
  * the standing keyframe   -> must be feasible, with small ankle/knee torques
  * the deep folded crouch  -> the pose the project measured as PD-holdable
"""
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wbik as W  # noqa: E402  (model, IK solver, stance definitions)

M, D = W.M, W.D
NV = M.nv

# Unitree G1 datasheet joint torque limits, N.m
SPEC = {"hip_pitch": 88, "hip_roll": 88, "hip_yaw": 88, "knee": 139,
        "ankle_pitch": 50, "ankle_roll": 50, "waist_yaw": 88, "waist_roll": 50,
        "waist_pitch": 50, "shoulder_pitch": 25, "shoulder_roll": 25,
        "shoulder_yaw": 25, "elbow": 25, "wrist_roll": 25, "wrist_pitch": 5,
        "wrist_yaw": 5}
# dof index -> (name, limit). Hands get a nominal 2 N.m: they are not
# load-bearing in a pregrasp and are excluded from the verdict.
DOF_NAME = {}
DOF_LIM = np.full(NV, np.inf)
CHECK = []
for j in range(M.njnt):
    if M.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
        continue
    n = mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_JOINT, j)
    d = M.jnt_dofadr[j]
    DOF_NAME[d] = n
    if "hand" in n:
        DOF_LIM[d] = 2.0
    else:
        lim = next((v for k, v in SPEC.items() if k in n), None)
        if lim:
            DOF_LIM[d] = lim
            CHECK.append(d)
MU = 0.6  # foot geom friction in the MJCF (class "foot")


def equilibrium(allowed_geoms, verbose_name=""):
    """Solve for contact forces + joint torques holding the CURRENT mj_forward
    state. Returns (feasible, info)."""
    # gather contacts against the floor from allowed (supporting) geoms
    pts, normals, bodies, rejected = [], [], [], []
    for i in range(D.ncon):
        c = D.contact[i]
        g1, g2 = c.geom1, c.geom2
        if W.FLOOR_G not in (g1, g2):
            continue
        g = g2 if g1 == W.FLOOR_G else g1
        if g not in allowed_geoms:
            rejected.append(mujoco.mj_id2name(
                M, mujoco.mjtObj.mjOBJ_BODY, M.geom_bodyid[g]))
            continue
        pts.append(c.pos.copy())
        # contact frame row 0 is the normal, pointing from geom1 to geom2
        n = c.frame[0:3].copy()
        if g1 == W.FLOOR_G:      # normal points floor -> body, i.e. up: good
            pass
        else:
            n = -n
        normals.append(n / max(np.linalg.norm(n), 1e-12))
        bodies.append(M.geom_bodyid[g])
    nc = len(pts)
    if nc == 0:
        return False, {"reason": "no supporting contact", "rejected": rejected}

    # contact Jacobians: generalised force of a world-frame force f at pts[i]
    # on bodies[i] is J_i^T f
    JT = np.zeros((NV, 3 * nc))
    jacp = np.zeros((3, NV))
    jacr = np.zeros((3, NV))
    for i in range(nc):
        mujoco.mj_jac(M, D, jacp, jacr, pts[i], bodies[i])
        JT[:, 3*i:3*i+3] = jacp.T

    g = D.qfrc_bias.copy()                 # qvel = 0 -> gravity only
    A = JT[0:6, :]                         # base rows: contacts alone
    b = g[0:6]
    Wt = 1.0 / np.clip(DOF_LIM[6:], 1e-9, 1e6)   # effort weighting
    B = JT[6:, :]

    # tau(lambda) = g[6:] - B lambda ; minimise ||Wt * tau|| s.t. A lambda = b
    # AND lambda inside its friction cone. The cone is not optional: an
    # unconstrained equality-least-squares happily returns NEGATIVE normal
    # forces (it "bolts" a foot to the floor to save joint torque), which reads
    # as a confident INFEASIBLE that is pure artifact. Projected gradient with
    # a per-contact cone projection each step.
    NRM = np.array(normals)                       # (nc, 3)

    def project(l):
        l = l.reshape(nc, 3).copy()
        fn = np.einsum("ij,ij->i", l, NRM)
        fn = np.maximum(fn, 0.0)
        ft = l - fn[:, None] * NRM
        mag = np.linalg.norm(ft, axis=1)
        lim = MU * fn
        scale = np.where(mag > lim, lim / np.maximum(mag, 1e-12), 1.0)
        return (fn[:, None] * NRM + ft * scale[:, None]).ravel()

    W_EQ = 1.0e4                                  # equality is the hard part
    BW = Wt[:, None] * B
    g6 = g[6:]

    def obj(l):
        r1 = A @ l - b
        r2 = Wt * (g6 - B @ l)
        return W_EQ * float(r1 @ r1) + float(r2 @ r2)

    def grad(l):
        return (2 * W_EQ * A.T @ (A @ l - b)
                - 2 * BW.T @ (Wt * (g6 - B @ l)))

    lam_p, *_ = np.linalg.lstsq(A, b, rcond=None)
    lam = project(lam_p)
    step = 1.0 / (W_EQ * float(np.linalg.norm(A, 2) ** 2) + 1.0)
    f0 = obj(lam)
    for _ in range(4000):
        cand = project(lam - step * grad(lam))
        fc = obj(cand)
        if fc < f0:
            lam, f0, step = cand, fc, step * 1.1
        else:
            step *= 0.5
            if step < 1e-18:
                break

    tau = g[6:] - B @ lam
    base_res = float(np.linalg.norm(A @ lam - b))

    # friction cones
    worst_cone, min_normal = 0.0, 1e9
    for i in range(nc):
        f = lam[3*i:3*i+3]
        fn = float(np.dot(f, normals[i]))
        ft = float(np.linalg.norm(f - fn * normals[i]))
        min_normal = min(min_normal, fn)
        if fn > 1e-6:
            worst_cone = max(worst_cone, ft / (MU * fn))
        elif ft > 1e-6:
            worst_cone = max(worst_cone, 99.0)

    over = [(DOF_NAME[6 + k], abs(tau[k]), DOF_LIM[6 + k])
            for k in range(NV - 6) if 6 + k in [d for d in CHECK]
            and abs(tau[k]) > DOF_LIM[6 + k]]
    fracs = sorted(((abs(tau[d - 6]) / DOF_LIM[d], DOF_NAME[d],
                     abs(tau[d - 6]), DOF_LIM[d]) for d in CHECK), reverse=True)
    # base_res is a wrench residual in N / N.m; judge it against body weight
    # (337 N), not against solver epsilon — projected gradient will not reach 1e-13
    feasible = (base_res < 2.0 and min_normal > -1e-6
                and worst_cone <= 1.01 and not over)
    return feasible, {"nc": nc, "base_res": base_res, "min_normal": min_normal,
                      "cone": worst_cone, "over": over, "worst": fracs[:5],
                      "rejected": sorted(set(rejected)),
                      "total_normal": sum(float(np.dot(lam[3*i:3*i+3], normals[i]))
                                          for i in range(nc))}


def show(name, ok, info):
    print(f"\n  {name}")
    if "reason" in info:
        print(f"    INFEASIBLE — {info['reason']}  rejected={info['rejected']}")
        return
    print(f"    contacts={info['nc']}  base residual={info['base_res']:.2e}  "
          f"normal load={info['total_normal']:6.1f} N "
          f"(weight {M.body_mass.sum()*9.81:.0f} N)")
    print(f"    friction cone usage={info['cone']*100:5.1f}% of mu={MU}   "
          f"min normal force={info['min_normal']:+7.1f} N")
    if info["rejected"]:
        print(f"    non-supporting bodies touching floor (excluded): "
              f"{info['rejected']}")
    print("    joint torque, worst 5 vs G1 spec:")
    for frac, n, t, l in info["worst"]:
        print(f"        {n:28s} {t:7.1f} / {l:3.0f} N.m  ({frac*100:5.1f}%)"
              f"{'   <-- OVER' if t > l else ''}")
    print(f"    -> {'FEASIBLE' if ok else 'INFEASIBLE'}")


def pose_from_x(x):
    W.fk(x, full=True)


def all_foot_geoms():
    s = set()
    for side in ("left", "right"):
        s.update(W.FOOT_SPH[side])
    return s


print("=" * 84)
print("STATIC FEASIBILITY — is there any torque assignment that holds the pose?")
print("=" * 84)
print("\n### CONTROLS (must reproduce known answers) ###")

# control 1: the standing keyframe
mujoco.mj_resetDataKeyframe(M, D, 0)
D.qvel[:] = 0
mujoco.mj_forward(M, D)
ok, info = equilibrium(all_foot_geoms())
show("CONTROL 1 — standing keyframe (expect: feasible, small torques)", ok, info)

# control 2: the deep folded crouch the project measured as PD-holdable
xc = np.zeros(6 + W.NJ)
xc[2] = 0.24
for pat, val in {"hip_pitch": -2.2, "knee": 2.8, "ankle_pitch": -0.87,
                 "hip_roll": 0.4}.items():
    for s in ("left", "right"):
        xc[W.IJ[f"{s}_{pat}_joint"]] = val
xc[W.IJ["left_elbow_joint"]] = 0.87
xc[W.IJ["right_elbow_joint"]] = 0.87
xc = np.clip(xc, W.LO, W.HI)
# A deep seiza rests on plantarflexed feet AND shins — allow the knee/shin links
# to bear load for this control, or it is being asked to balance on one point.
CROUCH_SUP = all_foot_geoms() | {
    g for g in range(M.ngeom)
    if M.geom_bodyid[g] in (W.KNEE_L, W.KNEE_R, W.bid("left_ankle_pitch_link"),
                            W.bid("right_ankle_pitch_link"))}
# drop until at least 3 supporting contact points exist (1 point cannot balance)
best_dz = None
for dz in np.arange(0.30, -0.10, -0.004):
    xc[2] = 0.20 + dz
    W.fk(xc, full=True)
    n = sum(1 for i in range(D.ncon)
            if W.FLOOR_G in (D.contact[i].geom1, D.contact[i].geom2)
            and (D.contact[i].geom2 if D.contact[i].geom1 == W.FLOOR_G
                 else D.contact[i].geom1) in CROUCH_SUP)
    if n >= 3:
        best_dz = dz
        break
if best_dz is not None:
    xc[2] = 0.20 + best_dz
pose_from_x(xc)
ok, info = equilibrium(CROUCH_SUP)
show(f"CONTROL 2 — deep folded crouch, pelvis={xc[2]:.3f} "
     f"(project measured this PD-holdable, up=0.98)", ok, info)

print("\n\n### THE POSES THE REDIRECT DEPENDS ON ###")
rng = np.random.default_rng(0)
S = W.seeds(rng, 6)
res = []
for h in (0.15,):
    for r in (0.30, 0.40, 0.50):
        for th in (-0.7, -0.45, -0.2):
            tgt = W.pregrasp_target(r, th, h)
            for stance in ("A_squat_upright", "B_squat_freetorso"):
                b = W.best_pose(stance, tgt, S)
                if b is None:
                    continue
                e, x, rp = b
                pose_from_x(x)
                ok, info = equilibrium(all_foot_geoms())
                res.append((ok, stance, r, th, rp, info))
                break

for ok, stance, r, th, rp, info in res:
    show(f"pregrasp r={r:.2f} th={np.degrees(th):+.0f}deg [{stance}]  "
         f"pelvis={rp['pelvis_z']:.3f} torso={np.degrees(rp['torso_pitch']):+.0f}deg "
         f"palm_z={rp['palm'][2]:.3f} tip_z={rp['tip'][2]:.3f}", ok, info)

n_ok = sum(1 for r_ in res if r_[0])
print(f"\n{'='*84}")
print(f"{n_ok}/{len(res)} pregrasp poses statically feasible within G1 spec torque")
print(f"{'='*84}")
