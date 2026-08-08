"""Self-authored whole-body reference trajectories for the G1 (Mac, CPU, $0).

For every certified pregrasp in refs/pregrasp_bank.npz (42 squats, wbik +
cone-QP certified), authors a 5 s / 50 Hz whole-body reference:

    P0 stand 0.5 s   certified default stance (keyframe "stand"), feet anchored
                     at THIS ref's certified squat foot positions
    P1 descend 1.5 s pelvis height + palm target scheduled stand -> squat,
                     every knot a full constrained IK re-solve (feet planted,
                     CoM in shrunk support polygon, capsule keep-out, joint
                     limits) warm-started from the previous knot
    P2 close 0.5 s   body holds the certified squat; fingers are PHASE-FLAG
                     only (Isaac-side Inspire targets — the local 3-finger
                     hand is not authored, per make_start_bank.py conventions)
    P3 lift 1.5 s    palm rises floor -> >=0.45 while pelvis rises to ~0.60,
                     per-knot IK, AND the statics cone-QP re-certifies every
                     knot WITH the 0.5 kg object weight applied at the object
                     CG on the palm body
    P4 hold 1.0 s    stand-hold of the lift endpoint, palm and object >=0.45

Zero human data: every frame is produced by the project's own certified
kinematics (tools/wbik.py) and statics (tools/statics.py) machinery.

Reuse notes / deviations from the parent tools, all deliberate:
  * statics.py has no __main__ guard — importing it re-runs its full pregrasp
    sweep (minutes) in EVERY worker. Its definitions are exec'd verbatim from
    source, truncated at the module-level run. The known-answer controls it
    would have run are reproduced ONCE in the parent (standing keyframe must
    be feasible), plus a new control for the object-weight modification: the
    same pose with 0.5 kg at the palm must stay feasible and its total normal
    load must rise by ~= m*g.
  * The wbik residual constrains foot HEIGHT only; a trajectory needs feet
    that do not slide, so this file's residual adds a foot-XY anchor (all 8
    sole spheres pinned to the certified squat's foot placement) plus a pelvis
    height target and a posture-blend regulariser. Everything else — feet-z,
    CoM-in-polygon, capsule keep-out, floor clearance, palm term, pins, the
    LM solve loop, report()/valid() — is wbik's machinery unchanged.
  * The bank squats carry ~0.28 rad base roll (stance A pins pitch only), so
    P1 keeps wbik's exact pin set; P3/P4 add a mild (w=1.0) roll pin so the
    hold straightens toward upright without fighting the certified endpoint.
  * The lift palm path is measured from the FEET CENTER (the bank's world
    frame has each ref's feet at a different place), pulled in to <=0.32 m
    horizontal radius, ending at z >= max(0.55, palm_grasp_z + 0.38) so the
    OBJECT (palm + rigid offset) also finishes >= 0.45.
  * Knot grid: 51 knots at 10 Hz over 5 s; exactly 30 are IK re-solves
    (stand, 14 P1 intermediates, 14 P3 intermediates, lift-hold endpoint);
    P1/P3 endpoints are the certified squat / solved hold verbatim; holds are
    knot copies. 50 Hz frames are cosine-eased between knots; schedules are
    also cosine-eased at phase level so phase boundaries have ~zero velocity.
  * The object is assumed rigidly palm-attached from the P2/P3 boundary on
    (constant world offset = object rest CG - grasp palm); before that it
    rests on the floor at the ref's certified (r, th). Its weight is applied
    at its CG on the palm body for the P3 statics certificates (16 knots:
    the loaded squat at t=2.5 plus every P3 knot through the hold).

Output refs/ref_traj_bank.npz (only refs whose every certificate passed):
    joint_pos[N,250,29]   wbik BODY_JOINTS order (Isaac remaps by name)
    joint_targets         copy of joint_pos
    root[N,250,13]        pos, quat wxyz, linvel, angvel (finite difference)
    obj[N,250,13]         object pos/quat/vels as above
    phase[N,250]          ints 0-4
    joint_names[29]       MuJoCo body-joint names
    bank_index/r/th/fps/obj_mass/obj_h   provenance

Also writes refs/ref_traj_bank_gc.npz — identical except joint_targets are
GRAVITY-COMPENSATED: joint_targets = joint_pos + tau_required/kp_eff, where
tau_required is the statics cone-QP certificate torque solved at EVERY knot
(unloaded through P2, object-loaded from the P2/P3 boundary) and kp_eff is
the Isaac-side actuator stiffness (see KP_EFF below). Open-loop PD playback
of joint_targets == joint_pos dies position-locked at max knee flexion: zero
PD error means zero torque exactly where holding needs the most, the plant
sags below the reference and topples. The gc bank carries `tau` and `kp_eff`
keys so the env can re-derive targets under a different GRASP_LEG_STIFF.

    .venv/bin/python -u tools/make_ref_traj.py
"""
import json
import os
import sys
import time
import types
import traceback
import multiprocessing as mp

import numpy as np
import mujoco

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
REFS = os.path.join(ROOT, "refs")
sys.path.insert(0, TOOLS)
import wbik as W  # noqa: E402


def _load_statics_defs():
    """statics.py definitions WITHOUT its module-level probe run (no guard
    there; importing it would re-run a multi-minute sweep per worker)."""
    path = os.path.join(TOOLS, "statics.py")
    with open(path) as f:
        src = f.read()
    cut = src.index('\nprint("=" * 84)')      # first module-level statement
    src = src[:cut]
    # equilibrium() computes the per-dof torque vector `tau` but does not
    # return it; the gravity-compensated targets need it, so it is added to
    # the info dict of the exec'd copy (statics.py itself untouched).
    anchor = '"cone": worst_cone, "over": over, "worst": fracs[:5],'
    assert src.count(anchor) == 1, "statics.py return-dict anchor changed"
    src = src.replace(anchor, anchor + ' "tau": tau.copy(),')
    mod = types.ModuleType("statics_defs")
    mod.__file__ = path
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


ST = _load_statics_defs()

# ---------------------------------------------------------------- constants
FPS, KDT = 50, 0.1
T, K = 250, 51                        # 250 frames @50 Hz, 51 knots @10 Hz
DT = 1.0 / FPS
OBJ_M, GRAV = 0.5, 9.81
PELV_HOLD = 0.60
HOLD_R = 0.32                          # max horizontal palm radius from feet
W_PALM, W_PZ, W_AXY, W_POST = 3.0, 5.0, 10.0, 0.02
PIN_DESCEND, PIN_LIFT = 4.0, 1.0
TOL_MID, TOL_END = 0.10, 0.05          # palm error gates (schedule / endpoint)
# Inter-knot joint step gate: catches solver configuration flips (observed
# 0.50+ rad in shoulder/elbow before the keep-out fixes). Set ABOVE the
# largest legitimate step — early-descent knee flexion, where dtheta/dz
# diverges as the leg straightens, measures up to 0.36 rad/knot (~5.7 rad/s
# peak eased; fine for the G1 knee).
JUMP_MAX = 0.45
MIN_OK = 30
# knot indices: P0 0..5 | P1 solves 6..19, 20=squat | P2 21..25 | P3 solves
# 26..39, 40=hold | P4 41..50
PHASE_F = np.zeros(T, np.int32)
PHASE_F[25:100], PHASE_F[100:125], PHASE_F[125:200], PHASE_F[200:] = 1, 2, 3, 4
P3F = 125
# wbik's capsule keep-out covers the RIGHT arm only; the raw keyframe hangs
# the LEFT hand into the left thigh (mesh-collision valid() catches it). The
# left arm has no task in any phase, so it holds the certified squat's left
# arm configuration for the entire trajectory.
LEFT_ARM_IX = [W.IJ[n] for n in (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint")]
# wbik's ARM_IDS misses the links whose meshes actually hit the legs during a
# descent (wrist pitch, thumb, index) — extend the same capsule keep-out. Two
# further gaps wbik's static queries never exercised but a descent/lift does:
# the hip motor housings (hip_roll/hip_yaw links) sweep into the hanging RIGHT
# thumb as the hip folds, and into the parked LEFT thumb as the leg unfolds.
# Both hands get point keep-outs against both hip clusters.
KEEPOUT_IDS = W.ARM_IDS + [W.bid(n) for n in (
    "right_wrist_pitch_link", "right_hand_thumb_2_link",
    "right_hand_index_1_link")]
KEEPOUT_L_IDS = [W.bid(n) for n in (
    "left_wrist_roll_link", "left_wrist_yaw_link",
    "left_hand_thumb_2_link", "left_hand_middle_1_link")]
HIP_PTS = [W.bid(f"{s}_hip_{a}_link") for s in ("left", "right")
           for a in ("roll", "yaw")]
HIP_CLR = 0.10    # m to the housing link origin (~mesh radius + margin)
BOW = 0.10        # m, outward bow of the P1 palm path (hand clears the thigh)

# ---- Isaac-side effective PD stiffness, for gravity-compensated targets ----
# joint_targets == joint_pos gives zero PD error exactly where deep flexion
# needs maximum static torque; the plant sags below the reference and topples
# ("the measured pose is the equilibrium under the targets that hold it, not
# under itself"). Fix: joint_targets = joint_pos + tau_required / kp_eff.
#
# kp source of truth (read 2026-07-28 from the pod, the machine that runs the
# playback): /workspace/IsaacLab/source/isaaclab_assets/isaaclab_assets/
# robots/unitree.py — G1_INSPIRE_FTP_CFG = G1_29DOF_CFG.copy() (actuators
# inherited unchanged): legs DCMotorCfg hip 100 / knee 200, feet DCMotorCfg
# ankle 20, waist Implicit 5000, arms Implicit 3000. pod/g1_grasp/grasp_env.py
# __post_init__ then multiplies every actuator group whose joint patterns
# mention hip/knee/ankle — the "legs" AND "feet" groups, waist untouched — by
# GRASP_LEG_STIFF (default 4.0).
LEG_STIFF = 4.0
def _kp_for(name):
    if "hip" in name:
        return 100.0 * LEG_STIFF
    if "knee" in name:
        return 200.0 * LEG_STIFF
    if "ankle" in name:
        return 20.0 * LEG_STIFF
    if "waist" in name:
        return 5000.0
    return 3000.0                              # arms group (shoulder/elbow/wrist)
KP_EFF = np.array([_kp_for(n) for n in W.BODY_JOINTS])
# statics tau vector is indexed by dof-6; map BODY_JOINTS into it
TAU_IX = []
for _n in W.BODY_JOINTS:
    _j = mujoco.mj_name2id(W.M, mujoco.mjtObj.mjOBJ_JOINT, _n)
    TAU_IX.append(int(W.M.jnt_dofadr[_j]) - 6)
TAU_IX = np.array(TAU_IX)
JGROUPS = {g: [k for k, n in enumerate(W.BODY_JOINTS) if key in n]
           for g, key in (("hip", "hip"), ("knee", "knee"), ("ankle", "ankle"),
                          ("waist", "waist"))}
JGROUPS["arm"] = [k for k, n in enumerate(W.BODY_JOINTS)
                  if not any(k in n for k in ("hip", "knee", "ankle", "waist"))]

_BANK = None


def bank():
    global _BANK
    if _BANK is None:
        _BANK = dict(np.load(os.path.join(REFS, "pregrasp_bank.npz"),
                             allow_pickle=False))
    return _BANK


def ease(s):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(s, 0.0, 1.0))


def quat2rpy(q):
    w, x, y, z = q
    return np.array([
        np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)),
        np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def keyframe_x():
    mujoco.mj_resetDataKeyframe(W.M, W.D, 0)
    q = W.D.qpos.copy()
    W.D.qvel[:] = 0
    x = np.zeros(6 + W.NJ)
    x[0:3] = q[0:3]
    x[3:6] = quat2rpy(q[3:7])
    x[6:] = q[W.JADR]
    return np.clip(x, W.LO, W.HI)


def foot_anchors(x):
    """(geom id, world xy) for all 8 sole spheres at pose x."""
    W.fk(x)
    return [(g, W.D.geom_xpos[g][:2].copy())
            for s in ("left", "right") for g in W.FOOT_SPH[s]]


# ------------------------------------------------------------- IK machinery
def traj_residual(x, anchors, palm_tgt, w_palm, pz_tgt, posture, w_post,
                  w_pin, pin_roll):
    """wbik's residual with feet additionally anchored in XY, a pelvis height
    target, and a posture-blend regulariser (weights from wbik where shared)."""
    W.fk(x)
    D = W.D
    res, pts = [], []
    for g, axy in anchors:
        p = D.geom_xpos[g]
        res.append(W.W_F * (p[2] - W.SOLE_Z))
        res.append(W_AXY * (p[0] - axy[0]))
        res.append(W_AXY * (p[1] - axy[1]))
        pts.append(p[:2].copy())
    poly = W.hull(pts)
    c = poly.mean(axis=0)
    res.append(W.W_C * W.dist_outside(D.subtree_com[1][:2], c + (poly - c) * 0.70))
    ko = 0.0
    for ai in KEEPOUT_IDS:
        p = D.xpos[ai]
        for a, b, clr in W.SEG_IDS:
            ko += max(0.0, clr - W.seg_dist(p, D.xpos[a], D.xpos[b]))
        ko += max(0.0, 0.13 - float(np.linalg.norm(p - D.xpos[W.PELVIS])))
        ko += max(0.0, 0.16 - float(np.linalg.norm(p[:2] - D.xpos[W.TORSO][:2])))
    for ai in KEEPOUT_IDS + KEEPOUT_L_IDS:
        p = D.xpos[ai]
        for h in HIP_PTS:
            ko += max(0.0, HIP_CLR - float(np.linalg.norm(p - D.xpos[h])))
    for ai in KEEPOUT_L_IDS:
        p = D.xpos[ai]
        for a, b, clr in W.SEG_IDS[:2]:            # left thigh + shin capsules
            ko += max(0.0, clr - W.seg_dist(p, D.xpos[a], D.xpos[b]))
    res.append(W.W_K * ko)
    fl = 0.0
    for ai in (W.ARM_IDS[0], W.ARM_IDS[1], W.PELVIS, W.TORSO, W.KNEE_L, W.KNEE_R):
        fl += max(0.0, 0.045 - D.xpos[ai][2])
    res.append(W.W_K * fl)
    if w_palm > 0.0:
        res += list(w_palm * (D.xpos[W.PALM] - palm_tgt))
    res.append(W_PZ * (x[2] - pz_tgt))
    res += list(w_post * (x[6:] - posture[6:]))
    res.append(w_pin * x[W.WP])            # waist pitch (wbik stance-A pin)
    res.append(w_pin * x[4])               # base pitch  (wbik stance-A pin)
    if pin_roll:
        res.append(w_pin * x[3])           # base roll — lift/hold only
    return np.array(res)


def lm_solve(res_fn, x0, iters):
    """wbik.solve's LM loop verbatim, parameterised by the residual."""
    x = x0.copy()
    r = res_fn(x)
    lam = 1e-2
    n = len(x)
    for _ in range(iters):
        J = np.zeros((len(r), n))
        for k in range(n):
            xp = x.copy()
            xp[k] = np.clip(xp[k] + 1e-5, W.LO[k], W.HI[k])
            dk = xp[k] - x[k]
            if abs(dk) < 1e-12:
                xp[k] = np.clip(x[k] - 1e-5, W.LO[k], W.HI[k])
                dk = xp[k] - x[k]
                if abs(dk) < 1e-12:
                    continue
            J[:, k] = (res_fn(xp) - r) / dk
        for _ls in range(10):
            try:
                dx = -np.linalg.solve(J.T @ J + lam * np.eye(n), J.T @ r)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            xn = np.clip(x + dx, W.LO, W.HI)
            rn = res_fn(xn)
            if rn @ rn < r @ r:
                x, r, lam = xn, rn, max(lam * 0.5, 1e-9)
                break
            lam *= 4
        else:
            break
        if r @ r < 1e-8:
            break
    return x, r


def solve_knot(x0, anchors, palm_tgt, w_palm, pz_tgt, posture, w_post,
               w_pin, pin_roll, iters):
    t0 = time.perf_counter()
    x, _ = lm_solve(lambda v: traj_residual(v, anchors, palm_tgt, w_palm,
                                            pz_tgt, posture, w_post, w_pin,
                                            pin_roll), x0, iters)
    rp = W.report(x, "A_squat_upright")
    okv = W.valid(rp)
    perr = (float(np.linalg.norm(rp["palm"] - palm_tgt)) if w_palm > 0 else 0.0)
    return x, perr, okv, rp, time.perf_counter() - t0


# ---------------------------------------------------------------- statics
def statics_obj(x, obj_point, mass):
    """statics.equilibrium at pose x, optionally with an extra `mass` kg
    hanging at world point obj_point on the palm body.

    tau + Jc^T lam = qfrc_bias - Jo^T f_obj with f_obj = (0,0,-m g), so the
    effective gravity load handed to equilibrium() is
    qfrc_bias + Jo^T (0,0,+m g)."""
    W.fk(x, full=True)
    if mass > 0.0:
        jacp = np.zeros((3, W.M.nv))
        jacr = np.zeros((3, W.M.nv))
        mujoco.mj_jac(W.M, W.D, jacp, jacr,
                      np.asarray(obj_point, dtype=float), W.PALM)
        W.D.qfrc_bias[:] = W.D.qfrc_bias + jacp.T @ np.array([0.0, 0.0,
                                                              mass * GRAV])
    return ST.equilibrium(ST.all_foot_geoms())


# --------------------------------------------------------------- velocities
def fdvel(P, dt):
    V = np.zeros_like(P)
    V[1:-1] = (P[2:] - P[:-2]) / (2 * dt)
    V[0] = (P[1] - P[0]) / dt
    V[-1] = (P[-1] - P[-2]) / dt
    return V


def quat_angvel(Q, dt):
    """World-frame angular velocity by central quaternion differences."""
    Wv = np.zeros((len(Q), 3))
    conj = np.array([1.0, -1.0, -1.0, -1.0])
    for t in range(1, len(Q) - 1):
        dq = quat_mul(Q[t + 1], Q[t - 1] * conj)
        if dq[0] < 0:
            dq = -dq
        s = float(np.linalg.norm(dq[1:4]))
        if s < 1e-12:
            continue
        Wv[t] = dq[1:4] / s * (2 * np.arctan2(s, dq[0])) / (2 * dt)
    Wv[0], Wv[-1] = Wv[1], Wv[-2]
    return Wv


# ----------------------------------------------------------------- one ref
def build_ref(i):
    t_start = time.time()
    b = bank()
    xsq = b["x"][i].astype(float)
    r, th = float(b["r"][i]), float(b["th"][i])
    obj_h = float(b["obj_h"])
    palm_sq = b["palm"][i].astype(float)
    pin1 = PIN_DESCEND if str(b["stance"][i]).startswith("A") else 0.0

    fails, times, stimes = [], [], []
    ik_ok = 0

    anchors = foot_anchors(xsq)
    c_feet = np.mean([a for _, a in anchors], axis=0)

    # ---- stand (P0 endpoint): keyframe stance re-planted on THIS ref's feet
    xkf = keyframe_x()
    xkf[LEFT_ARM_IX] = xsq[LEFT_ARM_IX]
    kf_c = np.mean([a for _, a in foot_anchors(xkf)], axis=0)
    x0 = xkf.copy()
    x0[0:2] += c_feet - kf_c
    x_stand, _, okv, rp, dt = solve_knot(
        x0, anchors, None, 0.0, xkf[2], xkf, 0.10, PIN_DESCEND, False, 70)
    times.append(dt)
    if okv:
        ik_ok += 1
    else:
        fails.append("STAND_IK_INVALID")
    W.fk(x_stand)
    palm_stand = W.D.xpos[W.PALM].copy()

    Xk = np.zeros((K, 6 + W.NJ))
    Xk[0:6] = x_stand
    maxperr = 0.0

    # ---- P1 descend: 14 intermediate re-solves, endpoint = certified squat.
    # The palm path bows outward (toward the grasp bearing) so the hand swings
    # ahead of the thigh instead of through it as the hip folds.
    u = palm_sq[:2] - c_feet
    u = u / max(float(np.linalg.norm(u)), 1e-9)
    prev = x_stand
    for j in range(1, 15):
        s = ease(j / 15)
        bow = BOW * np.sin(np.pi * s)
        pt = palm_stand + s * (palm_sq - palm_stand) \
            + np.array([u[0] * bow, u[1] * bow, 0.0])
        pz = x_stand[2] + s * (xsq[2] - x_stand[2])
        posture = x_stand + s * (xsq - x_stand)
        x, perr, okv, rp, dt = solve_knot(
            prev, anchors, pt, W_PALM, pz, posture, W_POST, pin1, False, 30)
        times.append(dt)
        maxperr = max(maxperr, perr)
        if okv and perr < TOL_MID:
            ik_ok += 1
        else:
            fails.append(f"P1_KNOT{j}(perr={perr:.3f},valid={okv})")
        Xk[5 + j] = x
        prev = x
    Xk[20] = xsq
    Xk[21:26] = xsq

    # ---- lift-hold endpoint (P3 end / P4): palm pulled in over the feet
    d = palm_sq[:2] - c_feet
    dn = float(np.linalg.norm(d))
    hold_xy = c_feet + d * (min(dn, HOLD_R) / max(dn, 1e-9))
    hold_z = max(0.55, palm_sq[2] + 0.38)  # object (palm+offset) >= ~0.455
    palm_hold = np.array([hold_xy[0], hold_xy[1], hold_z])
    lam_h = np.clip((PELV_HOLD - xsq[2]) / (x_stand[2] - xsq[2]), 0.0, 1.0)
    seed_h = xsq + lam_h * (x_stand - xsq)
    seed_h[2] = PELV_HOLD
    x_hold, perr_h, okv_h, rp_h, dt = solve_knot(
        seed_h, anchors, palm_hold, W_PALM, PELV_HOLD, seed_h, W_POST,
        PIN_LIFT, True, 80)
    times.append(dt)
    if okv_h and perr_h < TOL_END:
        ik_ok += 1
    else:
        fails.append(f"HOLD(perr={perr_h:.3f},valid={okv_h})")

    # ---- P3 lift: 14 intermediate re-solves squat -> hold
    prev = xsq
    for j in range(1, 15):
        s = ease(j / 15)
        pt = palm_sq + s * (palm_hold - palm_sq)
        pz = xsq[2] + s * (PELV_HOLD - xsq[2])
        posture = xsq + s * (x_hold - xsq)
        x, perr, okv, rp, dt = solve_knot(
            prev, anchors, pt, W_PALM, pz, posture, W_POST, PIN_LIFT, True, 30)
        times.append(dt)
        maxperr = max(maxperr, perr)
        if okv and perr < TOL_MID:
            ik_ok += 1
        else:
            fails.append(f"P3_KNOT{j}(perr={perr:.3f},valid={okv})")
        Xk[25 + j] = x
        prev = x
    Xk[40] = x_hold
    Xk[41:] = x_hold

    # ---- statics certificates + required torque at EVERY knot (unloaded
    # through P2 close, object-loaded from the P2/P3 boundary on). The QP's
    # tau vector is the certificate torque that holds each pose; it becomes
    # the PD feed-forward. Distinct poses only (holds copy their knot's tau).
    obj_rest = np.array([r * np.cos(th), r * np.sin(th), obj_h / 2])
    offset = obj_rest - palm_sq            # rigid palm->object CG, world frame
    TauK = np.zeros((K, W.NJ))
    stat_pass, stat_n, stat_fail_det = 0, 0, []

    def cert(x, loaded, tag):
        nonlocal stat_pass, stat_n
        t0 = time.perf_counter()
        if loaded:
            W.fk(x)
            pw = W.D.xpos[W.PALM].copy()
            ok_k, info = statics_obj(x, pw + offset, OBJ_M)
        else:
            ok_k, info = statics_obj(x, None, 0.0)
        stimes.append(time.perf_counter() - t0)
        stat_n += 1
        if ok_k:
            stat_pass += 1
        else:
            stat_fail_det.append((tag, {kk: info[kk] for kk in
                                        ("base_res", "min_normal", "cone",
                                         "over") if kk in info}
                                  if "reason" not in info else info["reason"]))
        tau = info.get("tau")
        return np.zeros(W.NJ) if tau is None else tau[TAU_IX]

    TauK[0:6] = cert(x_stand, False, "stand")
    for j in range(1, 15):
        TauK[5 + j] = cert(Xk[5 + j], False, f"P1k{j}")
    TauK[20:25] = cert(xsq, False, "squat")
    TauK[25] = cert(xsq, True, "squat+obj")
    for j in range(1, 15):
        TauK[25 + j] = cert(Xk[25 + j], True, f"P3k{j}")
    TauK[40:] = cert(x_hold, True, "hold+obj")
    if stat_fail_det:
        fails.append("STATICS:" + "; ".join(
            f"{t}={d}" for t, d in stat_fail_det))

    # ---- knot continuity
    jump = float(np.max(np.abs(np.diff(Xk[:, 6:], axis=0))))
    if jump > JUMP_MAX:
        fails.append(f"JUMP({jump:.2f}rad)")

    # ---- 50 Hz frames: cosine-eased between knots (tau exactly like poses)
    Xf = np.zeros((T, 6 + W.NJ))
    TauF = np.zeros((T, W.NJ))
    for f in range(T):
        tf = f * DT
        k = min(int(tf / KDT), K - 2)
        s = ease((tf - k * KDT) / KDT)
        Xf[f] = Xk[k] + s * (Xk[k + 1] - Xk[k])
        TauF[f] = TauK[k] + s * (TauK[k + 1] - TauK[k])

    palm_f = np.zeros((T, 3))
    for f in range(T):
        W.fk(Xf[f])
        palm_f[f] = W.D.xpos[W.PALM]

    root = np.zeros((T, 13), np.float64)
    root[:, 0:3] = Xf[:, 0:3]
    Q = np.stack([W.rpy2quat(*Xf[f, 3:6]) for f in range(T)])
    root[:, 3:7] = Q
    root[:, 7:10] = fdvel(Xf[:, 0:3], DT)
    root[:, 10:13] = quat_angvel(Q, DT)

    obj = np.zeros((T, 13), np.float64)
    obj[:, 0:3] = obj_rest
    obj[:, 3] = 1.0
    obj[P3F:, 0:3] = palm_f[P3F:] + offset
    obj[:, 7:10] = fdvel(obj[:, 0:3], DT)

    palm_p4 = float(palm_f[T - 1, 2])
    obj_p4 = float(obj[T - 1, 2])
    if palm_p4 < 0.45 or obj_p4 < 0.45:
        fails.append(f"P4_HEIGHT(palm={palm_p4:.3f},obj={obj_p4:.3f})")

    ok = not fails
    verdict = (f"ref {i:02d} r={r:.2f} th={np.degrees(th):+05.1f} "
               f"{'OK  ' if ok else 'FAIL'} ik={ik_ok}/30 "
               f"statics={stat_pass}/{stat_n} maxperr={maxperr:.3f} "
               f"jump={jump:.2f} palmP4={palm_p4:.3f} objP4={obj_p4:.3f} "
               f"t={time.time() - t_start:5.1f}s")
    if not ok:
        verdict += "\n        " + ("STATICS_FAIL " if stat_fail_det else "") \
                   + " | ".join(fails)
    return dict(i=i, ok=ok, fails=fails, verdict=verdict,
                statics_failed=bool(stat_fail_det),
                joint_pos=Xf[:, 6:].astype(np.float32),
                tau=TauF.astype(np.float32),
                root=root.astype(np.float32), obj=obj.astype(np.float32),
                ik_ok=ik_ok, stat_pass=stat_pass, stat_n=stat_n,
                maxperr=maxperr, jump=jump, palm_p4=palm_p4, obj_p4=obj_p4,
                times=times, stimes=stimes, wall=time.time() - t_start,
                r=r, th=th)


def _worker(i):
    try:
        return build_ref(i)
    except Exception:
        return dict(i=i, ok=False, fails=["EXC"], statics_failed=False,
                    verdict=f"ref {i:02d} FAIL EXC\n{traceback.format_exc()}",
                    times=[], stimes=[], wall=0.0)


# ------------------------------------------------------------------ parent
def controls():
    """Known-answer controls before any verdict is believed (standing law):
    statics must reproduce the standing-keyframe answer, and the object-weight
    modification must add ~= m*g of normal load without breaking feasibility."""
    mujoco.mj_resetDataKeyframe(W.M, W.D, 0)
    W.D.qvel[:] = 0
    mujoco.mj_forward(W.M, W.D)
    ok0, info0 = ST.equilibrium(ST.all_foot_geoms())
    n0 = info0.get("total_normal", 0.0)
    mujoco.mj_forward(W.M, W.D)            # fresh bias for the modified solve
    palm = W.D.xpos[W.PALM].copy()
    jacp = np.zeros((3, W.M.nv))
    jacr = np.zeros((3, W.M.nv))
    mujoco.mj_jac(W.M, W.D, jacp, jacr, palm, W.PALM)
    W.D.qfrc_bias[:] = W.D.qfrc_bias + jacp.T @ np.array([0., 0., OBJ_M * GRAV])
    ok1, info1 = ST.equilibrium(ST.all_foot_geoms())
    dn = info1.get("total_normal", 0.0) - n0
    print(f"CONTROL standing keyframe: {'FEASIBLE' if ok0 else 'INFEASIBLE'} "
          f"normal={n0:.1f}N", flush=True)
    print(f"CONTROL +{OBJ_M}kg at palm : {'FEASIBLE' if ok1 else 'INFEASIBLE'} "
          f"dN={dn:+.2f}N (expect {OBJ_M * GRAV:+.2f})", flush=True)
    if not (ok0 and ok1 and abs(dn - OBJ_M * GRAV) < 1.0):
        print("CONTROL_FAIL — statics machinery does not reproduce known "
              "answers; no trajectory verdict below can be believed.")
        sys.exit(2)
    print("CONTROL_OK", flush=True)


def main():
    t0 = time.time()
    n = len(bank()["x"])
    print(f"make_ref_traj: {n} certified refs -> 5 s / {FPS} Hz whole-body "
          f"references ({T} frames, {K} knots, 30 IK re-solves each)",
          flush=True)
    controls()

    nproc = max(1, min(8, (os.cpu_count() or 4) - 1))
    print(f"solving {n} refs on {nproc} processes ...", flush=True)
    results = {}
    with mp.get_context("spawn").Pool(nproc) as pool:
        for res in pool.imap_unordered(_worker, range(n)):
            results[res["i"]] = res
            print(res["verdict"], flush=True)

    ok_ids = [i for i in range(n) if results[i]["ok"]]
    n_statics_fail = sum(1 for i in range(n) if results[i]["statics_failed"])
    all_t = np.array([t for i in range(n) for t in results[i]["times"]])
    all_st = np.array([t for i in range(n) for t in results[i]["stimes"]])
    if len(all_t):
        print(f"\nIK solve time/knot: p50={np.percentile(all_t, 50):.2f}s "
              f"p90={np.percentile(all_t, 90):.2f}s max={all_t.max():.2f}s "
              f"(n={len(all_t)})", flush=True)
    if len(all_st):
        print(f"statics QP time/knot: p50={np.percentile(all_st, 50):.2f}s "
              f"p90={np.percentile(all_st, 90):.2f}s max={all_st.max():.2f}s "
              f"(n={len(all_st)})", flush=True)

    if len(ok_ids) >= MIN_OK:
        b = bank()
        os.makedirs(REFS, exist_ok=True)
        out = os.path.join(REFS, "ref_traj_bank.npz")
        jp = np.stack([results[i]["joint_pos"] for i in ok_ids])
        tau = np.stack([results[i]["tau"] for i in ok_ids])
        # poses must be untouched by the gravity-compensation rework: compare
        # against a previously written bank before overwriting it
        if os.path.exists(out):
            prev = np.load(out, allow_pickle=False)
            if (prev["joint_pos"].shape == jp.shape
                    and np.array_equal(prev["bank_index"],
                                       np.array(ok_ids, np.int32))):
                dd = float(np.abs(prev["joint_pos"] - jp).max())
                print(f"pose identity vs previous bank: max|d|={dd:.2e} "
                      f"{'IDENTICAL' if dd == 0.0 else 'CHANGED — investigate'}",
                      flush=True)
        common = dict(
            joint_pos=jp,
            root=np.stack([results[i]["root"] for i in ok_ids]),
            obj=np.stack([results[i]["obj"] for i in ok_ids]),
            phase=np.tile(PHASE_F, (len(ok_ids), 1)),
            joint_names=np.array(W.BODY_JOINTS),
            bank_index=np.array(ok_ids, np.int32),
            r=b["r"][ok_ids], th=b["th"][ok_ids],
            fps=float(FPS), obj_mass=OBJ_M, obj_h=float(b["obj_h"]))
        np.savez_compressed(out, joint_targets=jp, **common)
        # gravity-compensated bank: identical poses/phases/object path, but
        # joint_targets carry the feed-forward offset tau/kp so PD produces
        # the certificate torque AT the reference pose instead of zero
        off = tau / KP_EFF[None, None, :].astype(np.float32)
        out_gc = os.path.join(REFS, "ref_traj_bank_gc.npz")
        np.savez_compressed(out_gc, joint_targets=jp + off, tau=tau,
                            kp_eff=KP_EFF.astype(np.float32),
                            leg_stiff_mult=LEG_STIFF, **common)
        print(f"\ntarget offset |tau/kp| per joint group (max over bank):",
              flush=True)
        for g, ixs in JGROUPS.items():
            mx = float(np.abs(off[:, :, ixs]).max())
            kps = sorted({float(KP_EFF[k]) for k in ixs})
            flag = "  <-- EXCEEDS 0.5 rad, kp mis-read?" if mx > 0.5 else ""
            print(f"    {g:6s} kp={kps} max_offset={mx:.4f} rad{flag}",
                  flush=True)
        print(f"    peak |tau| knee={float(np.abs(tau[:, :, JGROUPS['knee']]).max()):.1f} "
              f"ankle={float(np.abs(tau[:, :, JGROUPS['ankle']]).max()):.1f} "
              f"hip={float(np.abs(tau[:, :, JGROUPS['hip']]).max()):.1f} N.m",
              flush=True)
        dj = np.abs(np.diff(jp, axis=1))
        with open(os.path.join(REFS, "ref_traj_bank_manifest.json"), "w") as f:
            json.dump(dict(
                date=time.strftime("%Y-%m-%d"),
                source="refs/pregrasp_bank.npz (42 certified squats)",
                spec="5s@50Hz stand/descend/close/lift/hold; 30 IK knots/ref; "
                     "P3 statics with 0.5kg object at palm-offset CG",
                model="menagerie g1_with_hands (BODY joints only; fingers are "
                      "phase-flagged for Isaac-side Inspire targets)",
                n_refs=len(ok_ids), frames=T,
                max_frame_joint_step_rad=float(dj.max()),
                refs=[dict(i=i, r=results[i]["r"],
                           th_deg=round(float(np.degrees(results[i]["th"])), 1),
                           ik=results[i]["ik_ok"],
                           statics=f"{results[i]['stat_pass']}/{results[i]['stat_n']}",
                           maxperr=round(results[i]["maxperr"], 4),
                           jump=round(results[i]["jump"], 3),
                           palm_p4=round(results[i]["palm_p4"], 3),
                           obj_p4=round(results[i]["obj_p4"], 3),
                           wall_s=round(results[i]["wall"], 1))
                      for i in ok_ids],
                failed=[dict(i=i, fails=results[i]["fails"])
                        for i in range(n) if not results[i]["ok"]]), f, indent=1)
        print(f"\nwrote {out}  ({len(ok_ids)} refs, "
              f"max frame joint step {dj.max():.3f} rad)", flush=True)
        print(f"wrote {out_gc}  (gravity-compensated targets + tau + kp_eff)",
              flush=True)
        code = 0
    else:
        code = 1

    print(f"\nREF_TRAJ_SUMMARY nrefs={n} ok={len(ok_ids)} "
          f"statics_fail={n_statics_fail}", flush=True)
    print(f"total wall {time.time() - t0:.0f}s", flush=True)
    if code:
        print(f"REF_TRAJ_FAIL only {len(ok_ids)}/{n} refs passed "
              f"(need >={MIN_OK})", flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
