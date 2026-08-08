"""CERT-BASEFRAME: known-answer certificates for GRASP_BASEFRAME_OBS.

Four certificates (see also the local numpy math certs, which verify the
rotation/hysteresis formulas themselves — this probe verifies the SHIPPED
env path):

  --mode id [--flag off]  (CERT-ID): 50 seeded zero-action steps from
      psi=0 spawns (no spawn-yaw flags), prints OBS_SHA over the full obs
      stream. PROTOCOL: run once with the flag on (default here) and once
      with --flag off — the two OBS_SHA lines must be IDENTICAL
      (R_z(-0) is the identity and the flag-off path never runs a kernel).
      The flag-on invocation additionally cross-checks the four blocks
      against a manual world-frame recomputation rotated by the env's own
      psi (max-abs-diff printed, needs < 1e-5).

  --mode rot [--flag off]  (CERT-ROT): settle, capture obs A, rigidly
      rotate ROBOT + OBJECT together by +90 deg about the env origin
      (positions, quaternions, velocities), step once, capture obs B.
        flag ON:  the four blocks (and hence the whole obs row) must be
                  INVARIANT — max|B - A| < 2e-2 (one physics step of
                  settle drift is the tolerance floor).
        flag OFF (control): the audited pathology as known answer — the
                  four world-frame blocks must match the +90-deg rotation
                  map (x,y) -> (-y,x) of A, NOT A itself.
      Run under GRASP_FREE=1 — root teleports require a floating base.

  --mode yawdeg  (CERT-YAWDEG): teleport-yaw the robot to psi ~ 1 rad,
      then pitch it to ~85 deg (fwd-axis ground projection 0.087 < 0.15):
      psi must HOLD its last-good value while pitched and every obs stay
      finite; then a forced flung reset must re-init psi from the fresh
      spawn quaternion (|psi| < 0.05, not the held ~1 rad).
      Run under GRASP_FREE=1.

  EMPNORM-guard (documented, not a mode): run the frozen-normalizer cert
      with this flag on —
        GRASP_EMPNORM=1 GRASP_BASEFRAME_OBS=1 PYTHONPATH=. \
        python -u g1_grasp/probe_empnorm_freeze.py --cert freeze \
            --checkpoint <mature.pt> --headless
      PASS = normalizer hash identical (frozen stats untouched by the
      rotated obs during CERT-ID-like traffic).

Examples:

    GRASP_FREE=1 PYTHONPATH=. python -u g1_grasp/probe_baseframe_obs.py \
        --mode id --headless
    GRASP_FREE=1 PYTHONPATH=. python -u g1_grasp/probe_baseframe_obs.py \
        --mode id --flag off --headless
    GRASP_FREE=1 PYTHONPATH=. python -u g1_grasp/probe_baseframe_obs.py \
        --mode rot --headless        # and again with --flag off
    GRASP_FREE=1 PYTHONPATH=. python -u g1_grasp/probe_baseframe_obs.py \
        --mode yawdeg --headless
"""

import argparse
import hashlib
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=["id", "rot", "yawdeg"])
parser.add_argument("--flag", default="on", choices=["on", "off"])
parser.add_argument("--num_envs", type=int, default=4)

_known, _ = parser.parse_known_args()
if _known.flag == "on":
    os.environ["GRASP_BASEFRAME_OBS"] = "1"
else:
    os.environ.pop("GRASP_BASEFRAME_OBS", None)
os.environ["GRASP_FORCE_STAGE"] = "3"
os.environ.pop("GRASP_SPAWN_YAW_MAX", None)   # id-cert needs psi=0 spawns
os.environ.pop("GRASP_PROTOCOL_SPAWN", None)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import g1_grasp  # noqa: F401

SEED = 3407

# obs layout offsets for the four blocks (after the optional FULLBODY
# 9-dim head): q(n) qd(n) palm(3) obj_rel(3) obj_vel(3) prev_a(n)
# tip_rel(15) ...
def block_slices(raw):
    from g1_grasp.grasp_env import FULLBODY
    n = len(raw.joint_ids)
    o = (9 if FULLBODY else 0) + 2 * n
    return {"palm": slice(o, o + 3), "obj_rel": slice(o + 3, o + 6),
            "obj_vel": slice(o + 6, o + 9),
            "tip_rel": slice(o + 9 + n, o + 24 + n)}


def rot90_blocks(row, sl):
    """Apply the +90-deg world map (x,y)->(-y,x) to the four blocks."""
    out = row.clone()
    for name, s in sl.items():
        v = row[s].reshape(-1, 3)
        out[s] = torch.stack([-v[:, 1], v[:, 0], v[:, 2]], -1).reshape(-1)
    return out


def rigid_rot90(raw):
    """Rotate robot + object together by +90 deg about each env origin."""
    dev = raw.device
    c, s = 0.0, 1.0  # cos/sin 90
    for asset in (raw.robot, raw.obj):
        pos = asset.data.root_pos_w.clone()
        quat = asset.data.root_quat_w.clone()
        vel = asset.data.root_vel_w.clone() if hasattr(asset.data, "root_vel_w") \
            else torch.cat([asset.data.root_lin_vel_w,
                            asset.data.root_ang_vel_w], dim=-1).clone()
        rel = pos - raw.scene.env_origins
        pos = raw.scene.env_origins + torch.stack(
            [c * rel[:, 0] - s * rel[:, 1],
             s * rel[:, 0] + c * rel[:, 1], rel[:, 2]], dim=-1)
        cy, sy = math.cos(math.pi / 4), math.sin(math.pi / 4)  # half angle
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        quat = torch.stack([cy * w - sy * z, cy * x - sy * y,
                            cy * y + sy * x, cy * z + sy * w], dim=-1)
        for a, b in ((0, 1), (3, 4)):  # lin xy, ang xy
            va, vb = vel[:, a].clone(), vel[:, b].clone()
            vel[:, a] = c * va - s * vb
            vel[:, b] = s * va + c * vb
        asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
        asset.write_root_velocity_to_sim(vel)


def main():
    torch.manual_seed(SEED)
    env_cfg = parse_env_cfg("G1-Grasp-v0", num_envs=args.num_envs)
    env_cfg.seed = SEED
    env = gym.make("G1-Grasp-v0", cfg=env_cfg)
    raw = env.unwrapped
    on = args.flag == "on"
    assert raw.baseframe_obs == on
    obs, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), None)
    acts = torch.zeros(raw.num_envs, raw.cfg.action_space, device=raw.device)
    sl = block_slices(raw)

    if args.mode == "id":
        h = hashlib.md5()
        xerr = 0.0
        for _ in range(50):
            out = env.step(acts)
            ob = out[0]["policy"] if isinstance(out[0], dict) else out[0]
            h.update(ob.detach().to("cpu", torch.float32).numpy().tobytes())
            if on:
                # cross-check block 'palm' against manual recomputation
                palm_w = (raw.robot.data.body_pos_w[:, raw.palm_id]
                          - raw.scene.env_origins)
                c0 = torch.cos(raw.bf_psi)
                s0 = torch.sin(raw.bf_psi)
                man = torch.stack([c0 * palm_w[:, 0] + s0 * palm_w[:, 1],
                                   -s0 * palm_w[:, 0] + c0 * palm_w[:, 1],
                                   palm_w[:, 2]], dim=-1)
                xerr = max(xerr, float((ob[:, sl["palm"]] - man).abs().max()))
        print(f"BASEFRAME_ID_SHA flag={args.flag} md5={h.hexdigest()}",
              flush=True)
        if on:
            print(f"BASEFRAME_ID xcheck max|obs-manual|={xerr:.2e} "
                  f"(need < 1e-5)", flush=True)
        print("PROBE_BF_ID_RECORDED (PASS = identical SHA across "
              "--flag on/off invocations"
              + (f"; xcheck {'ok' if xerr < 1e-5 else 'FAIL'}" if on else "")
              + ")", flush=True)
        return

    if args.mode == "rot":
        for _ in range(5):
            out = env.step(acts)
        a_row = (out[0]["policy"] if isinstance(out[0], dict)
                 else out[0])[0].clone()
        rigid_rot90(raw)
        out = env.step(acts)
        b_row = (out[0]["policy"] if isinstance(out[0], dict)
                 else out[0])[0].clone()
        if on:
            d = float((b_row - a_row).abs().max())
            ok = d < 2e-2
            print(f"PROBE_BF_ROT_{'PASS' if ok else 'FAIL'} flag=on "
                  f"(invariance: max|B-A|={d:.4f} need < 0.02)", flush=True)
        else:
            pred = rot90_blocks(a_row, sl)
            d_rot = max(float((b_row[s] - pred[s]).abs().max())
                        for s in sl.values())
            d_id = max(float((b_row[s] - a_row[s]).abs().max())
                       for s in sl.values())
            ok = d_rot < 2e-2 and d_id > 0.05
            print(f"PROBE_BF_ROT_{'PASS' if ok else 'FAIL'} flag=off "
                  f"(pathology known answer: |B-R90(A)|={d_rot:.4f} "
                  f"need<0.02, |B-A|={d_id:.4f} need>0.05)", flush=True)
        return

    # yawdeg
    assert on, "--mode yawdeg needs the flag on"
    # teleport-yaw to psi ~ 1 rad so 'held' and 'reinit' are distinguishable
    q1 = torch.tensor([[math.cos(0.5), 0.0, 0.0, math.sin(0.5)]],
                      device=raw.device).repeat(raw.num_envs, 1)
    pos = raw.robot.data.root_pos_w.clone()
    raw.robot.write_root_pose_to_sim(torch.cat([pos, q1], dim=-1))
    out = env.step(acts)
    psi_held = raw.bf_psi.clone()
    print(f"YAWDEG psi after yaw-1rad: {float(psi_held[0]):.3f}", flush=True)
    # pitch ~85 deg about y, composed on the yawed attitude: fwd-axis
    # ground projection = cos(85deg) = 0.087 < 0.15 -> psi must hold
    cp, sp = math.cos(85 / 2 * math.pi / 180), math.sin(85 / 2 * math.pi / 180)
    qp = torch.zeros_like(q1)
    w, x, y, z = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    # BODY-frame pitch: q1 (x) p, p = (cp, 0, sp, 0). The first version
    # flipped two signs and implemented p (x) q1 — a WORLD-frame pitch,
    # under which a yawed robot's forward axis keeps a ~0.84 ground
    # projection and the guard CORRECTLY recomputes (cert FAIL 2026-07-30
    # was a probe bug; the env guard was right). Body-frame pitch is the
    # physical "robot pitches forward" motion the guard exists for:
    # projection = cos(85 deg) = 0.087 < 0.15 -> hold.
    qp[:, 0] = w * cp - y * sp
    qp[:, 1] = x * cp - z * sp
    qp[:, 2] = y * cp + w * sp
    qp[:, 3] = z * cp + x * sp
    pos = raw.robot.data.root_pos_w.clone()
    raw.robot.write_root_pose_to_sim(torch.cat([pos, qp], dim=-1))
    out = env.step(acts)
    ob = out[0]["policy"] if isinstance(out[0], dict) else out[0]
    held_ok = float((raw.bf_psi - psi_held).abs().max()) < 1e-6
    finite_ok = bool(torch.isfinite(ob).all())
    # forced mid-probe reset of env 0 (flung) -> psi re-inits from spawn
    far = raw.obj.data.root_pos_w.clone()
    far[0, 0] = raw.scene.env_origins[0, 0] + 5.0
    quat = raw.obj.data.root_quat_w.clone()
    raw.obj.write_root_pose_to_sim(
        torch.cat([far[:1], quat[:1]], dim=-1),
        torch.tensor([0], device=raw.device))
    env.step(acts)   # flung fires, env 0 resets, psi re-seeded
    reinit_ok = (bool(raw.episode_length_buf[0] == 0 or
                      raw.episode_length_buf[0] == 1)
                 and abs(float(raw.bf_psi[0])) < 0.05)
    ok = held_ok and finite_ok and reinit_ok
    print(f"PROBE_BF_YAWDEG_{'PASS' if ok else 'FAIL'} "
          f"(held@85deg={held_ok} finite={finite_ok} "
          f"reinit_psi0={float(raw.bf_psi[0]):.3f} (need |.|<0.05, "
          f"was held at {float(psi_held[0]):.3f}): {reinit_ok})", flush=True)


if __name__ == "__main__":
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
