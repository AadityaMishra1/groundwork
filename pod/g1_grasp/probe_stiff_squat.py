"""STIFF-SQUAT probe: can the robot HOLD a squat at all, and does leg
stiffness decide it?

Context (2026-08-01). Driving the robot to the bank's squat pose over a
gentle 12 s sweep left it at pelvis 0.082 m — on the floor — at only 8%
slew saturation. Commanding a squat does not produce a squat; it produces a
collapse. Hypothesis: with GRASP_LEG_STIFF=4 the legs cannot hold body
weight in a bent-knee pose, so the body sags through the target. If true it
reframes three weeks of work: the learned splay-legged "lunge" is not a
degenerate choice, it is the only posture soft legs can support, because a
straight splayed stance carries load through geometry instead of joint
torque.

The bank cannot test this — bank_guard correctly refuses stiffness changes,
since its targets are gravity-compensated against x4 legs. So this probe
uses an ANALYTIC squat, no bank loaded, and sweeps GRASP_LEG_STIFF freely:

    hip_pitch -> -1.4 rad,  knee -> +2.2 rad,  ankle_pitch -> -0.85 rad
    (signs/magnitudes read off the bank's own frame-100 configuration)

Verdict per stiffness: does pelvis settle near the commanded squat height
(~0.39 m), or sag to the floor?

    PYTHONPATH=. GRASP_LEG_STIFF=<x> python -u g1_grasp/probe_stiff_squat.py --headless
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["GRASP_FORCE_STAGE"] = "3"
os.environ["GRASP_NO_VIDEO"] = "1"
os.environ.pop("GRASP_REF_BANK", None)      # no bank -> no stiffness guard
os.environ["GRASP_REF_TRACK"] = "0"
os.environ["GRASP_REF_FRAC"] = "0.0"

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--sweep", type=int, default=600)
parser.add_argument("--hold", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import g1_grasp  # noqa: F401, E402

SQUAT = {"hip_pitch": -1.40, "knee": 2.20, "ankle_pitch": -0.85}


def main():
    cfg = parse_env_cfg("G1-Grasp-v0", num_envs=args.num_envs)
    env = gym.make("G1-Grasp-v0", cfg=cfg)
    raw = env.unwrapped
    dev = raw.device
    org = raw.scene.env_origins
    names = [raw.robot.data.joint_names[i] for i in raw.joint_ids]
    stiff = os.environ.get("GRASP_LEG_STIFF", "?")

    # build the analytic squat target in ACTION-joint space
    env.reset()
    tgt = raw.targets.clone()
    hits = 0
    for a_i, nm in enumerate(names):
        for key, val in SQUAT.items():
            if key in nm:
                tgt[:, a_i] = val
                hits += 1
    print(f"STIFFSQUAT leg_stiff={stiff} squat joints set={hits} "
          f"sweep={args.sweep} hold={args.hold}", flush=True)
    if hits == 0:
        print("STIFFSQUAT_FAIL no leg joints matched")
        return

    start = raw.targets.clone()
    pz_hold, pz_min, sat_acc, n = [], 10.0, 0.0, 0
    total = args.sweep + args.hold
    with torch.inference_mode():
        for k in range(total):
            alpha = min(1.0, k / args.sweep)
            desired = start + alpha * (tgt - start)
            gear = (1.0 - 0.6 * raw.grasped_state.float()).unsqueeze(-1)
            step_lim = raw.delta_vec.unsqueeze(0) * gear
            a = (desired - raw.targets) / step_lim.clamp_min(1e-8)
            sat_acc += (a.abs() > 1.0).float().mean().item(); n += 1
            env.step(a.clamp(-1.0, 1.0))
            pz = float((raw.robot.data.root_pos_w[:, 2] - org[:, 2]).median())
            pz_min = min(pz_min, pz)
            if k >= args.sweep:
                pz_hold.append(pz)

    held = sum(pz_hold) / max(1, len(pz_hold))
    ok = held > 0.30
    print(f"STIFFSQUAT leg_stiff={stiff}: pelvis_during_hold={held:.3f} "
          f"pelvis_min={pz_min:.3f} saturation={sat_acc/max(1,n):.3f}",
          flush=True)
    print(f"STIFFSQUAT_{'HOLDS' if ok else 'COLLAPSES'} leg_stiff={stiff} "
          + ("— the legs support a bent-knee squat at this stiffness."
             if ok else
             "— the body sags through the commanded squat to the floor."))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
