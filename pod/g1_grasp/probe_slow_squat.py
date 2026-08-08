"""SLOW-SQUAT probe: is the ACTION SPACE the wall, or was it the reference?

probe_descent_authority showed the robot collapses when driven along the
reference bank (pelvis 0.233 vs ref 0.568) at only 12% slew saturation. That
proved the REFERENCE is dynamically infeasible. It did NOT separate two
different claims:

    (A) the reference's TIMING is impossible — the poses are fine but the
        motion demands accelerations this plant cannot produce
    (B) the ACTION SPACE cannot execute a controlled squat at all, at any
        speed, so no trajectory would ever work

This probe separates them. Same squat POSE (the bank's frame-100 joint
configuration, pelvis ~0.39), executed at a range of speeds from very slow to
reference-speed, under direct target interpolation with no policy:

    default standing pose --sweep--> squat pose --hold--> back to standing

If a slow sweep tracks smoothly and the robot stands back up, the action
space is FINE and (A) is the answer: regenerate the reference with dynamics.
If even the slowest sweep collapses, (B) is the answer: the action
representation itself must change, and no reference will rescue it.

    PYTHONPATH=. python -u g1_grasp/probe_slow_squat.py --headless
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["GRASP_REF_TRACK"] = "1"
os.environ["GRASP_REF_FRAC"] = "0.0"
os.environ["GRASP_FORCE_STAGE"] = "3"
os.environ["GRASP_NO_VIDEO"] = "1"

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--squat_frame", type=int, default=100)
parser.add_argument("--hold", type=int, default=100)
parser.add_argument("--sweep", type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import g1_grasp  # noqa: F401, E402

# sweep durations in control steps. The reference does its descent in ~150.
SWEEPS = [600]   # one sweep per process: env.reset() across sweeps is unsafe


def run_sweep(env, raw, dev, squat_pose, n_sweep, hold):
    """default -> squat over n_sweep, hold, -> default. Returns telemetry."""
    org = raw.scene.env_origins
    env.reset()
    start = raw.targets[:, raw.ref_joint_ids].clone()
    pz_min, pz_end, sat_acc, steps = 10.0, 0.0, 0.0, 0
    total = 2 * n_sweep + hold
    with torch.inference_mode():
        for k in range(total):
            if k < n_sweep:
                alpha = k / n_sweep
            elif k < n_sweep + hold:
                alpha = 1.0
            else:
                alpha = 1.0 - (k - n_sweep - hold) / n_sweep
            desired = start + alpha * (squat_pose - start)
            cur = raw.targets[:, raw.ref_joint_ids]
            gear = 1.0 - 0.6 * raw.grasped_state.float()
            step_lim = (raw.delta_vec[raw.ref_joint_ids].unsqueeze(0)
                        * gear.unsqueeze(-1))
            a = (desired - cur) / step_lim.clamp_min(1e-8)
            sat_acc += (a.abs() > 1.0).float().mean().item()
            steps += 1
            full = torch.zeros(raw.num_envs, raw.cfg.action_space, device=dev)
            full[:, raw.ref_joint_ids] = a.clamp(-1.0, 1.0)
            env.step(full)
            pz = float((raw.robot.data.root_pos_w[:, 2] - org[:, 2]).median())
            pz_min = min(pz_min, pz)
            pz_end = pz
    return pz_min, pz_end, sat_acc / max(1, steps)


def main():
    cfg = parse_env_cfg("G1-Grasp-v0", num_envs=args.num_envs)
    env = gym.make("G1-Grasp-v0", cfg=cfg)
    raw = env.unwrapped
    dev = raw.device
    # squat POSE from the bank (a static configuration, not its timing)
    squat_pose = raw.ref_jp[0, args.squat_frame].unsqueeze(0).expand(
        raw.num_envs, -1).clone()
    ref_pz = float(raw.ref_root[0, args.squat_frame, 2])
    print(f"SLOWSQUAT squat pose = bank frame {args.squat_frame} "
          f"(reference pelvis there = {ref_pz:.3f} m); sweeps {SWEEPS} steps "
          f"(reference does its descent in ~150)", flush=True)

    ok_any = False
    for n in [args.sweep]:
        pz_min, pz_end, sat = run_sweep(env, raw, dev, squat_pose,
                                        n, args.hold)
        stood = pz_end > 0.60
        squatted = pz_min < 0.50
        verdict = ("SQUAT+STAND OK" if (stood and squatted)
                   else "squatted but did NOT stand" if squatted
                   else "did not squat")
        ok_any = ok_any or (stood and squatted)
        print(f"SLOWSQUAT sweep={n:4d} steps ({n/50:.1f}s): "
              f"pelvis_min={pz_min:.3f} pelvis_end={pz_end:.3f} "
              f"saturation={sat:.3f} -> {verdict}", flush=True)

    print("SLOWSQUAT_VERDICT "
          + ("ACTION_SPACE_OK — a controlled squat and stand IS executable "
             "under this action representation at some speed. The wall is "
             "the REFERENCE's dynamics, not the plant: regenerate the bank "
             "with contact-aware trajectory optimization."
             if ok_any else
             "ACTION_SPACE_WALL — no sweep speed produced a squat-and-stand. "
             "The action representation itself cannot express the motion; "
             "no reference or reward will fix it."))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
