"""DESCENT-AUTHORITY probe: can the action space express the descent at all?

The question this settles (2026-08-01). A posture expert with ZERO competing
rewards — no object income, no lift, no stream, its only job to reproduce our
own computed squat-and-stand — tracked the RISE fine and got steadily WORSE
as the backward curriculum introduced the DESCENT (track_err 0.336 -> 0.512).
Five reward/curriculum interventions before it all failed the same way. That
eliminates reward design, exploration and architecture, and leaves one
suspect nobody has tested for the descent: the ACTION SPACE.

Controlled descent means yielding to gravity at a commanded rate. This env
commands *changes* to position targets, rate-limited:

    targets <- clamp(targets + a * delta_vec * gear, lo, hi)

If the reference descent needs target motion faster than delta_vec*gear
allows, the action SATURATES and no policy — however well trained — can
follow it. That is a plant limit, not a learning problem, and it would
explain every null tonight.

Method: no policy anywhere. Restore onto reference frame 0 (standing, object
on floor) and drive the delta integrator by exact inversion toward the bank's
recorded targets, i.e. the best any controller could do. Then compare, PER
PHASE:
    sat      fraction of joint-commands clamped at the slew limit
    pelvis   achieved pelvis_z vs the reference pelvis_z
A descent that saturates and falls behind is the wall; one that tracks
cleanly means the plant is fine and the failure is upstream.

    PYTHONPATH=. python -u g1_grasp/probe_descent_authority.py \
        --num_envs 64 --headless
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["GRASP_REF_TRACK"] = "1"
os.environ["GRASP_REF_FRAC"] = "1.0"
os.environ["GRASP_REF_T0ZERO"] = "1"      # every episode starts at frame 0
os.environ["GRASP_FORCE_STAGE"] = "3"
os.environ["GRASP_NO_VIDEO"] = "1"

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=0, help="0 = bank T")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import g1_grasp  # noqa: F401, E402


def main():
    cfg = parse_env_cfg("G1-Grasp-v0", num_envs=args.num_envs)
    env = gym.make("G1-Grasp-v0", cfg=cfg)
    raw = env.unwrapped
    dev = raw.device
    T = raw.ref_T
    steps = args.steps or T
    org = raw.scene.env_origins
    print(f"DESCENT_AUTH T={T} steps={steps} envs={raw.num_envs}", flush=True)

    env.reset()
    live = raw.ref_t0 >= 0
    if not bool(live.any()):
        print("DESCENT_AUTH_FAIL no reference episodes")
        return
    rid = raw.ref_id.clone()

    # per-step records
    sat_by_t, perr_by_t, pz_by_t, ref_pz_by_t = [], [], [], []
    with torch.inference_mode():
        for k in range(steps):
            t = min(k, T - 1)
            desired = raw.ref_jp[rid, t]                    # bank targets
            # raw.targets is the LIVE delta-integrator state (grasp_env:303,
            # written every step at :1194). ref_joint_ids index the tracked
            # body joints within the action set.
            cur = raw.targets[:, raw.ref_joint_ids]
            gear = 1.0 - 0.6 * raw.grasped_state.float()   # grasp_env:1193
            step_lim = (raw.delta_vec[raw.ref_joint_ids].unsqueeze(0)
                        * gear.unsqueeze(-1))
            need = desired - cur
            a = (need / step_lim.clamp_min(1e-8))
            sat = (a.abs() > 1.0).float().mean().item()
            a = a.clamp(-1.0, 1.0)
            full = torch.zeros(raw.num_envs, raw.cfg.action_space, device=dev)
            full[:, raw.ref_joint_ids] = a
            env.step(full)

            pz = (raw.robot.data.root_pos_w[:, 2] - org[:, 2])
            ref_pz = raw.ref_root[rid, t, 2]
            sat_by_t.append(sat)
            pz_by_t.append(float(pz.median()))
            ref_pz_by_t.append(float(ref_pz.median()))
            perr_by_t.append(float((pz - ref_pz).abs().median()))

    # phase report: descent = first half of the motion, rise = second
    def seg(lo, hi, name):
        s = sum(sat_by_t[lo:hi]) / max(1, hi - lo)
        e = sum(perr_by_t[lo:hi]) / max(1, hi - lo)
        print(f"DESCENT_AUTH {name:8s} frames {lo:3d}-{hi:3d}: "
              f"saturation={s:.3f}  pelvis_err={e:.3f} m  "
              f"achieved_p50={sum(pz_by_t[lo:hi])/max(1,hi-lo):.3f} "
              f"ref_p50={sum(ref_pz_by_t[lo:hi])/max(1,hi-lo):.3f}", flush=True)
        return s, e

    n = len(sat_by_t)
    d_sat, d_err = seg(0, min(150, n), "DESCENT")
    r_sat, r_err = seg(min(150, n), n, "RISE")

    wall = (d_sat > 0.25) or (d_err > 0.10)
    print(f"DESCENT_AUTH_VERDICT "
          + ("PLANT_WALL — the descent saturates the slew limit and/or the "
             "body cannot follow the reference. No reward or curriculum can "
             "fix this; the action representation must change."
             if wall else
             "PLANT_OK — the action space CAN express the descent open-loop. "
             "The failure is upstream: learning, not authority."))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
