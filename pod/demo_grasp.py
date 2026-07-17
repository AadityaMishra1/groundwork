"""Deterministic close-up eval of the grasp policy: N episodes, measured
success rate (object airborne >=2s with >=3 finger groups + thumb), and a
close-up video of the first few episodes.

    python -u demo_grasp.py --ckpt <model.pt> --headless --episodes 200
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--video_dir", type=str, default="/workspace/demo_videos")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# GRASP_NO_VIDEO=1: metrics-only eval for hosts without working Vulkan —
# with cameras enabled a failed renderer also kills the PhysX GPU solver
# (falls back to CPU: glacial and not physics-comparable to training)
import os as _os
NO_VIDEO = _os.environ.get("GRASP_NO_VIDEO", "0") == "1"
args.video = not NO_VIDEO
args.enable_cameras = not NO_VIDEO

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_tasks  # noqa: F401
import g1_grasp  # noqa: F401

import cli_args


def main():
    task = "G1-Grasp-v0"
    env_cfg = parse_env_cfg(task, num_envs=32)
    # close-up viewer: look at env 0's robot hand area
    env_cfg.viewer.eye = (1.3, -0.8, 0.7)
    env_cfg.viewer.lookat = (0.3, -0.1, 0.2)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task, args)

    env = gym.make(task, cfg=env_cfg,
                   render_mode=None if NO_VIDEO else "rgb_array")
    if not NO_VIDEO:
        env = gym.wrappers.RecordVideo(
            env, video_folder=args.video_dir, step_trigger=lambda s: s == 0,
            # short enough to finalize inside the eval — 800 never completed
            # for short hold-eval episodes, so no file was ever written
            video_length=150, name_prefix="grasp_demo", disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw = env.unwrapped
    n_envs = raw.num_envs
    done_count, success_count = 0, 0
    ep_success = torch.zeros(n_envs, dtype=torch.bool, device=raw.device)
    # failure taxonomy accumulators (per current episode)
    dev = raw.device
    got_near = torch.zeros(n_envs, dtype=torch.bool, device=dev)      # palm within 12cm
    got_contact = torch.zeros(n_envs, dtype=torch.bool, device=dev)   # any finger force
    got_grasp = torch.zeros(n_envs, dtype=torch.bool, device=dev)     # >=3 groups+thumb
    got_lift = torch.zeros(n_envs, dtype=torch.bool, device=dev)      # obj above 0.22
    max_objz = torch.zeros(n_envs, device=dev)
    ejected = torch.zeros(n_envs, dtype=torch.bool, device=dev)       # airborne w/o contact
    tax = {k: 0 for k in ["never_near", "near_no_contact", "contact_no_grasp",
                          "grasp_no_lift", "lift_no_hold", "success"]}
    eject_count = 0
    # hold-stage diagnosis: flicker vs genuine loss. cur/max = consecutive
    # (lifted & grasped) streak; tot = accumulated held steps; frag = number
    # of separate held fragments; loss_speed = |obj vel| when a streak breaks
    cur_streak = torch.zeros(n_envs, device=dev)
    max_streak = torch.zeros(n_envs, device=dev)
    tot_held = torch.zeros(n_envs, device=dev)
    frags = torch.zeros(n_envs, device=dev)
    loss_speeds = []
    hold_stats = []  # (max_streak, tot_held, frags) per lift-reaching episode
    # strict protocol bar (docs/EVAL_PROTOCOL.md): object >= 0.40 m with the
    # grasp held for 3 continuous seconds (150 steps at 50 Hz)
    strict_streak = torch.zeros(n_envs, device=dev)
    strict_success = torch.zeros(n_envs, dtype=torch.bool, device=dev)
    strict_count = 0

    obs, _ = env.get_observations(), None
    with torch.inference_mode():
        while done_count < args.episodes and simulation_app.is_running():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            ep_success |= raw.hold_counter >= 50
            palm = raw.robot.data.body_pos_w[:, raw.palm_id]
            objw = raw.obj.data.root_pos_w
            objz = objw[:, 2] - raw.scene.env_origins[:, 2]
            nf, th = raw._finger_contacts()
            got_near |= (palm - objw).norm(dim=-1) < 0.12
            got_contact |= nf >= 1
            got_grasp |= (nf >= 3) & th
            got_lift |= objz > 0.22
            max_objz = torch.maximum(max_objz, objz)
            ejected |= (objz > 0.30) & (nf == 0)
            held_now = (objz > 0.22) & (nf >= 3) & th
            broke = (~held_now) & (cur_streak > 0)
            if broke.any():
                bi = broke.nonzero(as_tuple=True)[0]
                loss_speeds += raw.obj.data.root_lin_vel_w[bi].norm(dim=-1).tolist()
                frags[bi] += 1
            cur_streak = torch.where(held_now, cur_streak + 1,
                                     torch.zeros_like(cur_streak))
            max_streak = torch.maximum(max_streak, cur_streak)
            tot_held += held_now.float()
            strict_now = (objz > 0.40) & (nf >= 3) & th
            strict_streak = torch.where(strict_now, strict_streak + 1,
                                        torch.zeros_like(strict_streak))
            strict_success |= strict_streak >= 150
            finished = dones.bool()
            if finished.any():
                idx = finished.nonzero(as_tuple=False).squeeze(-1)
                done_count += len(idx)
                success_count += int(ep_success[idx].sum())
                for i in idx.tolist():
                    if ep_success[i]:
                        tax["success"] += 1
                    elif got_lift[i]:
                        tax["lift_no_hold"] += 1
                    elif got_grasp[i]:
                        tax["grasp_no_lift"] += 1
                    elif got_contact[i]:
                        tax["contact_no_grasp"] += 1
                    elif got_near[i]:
                        tax["near_no_contact"] += 1
                    else:
                        tax["never_near"] += 1
                    eject_count += int(ejected[i])
                strict_count += int(strict_success[idx].sum())
                for i in idx.tolist():
                    if got_lift[i]:
                        hold_stats.append((float(max_streak[i]),
                                           float(tot_held[i]), float(frags[i])))
                for buf in (ep_success, got_near, got_contact, got_grasp,
                            got_lift, ejected, strict_success):
                    buf[idx] = False
                max_objz[idx] = 0.0
                for buf in (cur_streak, max_streak, tot_held, frags, strict_streak):
                    buf[idx] = 0.0

    rate = success_count / max(1, done_count)
    print(f"EVAL_RESULT episodes={done_count} successes={success_count} "
          f"success_rate={rate:.3f}")
    print("TAXONOMY " + " ".join(f"{k}={v}" for k, v in tax.items())
          + f" ejected={eject_count}")
    print(f"STRICT_RESULT successes={strict_count}/{done_count} "
          f"rate={strict_count / max(1, done_count):.3f} "
          f"(0.40m + 3s continuous hold)")
    if hold_stats:
        hs = torch.tensor(hold_stats)  # (N, 3): max_streak, tot_held, frags
        ms, tot, fr = hs[:, 0], hs[:, 1], hs[:, 2]
        # flicker signature: tot >> max_streak with many fragments
        q = lambda t, p: float(t.float().quantile(p))
        # a "carry" = >=5 consecutive held frames (0.1s) — separates real
        # lifts from throws that merely crossed the height threshold
        carried = int((ms >= 5).sum())
        print(f"CARRY n_lift_reached={len(hs)} carried={carried}")
        print(f"HOLD_DIAG n={len(hs)} "
              f"max_streak p25/50/75={q(ms,.25):.0f}/{q(ms,.5):.0f}/{q(ms,.75):.0f} "
              f"tot_held p25/50/75={q(tot,.25):.0f}/{q(tot,.5):.0f}/{q(tot,.75):.0f} "
              f"frags p50={q(fr,.5):.0f} "
              f"frac(tot>=50 but max<50)={float(((tot>=50)&(ms<50)).float().mean()):.3f}")
        if loss_speeds:
            ls = torch.tensor(loss_speeds)
            print(f"LOSS_SPEED n={len(ls)} p25/50/75="
                  f"{q(ls,.25):.2f}/{q(ls,.5):.2f}/{q(ls,.75):.2f} m/s "
                  f"frac>0.5={float((ls>0.5).float().mean()):.3f}")


if __name__ == "__main__":
    import os, sys
    main()
    sys.stdout.flush()
    os._exit(0)
