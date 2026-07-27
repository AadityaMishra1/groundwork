"""Kneel-quality eval: what depth can the LocoKneel policy actually HOLD?

Forces fixed commands (zero velocity + a target height) on the Play env and
measures the settled outcome per commanded depth: achieved pelvis height,
uprightness, and fall rate. The aggregate training height_error mixes
transitions with holds — this isolates the question that matters for the
composition handoff: can it reach and keep the grasp stance?

    python -u g1_loco/eval_kneel.py --ckpt <model.pt> --headless
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
# P2 step 0 (PLAN_2026-07-25): the certified pregrasp dwell is pelvis
# 0.387-0.396 — sweep hold quality there before aiming the chain at it
parser.add_argument("--targets", type=str, default="0.72,0.45,0.32,0.26",
                    help="comma-separated height commands to hold")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_tasks  # noqa: F401
import g1_loco  # noqa: F401

import cli_args


def main():
    task = "G1-LocoKneel-Play-v0"
    env_cfg = parse_env_cfg(task, num_envs=64)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task, args)
    env = gym.make(task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw = env.unwrapped
    robot = raw.scene["robot"]
    h_term = raw.command_manager.get_term("base_height")
    v_term = raw.command_manager.get_term("base_velocity")

    import os as _os
    if _os.environ.get("KNEEL_NO_OVERRIDE", "0") == "1":
        # discriminator: natural random commands — if heights look like
        # training here, the override path is the bug, not the policy
        obs, _ = env.get_observations(), None
        with torch.inference_mode():
            hs = []
            for step in range(400):
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                pz = robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
                hs.append(pz.clone())
        h = torch.stack(hs)
        print(f"NO_OVERRIDE height p25/50/75="
              f"{h.quantile(0.25):.3f}/{h.quantile(0.5):.3f}/{h.quantile(0.75):.3f} "
              f"final_mean={h[-1].mean():.3f}")
        return

    for target in [float(t) for t in args.targets.split(",")]:
        obs, _ = env.get_observations(), None
        fallen = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
        heights = []
        with torch.inference_mode():
            for step in range(500):  # 10 s
                h_term.height_command[:, 0] = target
                v_term.vel_command_b[:] = 0.0
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                # fall = STATE, never dones: dones include the 20 s episode
                # TRUNCATION, so with 10 s hold windows every second target
                # absorbed a mass timeout and read fell=62/64 while tracking
                # perfectly (2026-07-25 comb pattern; the recorded "0.45 weak
                # band" was very likely this same artifact).
                pz_now = robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
                q_now = robot.data.root_quat_w
                up_now = 1.0 - 2.0 * (q_now[:, 1] ** 2 + q_now[:, 2] ** 2)
                fallen |= (pz_now < 0.15) | (up_now < 0.3)
                if step >= 350:  # settled window: last 3 s
                    pz = robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
                    heights.append(pz.clone())
        h = torch.stack(heights)
        hm = h.mean(dim=0)
        q = robot.data.root_quat_w
        up = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        ok = (~fallen) & ((hm - target).abs() < 0.08) & (up > 0.5)
        print(f"KNEEL_EVAL target={target:.2f} "
              f"achieved p25/50/75={hm.quantile(0.25):.3f}/{hm.quantile(0.5):.3f}/{hm.quantile(0.75):.3f} "
              f"fell={int(fallen.sum())}/{raw.num_envs} "
              f"ok(within8cm+upright)={int(ok.sum())}/{raw.num_envs}")


if __name__ == "__main__":
    import os
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
