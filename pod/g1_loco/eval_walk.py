"""Can the LocoKneel checkpoint actually WALK, in its own env?

eval_kneel forces zero velocity for every target, so forward locomotion has
never been measured for this checkpoint. Without that number an embedding
failure ("the grasp env is wrong") is indistinguishable from a checkpoint
that simply never learned to walk. This forces a fixed forward command on
the native Play env and measures real displacement.

    python -u g1_loco/eval_walk.py --ckpt <model.pt> --headless
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--vx", type=float, default=0.5)
parser.add_argument("--tgt_h", type=float, default=0.72)
parser.add_argument("--steps", type=int, default=300)
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
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=agent_cfg.device)
    runner.load(args.ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw = env.unwrapped
    robot = raw.scene["robot"]
    h_term = raw.command_manager.get_term("base_height")
    v_term = raw.command_manager.get_term("base_velocity")

    obs, _ = env.get_observations(), None
    p0 = (robot.data.root_pos_w[:, :2] - raw.scene.env_origins[:, :2]).clone()
    fallen = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)

    with torch.inference_mode():
        for step in range(args.steps):
            h_term.height_command[:, 0] = args.tgt_h
            v_term.vel_command_b[:, 0] = args.vx
            v_term.vel_command_b[:, 1] = 0.0
            v_term.vel_command_b[:, 2] = 0.0
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            # a reset teleports the env home; rebase so it never reads as travel
            if dones.any():
                fallen |= dones.bool()
                cur = (robot.data.root_pos_w[:, :2]
                       - raw.scene.env_origins[:, :2])
                p0 = torch.where(dones.bool().unsqueeze(-1), cur, p0)
            if step in (100, 200, args.steps - 1):
                d = (robot.data.root_pos_w[:, :2]
                     - raw.scene.env_origins[:, :2] - p0).norm(dim=-1)
                z = robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
                q = robot.data.root_quat_w
                up = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
                print(f"NATIVE_WALK step={step} disp_p25/50/75="
                      f"{d.quantile(0.25):.3f}/{d.median():.3f}/"
                      f"{d.quantile(0.75):.3f} z_p50={z.median():.3f} "
                      f"up_p50={up.median():.3f} "
                      f"fell={int(fallen.sum())}/{raw.num_envs}", flush=True)

    d = (robot.data.root_pos_w[:, :2] - raw.scene.env_origins[:, :2] - p0).norm(dim=-1)
    secs = args.steps * 4 / 200.0
    print(f"NATIVE_WALK_FINAL vx_cmd={args.vx} disp_p50={d.median():.3f}m "
          f"over {secs:.1f}s => {d.median() / secs:.3f} m/s "
          f"fell={int(fallen.sum())}/{raw.num_envs}")


if __name__ == "__main__":
    import os
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
