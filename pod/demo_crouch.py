"""Deterministic demo of the crouch policy: walk -> stop -> deep crouch ->
hold -> stand. Records a video. Commands are scripted, actions are the
policy mean (no exploration noise).

    python demo_crouch.py --checkpoint <model.pt> --headless --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--video_dir", type=str, default="/workspace/demo_videos")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.video = True  # force offscreen render pipeline
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.hydra import *  # noqa: F401,F403  (keeps parity with play.py imports)

import isaaclab_tasks  # noqa: F401
import g1_crouch  # noqa: F401

import cli_args


def main():
    task = "G1-Crouch-Play-v0"
    env_cfg = parse_env_cfg(task, num_envs=9)
    env_cfg.episode_length_s = 30.0
    agent_cfg = cli_args.parse_rsl_rl_cfg(task, args)

    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env, video_folder=args.video_dir, step_trigger=lambda s: s == 0,
        video_length=1100, name_prefix="crouch_demo", disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    vel_term = env.unwrapped.command_manager.get_term("base_velocity")
    h_term = env.unwrapped.command_manager.get_term("base_height")

    # (duration_steps, vx, vy, wz, height)
    script = [
        (250, 0.6, 0.0, 0.0, 0.72),   # walk forward tall
        (100, 0.0, 0.0, 0.0, 0.72),   # stop
        (200, 0.0, 0.0, 0.0, 0.40),   # crouch deep
        (150, 0.0, 0.0, 0.0, 0.40),   # hold the crouch
        (200, 0.0, 0.0, 0.0, 0.72),   # stand back up
        (200, 0.6, 0.0, 0.0, 0.72),   # walk away tall
    ]

    obs, _ = env.get_observations(), None
    step = 0
    phase_idx, phase_left = 0, script[0][0]
    with torch.inference_mode():
        while simulation_app.is_running() and step < 1150:
            d, vx, vy, wz, h = script[phase_idx]
            vel_term.vel_command_b[:, 0] = vx
            vel_term.vel_command_b[:, 1] = vy
            vel_term.vel_command_b[:, 2] = wz
            h_term.height_command[:, 0] = h
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            step += 1
            phase_left -= 1
            if phase_left <= 0 and phase_idx < len(script) - 1:
                phase_idx += 1
                phase_left = script[phase_idx][0]
    print("DEMO_DONE")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
