"""Is the free-root grasp crouch statically stable? Zero actions (delta
space: zero = hold targets), GRASP_FREE=1, measure pelvis z and falls.

    GRASP_FREE=1 GRASP_FORCE_STAGE=3 python -u g1_grasp/probe_stance.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import g1_grasp  # noqa: F401


def main():
    env_cfg = parse_env_cfg("G1-Grasp-v0", num_envs=64)
    env = gym.make("G1-Grasp-v0", cfg=env_cfg)
    raw = env.unwrapped
    env.reset()
    robot = raw.robot
    z0 = robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
    acts = torch.zeros(raw.num_envs, raw.cfg.action_space, device=raw.device)
    fell = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
    for step in range(250):
        env.step(acts)
        z = robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
        q = robot.data.root_quat_w
        up = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        fell |= (z < 0.10) | (up < 0.0)
        if step in (5, 20, 60, 249):
            print(f"STANCE step={step} pelvis_z p25/50/75="
                  f"{z.quantile(0.25):.3f}/{z.median():.3f}/{z.quantile(0.75):.3f} "
                  f"up p50={up.median():.3f} fallen={int(fell.sum())}/64")
    tips = robot.data.body_pos_w[:, raw.tip_ids]
    cz = tips.mean(dim=1)[:, 2] - raw.scene.env_origins[:, 2]
    zf = robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
    print(f"STANCE_FINAL spawn_z p50={z0.median():.3f} settled_z p50={zf.median():.3f} "
          f"tip_z p50={cz.median():.3f} fallen={int(fell.sum())}/64")


if __name__ == "__main__":
    import os, sys
    main()
    sys.stdout.flush()
    os._exit(0)
