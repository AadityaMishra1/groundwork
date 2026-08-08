"""Where does the robot END UP after the arrival->crouch sink?

Run with GRASP_FREE=1 GRASP_FORCE_STAGE=3 GRASP_START_BANK=... GRASP_PARK_CROUCH=1.
Measures root xy translation, yaw change, and the palm position in the
SETTLED frame vs the object corridor the env actually spawns into.
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
    org = raw.scene.env_origins
    p0 = robot.data.root_pos_w[:, :2] - org[:, :2]
    q0 = robot.data.root_quat_w.clone()
    yaw0 = torch.atan2(2 * (q0[:, 0] * q0[:, 3] + q0[:, 1] * q0[:, 2]),
                       1 - 2 * (q0[:, 2] ** 2 + q0[:, 3] ** 2))
    acts = torch.zeros(raw.num_envs, raw.cfg.action_space, device=raw.device)
    for _ in range(200):  # 4 s settle
        env.step(acts)
    p1 = robot.data.root_pos_w[:, :2] - org[:, :2]
    q1 = robot.data.root_quat_w
    yaw1 = torch.atan2(2 * (q1[:, 0] * q1[:, 3] + q1[:, 1] * q1[:, 2]),
                       1 - 2 * (q1[:, 2] ** 2 + q1[:, 3] ** 2))
    drift = (p1 - p0).norm(dim=-1)
    dyaw = torch.atan2(torch.sin(yaw1 - yaw0), torch.cos(yaw1 - yaw0))
    palm = robot.data.body_pos_w[:, raw.palm_id] - org
    # object corridor center in env frame: r~0.285, th~-0.75
    obj = raw.obj.data.root_pos_w[:, :2] - org[:, :2]
    palm_obj = (palm[:, :2] - obj).norm(dim=-1)
    q = lambda t, p: float(t.quantile(p))
    print(f"SINK_DRIFT xy p50/p95={q(drift,.5):.3f}/{q(drift,.95):.3f} m "
          f"dyaw p50/p95={q(dyaw.abs(),.5):.2f}/{q(dyaw.abs(),.95):.2f} rad")
    print(f"SINK_PALM z p25/50/75={q(palm[:,2],.25):.3f}/{q(palm[:,2],.5):.3f}/{q(palm[:,2],.75):.3f} "
          f"palm_to_obj_xy p25/50/75={q(palm_obj,.25):.3f}/{q(palm_obj,.5):.3f}/{q(palm_obj,.75):.3f}")


if __name__ == "__main__":
    import os, sys
    main()
    sys.stdout.flush()
    os._exit(0)
