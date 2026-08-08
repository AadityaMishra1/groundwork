"""Diagnose the M1f spawn_at_random_height event: spawn a batch, step zero
actions, and report per-depth-bucket what actually happens — root z at spawn
vs after settling, which undesired bodies took contact force, and who died.

    python -u g1_loco/probe_spawn.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_tasks  # noqa: F401
import g1_loco  # noqa: F401


def main():
    task = "G1-LocoKneel-v0"
    env_cfg = parse_env_cfg(task, num_envs=64)
    env = gym.make(task, cfg=env_cfg)
    raw = env.unwrapped
    robot = raw.scene["robot"]
    sensor = raw.scene.sensors["contact_forces"]
    names = sensor.body_names

    env.reset()
    z0 = robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
    acts_base = torch.zeros(raw.num_envs, raw.action_manager.total_action_dim,
                       device=raw.device)
    torch.manual_seed(0)
    died = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
    first_hit = [""] * raw.num_envs
    for step in range(25):
        acts = acts_base + torch.randn_like(acts_base)  # fresh-policy flail
        _, _, term, trunc, _ = env.step(acts)
        f = sensor.data.net_forces_w.norm(dim=-1)
        for i in range(raw.num_envs):
            if not first_hit[i]:
                hits = [names[j] for j in range(len(names))
                        if f[i, j] > 5.0 and "ankle_roll" not in names[j]]
                if hits:
                    first_hit[i] = ",".join(hits[:3])
        died |= term.bool()
        if step == 0:
            z1 = robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
    z25 = robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]

    order = torch.argsort(z0)
    for k in range(0, 64, 8):
        i = order[k].item()
        print(f"PROBE z0={z0[i]:.3f} z_step1={z1[i]:.3f} z_step25={z25[i]:.3f} "
              f"died={bool(died[i])} first_nonfoot_contact=[{first_hit[i]}]")
    deep = z0 < 0.5
    print(f"DEEP died={int((died & deep).sum())}/{int(deep.sum())}  SHALLOW died={int((died & ~deep).sum())}/{int((~deep).sum())}")
    print(f"SUMMARY died={int(died.sum())}/64 "
          f"z0 range {z0.min():.3f}-{z0.max():.3f}")


if __name__ == "__main__":
    import os, sys
    main()
    sys.stdout.flush()
    os._exit(0)
