"""How much does the base carry the hand around during a 3s hold?

The grasp policy earned 35.5% from a static base. Under split control the
walker balances live for the whole episode, so the palm is in motion the
entire time the hand is supposed to be holding. This measures that motion
directly, with the walker live vs latched after a settle window.

    GRASP_FREE=1 GRASP_FORCE_STAGE=3 GRASP_LEG_STIFF=1.0 \
    GRASP_START_BANK=... GRASP_ARRIVAL_FRAC=1.0 GRASP_LOCO_CKPT=... \
    GRASP_LOCO_SETTLE=100 GRASP_LOCO_PARK=1 \
    python -u g1_grasp/probe_park.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import os

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import g1_grasp  # noqa: F401

SETTLE = int(os.environ.get("GRASP_LOCO_SETTLE", "100"))
HOLD = 150  # 3s at 50Hz — the strict-protocol hold duration


def main():
    env_cfg = parse_env_cfg("G1-Grasp-v0", num_envs=64)
    env = gym.make("G1-Grasp-v0", cfg=env_cfg)
    raw = env.unwrapped
    env.reset()
    robot = raw.robot
    acts = torch.zeros(raw.num_envs, raw.cfg.action_space, device=raw.device)

    palms, zs = [], []
    fell = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
    for step in range(SETTLE + HOLD):
        env.step(acts)
        z = robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
        q = robot.data.root_quat_w
        up = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        fell |= (z < 0.10) | (up < 0.0)
        if step >= SETTLE:
            palms.append(robot.data.body_pos_w[:, raw.palm_id].clone())
            zs.append(z.clone())

    P = torch.stack(palms)                      # (T, N, 3)
    drift = (P - P.mean(dim=0, keepdim=True)).norm(dim=-1)   # (T, N)
    excursion = drift.max(dim=0).values          # worst offset per env, metres
    span = (P.max(dim=0).values - P.min(dim=0).values).norm(dim=-1)
    Z = torch.stack(zs)
    zspan = Z.max(dim=0).values - Z.min(dim=0).values

    park = os.environ.get("GRASP_LOCO_PARK", "0")
    print(f"PARK_MODE park={park} settle={SETTLE} hold={HOLD}")
    print(f"PALM_EXCURSION_M p50/p75/p95="
          f"{excursion.median():.4f}/{excursion.quantile(0.75):.4f}/"
          f"{excursion.quantile(0.95):.4f}")
    print(f"PALM_SPAN_M p50/p95={span.median():.4f}/{span.quantile(0.95):.4f}")
    print(f"PELVIS_Z_SPAN_M p50/p95={zspan.median():.4f}/{zspan.quantile(0.95):.4f}")
    print(f"PELVIS_Z_FINAL p25/50/75={Z[-1].quantile(0.25):.3f}/"
          f"{Z[-1].median():.3f}/{Z[-1].quantile(0.75):.3f} "
          f"fallen={int(fell.sum())}/64")


if __name__ == "__main__":
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
