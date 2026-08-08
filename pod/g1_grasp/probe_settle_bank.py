"""Mine the POST-SINK equilibrium bank: spawn from the LocoKneel arrival bank
with crouch PD targets (GRASP_PARK_CROUCH=1), let each env settle, and save
the settled states in their own yaw-zero frame. These are the poses the grasp
policy actually operates from once the handoff sink finishes.

    GRASP_FREE=1 GRASP_FORCE_STAGE=3 GRASP_LEG_STIFF=4.0 \
    GRASP_START_BANK=/workspace/handoff_poses.pt GRASP_PARK_CROUCH=1 \
    python -u g1_grasp/probe_settle_bank.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default=f"{_WS}/settled_poses.pt")
parser.add_argument("--rounds", type=int, default=6)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import g1_grasp  # noqa: F401

import os as _os
_WS = _os.environ.get("GW_ROOT", "/workspace")


def main():
    env_cfg = parse_env_cfg("G1-Grasp-v0", num_envs=64)
    env = gym.make("G1-Grasp-v0", cfg=env_cfg)
    raw = env.unwrapped
    robot = raw.robot
    org = raw.scene.env_origins
    acts = torch.zeros(raw.num_envs, raw.cfg.action_space, device=raw.device)
    jps, rss = [], []
    for rnd in range(args.rounds):
        env.reset()
        for _ in range(200):  # 4 s settle
            env.step(acts)
        z = robot.data.root_pos_w[:, 2] - org[:, 2]
        q = robot.data.root_quat_w
        up = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        v = robot.data.root_lin_vel_w.norm(dim=-1)
        good = (z > 0.12) & (up > 0.5) & (v < 0.05)
        print(f"SETTLE_COND round={rnd} z>0.12: {int((z>0.12).sum())}/64 "
              f"up>0.5: {int((up>0.5).sum())}/64 v<0.05: {int((v<0.05).sum())}/64 "
              f"z p25/50={z.quantile(0.25):.3f}/{z.median():.3f} "
              f"up p25/50={up.quantile(0.25):.3f}/{up.median():.3f} "
              f"v p50/75={v.median():.3f}/{v.quantile(0.75):.3f}")
        if not good.any():
            continue
        jp = robot.data.joint_pos[good].clone()
        rs = robot.data.root_state_w[good].clone()
        rs[:, :3] -= org[good]
        # rotate the root into its own yaw-zero frame
        qg = rs[:, 3:7]
        yaw = torch.atan2(2 * (qg[:, 0] * qg[:, 3] + qg[:, 1] * qg[:, 2]),
                          1 - 2 * (qg[:, 2] ** 2 + qg[:, 3] ** 2))
        c, s = torch.cos(-yaw / 2), torch.sin(-yaw / 2)
        w, x, y, zz = qg[:, 0], qg[:, 1], qg[:, 2], qg[:, 3]
        rs[:, 3] = c * w - s * zz
        rs[:, 4] = c * x - s * y
        rs[:, 5] = c * y + s * x
        rs[:, 6] = c * zz + s * w
        rs[:, 7:] = 0.0
        rs[:, :2] = 0.0  # settled xy becomes the new origin
        jps.append(jp)
        rss.append(rs)
        print(f"SETTLE round={rnd} kept={int(good.sum())}/64 "
              f"z p50={z[good].median():.3f} up p50={up[good].median():.3f}")
    jp = torch.cat(jps)
    rs = torch.cat(rss)
    torch.save({"joint_pos": jp.cpu(), "root_state": rs.cpu(),
                "joint_names": list(robot.data.joint_names)}, args.out)
    print(f"SETTLE_SAVED n={len(jp)} -> {args.out} "
          f"pelvis_z p25/50/75={rs[:,2].quantile(0.25):.3f}/{rs[:,2].median():.3f}/{rs[:,2].quantile(0.75):.3f}")


if __name__ == "__main__":
    import os, sys
    main()
    sys.stdout.flush()
    os._exit(0)
