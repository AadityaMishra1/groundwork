"""M3 handoff probe: what kneel pose does LocoKneel actually ARRIVE in?

Commands 0.32 m (the composition dwell), lets the policy settle, and captures
the joint pose + root state of every upright on-target env at several settle
snapshots. Saves the arrival-pose bank for grasp-side fine-tuning and prints
the gap to the grasp env's hand-authored CROUCH_LEGS.

    python -u g1_loco/probe_handoff.py --ckpt <locokneel model.pt> --headless
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--out", type=str, default=f"{_WS}/handoff_poses.pt")
# PLAN_2026-07-25 P0: the composition dwell moved from the (impossible) deep
# kneel to the certified shallow squat band — capture arrivals at 0.50 too.
parser.add_argument("--target", type=float, default=0.32,
                    help="height command to settle at (chain bank: 0.50)")
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

import os as _os
_WS = _os.environ.get("GW_ROOT", "/workspace")

CROUCH_LEGS = {
    ".*_hip_pitch_joint": -2.2, ".*_knee_joint": 2.8, ".*_ankle_pitch_joint": -0.87,
    ".*_hip_roll_joint": 0.4, ".*_hip_yaw_joint": 0.0, ".*_ankle_roll_joint": 0.0,
}
TARGET = args.target


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

    poses, roots = [], []
    obs, _ = env.get_observations(), None
    fallen = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
    with torch.inference_mode():
        for step in range(500):
            h_term.height_command[:, 0] = TARGET
            v_term.vel_command_b[:] = 0.0
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            fallen |= dones.bool()
            # settled window: snapshot every 0.5 s over the last 3 s
            if step >= 350 and step % 25 == 0:
                pz = robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
                q = robot.data.root_quat_w
                up = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
                if step == 350:
                    print(f"HANDOFF_RAW z p10/25/50/75/90="
                          f"{pz.quantile(0.10):.3f}/{pz.quantile(0.25):.3f}/"
                          f"{pz.median():.3f}/{pz.quantile(0.75):.3f}/{pz.quantile(0.90):.3f} "
                          f"up p25/50={up.quantile(0.25):.3f}/{up.median():.3f} "
                          f"fallen={int(fallen.sum())}/64")
                good = (~fallen) & ((pz - TARGET).abs() < 0.15) & (up > 0.6)
                if good.any():
                    poses.append(robot.data.joint_pos[good].clone())
                    rs = robot.data.root_state_w[good].clone()
                    rs[:, :3] -= raw.scene.env_origins[good]
                    roots.append(rs)

    if not poses:
        print("HANDOFF no settled poses captured — policy never held 0.32")
        return
    jp = torch.cat(poses)
    rs = torch.cat(roots)
    print(f"HANDOFF captured n={len(jp)} pelvis_z p25/50/75="
          f"{rs[:, 2].quantile(0.25):.3f}/{rs[:, 2].median():.3f}/{rs[:, 2].quantile(0.75):.3f}")
    names = robot.data.joint_names
    for pattern, val in CROUCH_LEGS.items():
        ids, _ = robot.find_joints(pattern)
        got = jp[:, ids].mean(dim=0)
        print(f"HANDOFF_GAP {pattern:26s} crouch={val:+.2f} "
              f"arrival={' '.join(f'{v:+.2f}' for v in got.tolist())} "
              f"maxgap={float((got - val).abs().max()):.2f}")
    torch.save({"joint_pos": jp.cpu(), "root_state": rs.cpu(),
                "joint_names": list(names)}, args.out)
    print(f"HANDOFF_SAVED {args.out}")


if __name__ == "__main__":
    import os
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
