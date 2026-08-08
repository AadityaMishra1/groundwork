"""Does the M1i kneel survive its OWN freeze mechanism in the loco env?

Settle at 0.32 under policy control, then trigger the training freeze
machinery (all envs, 300 steps = 6 s) and watch pelvis z / uprightness.
Survives here + bows in the grasp env probe => env mismatch (gains),
not pose quality.

    LOCO_FREEZE=1 python -u g1_loco/probe_freeze_loco.py --ckpt <m1i.pt> --headless
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--out", type=str, default="")
parser.add_argument("--rounds", type=int, default=1)
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
    act_term = raw.action_manager.get_term("joint_pos")

    jps, tgts, rss = [], [], []
    with torch.inference_mode():
        for rnd in range(args.rounds):
            env.reset()
            obs, _ = env.get_observations(), None
            for step in range(300):  # settle at the kneel under policy control
                h_term.height_command[:, 0] = 0.32
                v_term.vel_command_b[:] = 0.0
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
            pz = robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
            print(f"FREEZE_PRE round={rnd} settled z p50={pz.median():.3f}")
            # trigger the training freeze machinery on every env
            frozen = robot.data.joint_pos[:, act_term._joint_ids].clone()
            raw.frozen_targets[:] = frozen
            raw.freeze_left[:] = 300
            p_ref = None
            for step in range(300):  # 6 s frozen
                h_term.height_command[:, 0] = 0.32
                v_term.vel_command_b[:] = 0.0
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                if step == 250:
                    p_ref = (robot.data.root_pos_w[:, :2]
                             - raw.scene.env_origins[:, :2]).clone()
                if step in (25, 150, 299):
                    pz = robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
                    q = robot.data.root_quat_w
                    up = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
                    print(f"FREEZE step={step} z p25/50/75="
                          f"{pz.quantile(0.25):.3f}/{pz.median():.3f}/{pz.quantile(0.75):.3f} "
                          f"up p50={up.median():.3f}")
            if not args.out:
                continue
            # capture frozen-settled equilibria: spawn state = settled pose,
            # PD state = the frozen targets that produced it. Stillness is
            # judged by DISPLACEMENT over the last second — instantaneous
            # root velocity reads 0.2-0.8 m/s of PD chatter on still robots.
            pz = robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
            q = robot.data.root_quat_w
            up = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
            disp = ((robot.data.root_pos_w[:, :2]
                     - raw.scene.env_origins[:, :2]) - p_ref).norm(dim=-1)
            print(f"FREEZE_DISP z p25/50={pz.quantile(0.25):.3f}/{pz.median():.3f} "
                  f"up p25/50={up.quantile(0.25):.3f}/{up.median():.3f} "
                  f"disp p10/25/50={disp.quantile(0.1):.4f}/{disp.quantile(0.25):.4f}/{disp.median():.4f}")
            good = (pz > 0.20) & (up > 0.6) & (disp < 0.05)
            print(f"FREEZE_KEPT round={rnd} {int(good.sum())}/64")
            if not good.any():
                continue
            jp = robot.data.joint_pos[good].clone()
            # full-width targets: measured pose everywhere, frozen targets on
            # the action joints (hands stay at their measured/parked pose)
            tg = robot.data.joint_pos[good].clone()
            tg[:, act_term._joint_ids] = frozen[good]
            rs = robot.data.root_state_w[good].clone()
            rs[:, :3] -= raw.scene.env_origins[good]
            yaw = torch.atan2(2 * (rs[:, 3] * rs[:, 6] + rs[:, 4] * rs[:, 5]),
                              1 - 2 * (rs[:, 5] ** 2 + rs[:, 6] ** 2))
            c, s = torch.cos(-yaw / 2), torch.sin(-yaw / 2)
            w, x, y, zz = rs[:, 3].clone(), rs[:, 4].clone(), rs[:, 5].clone(), rs[:, 6].clone()
            rs[:, 3] = c * w - s * zz
            rs[:, 4] = c * x - s * y
            rs[:, 5] = c * y + s * x
            rs[:, 6] = c * zz + s * w
            rs[:, :2] = 0.0
            rs[:, 7:] = 0.0
            jps.append(jp)
            tgts.append(tg)
            rss.append(rs)
    if args.out and jps:
        jp = torch.cat(jps)
        tg = torch.cat(tgts)
        rs = torch.cat(rss)
        torch.save({"joint_pos": jp.cpu(), "joint_targets": tg.cpu(),
                    "root_state": rs.cpu(),
                    "joint_names": list(robot.data.joint_names)}, args.out)
        print(f"FREEZE_SAVED n={len(jp)} -> {args.out} "
              f"z p25/50/75={rs[:, 2].quantile(0.25):.3f}/{rs[:, 2].median():.3f}/{rs[:, 2].quantile(0.75):.3f}")


if __name__ == "__main__":
    import os
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
