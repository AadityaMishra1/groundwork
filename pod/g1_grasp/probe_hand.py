"""Measure the Inspire right hand's grasp geometry in the crouch pose:
palm frame axes, finger joint limits/signs, and the finger-centroid offset
from the palm when the hand closes. Output feeds the reverse-curriculum
in-hand reset. Runs a tiny env; safe next to a training job.

    python -u probe_hand.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from g1_grasp.grasp_env import (
    ACTION_JOINTS, FINGER_BODIES, PALM_BODY, G1GraspEnv, G1GraspEnvCfg,
)

FINGER_JOINTS = ACTION_JOINTS[8:]  # 12 hand joints


def main():
    cfg = G1GraspEnvCfg()
    cfg.scene.num_envs = 4
    cfg.scene.env_spacing = 3.0
    env = G1GraspEnv(cfg)
    env.reset()
    robot = env.robot

    fj_ids, fj_names = robot.find_joints(FINGER_JOINTS, preserve_order=True)
    limits = robot.data.joint_pos_limits[0, fj_ids]
    print("=== finger joint limits (lo, hi) ===")
    for n, (lo, hi) in zip(fj_names, limits.tolist()):
        print(f"  {n}: ({lo:.3f}, {hi:.3f})")

    finger_ids, _ = robot.find_bodies(FINGER_BODIES, preserve_order=True)

    zero = torch.zeros(cfg.scene.num_envs, len(ACTION_JOINTS), device=env.device)

    def settle(actions, steps=120):
        for _ in range(steps):
            env._pre_physics_step(actions)
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)

    def report(tag):
        palm_p = robot.data.body_pos_w[0, env.palm_id] - env.scene.env_origins[0]
        palm_q = robot.data.body_quat_w[0, env.palm_id]
        cent = (robot.data.body_pos_w[0, finger_ids].mean(dim=0)
                - env.scene.env_origins[0])
        print(f"[{tag}] palm_pos={palm_p.tolist()}")
        print(f"[{tag}] palm_quat={palm_q.tolist()}")
        print(f"[{tag}] finger_centroid={cent.tolist()}")
        print(f"[{tag}] centroid-palm={ (cent - palm_p).tolist()}")

    settle(zero)
    report("open")

    # close: drive all 12 hand actions to +1 then -1, see which flexes
    for direction in (1.0, -1.0):
        a = zero.clone()
        a[:, 8:] = direction
        settle(a)
        q = robot.data.joint_pos[0, fj_ids]
        print(f"=== hand action {direction:+.0f}: joint positions ===")
        for n, v in zip(fj_names, q.tolist()):
            print(f"  {n}: {v:.3f}")
        report(f"closed{direction:+.0f}")

    print("PROBE_DONE")


if __name__ == "__main__":
    import os, sys
    main()
    sys.stdout.flush()
    os._exit(0)
