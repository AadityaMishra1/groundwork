"""Verify the arm can reach floor objects with full joint authority.
Sweeps arm-down poses, reports finger-centroid position for each.
    PYTHONPATH=/workspace/humanoid/pod python -u g1_grasp/probe_reach.py --headless
"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from g1_grasp.grasp_env import ACTION_JOINTS, FINGER_BODIES, G1GraspEnv, G1GraspEnvCfg

def main():
    cfg = G1GraspEnvCfg()
    cfg.scene.num_envs = 4
    env = G1GraspEnv(cfg)
    env.reset()
    robot = env.robot
    finger_ids, _ = robot.find_bodies(FINGER_BODIES, preserve_order=True)
    names = ACTION_JOINTS
    wp, sp, sr, el = (names.index("waist_pitch_joint"),
                      names.index("right_shoulder_pitch_joint"),
                      names.index("right_shoulder_roll_joint"),
                      names.index("right_elbow_joint"))

    def try_pose(vals, steps=180):
        a = torch.zeros(cfg.scene.num_envs, len(names), device=env.device)
        for k, v in vals.items():
            a[:, k] = v
        for _ in range(steps):
            env._pre_physics_step(a)
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
        cent = (robot.data.body_pos_w[0, finger_ids].mean(dim=0) - env.scene.env_origins[0])
        print(f"  wp={vals.get(wp,0):+.2f} sp={vals.get(sp,0):+.2f} el={vals.get(el,0):+.2f}"
              f" -> centroid=({cent[0]:.3f}, {cent[1]:.3f}, {cent[2]:.3f})")

    print("=== reach sweep (finger centroid, env frame) ===")
    for wpv in (0.5, -0.5):
        for spv in (-0.3, -0.6, -0.9):
            for elv in (-0.7, -0.3, 0.2):
                try_pose({wp: wpv, sp: spv, el: elv})
    print("PROBE_DONE")

if __name__ == "__main__":
    import os, sys
    main(); sys.stdout.flush(); os._exit(0)
