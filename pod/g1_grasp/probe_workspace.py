"""Random-search the reachable workspace at several pelvis heights at once.

4 envs = 4 pelvis heights. Sample random full-authority arm/hand actions,
settle each briefly, record the lowest finger-centroid per env (with its
xy), plus the best pose's action vector for the lowest env. One run answers
whether floor reach is feasible and at which crouch height.

    PYTHONPATH=/workspace/humanoid/pod python -u g1_grasp/probe_workspace.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from g1_grasp.grasp_env import ACTION_JOINTS, FINGER_BODIES, G1GraspEnv, G1GraspEnvCfg

HEIGHTS = [0.55, 0.48, 0.42, 0.36]
N_SAMPLES = 150
SETTLE_STEPS = 60  # 1.2 s


def main():
    cfg = G1GraspEnvCfg()
    cfg.scene.num_envs = len(HEIGHTS)
    cfg.scene.env_spacing = 4.0
    env = G1GraspEnv(cfg)
    env.reset()
    robot = env.robot

    # set per-env pelvis heights (root is fixed; write once)
    root = robot.data.default_root_state.clone()
    for i, h in enumerate(HEIGHTS):
        root[i, 2] = h
    root[:, :3] += env.scene.env_origins
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])

    finger_ids, _ = robot.find_bodies(FINGER_BODIES, preserve_order=True)
    n_act = len(ACTION_JOINTS)
    best_z = torch.full((len(HEIGHTS),), 99.0)
    best_xy = torch.zeros(len(HEIGHTS), 2)
    best_act = torch.zeros(len(HEIGHTS), n_act)

    gen = torch.Generator(device=env.device).manual_seed(0)
    for s in range(N_SAMPLES):
        a = 2 * torch.rand(cfg.scene.num_envs, n_act, generator=gen, device=env.device) - 1
        # bias hand open during reach probing; fingers don't affect reach much
        a[:, 8:] = -0.8
        for _ in range(SETTLE_STEPS):
            env._pre_physics_step(a)
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
        cent = robot.data.body_pos_w[:, finger_ids].mean(dim=1) - env.scene.env_origins
        # require the hand to be in front of the robot where objects spawn
        valid = (cent[:, 0] > 0.15) & (cent[:, 0] < 0.55)
        z = torch.where(valid, cent[:, 2], torch.full_like(cent[:, 2], 99.0)).cpu()
        for i in range(len(HEIGHTS)):
            if z[i] < best_z[i]:
                best_z[i] = z[i]
                best_xy[i] = cent[i, :2].cpu()
                best_act[i] = a[i].cpu()

    print("=== workspace floor-reach by pelvis height ===")
    for i, h in enumerate(HEIGHTS):
        print(f"  pelvis={h:.2f}: min_z={best_z[i]:.3f} at "
              f"xy=({best_xy[i,0]:.3f},{best_xy[i,1]:.3f})")
    # dump the best action vector for the deepest-reaching feasible height
    feas = [i for i in range(len(HEIGHTS)) if best_z[i] < 0.12]
    pick = feas[0] if feas else int(best_z.argmin())
    print(f"BEST_HEIGHT={HEIGHTS[pick]}")
    print("BEST_ACTIONS=" + ",".join(f"{v:.3f}" for v in best_act[pick].tolist()))
    print("PROBE_DONE")


if __name__ == "__main__":
    import os
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
