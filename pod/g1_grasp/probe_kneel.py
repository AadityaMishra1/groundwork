"""Probe: does a kneeling / folded-leg stance put protocol-spec objects
(10-25 cm) inside the finger workspace?

Round-1 finding: pelvis 0.36 (deepest squat) -> finger centroid bottoms out
at z~0.25, so only a 35 cm bottle was graspable. Hypothesis: folding the legs
(kneel / seiza) lets the fixed pelvis sit far lower, dropping the whole arm
workspace by the same amount.

4 envs = 4 pelvis heights with legs folded to joint limits. Random-search
full-authority arm actions (waist pitch included), settle, record min finger
centroid z in the forward annulus where objects spawn, plus the best action
vector per height (seed for scripted demos later).

    PYTHONPATH=/workspace/humanoid/pod python -u g1_grasp/probe_kneel.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from g1_grasp.grasp_env import ACTION_JOINTS, FINGER_BODIES, G1GraspEnv, G1GraspEnvCfg

HEIGHTS = [0.30, 0.25, 0.20, 0.16]
N_SAMPLES = 200
SETTLE_STEPS = 60  # 1.2 s

# fold the legs as far as limits allow (kneel/seiza-like); clamped to the
# asset's actual joint limits at runtime, so values here can over-ask
FOLD_LEGS = {
    ".*_hip_pitch_joint": -2.2, ".*_knee_joint": 2.8, ".*_ankle_pitch_joint": -0.9,
    ".*_hip_roll_joint": 0.4, ".*_hip_yaw_joint": 0.0, ".*_ankle_roll_joint": 0.0,
}


def main():
    cfg = G1GraspEnvCfg()
    cfg.scene.num_envs = len(HEIGHTS)
    cfg.scene.env_spacing = 4.0
    env = G1GraspEnv(cfg)

    # print actual leg limits once, then bake folded legs into defaults
    for pattern, val in FOLD_LEGS.items():
        ids, names = env.robot.find_joints(pattern)
        lim = env.robot.data.joint_pos_limits[0, ids]
        clamped = torch.clamp(torch.full((len(ids),), val, device=env.device),
                              lim[:, 0], lim[:, 1])
        env.robot.data.default_joint_pos[:, ids] = clamped
        print(f"  {pattern}: ask={val} limits={lim[0].tolist()} -> {clamped[0].item():.2f}")

    env.reset()
    robot = env.robot

    root = robot.data.default_root_state.clone()
    for i, h in enumerate(HEIGHTS):
        root[i, 2] = h
    root[:, :3] += env.scene.env_origins
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    # legs to folded pose
    jp = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))

    finger_ids, _ = robot.find_bodies(FINGER_BODIES, preserve_order=True)
    n_act = len(ACTION_JOINTS)
    best_z = torch.full((len(HEIGHTS),), 99.0)
    best_xy = torch.zeros(len(HEIGHTS), 2)
    best_act = torch.zeros(len(HEIGHTS), n_act)

    gen = torch.Generator(device=env.device).manual_seed(0)
    for s in range(N_SAMPLES):
        a = 2 * torch.rand(cfg.scene.num_envs, n_act, generator=gen, device=env.device) - 1
        a[:, 8:] = -0.8  # hand open while probing reach
        for _ in range(SETTLE_STEPS):
            env._pre_physics_step(a)
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
        cent = robot.data.body_pos_w[:, finger_ids].mean(dim=1) - env.scene.env_origins
        valid = (cent[:, 0] > 0.15) & (cent[:, 0] < 0.55)
        z = torch.where(valid, cent[:, 2], torch.full_like(cent[:, 2], 99.0)).cpu()
        for i in range(len(HEIGHTS)):
            if z[i] < best_z[i]:
                best_z[i] = z[i]
                best_xy[i] = cent[i, :2].cpu()
                best_act[i] = a[i].cpu()
        if (s + 1) % 50 == 0:
            print(f"  sample {s+1}/{N_SAMPLES}: " +
                  " ".join(f"h{h}={best_z[i]:.3f}" for i, h in enumerate(HEIGHTS)))

    print("=== kneel workspace floor-reach by pelvis height ===")
    for i, h in enumerate(HEIGHTS):
        print(f"  pelvis={h:.2f}: min_z={best_z[i]:.3f} at "
              f"xy=({best_xy[i,0]:.3f},{best_xy[i,1]:.3f})")
    for i, h in enumerate(HEIGHTS):
        print(f"BEST_ACTIONS_{h}=" + ",".join(f"{v:.3f}" for v in best_act[i].tolist()))
    print("PROBE_DONE")


if __name__ == "__main__":
    import os
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
