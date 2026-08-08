"""Render a close-up video of scripted grasp successes from the mined bank.

Replays bank entries exactly: kneel pose -> cage joints -> object placed ->
stored close targets ramped -> stored lift delta ramped -> hold. Frames from
the viewport at 50 fps.

    PYTHONPATH=/workspace/humanoid/pod python -u g1_grasp/replay_bank.py \
        --headless --enable_cameras --entries 6
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--entries", type=int, default=6)
parser.add_argument("--out", type=str, default="/workspace/demo_videos/scripted_grasp.mp4")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import imageio
import torch

from g1_grasp.grasp_env import ACTION_JOINTS, G1GraspEnv, G1GraspEnvCfg

# keep in sync with mine_demos.py (not imported: its module top-level
# launches its own AppLauncher)
PELVIS_Z = 0.20
FOLD_LEGS = {
    ".*_hip_pitch_joint": -2.2, ".*_knee_joint": 2.8, ".*_ankle_pitch_joint": -0.87,
    ".*_hip_roll_joint": 0.4, ".*_hip_yaw_joint": 0.0, ".*_ankle_roll_joint": 0.0,
}
OBJ_H, OBJ_R, OBJ_M = 0.15, 0.03, 0.25

STEPS_CLOSE, STEPS_LIFT, STEPS_HOLD = 120, 300, 200
FRAME_EVERY = 4  # 200 Hz physics -> 50 fps


def main():
    bank = torch.load("/workspace/pregrasp_bank.pt", weights_only=True)
    print(f"bank size: {len(bank['joint_pos'])}")

    cfg = G1GraspEnvCfg()
    cfg.scene.num_envs = 1
    cfg.object_cfg.spawn.height = OBJ_H
    cfg.object_cfg.spawn.radius = OBJ_R
    cfg.object_cfg.spawn.mass_props.mass = OBJ_M
    cfg.robot.init_state.pos = (0.0, 0.0, PELVIS_Z)
    cfg.viewer.eye = (1.1, -0.9, 0.55)
    cfg.viewer.lookat = (0.25, -0.1, 0.15)
    env = G1GraspEnv(cfg, render_mode="rgb_array")
    dev = env.device
    for pattern, val in FOLD_LEGS.items():
        ids, _ = env.robot.find_joints(pattern)
        lim = env.robot.data.joint_pos_limits[0, ids]
        env.robot.data.default_joint_pos[:, ids] = torch.clamp(
            torch.full((len(ids),), val, device=dev), lim[:, 0], lim[:, 1])
    env.reset()
    robot, obj = env.robot, env.obj

    frames = []

    def step_targets(act, steps):
        env._pre_physics_step(act)
        for k in range(steps):
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=(k % FRAME_EVERY == 0))
            if k % FRAME_EVERY == 0:
                frames.append(env.render())
            env.scene.update(env.physics_dt)

    n_bank = len(bank["joint_pos"])
    picks = torch.linspace(0, n_bank - 1, min(args.entries, n_bank)).long()
    adc = bank["arm_delta_close"]
    for i in picks.tolist():
        jp = bank["joint_pos"][i : i + 1].to(dev)
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        root = robot.data.default_root_state.clone()
        root[:, :3] += env.scene.env_origins
        robot.write_root_pose_to_sim(root[:, :7])
        robot.write_root_velocity_to_sim(root[:, 7:])
        st = torch.zeros(1, 13, device=dev)
        st[0, :3] = bank["obj_pos"][i].to(dev) + env.scene.env_origins[0]
        st[0, 3] = 1.0
        obj.write_root_pose_to_sim(st[:, :7])
        obj.write_root_velocity_to_sim(st[:, 7:])

        arm = adc[i, :8].to(dev)
        delta = adc[i, 8:16].to(dev)
        close = adc[i, 16:].to(dev)
        a = torch.zeros(1, len(ACTION_JOINTS), device=dev)
        a[0, :8] = arm
        a[0, 8:] = -0.8
        step_targets(a, 30)
        for k in range(6):
            b = (k + 1) / 6
            a[0, 8:] = (1 - b) * (-0.8) + b * close
            step_targets(a, STEPS_CLOSE // 6)
        for k in range(10):
            b = (k + 1) / 10
            a2 = a.clone()
            a2[0, :8] = arm + b * delta
            step_targets(a2, STEPS_LIFT // 10)
        step_targets(a2, STEPS_HOLD)
        obj_z = (obj.data.root_pos_w - env.scene.env_origins)[0, 2]
        nf, th = env._finger_contacts()
        print(f"entry {i}: end obj_z={obj_z:.3f} fingers={int(nf[0])} "
              f"thumb={bool(th[0])}")

    frames_ok = [f for f in frames if f is not None]
    imageio.mimwrite(args.out, frames_ok, fps=50, quality=8)
    print(f"VIDEO_SAVED {args.out} frames={len(frames_ok)}")


if __name__ == "__main__":
    import os
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
