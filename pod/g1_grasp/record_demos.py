"""Record full-task (obs, action) demonstration trajectories for BC.

Each env replays a bank entry as a complete honest episode: object placed at
its bank position FROM THE START, arm ramps from the default sit to the bank
pre-grasp action (a real approach with the object present), stored close and
lift ramps follow. Obs computed exactly as G1GraspEnv._get_observations.
Only strictly successful episodes (final: lifted + >=3 groups + thumb over a
hold window) are kept.

    PYTHONPATH=/workspace/humanoid/pod python -u g1_grasp/record_demos.py \
        --headless --rounds 20 --num_envs 512
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--rounds", type=int, default=20)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--out", type=str, default=f"{_WS}/bc_demos.pt")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from g1_grasp.grasp_env import ACTION_JOINTS, G1GraspEnv, G1GraspEnvCfg

import os as _os
_WS = _os.environ.get("GW_ROOT", "/workspace")

PELVIS_Z = 0.20
FOLD_LEGS = {
    ".*_hip_pitch_joint": -2.2, ".*_knee_joint": 2.8, ".*_ankle_pitch_joint": -0.87,
    ".*_hip_roll_joint": 0.4, ".*_hip_yaw_joint": 0.0, ".*_ankle_roll_joint": 0.0,
}
OBJ_H = 0.15
APPROACH_TICKS = 50   # 1 s: ramp default -> pre-grasp action
CLOSE_TICKS = 30      # 0.6 s
LIFT_TICKS = 75       # 1.5 s
HOLD_TICKS = 50       # 1 s
PHYS_PER_TICK = 4     # 200 Hz physics, 50 Hz control


def main():
    bank = torch.load(f"{_WS}/pregrasp_bank.pt", weights_only=True)
    n_bank = len(bank["obj_pos"])
    print(f"bank entries: {n_bank}")

    cfg = G1GraspEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.robot.init_state.pos = (0.0, 0.0, PELVIS_Z)
    env = G1GraspEnv(cfg)
    dev = env.device
    n = args.num_envs
    for pattern, val in FOLD_LEGS.items():
        ids, _ = env.robot.find_joints(pattern)
        lim = env.robot.data.joint_pos_limits[0, ids]
        env.robot.data.default_joint_pos[:, ids] = torch.clamp(
            torch.full((len(ids),), val, device=dev), lim[:, 0], lim[:, 1])
    env.reset()
    robot, obj = env.robot, env.obj

    adc = bank["arm_delta_close"].to(dev)      # (B, 8+8+12)
    obj_pos_bank = bank["obj_pos"].to(dev)     # (B, 3)
    n_act = len(ACTION_JOINTS)
    # default-pose action (inverse affine of default targets), (n_envs, 20)
    default_a = ((env.default_targets - env.act_mid) / env.act_half).clamp(-1, 1)
    default_arm = default_a[:, :8]

    gen = torch.Generator(device=dev).manual_seed(7)
    all_obs, all_act, kept = [], [], 0

    def obs_now(prev_a):
        q = robot.data.joint_pos[:, env.joint_ids]
        qd = robot.data.joint_vel[:, env.joint_ids]
        palm = robot.data.body_pos_w[:, env.palm_id] - env.scene.env_origins
        obj_rel = obj.data.root_pos_w - robot.data.body_pos_w[:, env.palm_id]
        return torch.cat([q, qd, palm, obj_rel,
                          obj.data.root_lin_vel_w, prev_a], dim=-1)

    for rd in range(args.rounds):
        pick = torch.randint(n_bank, (n,), generator=gen, device=dev)
        arm = adc[pick, :8] + 0.05 * (2 * torch.rand(n, 8, generator=gen, device=dev) - 1)
        delta = adc[pick, 8:16]
        close = adc[pick, 16:]

        # reset: default sit, object at its bank location from t=0
        jp = robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        root = robot.data.default_root_state.clone()
        root[:, :3] += env.scene.env_origins
        robot.write_root_pose_to_sim(root[:, :7])
        robot.write_root_velocity_to_sim(root[:, 7:])
        st = torch.zeros(n, 13, device=dev)
        st[:, :3] = obj_pos_bank[pick] + env.scene.env_origins
        st[:, 3] = 1.0
        obj.write_root_pose_to_sim(st[:, :7])
        obj.write_root_velocity_to_sim(st[:, 7:])
        env.targets[:] = jp[:, env.joint_ids]

        ep_obs = torch.zeros(n, APPROACH_TICKS + CLOSE_TICKS + LIFT_TICKS + HOLD_TICKS,
                             69, device=dev)
        ep_act = torch.zeros(n, ep_obs.shape[1], n_act, device=dev)
        prev_a = default_a.clone()
        t = 0

        def tick(a):
            nonlocal t, prev_a
            ep_obs[:, t] = obs_now(prev_a)
            ep_act[:, t] = a
            env._pre_physics_step(a)
            for _ in range(PHYS_PER_TICK):
                env._apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)
            prev_a = a.clone()
            t += 1

        # approach FROM ABOVE: ramp to the lifted pose (hand high over the
        # pocket), then descend vertically — a direct ramp to the grasp pose
        # sweeps the arm through the object and knocks it (v1: 9/10240 kept)
        a = torch.zeros(n, n_act, device=dev)
        high = arm + delta
        for k in range(APPROACH_TICKS * 2 // 3):
            b = (k + 1) / (APPROACH_TICKS * 2 // 3)
            a[:, :8] = (1 - b) * default_arm + b * high
            a[:, 8:] = -0.8
            tick(a.clone())
        for k in range(APPROACH_TICKS - APPROACH_TICKS * 2 // 3):
            b = (k + 1) / (APPROACH_TICKS - APPROACH_TICKS * 2 // 3)
            a[:, :8] = (1 - b) * high + b * arm
            a[:, 8:] = -0.8
            tick(a.clone())
        for k in range(CLOSE_TICKS):
            b = (k + 1) / CLOSE_TICKS
            a[:, 8:] = (1 - b) * (-0.8) + b * close
            tick(a.clone())
        for k in range(LIFT_TICKS):
            b = (k + 1) / LIFT_TICKS
            a[:, :8] = arm + b * delta
            tick(a.clone())
        good = torch.zeros(n, device=dev)
        for k in range(HOLD_TICKS):
            tick(a.clone())
            if k % 10 == 0:
                oz = (obj.data.root_pos_w - env.scene.env_origins)[:, 2]
                nf, th = env._finger_contacts()
                good += ((oz > 0.22) & (nf >= 3) & th).float()
        success = good >= 4
        ns = int(success.sum())
        kept += ns
        print(f"round {rd+1}/{args.rounds}: demo-success {ns}/{n} (total {kept})")
        if ns:
            idx = success.nonzero(as_tuple=True)[0]
            all_obs.append(ep_obs[idx].cpu())
            all_act.append(ep_act[idx].cpu())
            torch.save({"obs": torch.cat(all_obs), "act": torch.cat(all_act)},
                       args.out)

    print(f"DEMOS_DONE kept={kept}")


if __name__ == "__main__":
    import os
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
