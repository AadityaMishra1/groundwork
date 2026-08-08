"""Probe B / demo miner: build a bank of physics-verified pre-grasp poses.

Kneel stance (probe_kneel 2026-07-11: pelvis 0.20, folded legs -> finger
centroid reaches z~0.07). Protocol-spec cylinder (h=0.15, r=0.03).

Per round, in each env:
  1. object parked far away; sample a random arm action near the probe's
     best-reach seed, settle 1.2 s, measure finger centroid
  2. if the centroid is in the graspable band, teleport the object under the
     hand (demo mining only — training/eval never teleports after t=0)
  3. ramp fingers closed 0.6 s
  4. ramp a sampled lift delta on waist/shoulder/elbow 1.5 s, hold 0.6 s
  5. success = object z > lift threshold AND >=3 finger groups + thumb in
     contact (filtered PhysX contact report) at the end of the hold

Successful (joint pose, object rel-pose) pairs are appended to
/workspace/pregrasp_bank.pt. Prints per-round acceptance so we know early
whether scripted close-and-lift is physically workable at all.

    PYTHONPATH=/workspace/humanoid/pod python -u g1_grasp/mine_demos.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--rounds", type=int, default=30)
parser.add_argument("--num_envs", type=int, default=512)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from g1_grasp.grasp_env import (
    ACTION_JOINTS, FINGER_BODIES, G1GraspEnv, G1GraspEnvCfg,
)

import os as _os
_WS = _os.environ.get("GW_ROOT", "/workspace")

PELVIS_Z = 0.20
FOLD_LEGS = {
    ".*_hip_pitch_joint": -2.2, ".*_knee_joint": 2.8, ".*_ankle_pitch_joint": -0.87,
    ".*_hip_roll_joint": 0.4, ".*_hip_yaw_joint": 0.0, ".*_ankle_roll_joint": 0.0,
}
# probe_kneel best action at pelvis 0.20, spread legs + self-collision
# (2026-07-11 probe_kneel2: min_z=0.054 at xy 0.244,-0.038)
SEED_ARM = [0.090, 0.490, 0.494, -0.940, 0.412, 0.292, 0.339, -0.160]
ARM_NOISE = 0.35
OBJ_H, OBJ_R, OBJ_M = 0.15, 0.03, 0.25
# graspable band for the finger centroid: around/below the cylinder top
BAND_LO, BAND_HI = 0.05, 0.16
CLOSE_FING = [0.9, 0.7, 0.9, 0.7, 0.9, 0.7, 0.9, 0.7, 0.9, 0.4, 0.5, 0.7]
LIFT_Z_SUCCESS = 0.22  # object center (spawn 0.075) raised ~15 cm

STEPS_SETTLE = 60   # 1.2 s at 50 Hz control? (we step physics directly, 200 Hz)
STEPS_CLOSE = 120
STEPS_LIFT = 300
STEPS_HOLD = 120


def main():
    cfg = G1GraspEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.object_cfg.spawn.height = OBJ_H
    cfg.object_cfg.spawn.radius = OBJ_R
    cfg.object_cfg.spawn.mass_props.mass = OBJ_M
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
    finger_ids, _ = robot.find_bodies(FINGER_BODIES, preserve_order=True)
    seed_arm = torch.tensor(SEED_ARM, device=dev)
    fing_close = torch.tensor(CLOSE_FING, device=dev)
    # finger close targets as actions: invert affine map
    fing_close_a = ((fing_close - env.act_mid[8:]) / env.act_half[8:]).clamp(-1, 1)

    gen = torch.Generator(device=dev).manual_seed(1)
    bank_q, bank_obj, bank_lift, bank_qh, bank_objh = [], [], [], [], []
    archive = []  # accepted in-band arm actions, resampled in later rounds

    def step_targets(act, steps):
        env._pre_physics_step(act)
        for _ in range(steps):
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)

    for rd in range(args.rounds):
        # reset robots to kneel, park objects far away
        jp = robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        root = robot.data.default_root_state.clone()
        root[:, :3] += env.scene.env_origins
        robot.write_root_pose_to_sim(root[:, :7])
        robot.write_root_velocity_to_sim(root[:, 7:])
        far = torch.tensor([1.8, 0.0, OBJ_H / 2], device=dev).expand(n, 3)
        st = torch.zeros(n, 13, device=dev)
        st[:, :3] = far + env.scene.env_origins
        st[:, 3] = 1.0
        obj.write_root_pose_to_sim(st[:, :7])
        obj.write_root_velocity_to_sim(st[:, 7:])

        # 1. sample + settle arm pose, hand open. Fully random exploration
        # plus CEM-style resampling around previously ACCEPTED poses (these
        # are reproducible: settled from the default sit, unlike the probe's
        # path-dependent "best actions" which put the hand behind the back).
        a = torch.zeros(n, len(ACTION_JOINTS), device=dev)
        a[:, :8] = 2 * torch.rand(n, 8, generator=gen, device=dev) - 1
        if archive:
            arch = torch.stack(archive)
            k = n // 2
            ridx = torch.randint(len(arch), (k,), generator=gen, device=dev)
            a[:k, :8] = (arch[ridx] + 0.12 * (
                2 * torch.rand(k, 8, generator=gen, device=dev) - 1)).clamp(-1, 1)
        a[:, 8:] = -0.8
        step_targets(a.clamp(-1, 1), STEPS_SETTLE + 40)
        cent = robot.data.body_pos_w[:, finger_ids].mean(dim=1) - env.scene.env_origins
        ok = (cent[:, 2] > BAND_LO) & (cent[:, 2] < BAND_HI) \
             & (cent[:, 0] > 0.15) & (cent[:, 0] < 0.55)
        if rd == 0:
            q = torch.tensor([0.05, 0.5, 0.95], device=dev)
            print("  [debug] cent x q:", torch.quantile(cent[:, 0], q).tolist(),
                  "y q:", torch.quantile(cent[:, 1], q).tolist(),
                  "z q:", torch.quantile(cent[:, 2], q).tolist(),
                  "in_z:", int(((cent[:, 2] > BAND_LO) & (cent[:, 2] < BAND_HI)).sum()),
                  "in_x:", int(((cent[:, 0] > 0.15) & (cent[:, 0] < 0.55)).sum()))

        # 2. teleport object under the hand, offset from the finger centroid
        #    toward the palm so the barrel sits in the grasp pocket instead of
        #    interpenetrating finger colliders (v1: pop-ejection, obj_z ~0.9)
        palm = robot.data.body_pos_w[:, env.palm_id] - env.scene.env_origins
        to_palm = palm[:, :2] - cent[:, :2]
        to_palm = to_palm / to_palm.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        pos = torch.zeros(n, 3, device=dev)
        pos[:, :2] = cent[:, :2] + 0.025 * to_palm
        pos[:, 2] = OBJ_H / 2
        st = torch.zeros(n, 13, device=dev)
        st[:, :3] = pos + env.scene.env_origins
        st[:, 3] = 1.0
        obj.write_root_pose_to_sim(st[:, :7])
        obj.write_root_velocity_to_sim(st[:, 7:])
        # settle with hand still open, then require the placement to have
        # survived (still near placement, still upright) — filters pops
        step_targets(a.clamp(-1, 1), 40)
        opos = obj.data.root_pos_w - env.scene.env_origins
        oq = obj.data.root_quat_w
        up_z = 1.0 - 2.0 * (oq[:, 1] ** 2 + oq[:, 2] ** 2)
        placed = (opos[:, :2] - pos[:, :2]).norm(dim=-1) < 0.05
        placed &= (up_z > 0.9) & (opos[:, 2] < 0.12)
        ok &= placed
        for i in ok.nonzero(as_tuple=True)[0].tolist():
            if len(archive) < 800:
                archive.append(a[i, :8].clone())

        # snapshot the cage pose for the bank
        q_cage = robot.data.joint_pos.clone()

        # 3. close fingers (ramp) with per-env noise on the close targets so
        #    the search covers grip variations (vice-crush vs loose cage)
        close_a = (fing_close_a + 0.15 * (
            2 * torch.rand(n, 12, generator=gen, device=dev) - 1)).clamp(-1, 1)
        for k in range(6):
            blend = (k + 1) / 6
            a2 = a.clone()
            a2[:, 8:] = (1 - blend) * (-0.8) + blend * close_a
            step_targets(a2.clamp(-1, 1), STEPS_CLOSE // 6)

        # 4. lift: sampled joint-space delta on waist pitch / shoulder pitch /
        #    elbow — random search finds the direction that raises the hand
        delta = torch.zeros(n, 8, device=dev)
        delta[:, 0] = 0.6 * (2 * torch.rand(n, generator=gen, device=dev) - 1)  # waist
        delta[:, 1] = 0.8 * (2 * torch.rand(n, generator=gen, device=dev) - 1)  # sh pitch
        delta[:, 4] = 0.6 * (2 * torch.rand(n, generator=gen, device=dev) - 1)  # elbow
        for k in range(10):
            blend = (k + 1) / 10
            a3 = a2.clone()
            a3[:, :8] = a[:, :8] + blend * delta
            step_targets(a3.clamp(-1, 1), STEPS_LIFT // 10)

        # 5. verdict sampled across the hold window (not one noisy end step)
        good = torch.zeros(n, device=dev)
        for _ in range(6):
            step_targets(a3.clamp(-1, 1), STEPS_HOLD // 6)
            obj_z = (obj.data.root_pos_w - env.scene.env_origins)[:, 2]
            n_fing, thumb = env._finger_contacts()
            good += ((obj_z > LIFT_Z_SUCCESS) & (n_fing >= 3) & thumb).float()
        opos = obj.data.root_pos_w - env.scene.env_origins
        near = (opos[:, :2] - pos[:, :2]).norm(dim=-1) < 0.35  # not flung
        success = ok & (good >= 5) & near
        # failure decomposition among in-band placed envs
        lifted_end = ok & (opos[:, 2] > LIFT_Z_SUCCESS)
        grasped_end = ok & (n_fing >= 3) & thumb
        n_ok, n_succ = int(ok.sum()), int(success.sum())
        print(f"round {rd+1}/{args.rounds}: placed {n_ok}/{n} "
              f"success {n_succ} | end: lifted {int(lifted_end.sum())} "
              f"grasped {int(grasped_end.sum())} flung {int((ok & ~near).sum())} "
              f"objz_med {opos[ok, 2].median() if n_ok else 0:.3f}")

        if n_succ:
            idx = success.nonzero(as_tuple=True)[0]
            bank_q.append(q_cage[idx].cpu())
            bank_obj.append(pos[idx].cpu())  # object pos in env frame at cage
            bank_lift.append(torch.cat(
                [a[idx, :8], delta[idx], close_a[idx]], dim=-1).cpu())
            # held end-state: backward-chaining stage 0 starts here
            bank_qh.append(robot.data.joint_pos[idx].cpu())
            bank_objh.append((obj.data.root_pos_w[idx]
                              - env.scene.env_origins[idx]).cpu())
            torch.save(
                {"joint_pos": torch.cat(bank_q), "obj_pos": torch.cat(bank_obj),
                 "arm_delta_close": torch.cat(bank_lift),
                 "held_joint_pos": torch.cat(bank_qh),
                 "held_obj_pos": torch.cat(bank_objh),
                 "joint_names": robot.joint_names,
                 "meta": {"pelvis_z": PELVIS_Z, "obj_h": OBJ_H, "obj_r": OBJ_R,
                          "obj_m": OBJ_M, "fold_legs": FOLD_LEGS}},
                f"{_WS}/pregrasp_bank.pt")

    total = sum(len(b) for b in bank_q) if bank_q else 0
    print(f"BANK_SIZE={total}")
    print("MINE_DONE")


if __name__ == "__main__":
    import os
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
