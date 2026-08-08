"""Trajectory harvester: bank m2d's SUCCESSFUL episodes as phase-labeled
start states — the distillation-lite lever for closure.

harvest_success.py banks instants (held). This banks the LADDER: for every
episode that ultimately achieves a 1 s hold, commit snapshots taken along the
way, labeled by phase at capture time:

    approach  — no finger contact yet, palm within 0.45 m of the object
    closing   — >=1 finger contact, object still below lift height
    lift      — grasped, object rising, hold_counter < 25
    hold      — grasped, hold_counter >= 25 (same as harvest_success 'held')

Buffer-and-commit: states accumulate per env in a ring buffer and are written
to the bank ONLY when that env's episode reaches success (hold_counter >= 50
crossing); on reset the buffer is discarded. So every banked state provably
lies on a trajectory that ends in a hold — RSI from these states starts the
policy inside demonstrated-successful motion.

All the hard-won rules apply: reset-boundary guard (never capture on dones or
young episodes), up > 0.3 (m2d's real posture), incremental saves every 1000
steps, per-phase diagnostics with filter quantiles.

    GRASP_FREE=1 GRASP_FORCE_STAGE=3 GRASP_LEG_STIFF=4.0 GRASP_NO_VIDEO=1 \
    python -u g1_grasp/harvest_traj.py \
        --checkpoint /workspace/g1_grasp_m2d_free_35pct_strict_model_6997.pt \
        --num_envs 64 --steps 6000 --out /workspace/traj_bank.pt --headless
"""

import argparse
import os

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=6000)
parser.add_argument("--out", type=str, default="/workspace/traj_bank.pt")
parser.add_argument("--snap_every", type=int, default=10,
                    help="buffer a snapshot every N steps per env")
parser.add_argument("--max_per_phase", type=int, default=3000)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.checkpoint:
    parser.error("--checkpoint required")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_tasks  # noqa: F401
import g1_grasp  # noqa: F401

PHASES = ("approach", "closing", "lift", "hold")


def main():
    task = "G1-Grasp-v0"
    env_cfg = parse_env_cfg(task, num_envs=args.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task, args)
    env = gym.make(task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw = env.unwrapped
    dev = raw.device
    org = raw.scene.env_origins
    robot = raw.robot
    n = raw.num_envs

    def full_targets():
        t = robot.data.default_joint_pos.clone()
        t[:, raw.joint_ids] = raw.targets
        return t

    # per-env pending buffers: lists of (phase, snapshot dict rows on cpu)
    pending = [[] for _ in range(n)]
    bank = {p: {"jp": [], "jv": [], "tgt": [], "root": [], "obj": []}
            for p in PHASES}
    counts = {p: 0 for p in PHASES}

    def snapshot(i):
        rs = robot.data.root_state_w[i].clone()
        rs[:3] -= org[i]
        ob = raw.obj.data.root_state_w[i].clone()
        ob[:3] -= org[i]
        return {"jp": robot.data.joint_pos[i].cpu().clone(),
                "jv": robot.data.joint_vel[i].cpu().clone(),
                "tgt": full_targets()[i].cpu().clone(),
                "root": rs.cpu(), "obj": ob.cpu()}

    def commit(i):
        for phase, snap in pending[i]:
            if counts[phase] >= args.max_per_phase:
                continue
            for k, v in snap.items():
                bank[phase][k].append(v.unsqueeze(0))
            counts[phase] += 1
        pending[i] = []

    def save_bank():
        out = {"joint_names": list(robot.data.joint_names)}
        for p, d in bank.items():
            if d["jp"]:
                out[p] = {k: torch.cat(v) for k, v in d.items()}
        torch.save(out, args.out)

    obs, _ = env.get_observations(), None
    committed_eps = 0
    with torch.inference_mode():
        for step in range(args.steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            nf, th = raw._finger_contacts()
            grasped = (nf >= 3) & th
            objz = raw.obj.data.root_pos_w[:, 2] - org[:, 2]
            palm = robot.data.body_pos_w[:, raw.palm_id]
            palm_d = (palm - raw.obj.data.root_pos_w).norm(dim=-1)
            q = robot.data.root_quat_w
            up = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
            alive = (~dones.bool()) & (raw.episode_length_buf > 10)
            ok = (up > 0.3) & alive

            # success crossing: commit this env's buffered ladder
            crossed = (raw.hold_counter == 50).nonzero(as_tuple=False).squeeze(-1)
            for i in crossed.tolist():
                commit(i)
                committed_eps += 1
            # resets discard (reset-boundary rule: dones-step state is invalid)
            for i in dones.bool().nonzero(as_tuple=False).squeeze(-1).tolist():
                pending[i] = []

            if step % args.snap_every == 0:
                lifted = objz > 0.22
                phase_t = torch.full((n,), -1, dtype=torch.long, device=dev)
                phase_t[ok & (nf == 0) & (palm_d < 0.45)] = 0
                phase_t[ok & (nf >= 1) & ~lifted] = 1
                phase_t[ok & grasped & lifted & (raw.hold_counter < 25)] = 2
                phase_t[ok & grasped & (raw.hold_counter >= 25)] = 3
                for i in (phase_t >= 0).nonzero(as_tuple=False).squeeze(-1).tolist():
                    if len(pending[i]) < 200:
                        pending[i].append((PHASES[int(phase_t[i])], snapshot(i)))

            if step % 1000 == 999:
                save_bank()
                print(f"TRAJ checkpointed step {step}", flush=True)
            if step % 500 == 0:
                print(f"TRAJ step={step} eps={committed_eps} "
                      + " ".join(f"{p}={counts[p]}" for p in PHASES)
                      + f" | up_p50={up.median():.2f} grasped={int(grasped.sum())}",
                      flush=True)

    for p in PHASES:
        print(f"TRAJ {p}: n={counts[p]}", flush=True)
    save_bank()
    print(f"TRAJ_SAVED {args.out}", flush=True)


if __name__ == "__main__":
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
