"""P0 chain-bank harvester: snapshot the m2d grasp policy's own SUCCESS states.

The twice-proven recipe (grasp pose bank r2c, walker spawn-heights M1f) needs
start states along the WHOLE chain. The caged/held segments come from the
policy that already owns them: run m2d free-root, full-task starts, and bank
two families of physics-earned states:

  cage: grasped (>=3 groups + thumb), object still on the floor, obj speed
        < 0.2  — the moment before the lift
  held: grasped and lifted with hold_counter >= 25 (0.5 s of continuous hold)
        — the mid-air carry

Each snapshot stores the full joint state, the PD targets ACTUALLY applied
(action joints from raw.targets, parked joints from default — a settled pose
is the equilibrium under its targets, not under itself; probe_settle_bank
lesson), root state, and object pose in the env frame.

    GRASP_FREE=1 GRASP_FORCE_STAGE=3 GRASP_LEG_STIFF=4.0 \
    python -u g1_grasp/harvest_success.py \
        --checkpoint /workspace/g1_grasp_m2d_free_35pct_strict_model_6997.pt \
        --num_envs 64 --steps 6000 --out /workspace/chain_bank_grasp.pt --headless
"""

import argparse
import os

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=6000)
parser.add_argument("--out", type=str, default=f"{_WS}/chain_bank_grasp.pt")
parser.add_argument("--max_per_family", type=int, default=2000)
# snapshot cadence per env: don't bank near-duplicate consecutive frames
parser.add_argument("--min_gap", type=int, default=25)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.checkpoint:
    parser.error("--checkpoint (grasp policy .pt) is required")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_tasks  # noqa: F401
import g1_grasp  # noqa: F401

import os as _os
_WS = _os.environ.get("GW_ROOT", "/workspace")


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
    n_dof = robot.data.joint_pos.shape[1]
    # full-body PD targets actually applied: parked joints hold defaults,
    # action joints hold raw.targets (kept current every step by the env)
    def full_targets():
        t = robot.data.default_joint_pos.clone()
        t[:, raw.joint_ids] = raw.targets
        return t

    bank = {"cage": {"jp": [], "jv": [], "tgt": [], "root": [], "obj": []},
            "held": {"jp": [], "jv": [], "tgt": [], "root": [], "obj": []}}
    counts = {"cage": 0, "held": 0}
    last_snap = torch.full((n,), -10**9, dtype=torch.long, device=dev)
    step_t = torch.zeros(n, dtype=torch.long, device=dev)

    def grab(family, mask):
        m = mask & (step_t - last_snap >= args.min_gap)
        if not m.any() or counts[family] >= args.max_per_family:
            return 0
        idx = m.nonzero(as_tuple=False).squeeze(-1)
        last_snap[idx] = step_t[idx]
        rs = robot.data.root_state_w[idx].clone()
        rs[:, :3] -= org[idx]
        ob = raw.obj.data.root_state_w[idx].clone()
        ob[:, :3] -= org[idx]
        bank[family]["jp"].append(robot.data.joint_pos[idx].cpu().clone())
        bank[family]["jv"].append(robot.data.joint_vel[idx].cpu().clone())
        bank[family]["tgt"].append(full_targets()[idx].cpu().clone())
        bank[family]["root"].append(rs.cpu())
        bank[family]["obj"].append(ob.cpu())
        return len(idx)

    def save_bank():
        out = {"joint_names": list(robot.data.joint_names)}
        for fam, d in bank.items():
            if d["jp"]:
                out[fam] = {k: torch.cat(v) for k, v in d.items()}
        torch.save(out, args.out)

    obs, _ = env.get_observations(), None
    with torch.inference_mode():
        for step in range(args.steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            step_t += 1
            nf, th = raw._finger_contacts()
            grasped = (nf >= 3) & th
            objz = raw.obj.data.root_pos_w[:, 2] - org[:, 2]
            obj_speed = raw.obj.data.root_lin_vel_w.norm(dim=-1)
            # upright check — never bank a falling robot
            q = robot.data.root_quat_w
            up = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
            # RESET-BOUNDARY GUARD (2026-07-26, the bug that poisoned the
            # first bank): on the step an episode ends, robot/object buffers
            # hold the freshly-written RESET state while contact sensors
            # still hold STALE pre-reset forces — "grasped" reads true on a
            # spawn pose. All 435 first-bank cage states were this artifact
            # (root z identically 0.320 = spawn, zero velocities). Never
            # capture a just-reset or young episode.
            alive = (~dones.bool()) & (raw.episode_length_buf > 10)
            # up bar 0.7 -> 0.3 (2026-07-26): instrumented run proved m2d
            # HOLDS with the pelvis pitched past 45 deg (held components
            # intersected >=56/64 yet heldmask=0 — only the unprinted up
            # conjunct could zero it). Leaning IS the real hold posture; the
            # 0.7 bar banked upright reset-frames and rejected every genuine
            # grasp. Bank whatever the env itself considers alive.
            ok = (up > 0.3) & alive
            cage = ok & grasped & (objz < raw.obj_h2 + 0.03) & (obj_speed < 0.2)
            held = ok & grasped & (objz > 0.22) & (raw.hold_counter >= 25)
            counts["cage"] += grab("cage", cage)
            counts["held"] += grab("held", held)
            fin = dones.bool()
            if fin.any():
                step_t[fin.nonzero(as_tuple=False).squeeze(-1)] = 0
                last_snap[fin.nonzero(as_tuple=False).squeeze(-1)] = -10**9
            if step % 1000 == 999:
                # incremental save — an Isaac assert at step ~4700 destroyed
                # a full harvest once ("Recursion not allowed"); never again
                save_bank()
                print(f"HARVEST checkpointed at step {step}", flush=True)
            if step % 500 == 0:
                # per-condition diagnostics — explains empty families.
                # gap_ok added 2026-07-26: run 2 showed held-mask components
                # ~58/64 with ZERO captures — the dead conjunct is inside
                # grab()'s gap bookkeeping; observe it, don't re-infer it.
                gap_ok = int((step_t - last_snap >= args.min_gap).sum())
                print(f"HARVEST step={step} cage={counts['cage']} "
                      f"held={counts['held']} | now: grasped={int(grasped.sum())} "
                      f"lifted={int((objz > 0.22).sum())} "
                      f"holdc25={int((raw.hold_counter >= 25).sum())} "
                      f"alive={int(alive.sum())} floorobj={int((objz < raw.obj_h2 + 0.03).sum())} "
                      f"heldmask={int(held.sum())} cagemask={int(cage.sum())} "
                      f"gap_ok={gap_ok} step_t_p50={int(step_t.median())} "
                      f"up_p25/50={up.quantile(0.25):.2f}/{up.median():.2f}",
                      flush=True)

    for fam, d in bank.items():
        if not d["jp"]:
            print(f"HARVEST {fam}: EMPTY — policy never reached this state")
        else:
            print(f"HARVEST {fam}: n={sum(len(x) for x in d['jp'])}")
    save_bank()
    print(f"HARVEST_SAVED {args.out}")


if __name__ == "__main__":
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
