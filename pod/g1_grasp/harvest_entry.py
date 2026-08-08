"""Entry harvester: bank the unified policy's OWN successful entry motion as
phase-labeled start states — the self-RSI lever against income capture.

harvest_traj.py banks m2d's (seated teacher) ladder behind a hold_counter=50
commit. The unified standing-start lineage almost never reaches 50 from a
standing spawn, so that bar would reproduce the counterfeit-bank rake in
reverse (an empty bank from a real skill). This harvester uses TWO commit
tiers matched to the lineage's actual competence:

    grasp-formed crossing (first nf>=3 & thumb in the episode)
        -> commit buffered "approach" frames (pre-contact, palm closing in)
           and "closing" frames (partial contact) — they provably led to a
           formed grasp from a standing start.
    hold_counter crossing 25 / 50
        -> commit "lift" / "hold" frames (rare from standing; every one is
           gold).

Families are named approach/closing/lift/hold so grasp_env's bank2 loader
consumes the bank with zero code changes.

Run STOCHASTIC by default: entry behavior lives in the sampled policy at far
higher rates than in the deterministic mean (train grasped 0.88 vs mean 12/100
at run-8 model_2200). Obs normalization (GRASP_EMPNORM lineage) is applied
explicitly before act() — the raw actor has never seen unnormalized obs.

All standing rules apply: reset-boundary guard (never capture on dones or
young episodes), filter-quantile prints for every conjunct, incremental saves,
zero-action probe validation (probe_cage.py) REQUIRED before any training run
consumes the output.

    GRASP_FULLBODY=1 GRASP_FREE=1 GRASP_LEG_STIFF=4.0 GRASP_EP_S=30 \
    GRASP_FLUNG_REF=robot GRASP_LEG_BLEND=1.0 GRASP_SPAWN_Z=0.80 \
    GRASP_EMPNORM=1 GRASP_FORCE_STAGE=3 GRASP_FAR_FRAC=0.0 GRASP_NO_VIDEO=1 \
    python -u g1_grasp/harvest_entry.py \
        --checkpoint .../model_1400.pt --num_envs 256 --steps 6000 \
        --out /workspace/entry_bank_1400.pt --headless
"""

import argparse
import os

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=6000)
parser.add_argument("--out", type=str, default=f"{_WS}/entry_bank.pt")
parser.add_argument("--snap_every", type=int, default=10,
                    help="buffer a snapshot every N steps per env")
parser.add_argument("--max_per_phase", type=int, default=3000)
parser.add_argument("--deterministic", action="store_true",
                    help="use the mean policy instead of sampling")
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

import os as _os
_WS = _os.environ.get("GW_ROOT", "/workspace")

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

    if args.deterministic:
        policy = runner.get_inference_policy(device=env.unwrapped.device)
    else:
        # stochastic: this rsl_rl's ActorCritic.act() applies
        # actor_obs_normalizer INTERNALLY (verified in site-packages source),
        # so raw obs go straight in. eval_mode() freezes the running stats.
        runner.eval_mode()
        ac = (runner.alg.policy if hasattr(runner.alg, "policy")
              else runner.alg.actor_critic)
        policy = lambda o: ac.act(o).detach()  # noqa: E731

    raw = env.unwrapped
    dev = raw.device
    org = raw.scene.env_origins
    robot = raw.robot
    n = raw.num_envs

    def full_targets():
        t = robot.data.default_joint_pos.clone()
        t[:, raw.joint_ids] = raw.targets
        return t

    pending = [[] for _ in range(n)]
    got_grasp = torch.zeros(n, dtype=torch.bool, device=dev)
    hold25_done = torch.zeros(n, dtype=torch.bool, device=dev)
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

    def commit(i, phases):
        keep = []
        for phase, snap in pending[i]:
            if phase not in phases:
                keep.append((phase, snap))
                continue
            if counts[phase] >= args.max_per_phase:
                continue
            for k, v in snap.items():
                bank[phase][k].append(v.unsqueeze(0))
            counts[phase] += 1
        pending[i] = keep

    def save_bank():
        out = {"joint_names": list(robot.data.joint_names)}
        for p, d in bank.items():
            if d["jp"]:
                out[p] = {k: torch.cat(v) for k, v in d.items()}
        torch.save(out, args.out)

    obs, _ = env.get_observations(), None
    grasp_eps = hold_eps = 0
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

            # tier 1: first formed grasp -> the entry that got here is proven
            crossing = grasped & alive & ~got_grasp
            for i in crossing.nonzero(as_tuple=False).squeeze(-1).tolist():
                commit(i, ("approach", "closing"))
                got_grasp[i] = True
                grasp_eps += 1
            # tier 2: sustained hold -> the closure that got here is proven
            h25 = (raw.hold_counter >= 25) & alive & ~hold25_done
            for i in h25.nonzero(as_tuple=False).squeeze(-1).tolist():
                commit(i, ("lift", "hold"))
                hold25_done[i] = True
                hold_eps += 1
            # resets discard (reset-boundary rule)
            d_idx = dones.bool()
            for i in d_idx.nonzero(as_tuple=False).squeeze(-1).tolist():
                pending[i] = []
            got_grasp[d_idx] = False
            hold25_done[d_idx] = False

            if step % args.snap_every == 0:
                lifted = objz > 0.22
                phase_t = torch.full((n,), -1, dtype=torch.long, device=dev)
                phase_t[ok & (nf == 0) & (palm_d < 0.60)] = 0
                phase_t[ok & (nf >= 1) & ~lifted] = 1
                phase_t[ok & grasped & lifted & (raw.hold_counter < 25)] = 2
                phase_t[ok & grasped & (raw.hold_counter >= 25)] = 3
                for i in (phase_t >= 0).nonzero(as_tuple=False).squeeze(-1).tolist():
                    if len(pending[i]) < 200:
                        pending[i].append((PHASES[int(phase_t[i])], snapshot(i)))

            if step % 1000 == 999:
                save_bank()
                print(f"ENTRY checkpointed step {step}", flush=True)
            if step % 500 == 0:
                print(f"ENTRY step={step} grasp_eps={grasp_eps} "
                      f"hold_eps={hold_eps} "
                      + " ".join(f"{p}={counts[p]}" for p in PHASES)
                      + f" | up_q10={up.quantile(0.1):.2f}"
                      f" up_p50={up.median():.2f}"
                      f" palm_p50={palm_d.median():.2f}"
                      f" grasped_now={int(grasped.sum())}", flush=True)

    for p in PHASES:
        print(f"ENTRY {p}: n={counts[p]}", flush=True)
    save_bank()
    print(f"ENTRY_SAVED {args.out} grasp_eps={grasp_eps} hold_eps={hold_eps}",
          flush=True)


if __name__ == "__main__":
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
