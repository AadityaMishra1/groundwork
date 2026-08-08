"""Why is feet_only_contact paying ~0 under the trained M1f policy?

Loads a checkpoint, rolls the policy in the Play env, and decomposes the
gate per step: foot-force condition pass rate, pelvis/torso veto rate, and
the force distributions on the veto bodies.

    python -u g1_loco/probe_gate_policy.py --ckpt <model.pt> --headless
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import g1_loco  # noqa: F401
import cli_args


def main():
    task = "G1-LocoKneel-Play-v0"
    env_cfg = parse_env_cfg(task, num_envs=64)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task, args)
    env = gym.make(task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=agent_cfg.device)
    runner.load(args.ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw = env.unwrapped
    sensor = raw.scene.sensors["contact_forces"]
    feet, _ = sensor.find_bodies(".*_ankle_roll_link")
    bad, bad_names = sensor.find_bodies(["pelvis", "torso_link"])
    fids = torch.tensor(feet, device=raw.device)
    bids = torch.tensor(bad, device=raw.device)

    n_steps = 0
    foot_pass = veto = gate_pass = 0.0
    bad_force_hi = []
    obs, _ = env.get_observations(), None
    with torch.inference_mode():
        for step in range(400):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            f = sensor.data.net_forces_w.norm(dim=-1)
            some_foot = (f[:, fids] > 1.0).any(dim=1)
            bmax = f[:, bids].max(dim=1).values
            not_collapsed = bmax < 5.0
            foot_pass += some_foot.float().mean().item()
            veto += (~not_collapsed).float().mean().item()
            gate_pass += (some_foot & not_collapsed).float().mean().item()
            if step % 20 == 0:
                bad_force_hi.append(float(bmax.median()))
            n_steps += 1
    q = torch.tensor(bad_force_hi)
    print(f"GATE_PROBE steps={n_steps} foot_cond_pass={foot_pass/n_steps:.3f} "
          f"badbody_veto={veto/n_steps:.3f} gate_pass={gate_pass/n_steps:.3f}")
    print(f"GATE_PROBE badbody force median-of-medians={q.median():.1f}N "
          f"p90={q.quantile(0.9):.1f}N bodies={bad_names}")


if __name__ == "__main__":
    import os, sys
    main()
    sys.stdout.flush()
    os._exit(0)
