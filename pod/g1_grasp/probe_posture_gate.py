"""CERT-POS: known-answer certificates for the 9S-d posture gate.

Two modes, both run with GRASP_POSTURE_GATE=1:

  A (no --ckpt): CERTIFIED SQUATS, zero action. Env restores the 42
    wbik-certified upright floor-pregrasp poses (GRASP_START_BANK =
    certified-only bank, GRASP_ARRIVAL_FRAC=1.0). The gate MUST pass them:
    posture_ok fraction >= 0.90 at settle, zero c5 terminations, and the
    settled p5 pelvis height is printed — it fixes GRASP_GATE_PELVIS per the
    pre-registered rule min(0.35, settled_p5 - 0.02).

  B (--ckpt <model_3000>): SPRAWL CONTROL, deterministic policy. The killed
    9S-c policy sprawls by construction. The gate MUST fail it: zero
    qualification (high_counter never leaves 0 => no stream/bonus ever
    paid) and c5term terminations MUST fire (sustained floor-lying
    disqualified). Either direction failing = the gate does not ship.

Per-conjunct quantiles printed throughout (the up>0.7-filter law).
"""

import argparse
import os

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=300)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import g1_grasp  # noqa: F401

MODE = "SPRAWL" if args.checkpoint else "CERT"


def main():
    env_cfg = parse_env_cfg("G1-Grasp-v0", num_envs=64)
    env = gym.make("G1-Grasp-v0", cfg=env_cfg)
    raw = env.unwrapped
    assert raw.posture_gate, "GRASP_POSTURE_GATE=1 required"
    env.reset()

    if args.checkpoint:
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        wenv = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(wenv, cli_args.parse_rsl_rl_cfg(
            "G1-Grasp-v0", args).to_dict(), log_dir=None, device=raw.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=raw.device)
        obs = wenv.get_observations()
        stepper = lambda: wenv.step(policy(obs))
    else:
        acts = torch.zeros(raw.num_envs, raw.cfg.action_space, device=raw.device)
        stepper = lambda: env.step(acts)

    n = raw.num_envs
    max_hc = 0
    settled_pelv = None
    marks = {1, 5, 25, 60, 100, 150, 250, 299}
    for step in range(args.steps):
        out = stepper()
        if args.checkpoint:
            obs = out[0]
        q = raw.robot.data.root_quat_w
        up = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        pelv = raw.robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
        minz = (raw.robot.data.body_pos_w[:, raw.c5_bodies, 2]
                - raw.scene.env_origins[:, 2:3]).min(dim=1).values
        ok = (up > raw.gate_up) & (pelv > raw.gate_pelv) & (minz > raw.gate_minz)
        max_hc = max(max_hc, int(raw.high_counter.max()))
        if step == 60 and MODE == "CERT":
            settled_pelv = float(pelv.quantile(0.05))
        if step in marks:
            print(f"POSGATE t={step:3d} ok={ok.float().mean():.2f} "
                  f"up_p10/50={up.quantile(.1):.2f}/{up.median():.2f} "
                  f"pelv_p5/50={pelv.quantile(.05):.3f}/{pelv.median():.3f} "
                  f"minz_p10/50={minz.quantile(.1):.3f}/{minz.median():.3f} "
                  f"c5term={raw.term_counts[5]} succ={raw.term_counts[4]} "
                  f"max_hc={max_hc}", flush=True)

    c5t, succ = raw.term_counts[5], raw.term_counts[4]
    if MODE == "CERT":
        # final ok measured at the last mark; recompute now
        okf = float(ok.float().mean())
        ok_pass = okf >= 0.90 and c5t == 0
        print(f"POSGATE_SETTLED_P5_PELVIS={settled_pelv:.3f} "
              f"(rule: GRASP_GATE_PELVIS = min(0.35, this-0.02))", flush=True)
        print(f"PROBE_POSGATE_CERT_{'PASS' if ok_pass else 'FAIL'} "
              f"(ok={okf:.2f} need>=0.90, c5term={c5t} need 0)", flush=True)
    else:
        ok_pass = max_hc == 0 and succ == 0 and c5t > 0
        print(f"PROBE_POSGATE_SPRAWL_{'PASS' if ok_pass else 'FAIL'} "
              f"(max_hc={max_hc} need 0, succ={succ} need 0, "
              f"c5term={c5t} need >0)", flush=True)


if __name__ == "__main__":
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
