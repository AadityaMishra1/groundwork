"""CERT-EMPNORM: known-answer certificates for GRASP_EMPNORM_FREEZE.

The freeze gate (agents.apply_empnorm_freeze, called by train_grasp.py
after runner.load()) must (1) change nothing when the flag is unset,
(2) actually pin the normalizer statistics through real learn() iterations,
and (3) not break learning. Three certificates, one per invocation — this
probe calls the SAME apply_empnorm_freeze the trainer ships, so the certs
exercise the shipped code path, not a re-implementation beside it.

  bitexact  (--cert bitexact): 100 seeded env steps, prints md5 over the
      obs+reward stream. PROTOCOL: run on the pre-sync build and on the
      synced build (same pod+GPU, same env vars) — the EMPNORM_SHA lines
      must be IDENTICAL. Structural half of the cert: with the flag unset
      the trainer-side gate executes zero code and the env is untouched by
      this feature entirely, so flag-off bit-exactness is by construction;
      this stream is the regression harness that proves the sync itself
      changed nothing.

  freeze    (--cert freeze --checkpoint <mature.pt> [--no_freeze]): builds
      the real OnPolicyRunner (GRASP_EMPNORM=1 required — the cfg gates
      empirical_normalization on it), loads the checkpoint, hashes every
      normalizer tensor (_mean/_var/_std + count), runs N=--iters real
      learn() iterations, hashes again.
        default arm: freeze applied -> hashes MUST be identical.
        --no_freeze control arm: freeze NOT applied -> hashes MUST differ
        (proves the detector can see drift at all; a pass on the default
        arm without this control would be the classic vacuous cert).

  learn     (--cert learn --checkpoint <mature.pt> --iters 10): freeze
      applied, then real learn() iterations. PASS = normalizer hash still
      identical AND the actor weights moved (md5 of the policy state_dict
      changed) — optimization proceeds under a frozen normalizer. Mean
      on-policy reward over 100 steps is printed before/after as the
      "returns move" readout (short-horizon, so it is a readout, not a
      pass bar — 3-10 iters of PPO on 64 envs is noise-dominated).

Run on the pod (short, low-env; does NOT touch the running trainers):

    cd /workspace/humanoid/pod
    PYTHONPATH=. python -u g1_grasp/probe_empnorm_freeze.py \
        --cert bitexact --headless
    GRASP_EMPNORM=1 PYTHONPATH=. python -u g1_grasp/probe_empnorm_freeze.py \
        --cert freeze --checkpoint <mature.pt> --headless
    GRASP_EMPNORM=1 PYTHONPATH=. python -u g1_grasp/probe_empnorm_freeze.py \
        --cert freeze --checkpoint <mature.pt> --no_freeze --headless
    GRASP_EMPNORM=1 PYTHONPATH=. python -u g1_grasp/probe_empnorm_freeze.py \
        --cert learn --checkpoint <mature.pt> --iters 10 --headless
"""

import argparse
import hashlib
import os

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--cert", required=True,
                    choices=["bitexact", "freeze", "learn"])
parser.add_argument("--iters", type=int, default=3)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--no_freeze", action="store_true",
                    help="freeze-cert control arm: expect the hash to MOVE")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import g1_grasp  # noqa: F401
from g1_grasp.agents import apply_empnorm_freeze, find_empnorm_modules

SEED = 3407


def norm_hash(runner):
    """md5 over every normalizer buffer the runner owns (+ sample count).

    A missing buffer contributes an explicit ABSENT marker instead of being
    skipped: this certificate exists to detect a normalizer that silently
    became nn.Identity, so absence MUST change the digest. Raises if no
    normalizer module is found at all -- a clean hexdigest computed over
    nothing is precisely the false certificate this probe was written to
    catch (see GRASP_EMPNORM in docs/POSTMORTEM_failure_catalogue.md).
    """
    h = hashlib.md5()
    found = 0
    for tag, m in find_empnorm_modules(runner):
        found += 1
        h.update(tag.encode())
        for b in ("_mean", "_var", "_std"):
            t = getattr(m, b, None)
            if t is None:
                h.update(f"<ABSENT:{b}>".encode())
            else:
                h.update(t.detach().to("cpu", torch.float64).numpy().tobytes())
        count = getattr(m, "count", None)
        h.update(b"<ABSENT:count>" if count is None
                 else str(int(count)).encode())
    if found == 0:
        raise RuntimeError(
            "norm_hash: no empirical-normalization module on the runner. "
            "Refusing to return a digest over nothing."
        )
    return h.hexdigest()


def policy_module(runner):
    return (getattr(runner.alg, "policy", None)
            or getattr(runner.alg, "actor_critic", None))


def actor_hash(runner):
    """md5 over the whole policy state_dict — with the normalizer frozen,
    any change here is the OPTIMIZER moving weights."""
    h = hashlib.md5()
    for k, v in sorted(policy_module(runner).state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().to("cpu", torch.float64).numpy().tobytes())
    return h.hexdigest()


def eval_mean_reward(wenv, runner, steps=100):
    policy = runner.get_inference_policy(device=wenv.unwrapped.device)
    obs = wenv.get_observations()
    tot = 0.0
    for _ in range(steps):
        with torch.no_grad():
            acts = policy(obs)
        obs, rew, _, _ = wenv.step(acts)
        tot += float(rew.mean())
    return tot / steps


def make_runner():
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    torch.manual_seed(SEED)
    env_cfg = parse_env_cfg("G1-Grasp-v0", num_envs=args.num_envs)
    env_cfg.seed = SEED
    env = gym.make("G1-Grasp-v0", cfg=env_cfg)
    raw = env.unwrapped
    wenv = RslRlVecEnvWrapper(env)
    acfg = cli_args.parse_rsl_rl_cfg("G1-Grasp-v0", args).to_dict()
    # log_dir=None: no writer, no saves — nothing lands on disk
    runner = OnPolicyRunner(wenv, acfg, log_dir=None, device=raw.device)
    if args.checkpoint:
        runner.load(args.checkpoint)
    return env, raw, wenv, runner


def main():
    if args.cert == "bitexact":
        # CERT 1 — seeded env stream; flag-off trainer gate executes zero
        # code, so this is the sync-regression half (see docstring protocol)
        torch.manual_seed(SEED)
        env_cfg = parse_env_cfg("G1-Grasp-v0", num_envs=args.num_envs)
        env_cfg.seed = SEED
        env = gym.make("G1-Grasp-v0", cfg=env_cfg)
        raw = env.unwrapped
        env.reset()
        gen = torch.Generator().manual_seed(SEED)
        h = hashlib.md5()
        tot = 0.0
        for _ in range(100):
            a = (torch.rand(raw.num_envs, raw.cfg.action_space,
                            generator=gen) * 2 - 1).to(raw.device)
            out = env.step(a)
            obs = out[0]["policy"] if isinstance(out[0], dict) else out[0]
            h.update(obs.detach().to("cpu", torch.float32).numpy().tobytes())
            h.update(out[1].detach().to("cpu", torch.float32).numpy().tobytes())
            tot += float(out[1].to(torch.float64).sum())
        print(f"EMPNORM_SHA md5={h.hexdigest()} rew_sum64={tot:.10f} "
              f"envs={raw.num_envs}", flush=True)
        print("PROBE_EMPNORM_BITEXACT_RECORDED (identical line pre/post "
              "sync, same pod+GPU = PASS)", flush=True)
        return

    assert args.checkpoint, f"--cert {args.cert} needs --checkpoint"
    env, raw, wenv, runner = make_runner()
    mods = find_empnorm_modules(runner)
    assert mods, ("no EmpiricalNormalization on this runner — run with "
                  "GRASP_EMPNORM=1 (agents.py gates empirical_normalization "
                  "on it)")
    print(f"EMPNORM found {len(mods)} normalizer(s): "
          + ", ".join(t for t, _ in mods), flush=True)

    if args.cert == "freeze":
        # CERT 2 — statistics pinned through real learn() iterations
        if not args.no_freeze:
            frozen = apply_empnorm_freeze(runner)
            assert frozen, "freeze applied to zero modules"
        h0 = norm_hash(runner)
        runner.learn(num_learning_iterations=args.iters,
                     init_at_random_ep_len=True)
        h1 = norm_hash(runner)
        if args.no_freeze:
            ok = h0 != h1
            print(f"EMPNORM_FREEZE_CONTROL before={h0} after={h1}", flush=True)
            print(f"PROBE_EMPNORM_DRIFT_{'PASS' if ok else 'FAIL'} "
                  f"(control arm: unfrozen stats MUST move over "
                  f"{args.iters} iters — detector validity)", flush=True)
        else:
            ok = h0 == h1
            print(f"EMPNORM_FREEZE before={h0} after={h1}", flush=True)
            print(f"PROBE_EMPNORM_FREEZE_{'PASS' if ok else 'FAIL'} "
                  f"(frozen stats MUST be identical after {args.iters} "
                  f"learn iters)", flush=True)
        return

    # CERT 3 — learning proceeds under a frozen normalizer
    frozen = apply_empnorm_freeze(runner)
    assert frozen, "freeze applied to zero modules"
    n0 = norm_hash(runner)
    a0 = actor_hash(runner)
    r0 = eval_mean_reward(wenv, runner)
    runner.learn(num_learning_iterations=args.iters,
                 init_at_random_ep_len=True)
    n1 = norm_hash(runner)
    a1 = actor_hash(runner)
    r1 = eval_mean_reward(wenv, runner)
    ok = (n0 == n1) and (a0 != a1)
    print(f"EMPNORM_LEARN norm {n0} -> {n1} actor {a0[:12]} -> {a1[:12]} "
          f"mean_rew {r0:.3f} -> {r1:.3f} over {args.iters} iters", flush=True)
    print(f"PROBE_EMPNORM_LEARN_{'PASS' if ok else 'FAIL'} "
          f"(norm hash unchanged: {n0 == n1}; actor weights moved: "
          f"{a0 != a1}; reward readout above is informational — short "
          f"horizon)", flush=True)


if __name__ == "__main__":
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
