"""CERT-GRACE-ANNEAL: known-answer certificates for GRASP_HC_GRACE_ANNEAL.

Three certificates:

  --mode schedule: builds the env with START=5, GRACE=2, ANNEAL=10000,
      then reads the SHIPPED _hc_grace_k() at forced common_step_counter
      values. Known answers (int(round(lerp)), banker's rounding):
        t=0 -> 5   t=2500 -> 4 (4.25)   t=5000 -> 4 (3.5, half-even)
        t=7500 -> 3 (2.75)   t=10000 -> 2   t=10^9 -> 2
      plus a 101-point monotonicity sweep. ANNEAL=0 bit-exactness is
      structural (_hc_grace_k returns self.hc_grace before any arithmetic;
      the reward-site gate and threshold then reduce to the pre-anneal
      expressions) and measured by the shared seeded-stream harness
      (probe_empnorm_freeze --cert bitexact, all flags unset).

  --mode counter-anneal: START=3, GRACE=0, ANNEAL=1e9 (so K(t)=3 and
      STABLE over the probe window while the ANNEAL code path — not the
      static branch — is the one executing). Presets high_counter=10,
      _hc_miss=0 on all envs, objects parked on the floor (_hc_ok False
      every step), then steps and records the high_counter trajectory:
      known answer HOLD 10,10,10 for misses 1..3 then ZERO from miss 4.

  --mode counter-static: GRACE=3, ANNEAL=0 (the pre-existing static
      path). Identical procedure — the printed trajectory must EQUAL the
      counter-anneal one line for line: a mid-anneal K behaves identically
      to a static K of the same value (the forced-counter pattern from
      the posture-gate battery).

Run (low-env, short):

    PYTHONPATH=. python -u g1_grasp/probe_hc_grace.py --mode schedule --headless
    PYTHONPATH=. python -u g1_grasp/probe_hc_grace.py --mode counter-anneal --headless
    PYTHONPATH=. python -u g1_grasp/probe_hc_grace.py --mode counter-static --headless
"""

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True,
                    choices=["schedule", "counter-anneal", "counter-static"])
parser.add_argument("--num_envs", type=int, default=4)

_known, _ = parser.parse_known_args()
os.environ["GRASP_FORCE_STAGE"] = "3"
if _known.mode == "schedule":
    os.environ["GRASP_HC_GRACE"] = "2"
    os.environ["GRASP_HC_GRACE_START"] = "5"
    os.environ["GRASP_HC_GRACE_ANNEAL"] = "10000"
elif _known.mode == "counter-anneal":
    os.environ["GRASP_HC_GRACE"] = "0"
    os.environ["GRASP_HC_GRACE_START"] = "3"
    os.environ["GRASP_HC_GRACE_ANNEAL"] = "1000000000"
else:  # counter-static
    os.environ["GRASP_HC_GRACE"] = "3"
    os.environ["GRASP_HC_GRACE_ANNEAL"] = "0"

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import g1_grasp  # noqa: F401


def main():
    env_cfg = parse_env_cfg("G1-Grasp-v0", num_envs=args.num_envs)
    env = gym.make("G1-Grasp-v0", cfg=env_cfg)
    raw = env.unwrapped
    env.reset()

    if args.mode == "schedule":
        known = {0: 5, 2500: 4, 5000: 4, 7500: 3, 10000: 2, 10 ** 9: 2}
        saved = raw.common_step_counter
        fails = []
        for t, want in known.items():
            raw.common_step_counter = t
            got = raw._hc_grace_k()
            print(f"GRACE_K t={t} K={got} want={want}", flush=True)
            if got != want:
                fails.append(t)
        prev = 10 ** 9
        mono = True
        for i in range(101):
            raw.common_step_counter = int(10000 * i / 100)
            k = raw._hc_grace_k()
            if k > prev:
                mono = False
            prev = k
        raw.common_step_counter = saved
        ok = not fails and mono
        print(f"PROBE_GRACE_SCHEDULE_{'PASS' if ok else 'FAIL'} "
              f"(known-answer misses={fails} monotone={mono})", flush=True)
        return

    # counter modes: identical scripted trajectory, printed for line-diff
    acts = torch.zeros(raw.num_envs, raw.cfg.action_space, device=raw.device)
    env.step(acts)  # one settle step so the print starts on a clean state
    k_now = raw._hc_grace_k()
    raw.high_counter[:] = 10
    raw._hc_miss[:] = 0
    traj = []
    for step in range(8):
        env.step(acts)  # object on floor -> _hc_ok False -> pure misses
        traj.append(int(raw.high_counter[0]))
        print(f"GRACE_TRAJ step={step + 1} miss={int(raw._hc_miss[0])} "
              f"hc={int(raw.high_counter[0])}", flush=True)
    # known answer for K=3: held for misses 1..3, zeroed from miss 4
    want = [10, 10, 10, 0, 0, 0, 0, 0]
    ok = k_now == 3 and traj == want
    print(f"PROBE_GRACE_COUNTER_{'PASS' if ok else 'FAIL'} mode={args.mode} "
          f"(K={k_now} need 3; traj={traj} want={want}; the two counter-"
          f"mode invocations must print identical GRACE_TRAJ lines)",
          flush=True)


if __name__ == "__main__":
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
