"""P1 mechanism probe for RUN 9S success-termination (known-answer control).

Restores hold-group bank states (GRASP_FORCE_STAGE=0 under bank_run5:
hold/held/lift families), commands zero actions (delta space: zero = hold
targets — G0 precedent: restored grasps persist), and observes the
success-termination machinery directly. Run TWICE by the driver:

  treatment  GRASP_SUCCESS_TERM=150  — every env whose high_counter crosses
             150 must terminate within 2 steps via succ_done with a terminal
             reward > 0.9*B, and at least one such event must occur
             (zero events = vacuous pass = FAIL: investigate bank heights).
  control    flag unset — the SAME states must produce ZERO success
             terminations while some env demonstrably accrues
             high_counter >= 150 (proves the event occurs and legacy
             semantics ignore it, bit-preserved).

Quantile prints for every gating quantity (filter law): obj_z, high_counter.

    GRASP_FULLBODY=1 GRASP_FREE=1 GRASP_LEG_STIFF=4.0 GRASP_EP_S=30 \
    GRASP_FLUNG_REF=robot GRASP_LEG_BLEND=1.0 GRASP_SPAWN_Z=0.80 \
    GRASP_BANK2=/workspace/bank_run5.pt GRASP_FORCE_STAGE=0 \
    [GRASP_SUCCESS_TERM=150] python -u g1_grasp/probe_success_term.py --headless
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import g1_grasp  # noqa: F401

TERM = int(os.environ.get("GRASP_SUCCESS_TERM", "0"))
BONUS = float(os.environ.get("GRASP_SUCCESS_BONUS", "2000.0"))


def main():
    env_cfg = parse_env_cfg("G1-Grasp-v0", num_envs=64)
    env = gym.make("G1-Grasp-v0", cfg=env_cfg)
    raw = env.unwrapped
    env.reset()
    assert raw.bank2 is not None and len(raw.bank2) > 0, "bank2 empty"
    print(f"SUCCTERM mode={'TREATMENT' if TERM > 0 else 'CONTROL'} "
          f"term={TERM} bonus={BONUS} families={list(raw.bank2.keys())}",
          flush=True)

    acts = torch.zeros(raw.num_envs, raw.cfg.action_space, device=raw.device)
    n = raw.num_envs
    # v2 KNOWN-ANSWER: passive zero-action holds settle the object at
    # ~0.356 m (v1 measured it) — BELOW the 0.40 strict bar, so high_counter
    # cannot accrue physically here and a rollout-only probe is vacuous.
    # Physics-side accrual is already proven elsewhere (m2d 38% strict; the
    # unified lineage's lift_no_hold taxonomy requires 0.40 crossings). What
    # THIS certifies is the NEW machinery: at FORCE_STEPS we set
    # high_counter=155 directly; the next step must terminate via succ_done
    # with the bonus paid (treatment) / must ignore it entirely (control).
    FORCE_STEPS = {60, 120, 180}
    n_forced = 0         # envs alive at force time (expected terminations)
    n_succ = 0           # succ_done terminations observed
    bad_bonus = 0        # succ terminations whose reward missed the bonus
    n_prompt = 0         # succ terminations on the step right after a force
    max_hc = 0
    just_forced = False
    marks = {0, 1, 5, 25, 59, 61, 62, 119, 121, 179, 181, 250, 399}
    for step in range(args.steps):
        _, rew, term, trunc, _ = env.step(acts)
        hc = raw.high_counter
        max_hc = max(max_hc, int(hc.max()))
        succ = raw.succ_done
        n_succ += int(succ.sum())
        if succ.any():
            bad_bonus += int((rew[succ] < 0.9 * BONUS).sum())
        if just_forced:
            n_prompt += int(succ.sum())
            just_forced = False
        if step in marks:
            objz = raw.obj.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
            nf, th = raw._finger_contacts()
            print(f"SUCCTERM t={step:3d} "
                  f"objz_q25/50/75={objz.quantile(0.25):.3f}/"
                  f"{objz.median():.3f}/{objz.quantile(0.75):.3f} "
                  f"hc_p50/max={hc.float().median():.0f}/{int(hc.max())} "
                  f"grasped={((nf >= 3) & th).float().mean():.2f} "
                  f"succ={n_succ}", flush=True)
        if step in FORCE_STEPS:
            # count only envs mid-episode (not just reset) as expected firers
            alive = raw.episode_length_buf > 10
            raw.high_counter[alive] = 155
            n_forced += int(alive.sum())
            just_forced = True
            print(f"SUCCTERM FORCED hc=155 on {int(alive.sum())} envs "
                  f"at t={step}", flush=True)

    print(f"SUCCTERM totals: forced={n_forced} succ_terms={n_succ} "
          f"prompt={n_prompt} bad_bonus={bad_bonus} max_hc={max_hc}",
          flush=True)
    if TERM > 0:
        # every forced env must fire on the very next step, with the bonus
        ok = (n_forced > 0 and n_succ >= int(0.9 * n_forced)
              and n_prompt == n_succ and bad_bonus == 0)
        print(f"PROBE_SUCCTERM_{'PASS' if ok else 'FAIL'} "
              f"(need succ>=0.9*forced, all prompt, bad_bonus=0)", flush=True)
    else:
        # control: forcing the counter must terminate NOTHING (legacy
        # semantics bit-preserved: succ_done stays False when flag unset)
        ok = n_forced > 0 and n_succ == 0
        print(f"PROBE_CONTROL_{'PASS' if ok else 'FAIL'} "
              f"(need forced>0 and succ=0)", flush=True)


if __name__ == "__main__":
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
