"""CERT-PBRS: known-answer certificates for the miss-branch #2 PBRS term.

Five certificates, one per invocation (--cert), all low-env and short —
certificates before training, per the founding directives. The probe reads
raw._pbrs_last (the env's OWN per-step term) so every number certifies the
shipped code path — formula, lag buffer, and reset seeding — not a formula
re-derived beside it. Expected values are recomputed independently from the
MEASURED object z after each step, so physics slop (the ~2 mm gravity sag
between a kinematic pose write and the reward read) cancels out of the
comparison instead of eating the tolerance.

  ctrl   GRASP_PBRS_C forcibly unset. 100 seeded steps; prints the reward
         stream's float64 sum + md5 of its float32 bytes. PROTOCOL: run
         this cert on the PRE-patch build and on the patched build (same
         pod, same GPU) — the two CTRL_SHA lines must be IDENTICAL. Also
         asserts no PBRS buffers were allocated (C=0 = "no buffers
         touched", not "buffers of zeros").

  raise  C=300: env 0's object kinematically forced floor -> 0.45 m over
         75 steps (pose written before each step, zero velocity). Summed
         r_pbrs must equal the closed form  d_phi - (1-gamma)*sum(phi)
         within +-10% (the coordinator's ~+139 estimate at C=300 is
         reported next to the measured and closed-form numbers).

  pump   C=300: up over 75 steps then back down over 75. sum must be
         STRICTLY NEGATIVE and ~= -(1-gamma)*sum(phi) +-10% — pumping the
         object cannot be income.

  camp   C=300: nobody touches anything for 500 steps. |sum| < eps
         (eps = 0.02*C): a parked floor object earns nothing.

  reset  C=300, GRASP_FORCE_STAGE=0 (bank2 hold/held/lift machinery — the
         aloft-restore start family; requires GRASP_BANK2 pointing at the
         gc2 bank). Asserts the first-step term is ~ -(1-gamma)*phi0 with
         NO +-phi0-scale spike, then forces a mid-probe reset (teleports
         env 0's object past the 1.2 m flung line) and asserts the first
         post-reset step is spike-free too. This is the radioactive
         boundary: prev_phi must be re-seeded from the RESTORED object,
         never carried across a done.

Run on pod B alongside training (probes short, low-env, separate session):

    cd /workspace/humanoid/pod
    PYTHONPATH=. python -u g1_grasp/probe_pbrs.py --cert ctrl  --headless
    PYTHONPATH=. python -u g1_grasp/probe_pbrs.py --cert raise --headless
    PYTHONPATH=. python -u g1_grasp/probe_pbrs.py --cert pump  --headless
    PYTHONPATH=. python -u g1_grasp/probe_pbrs.py --cert camp  --headless
    GRASP_BANK2=/workspace/chain_bank_grasp.pt \
    PYTHONPATH=. python -u g1_grasp/probe_pbrs.py --cert reset --headless
"""

import argparse
import hashlib
import os

parser = argparse.ArgumentParser()
parser.add_argument("--cert", required=True,
                    choices=["ctrl", "raise", "pump", "camp", "reset"])
parser.add_argument("--num_envs", type=int, default=4)

# env vars BEFORE the app/env import chain — per-cert configuration
_known, _ = parser.parse_known_args()
if _known.cert == "ctrl":
    # the control is the CURRENT build: term fully off, C genuinely unset
    os.environ.pop("GRASP_PBRS_C", None)
else:
    os.environ.setdefault("GRASP_PBRS_C", "300")
if _known.cert == "reset":
    os.environ["GRASP_FORCE_STAGE"] = "0"   # aloft bank2 restores
else:
    os.environ.setdefault("GRASP_FORCE_STAGE", "3")  # plain corridor spawns

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import g1_grasp  # noqa: F401

SEED = 3407


def make_env():
    torch.manual_seed(SEED)
    env_cfg = parse_env_cfg("G1-Grasp-v0", num_envs=args.num_envs)
    env_cfg.seed = SEED
    env = gym.make("G1-Grasp-v0", cfg=env_cfg)
    raw = env.unwrapped
    env.reset()
    return env, raw


def phi_of(raw, z):
    """Independent recomputation of phi from measured z (same constants)."""
    return raw.pbrs_c * min(1.0, max(0.0, (z - raw.pbrs_z0)
                                     / (raw.pbrs_z1 - raw.pbrs_z0)))


def obj_z(raw, i=0):
    return float(raw.obj.data.root_pos_w[i, 2] - raw.scene.env_origins[i, 2])


def force_z(raw, z_env, i=0):
    """Kinematically place env i's object at env-frame z (xy kept, zero vel)."""
    ids = torch.tensor([i], device=raw.device)
    pose = torch.zeros(1, 7, device=raw.device)
    pose[0, :2] = raw.obj.data.root_pos_w[i, :2]
    pose[0, 2] = raw.scene.env_origins[i, 2] + z_env
    pose[0, 3] = 1.0  # upright — never trips the tipped termination
    raw.obj.write_root_pose_to_sim(pose, ids)
    raw.obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=raw.device), ids)


def drive_profile(env, raw, z_cmds, label):
    """Force env 0 through z_cmds; return (measured_sum, expected_sum, sum_phi).

    expected_sum = sum over steps of gamma*phi(z_t) - phi(z_{t-1}) computed
    from MEASURED z — independent of the env's lag buffer, same physics.
    """
    acts = torch.zeros(raw.num_envs, raw.cfg.action_space, device=raw.device)
    z_prev = obj_z(raw)
    measured, expected, sum_phi = 0.0, 0.0, 0.0
    g = raw.pbrs_gamma
    for k, zc in enumerate(z_cmds):
        force_z(raw, zc)
        env.step(acts)
        z_now = obj_z(raw)
        measured += float(raw._pbrs_last[0])
        expected += g * phi_of(raw, z_now) - phi_of(raw, z_prev)
        sum_phi += phi_of(raw, z_now)
        z_prev = z_now
        if k % 25 == 0 or k == len(z_cmds) - 1:
            print(f"{label} t={k:3d} z_cmd={zc:.3f} z_meas={z_now:.3f} "
                  f"r_pbrs={float(raw._pbrs_last[0]):+.3f} "
                  f"sum={measured:+.2f}", flush=True)
    return measured, expected, sum_phi


def main():
    env, raw = make_env()
    n = raw.num_envs

    if args.cert == "ctrl":
        # CERT 1 — C unset must mean untouched: no buffers, no term
        assert not hasattr(raw, "prev_phi"), "C=0 allocated prev_phi"
        assert not hasattr(raw, "_pbrs_last"), "C=0 allocated _pbrs_last"
        # getattr: the pre-patch build (CTRL-PRE leg) has no pbrs attributes
        # at all — that absence IS the control condition, not an error
        assert getattr(raw, "pbrs_c", 0.0) == 0.0
        gen = torch.Generator().manual_seed(SEED)
        stream = []
        for _ in range(100):
            a = (torch.rand(n, raw.cfg.action_space, generator=gen) * 2
                 - 1).to(raw.device)
            out = env.step(a)
            stream.append(out[1].detach().to("cpu", torch.float32))
        buf = torch.cat(stream).numpy()
        sha = hashlib.md5(buf.tobytes()).hexdigest()
        print(f"CTRL_SHA md5={sha} sum64={float(buf.astype(np.float64).sum()):.10f} "
              f"n={buf.size}", flush=True)
        print("PROBE_PBRS_CTRL_RECORDED (bit-exactness = this line identical "
              "on pre-patch and patched builds, same pod+GPU)", flush=True)
        return

    if args.cert == "raise":
        # CERT 2 — floor -> 0.45 over 75 steps
        z_cmds = [0.075 + (0.45 - 0.075) * (k + 1) / 75 for k in range(75)]
        m, e, sp = drive_profile(env, raw, z_cmds, "RAISE")
        ok = e > 0 and abs(m - e) <= 0.10 * abs(e)
        print(f"RAISE_SUM measured={m:+.2f} closed_form={e:+.2f} "
              f"(coordinator est ~+139 at C=300; leak=(1-g)*sum_phi="
              f"{(1 - raw.pbrs_gamma) * sp:.2f})", flush=True)
        print(f"PROBE_PBRS_RAISE_{'PASS' if ok else 'FAIL'} "
              f"(|{m:.2f} - {e:.2f}| need <= {0.10 * abs(e):.2f})", flush=True)
        return

    if args.cert == "pump":
        # CERT 3 — up 75, down 75: net height zero, sum must be negative
        up = [0.075 + (0.45 - 0.075) * (k + 1) / 75 for k in range(75)]
        z_cmds = up + up[::-1][1:] + [0.075]
        m, e, sp = drive_profile(env, raw, z_cmds, "PUMP")
        ok = m < 0 and abs(m - e) <= 0.10 * abs(e)
        print(f"PUMP_SUM measured={m:+.2f} closed_form={e:+.2f} "
              f"-(1-g)*sum_phi={-(1 - raw.pbrs_gamma) * sp:+.2f}", flush=True)
        print(f"PROBE_PBRS_PUMP_{'PASS' if ok else 'FAIL'} "
              f"(need strictly negative and within 10% of closed form)",
              flush=True)
        return

    if args.cert == "camp":
        # CERT 4 — parked floor object, 500 steps, no income
        acts = torch.zeros(n, raw.cfg.action_space, device=raw.device)
        total = 0.0
        resets = 0
        prev_ep = raw.episode_length_buf.clone()
        for k in range(500):
            env.step(acts)
            total += float(raw._pbrs_last[0])
            resets += int((raw.episode_length_buf < prev_ep).sum())
            prev_ep = raw.episode_length_buf.clone()
            if k % 100 == 0 or k == 499:
                print(f"CAMP t={k:3d} z={obj_z(raw):.4f} sum={total:+.4f} "
                      f"resets={resets}", flush=True)
        eps = 0.02 * abs(raw.pbrs_c)
        ok = abs(total) < eps and resets == 0
        print(f"CAMP_SUM measured={total:+.4f} eps={eps:.2f} resets={resets}",
              flush=True)
        print(f"PROBE_PBRS_CAMP_{'PASS' if ok else 'FAIL'} "
              f"(|sum| need < {eps:.2f}, resets need 0)", flush=True)
        return

    # CERT 5 — reset boundary with aloft restores (FORCE_STAGE=0 + bank2)
    assert raw.bank2 is not None and raw.bank2g.get(0) is not None, \
        "reset cert needs GRASP_BANK2 with a hold/held/lift group"
    acts = torch.zeros(n, raw.cfg.action_space, device=raw.device)
    phi0 = raw.prev_phi.clone()          # phi as seeded by the restore
    assert float(phi0.max()) > 0, "aloft restore produced phi0=0 — not aloft?"
    env.step(acts)
    first = raw._pbrs_last.clone()
    leak = (1 - raw.pbrs_gamma) * phi0
    # spike test per env with phi0 meaningfully aloft: the bug modes are
    # ~ +gamma*phi0 (prev_phi lost) or ~ -phi0 (double-counted) — both
    # phi0-scale; correct is ~ -(1-gamma)*phi0 plus ~mm settling slop
    sel = phi0 > 10.0
    if not bool(sel.any()):
        sel = phi0 > 0.0  # degenerate bank: fall back to any-aloft envs
    spike_free = bool((first[sel].abs() <= 0.10 * phi0[sel]).all())
    print(f"RESET first-step: phi0_p50={float(phi0.median()):.1f} "
          f"term_p50={float(first.median()):+.3f} "
          f"-(1-g)*phi0_p50={float(-leak.median()):+.3f} "
          f"max|term|/phi0={float((first[sel].abs() / phi0[sel]).max()):.4f}",
          flush=True)
    # settle a few steps, then force a mid-probe reset of env 0 via the
    # flung line (teleport 5 m out at the SAME z — phi is z-only, so even
    # the teleport step itself must be spike-free)
    for _ in range(20):
        env.step(acts)
    z_keep = obj_z(raw)
    ids = torch.tensor([0], device=raw.device)
    pose = torch.zeros(1, 7, device=raw.device)
    pose[0, 0] = raw.scene.env_origins[0, 0] + 5.0
    pose[0, 1] = raw.scene.env_origins[0, 1]
    pose[0, 2] = raw.scene.env_origins[0, 2] + z_keep
    pose[0, 3] = 1.0
    raw.obj.write_root_pose_to_sim(pose, ids)
    raw.obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=raw.device), ids)
    env.step(acts)                       # flung fires -> env 0 resets
    reset_happened = bool(raw.episode_length_buf[0] == 0)
    phi_new = float(raw.prev_phi[0])     # seeded from the fresh restore
    env.step(acts)
    post = float(raw._pbrs_last[0])
    post_ok = abs(post) <= max(0.10 * phi_new, 1.0)
    print(f"RESET mid-probe: reset_fired={reset_happened} "
          f"phi_new={phi_new:.1f} first_post_term={post:+.3f} "
          f"(phi-scale spike would be ~{phi_new:.0f})", flush=True)
    ok = spike_free and reset_happened and post_ok
    print(f"PROBE_PBRS_RESET_{'PASS' if ok else 'FAIL'} "
          f"(first|term|<=0.10*phi0 all aloft envs: {spike_free}; "
          f"mid-probe reset fired: {reset_happened}; post-reset "
          f"|{post:.3f}| <= max(0.10*{phi_new:.1f}, 1.0): {post_ok})",
          flush=True)


if __name__ == "__main__":
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
