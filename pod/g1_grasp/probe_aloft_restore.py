"""ALOFT-RESTORE certificate: every aloft (phase>2) reference frame must
restore into a LIVE grasp — engine grasped at t=1 and the object NOT
free-falling under frozen restored targets.

Why: aloft restores from the gc bank showed engine grasp 0.000 at t=1 and the
object free-falling from ~0.445 m — the MuJoCo-derived palm offset places the
object outside the Inspire hand's actual grip, so the 15% REF_TRACK aloft
starts are inert. After the bank's aloft object placements are rewritten from
the measured in-grasp offset (probe_grasp_offset.py), THIS probe is the
pass/fail gate.

Frozen restore conditions: the env's own REF_TRACK reset does the full-state
restore (root+vel, joints, recorded PD targets, CLOSE_FING grip, object
pose+vel); the probe then steps ZERO actions, which freezes the delta
integrator at the restored targets. No policy anywhere.

    GRASP_FULLBODY=1 GRASP_FREE=1 GRASP_LEG_STIFF=4.0 GRASP_EP_S=30 \
    GRASP_FLUNG_REF=robot GRASP_POSTURE_GATE=1 GRASP_SPAWN_Z=0.80 \
    GRASP_REF_BANK=/workspace/ref_traj_bank_gc2.npz GRASP_NO_VIDEO=1 \
    python -u g1_grasp/probe_aloft_restore.py --headless \
        --num_envs 48 --rounds 60

Per round: reset all envs (uniform (ref, t0) with t0 >= --t0min, default 125
= first aloft frame), read each env's (ref, t0), then hold zero actions for
--hold_steps (default 25). Sample verdict:
    grasp1   engine grasp (>=3 groups + thumb) after the FIRST step
    drift    max |obj_z - z0| over the window, z0 = the bank's restore height
    fell     the env terminated inside the window
    PASS     grasp1 AND drift <= 0.05 AND not fell
Reports pass rates per t0 frame index (banked into 5-frame bins) plus the
aggregate PROBE_ALOFT_{PASS,FAIL} line (pass bar: >= 90% overall).
"""

import argparse
import os
import sys

# pod root on sys.path so `import g1_grasp` resolves (this file lives inside
# the package directory; sys.path[0] is g1_grasp/ when run as a script)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# forced BEFORE g1_grasp import — the module reads these at import time
os.environ["GRASP_REF_TRACK"] = "1"
os.environ["GRASP_REF_FRAC"] = "1.0"
os.environ["GRASP_FORCE_STAGE"] = "3"
os.environ["GRASP_FAR_FRAC"] = "0.0"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=48)
parser.add_argument("--rounds", type=int, default=60)
parser.add_argument("--hold_steps", type=int, default=25)
parser.add_argument("--t0min", type=int, default=125)
parser.add_argument("--drift", type=float, default=0.05)
parser.add_argument("--diag", action="store_true",
                    help="print per-step object telemetry + env term-type "
                         "counters for round 0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
os.environ["GRASP_REF_T0MIN"] = str(args.t0min)
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import g1_grasp  # noqa: F401, E402


def main():
    env_cfg = parse_env_cfg("G1-Grasp-v0", num_envs=args.num_envs)
    env = gym.make("G1-Grasp-v0", cfg=env_cfg)
    raw = env.unwrapped
    assert raw.ref_track and raw.ref_frac >= 1.0
    dev = raw.device
    n = raw.num_envs
    org = raw.scene.env_origins
    acts = torch.zeros(n, raw.cfg.action_space, device=dev)
    T = raw.ref_T

    stats = {}  # t0 -> [n, grasp1, drift_ok, fell, pass]
    tc0 = list(raw.term_counts)
    print(f"ALOFT_RESTORE bank R={raw.ref_R} T={T} t0min={args.t0min} "
          f"envs={n} rounds={args.rounds} hold={args.hold_steps}", flush=True)
    with torch.inference_mode():
        for rnd in range(args.rounds):
            env.reset()
            rid = raw.ref_id.clone()
            t0 = raw.ref_t0.clone()
            live = t0 >= 0
            z0 = raw.ref_obj[rid.clamp(min=0), t0.clamp(min=0), 2]
            # restore sanity: post-reset object z must match the bank
            objz = raw.obj.data.root_pos_w[:, 2] - org[:, 2]
            r0err = (objz - z0)[live].abs().max() if live.any() else 0.0
            grasp1 = torch.zeros(n, dtype=torch.bool, device=dev)
            maxdrift = torch.zeros(n, device=dev)
            fell = torch.zeros(n, dtype=torch.bool, device=dev)
            for k in range(args.hold_steps):
                out = env.step(acts)
                done = out[2] if isinstance(out, tuple) else out.terminated
                # a done env restores onto a NEW (ref,t0) mid-window: its
                # sample is a failure and it stops contributing
                newly = done & live & ~fell
                fell |= newly
                track = live & ~fell
                objz = raw.obj.data.root_pos_w[:, 2] - org[:, 2]
                maxdrift[track] = torch.maximum(maxdrift[track],
                                                (objz - z0)[track].abs())
                if k == 0:
                    nf, th = raw._finger_contacts()
                    grasp1 = (nf >= 3) & th & track
                if rnd == 0 and args.diag:
                    pd = (raw.robot.data.body_pos_w[:, raw.palm_id]
                          - raw.obj.data.root_pos_w).norm(dim=-1)
                    sp = raw.obj.data.root_lin_vel_w.norm(dim=-1)
                    dz = objz - z0
                    tcd = [raw.term_counts[i] - tc0[i] for i in range(6)]
                    print(f"DIAG k={k:2d} objz_p50={objz.median():.3f} "
                          f"dz_p10/50/90={dz.quantile(.1):+.3f}/"
                          f"{dz.median():+.3f}/{dz.quantile(.9):+.3f} "
                          f"palmd_p50/90={pd.median():.3f}/{pd.quantile(.9):.3f} "
                          f"spd_p50/max={sp.median():.2f}/{sp.max():.2f} "
                          f"terms(held/flung/tip/to/succ/c5)={tcd}", flush=True)
            ok = grasp1 & (maxdrift <= args.drift) & ~fell & live
            for i in range(n):
                if not bool(live[i]):
                    continue
                t = int(t0[i])
                s = stats.setdefault(t, [0, 0, 0, 0, 0])
                s[0] += 1
                s[1] += int(grasp1[i])
                s[2] += int(maxdrift[i] <= args.drift and not fell[i])
                s[3] += int(fell[i])
                s[4] += int(ok[i])
            if rnd % 10 == 0 or rnd == args.rounds - 1:
                tot = sum(s[0] for s in stats.values())
                g = sum(s[1] for s in stats.values())
                p = sum(s[4] for s in stats.values())
                print(f"ALOFT rnd={rnd:3d} r0err={float(r0err):.4f} "
                      f"samples={tot} grasp1={g/max(tot,1):.3f} "
                      f"pass={p/max(tot,1):.3f}", flush=True)

    # per-frame-index verdicts, binned by 5 frames for the printout
    bins = {}
    for t, s in sorted(stats.items()):
        b = (t // 5) * 5
        acc = bins.setdefault(b, [0, 0, 0, 0, 0])
        for j in range(5):
            acc[j] += s[j]
    print("ALOFT_BY_FRAME  bin      n  grasp1  driftok  fell  pass", flush=True)
    for b, s in sorted(bins.items()):
        print(f"ALOFT_BIN {b:4d}-{b+4:3d} {s[0]:6d}  {s[1]/max(s[0],1):.3f}"
              f"   {s[2]/max(s[0],1):.3f}  {s[3]:4d}  {s[4]/max(s[0],1):.3f}",
              flush=True)
    tcd = [raw.term_counts[i] - tc0[i] for i in range(6)]
    print(f"ALOFT_TERMS held/flung/tip/timeout/succ/c5 = {tcd} "
          f"(env-accumulated over the probe; zeroed only by the env's own "
          f"2400-step telemetry, which a short probe never reaches)", flush=True)
    tot = sum(s[0] for s in stats.values())
    g = sum(s[1] for s in stats.values())
    d = sum(s[2] for s in stats.values())
    f = sum(s[3] for s in stats.values())
    p = sum(s[4] for s in stats.values())
    frac = p / max(tot, 1)
    print(f"ALOFT_SUMMARY n={tot} grasp1={g/max(tot,1):.3f} "
          f"driftok={d/max(tot,1):.3f} fell={f} pass={frac:.3f} "
          f"frames_covered={len(stats)}/{T - args.t0min}", flush=True)
    print(f"PROBE_ALOFT_{'PASS' if frac >= 0.90 else 'FAIL'} "
          f"(pass={frac:.3f} need >=0.90)", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
