"""CERT-SPAWN: known-answer certificates for the protocol spawn axes.

Two modes, one per invocation (low-env, resets only — no training):

  --mode protocol  (forces GRASP_PROTOCOL_SPAWN=1, GRASP_FORCE_STAGE=3):
      histogram over >=4096 fresh resets. PASS requires ALL of:
        dist   min >= 1.0 and max <= 4.0 (+5 mm write slop) and
               mean in [2.35, 2.65]  (U(1,4) -> 2.5)
        bearing (object bearing in the ROBOT frame) circular-uniform:
               resultant length Rbar <= 0.06  (Rayleigh E[Rbar] ~ 0.014
               at n=4096; 0.06 is > 4x that)
        yaw    robot yaw circular-uniform over (-pi, pi]: Rbar <= 0.06
        up     roll/pitch untouched: min over all spawns of the root
               up-component (1 - 2(qx^2+qy^2)) >= 0.9999
      The env prints its own [protocol-spawn] standing-only disclaimer at
      init — this cert intentionally does NOT test a lying object.

  --mode yaw  (forces GRASP_SPAWN_YAW_MAX=pi, GRASP_FORCE_STAGE=3):
      same yaw + up assertions as above, PLUS the object placement must
      remain the TRAINED corridor (dist within [rmin, rmax] + slop) —
      YAW_MAX must randomize the robot heading and touch nothing else.
      (YAW_MAX=0 bit-exactness is structural — the sampler never runs —
      and is measured by the shared seeded-stream harness:
      probe_empnorm_freeze --cert bitexact with all flags unset.)

Run with NO bank env vars (GRASP_BANK2/GRASP_START_BANK unset — fresh
spawns only; the probe asserts this) e.g.:

    PYTHONPATH=. python -u g1_grasp/probe_spawn_protocol.py \
        --mode protocol --headless
    PYTHONPATH=. python -u g1_grasp/probe_spawn_protocol.py \
        --mode yaw --headless
"""

import argparse
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=["protocol", "yaw"])
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--samples", type=int, default=4096)

_known, _ = parser.parse_known_args()
os.environ["GRASP_FORCE_STAGE"] = "3"
if _known.mode == "protocol":
    os.environ["GRASP_PROTOCOL_SPAWN"] = "1"
else:
    os.environ["GRASP_SPAWN_YAW_MAX"] = str(math.pi)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import g1_grasp  # noqa: F401


def circ_resultant(ang):
    """Mean resultant length of a circular sample — ~0 for uniform."""
    return float(torch.sqrt(torch.cos(ang).mean() ** 2
                            + torch.sin(ang).mean() ** 2))


def main():
    env_cfg = parse_env_cfg("G1-Grasp-v0", num_envs=args.num_envs)
    env = gym.make("G1-Grasp-v0", cfg=env_cfg)
    raw = env.unwrapped
    # fresh spawns only — any restore family would legitimately violate
    # the spawn law and corrupt the histogram
    assert raw.arrival_jp is None, "unset GRASP_START_BANK for this cert"
    assert raw.bank2 is None or len(raw.bank2) == 0, \
        "unset GRASP_BANK2 for this cert"
    assert not raw.ref_track, "unset GRASP_REF_TRACK for this cert"
    if args.mode == "protocol":
        assert raw.protocol_spawn and raw.flung_from_robot
    else:
        assert raw.spawn_yaw_max > 3.14 and not raw.protocol_spawn

    # STEP-SURVIVAL check (outside-review 2026-07-30): the v1 cert only
    # reset and histogrammed — it structurally could not see the flung
    # trap (robot-referenced flung at 3.0 m < the protocol's 4.0 m max
    # spawn), which killed ~1/3 of protocol episodes at step 1 before the
    # robot moved. Certify by STEPPING: over 5 zero-action steps, no env
    # spawned inside the legal distance range may terminate for any
    # non-truncation reason. Zero actions cannot fling an object 4 m.
    if args.mode == "protocol":
        env.reset()
        d0 = (raw.obj.data.root_pos_w[:, :2]
              - raw.robot.data.root_pos_w[:, :2]).norm(dim=-1)
        legal = (d0 >= 1.0) & (d0 <= 4.005)
        acts0 = torch.zeros(raw.num_envs, raw.cfg.action_space,
                            device=raw.device)
        died = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
        for _k in range(5):
            out = env.step(acts0)
            died |= (out[2] if isinstance(out, tuple) else out.terminated)
        n_dead = int((died & legal).sum())
        print(f"STEP_SURVIVAL legal={int(legal.sum())}/{raw.num_envs} "
              f"dead_by_step5={n_dead} "
              f"d0_max={float(d0.max()):.3f} "
              f"flung_dist={raw.flung_robot_dist:g}", flush=True)
        if n_dead > 0:
            print("PROBE_SPAWNPROT_FAIL (step-survival: legal spawns "
                  "terminated under zero action — the step-1 flung trap "
                  "or an equivalent is live)", flush=True)
            import sys
            sys.stdout.flush()
            os._exit(1)

    dists, bearings, yaws, ups = [], [], [], []
    n_resets = (args.samples + args.num_envs - 1) // args.num_envs
    for k in range(n_resets):
        env.reset()  # re-runs _reset_idx for every env
        q = raw.robot.data.root_quat_w
        yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        up = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        d_xy = raw.obj.data.root_pos_w[:, :2] - raw.robot.data.root_pos_w[:, :2]
        dist = d_xy.norm(dim=-1)
        # object bearing in the ROBOT frame (protocol: bearing 0-360 deg)
        bear = torch.atan2(d_xy[:, 1], d_xy[:, 0]) - yaw
        bear = torch.atan2(torch.sin(bear), torch.cos(bear))  # wrap
        dists.append(dist.cpu())
        bearings.append(bear.cpu())
        yaws.append(yaw.cpu())
        ups.append(up.cpu())
    dist = torch.cat(dists)[:args.samples]
    bear = torch.cat(bearings)[:args.samples]
    yaw = torch.cat(yaws)[:args.samples]
    up = torch.cat(ups)[:args.samples]

    r_yaw = circ_resultant(yaw)
    r_bear = circ_resultant(bear)
    print(f"SPAWN_HIST n={len(dist)} dist_min/mean/max="
          f"{dist.min():.3f}/{dist.mean():.3f}/{dist.max():.3f} "
          f"bear_Rbar={r_bear:.4f} yaw_Rbar={r_yaw:.4f} "
          f"yaw_min/max={yaw.min():.3f}/{yaw.max():.3f} "
          f"up_min={up.min():.6f}", flush=True)
    # decile counts — the eyeball histogram for the report
    edges = torch.linspace(-math.pi, math.pi, 11)
    print("YAW_DECILES " + " ".join(
        str(int(((yaw >= edges[i]) & (yaw < edges[i + 1])).sum()))
        for i in range(10)), flush=True)

    up_ok = float(up.min()) >= 0.9999
    yaw_ok = r_yaw <= 0.06
    if args.mode == "protocol":
        d_ok = (float(dist.min()) >= 1.0
                and float(dist.max()) <= 4.005
                and 2.35 <= float(dist.mean()) <= 2.65)
        b_ok = r_bear <= 0.06
        ok = d_ok and b_ok and yaw_ok and up_ok
        print(f"PROBE_SPAWNPROT_{'PASS' if ok else 'FAIL'} "
              f"(dist_ok={d_ok} bear_ok={b_ok} yaw_ok={yaw_ok} "
              f"up_ok={up_ok}; REMINDER: standing-only — lying-object "
              f"clause not implemented)", flush=True)
    else:
        # corridor untouched: rmin/rmax from the env (defaults 0.20-0.35),
        # + slop for the base-xy vs env-origin measurement
        d_ok = (float(dist.min()) >= raw.obj_rmin - 0.02
                and float(dist.max()) <= raw.obj_rmax + 0.02)
        ok = d_ok and yaw_ok and up_ok
        print(f"PROBE_SPAWNYAW_{'PASS' if ok else 'FAIL'} "
              f"(corridor_ok={d_ok} [{raw.obj_rmin:.2f},{raw.obj_rmax:.2f}] "
              f"yaw_ok={yaw_ok} up_ok={up_ok})", flush=True)


if __name__ == "__main__":
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
