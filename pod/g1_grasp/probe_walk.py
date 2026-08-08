"""Can the embedded walker actually walk inside the grasp env?

The chain runner reached kneel 0/128 with zero falls — the robot stands but
never arrives. This drives a constant forward velocity command straight into
`loco_vel_cmd` and measures base displacement, which separates "the walker
cannot walk here" from "the runner's standoff servo is wrong".

    GRASP_FREE=1 GRASP_FORCE_STAGE=3 GRASP_LEG_STIFF=1.0 GRASP_LEG_BLEND=1.0 \
    GRASP_SPAWN_Z=0.80 GRASP_LOCO_CKPT=... python -u g1_grasp/probe_walk.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--vx", type=float, default=0.5)
parser.add_argument("--sweep", type=str, default="",
                    help="comma-separated vx values, all in ONE boot "
                         "(a 5 min Isaac Sim start per hypothesis is the "
                         "real bottleneck, not GPU time)")
parser.add_argument("--hcmd", type=float, default=0.72)
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import os

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import g1_grasp  # noqa: F401


def main():
    env_cfg = parse_env_cfg("G1-Grasp-v0", num_envs=64)
    env = gym.make("G1-Grasp-v0", cfg=env_cfg)
    raw = env.unwrapped
    env.reset()
    robot = raw.robot

    for k in ("loco_vel_cmd", "loco_hcmd_t", "suppress_grasp", "park_now"):
        if not hasattr(raw, k):
            raise RuntimeError(f"env missing {k} — walker not loaded?")

    acts = torch.zeros(raw.num_envs, raw.cfg.action_space, device=raw.device)

    def _stash_object():
        # The object spawns 0.20-0.35 m front-right, i.e. straight in the
        # walking path. Park it BEHIND the robot — and inside 1.2 m of the
        # env origin, because _get_dones treats >1.2 m as "flung" and would
        # reset every single step.
        st = raw.obj.data.root_state_w.clone()
        st[:, 0] = raw.scene.env_origins[:, 0] - 1.0
        st[:, 1] = raw.scene.env_origins[:, 1]
        st[:, 2] = raw.scene.env_origins[:, 2] + 0.075
        st[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=raw.device)
        raw.obj.write_root_pose_to_sim(st[:, :7])
        raw.obj.write_root_velocity_to_sim(torch.zeros_like(st[:, 7:]))

    # object state must not be able to end an episode while we measure gait
    def _walk_dones():
        pz = robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
        q = robot.data.root_quat_w
        u = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        trunc = raw.episode_length_buf >= raw.max_episode_length - 1
        return (pz < 0.10) | (u < 0.0), trunc

    raw._get_dones = _walk_dones

    def run(vx):
        env.reset()
        _stash_object()
        n = raw.num_envs
        p0 = (robot.data.root_pos_w[:, :2] - raw.scene.env_origins[:, :2]).clone()
        alive = torch.ones(n, dtype=torch.bool, device=raw.device)
        # steps survived before the first fall — the number that decides
        # whether a 1.2 m approach is reachable at this speed
        life = torch.zeros(n, device=raw.device)
        best = torch.zeros(n, device=raw.device)
        for step in range(args.steps):
            _stash_object()
            raw.loco_vel_cmd[:, 0] = vx
            raw.loco_vel_cmd[:, 1] = 0.0
            raw.loco_vel_cmd[:, 2] = 0.0
            raw.loco_hcmd_t[:] = args.hcmd
            raw.suppress_grasp[:] = True
            raw.park_now[:] = False
            env.step(acts)
            z = robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
            q = robot.data.root_quat_w
            up = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
            down = (z < 0.35) | (up < 0.6)
            life = torch.where(alive & ~down, life + 1.0, life)
            alive = alive & ~down
            d = (robot.data.root_pos_w[:, :2]
                 - raw.scene.env_origins[:, :2] - p0).norm(dim=-1)
            best = torch.where(alive, torch.maximum(best, d), best)
        secs = life * 4 / 200.0
        print(f"SWEEP vx={vx:.2f} alive_end={int(alive.sum())}/{n} "
              f"upright_secs_p25/50/75={secs.quantile(0.25):.2f}/"
              f"{secs.median():.2f}/{secs.quantile(0.75):.2f} "
              f"dist_while_upright_p50={best.median():.3f}m "
              f"p75={best.quantile(0.75):.3f}m", flush=True)

    vxs = ([float(v) for v in args.sweep.split(",")] if args.sweep
           else [args.vx])
    for vx in vxs:
        run(vx)


if __name__ == "__main__":
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
