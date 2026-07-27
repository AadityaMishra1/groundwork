"""Zero-action cage-restore probe — the three-strikes observation.

Three unified runs (baseline / GRASP_LEG_STIFF=4.0 / init_noise_std 0.5)
produced the IDENTICAL grasped_now crawl (~0.008 at iter 300), so the cage
starts' failure is upstream of plant and exploration. This probe observes the
restore directly: force every reset to stage 1 (bank2 cage states), command
zero actions (delta space: zero = hold targets), and log the grasp/stance
trajectory. No inference — the state either survives physics or it does not.

Run under the EXACT training env config:

    GRASP_FULLBODY=1 GRASP_FREE=1 GRASP_LEG_STIFF=4.0 GRASP_EP_S=30 \
    GRASP_FLUNG_REF=robot GRASP_LEG_BLEND=1.0 GRASP_SPAWN_Z=0.80 \
    GRASP_BANK2=/workspace/chain_bank_grasp.pt GRASP_FORCE_STAGE=1 \
    python -u g1_grasp/probe_cage.py --headless

Readout guide:
  grasped@step1-2 ~0  -> restore itself broken (states never re-enter the
                         world grasped: frames, ordering, targets, or the
                         contact sensor needs steps to re-register)
  grasped high then decaying over 10-50 steps with pelvis sinking
                      -> stance collapse under the restored targets
  grasped high and FLAT with pelvis stable
                      -> restore fine; the training-time killer is the
                         policy's own actions and none of the three runs'
                         telemetry sampled early-episode steps densely
                         enough to see it (check sampling next, not physics)
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=250)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

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
    assert raw.bank2 is not None and len(raw.bank2) > 0, "bank2 empty"
    print(f"CAGE families: {list(raw.bank2.keys())}", flush=True)

    acts = torch.zeros(raw.num_envs, raw.cfg.action_space, device=raw.device)
    marks = {0, 1, 2, 5, 10, 25, 50, 100, 150, 200, 249}
    resets = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
    for step in range(args.steps):
        env.step(acts)
        nf, th = raw._finger_contacts()
        grasped = (nf >= 3) & th
        pz = robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
        q = robot.data.root_quat_w
        up = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        objz = raw.obj.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
        palm = robot.data.body_pos_w[:, raw.palm_id]
        d = (palm - raw.obj.data.root_pos_w).norm(dim=-1)
        resets |= raw.episode_length_buf < step  # env re-spawned mid-probe
        if step in marks:
            print(f"CAGE t={step:3d} grasped={grasped.float().mean():.3f} "
                  f"nf_p50={nf.float().median():.1f} thumb={th.float().mean():.2f} "
                  f"pelvis_p25/50={pz.quantile(0.25):.3f}/{pz.median():.3f} "
                  f"up_p50={up.median():.3f} objz_p50={objz.median():.3f} "
                  f"palm_d_p50={d.median():.3f} resets={int(resets.sum())}/64",
                  flush=True)
    print("CAGE_PROBE_DONE", flush=True)


if __name__ == "__main__":
    import os
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
