"""Introspect the feet_only_contact gate: what does the sensor actually see?

    python -u g1_loco/probe_gate.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import g1_loco  # noqa: F401
from g1_loco.loco_env_cfg import feet_only_contact

env_cfg = parse_env_cfg("G1-LocoKneel-Play-v0", num_envs=4)
env = gym.make("G1-LocoKneel-Play-v0", cfg=env_cfg)
env.reset()
raw = env.unwrapped
sensor = raw.scene.sensors["contact_forces"]
print("SENSOR BODIES:", sensor.body_names if hasattr(sensor, "body_names") else "NO body_names attr")

feet, names = sensor.find_bodies(".*_ankle_roll_link")
print("FEET IDS:", feet, names)
bad, bnames = sensor.find_bodies(["pelvis", "torso_link", ".*_hip_.*_link", ".*_knee_link"])
print("BAD IDS:", bad, bnames)

zero = torch.zeros(raw.num_envs, raw.action_manager.total_action_dim, device=raw.device)
for step in range(150):
    env.step(zero)
    if step % 50 == 49:
        f = sensor.data.net_forces_w.norm(dim=-1)
        pz = raw.scene["robot"].data.root_pos_w[:, 2]
        print(f"step {step}: pelvis_z={pz.tolist()}")
        print(f"  feet forces: {f[:, feet].tolist()}")
        print(f"  max bad-body force: {f[:, bad].max(dim=1).values.tolist()}")
        print(f"  gate: {feet_only_contact(raw).tolist()}")
        top = f[0].argsort(descending=True)[:5]
        print(f"  top contact bodies env0: {[(sensor.body_names[i], round(float(f[0,i]),1)) for i in top.tolist()]}")

print("PROBE_GATE_DONE")
import os, sys
sys.stdout.flush()
os._exit(0)
