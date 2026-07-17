"""Probe: scripted stand->kneel->stand transition stability, free root.

The composition lowers into the grasp kneel via a slow joint-target ramp
(stereotyped motion, like a human sitting down) rather than asking the RL
walker to master dynamic deep-descent. This probe verifies under honest
physics that (a) the ramp reaches the kneel without falling, (b) the kneel
is statically stable, (c) the reverse ramp stands back up, (d) all of the
above with a payload mass on the right wrist. Reports success rates over
randomized initial noise.

    PYTHONPATH=/workspace/humanoid/pod python -u g1_loco/probe_transition.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG

FREE_USD = "/workspace/assets/G1/g1_29dof_inspire_hand_free.usd"

STAND = {
    ".*_hip_pitch_joint": -0.20, ".*_knee_joint": 0.42,
    ".*_ankle_pitch_joint": -0.23,
}
KNEEL = {
    ".*_hip_pitch_joint": -2.2, ".*_knee_joint": 2.8, ".*_ankle_pitch_joint": -0.87,
    ".*_hip_roll_joint": 0.4, ".*_hip_yaw_joint": 0.0, ".*_ankle_roll_joint": 0.0,
}
RAMP_S = 1.5          # seconds per transition
HOLD_S = 2.0          # settle time at each end
CTRL_HZ = 50
PHYS_PER_CTRL = 4


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = G1_INSPIRE_FTP_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    def __post_init__(self):
        self.robot.spawn.usd_path = FREE_USD
        self.robot.spawn.articulation_props.fix_root_link = False
        self.robot.spawn.articulation_props.enabled_self_collisions = True
        self.robot.spawn.rigid_props.disable_gravity = False
        self.robot.init_state.pos = (0.0, 0.0, 0.80)
        self.robot.init_state.joint_pos = dict(STAND)


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 200, render_interval=4))
    scene = InteractiveScene(SceneCfg(num_envs=args.num_envs, env_spacing=2.5))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0)
    light.func("/World/Light", light)
    sim.reset()
    robot: Articulation = scene["robot"]
    dev = robot.device
    n = args.num_envs

    def targets_for(pose: dict) -> torch.Tensor:
        t = robot.data.default_joint_pos.clone()
        for pattern, val in pose.items():
            ids, _ = robot.find_joints(pattern)
            t[:, ids] = val
        return t

    stand_t = targets_for(STAND)
    kneel_t = targets_for(KNEEL)

    # payload: second half of envs carries extra mass on the right wrist
    wrist_id = robot.find_bodies("right_wrist_yaw_link")[0][0]
    masses = robot.root_physx_view.get_masses().clone()
    masses[n // 2:, wrist_id] += 0.5
    robot.root_physx_view.set_masses(masses, torch.arange(n))

    def run_phase(a: torch.Tensor, b: torch.Tensor, seconds: float):
        steps = int(seconds * CTRL_HZ)
        for k in range(steps):
            alpha = min(1.0, (k + 1) / (RAMP_S * CTRL_HZ)) if seconds > HOLD_S - 1e-6 else 1.0
            tgt = (1 - alpha) * a + alpha * b
            robot.set_joint_position_target(tgt)
            for _ in range(PHYS_PER_CTRL):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(sim.get_physics_dt())

    def upright() -> torch.Tensor:
        q = robot.data.root_quat_w
        return 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)

    # settle at stand
    run_phase(stand_t, stand_t, 1.0)
    ok = upright() > 0.7
    print(f"phase stand-settle: upright {int(ok.sum())}/{n}")

    # stand -> kneel ramp + hold
    run_phase(stand_t, kneel_t, RAMP_S + HOLD_S)
    pelvis_z = robot.data.root_pos_w[:, 2] - scene.env_origins[:, 2]
    ok_kneel = (upright() > 0.5) & (pelvis_z > 0.12) & (pelvis_z < 0.45)
    print(f"phase kneel: upright+in-band {int(ok_kneel.sum())}/{n} "
          f"pelvis_z p25/50/75="
          f"{pelvis_z.quantile(0.25):.3f}/{pelvis_z.quantile(0.5):.3f}/{pelvis_z.quantile(0.75):.3f}")

    # kneel -> stand ramp + hold
    run_phase(kneel_t, stand_t, RAMP_S + HOLD_S)
    pelvis_z = robot.data.root_pos_w[:, 2] - scene.env_origins[:, 2]
    ok_stand = (upright() > 0.7) & (pelvis_z > 0.55)
    both = ok_kneel & ok_stand
    half = n // 2
    print(f"phase stand-up: upright+tall {int(ok_stand.sum())}/{n}")
    print(f"PROBE_RESULT full-cycle ok={int(both.sum())}/{n} "
          f"no-payload={int(both[:half].sum())}/{half} "
          f"payload0.5kg={int(both[half:].sum())}/{n - half}")


if __name__ == "__main__":
    import os
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
