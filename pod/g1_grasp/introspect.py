"""Dump joint/body/actuator names of the G1 + Inspire-hand asset.

Run headless on the pod; output drives the grasp env's action-space regexes.
    python introspect.py --headless
"""

import argparse
import faulthandler
faulthandler.dump_traceback_later(120, exit=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 200, device="cuda:0"))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    cfg = G1_INSPIRE_FTP_CFG.replace(prim_path="/World/G1")
    cfg.spawn.usd_path = "/workspace/assets/G1/g1_29dof_inspire_hand.usd"
    robot = Articulation(cfg)
    sim.reset()

    print(f"=== {robot.num_joints} joints ===")
    for n in robot.joint_names:
        print(" joint:", n)
    print(f"=== {robot.num_bodies} bodies ===")
    for n in robot.body_names:
        print(" body:", n)
    print("=== actuators ===")
    for k, v in robot.actuators.items():
        print(" actuator group:", k, "->", v.joint_names)
    print("=== default joint pos (nonzero) ===")
    import torch
    dp = robot.data.default_joint_pos[0]
    for n, q in zip(robot.joint_names, dp.tolist()):
        if abs(q) > 1e-6:
            print(f" {n}: {q:.3f}")
    print("INTROSPECT_DONE")


if __name__ == "__main__":
    import os, sys
    main()
    sys.stdout.flush()
    os._exit(0)  # simulation_app.close() hangs in this container; skip it
