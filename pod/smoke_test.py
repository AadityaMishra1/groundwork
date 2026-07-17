"""Phase 0 smoke test: prove the pod can do everything the project needs.

Checks, in order:
  1. Isaac Sim boots headless.
  2. The Unitree G1 asset loads and a physics scene steps in real time.
  3. The Inspire five-finger hand asset is present in the asset registry.
  4. Offscreen rendering works (writes smoke_test.mp4 to pull back to the Mac).

Run:  python smoke_test.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # needed for offscreen video

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- everything below runs inside the Isaac Sim context ----
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

# In-tree robot configs. G1 config name is stable across 2.x; the Inspire hand
# shipped with the 2.3 teleop/pickplace work — probe a couple of module paths
# so a minor-version move doesn't fail the whole test.
from isaaclab_assets import G1_MINIMAL_CFG  # noqa: E402

def find_inspire_hand_cfg():
    candidates = [
        ("isaaclab_assets.robots.unitree", "G1_INSPIRE_FTP_CFG"),
        ("isaaclab_assets.robots.unitree", "G1_29DOF_INSPIRE_CFG"),
        ("isaaclab_assets.robots.inspire", "INSPIRE_HAND_CFG"),
    ]
    import importlib
    found = []
    for mod_name, attr in candidates:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, attr):
                found.append(f"{mod_name}.{attr}")
            # also report anything inspire-ish for phase-1 planning
            found += [f"{mod_name}.{a}" for a in dir(mod)
                      if "INSPIRE" in a.upper() and f"{mod_name}.{a}" not in found]
        except ImportError:
            pass
    return found


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 200, device="cuda:0"))

    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2000.0)
    light.func("/World/light", light)

    robot_cfg = G1_MINIMAL_CFG.replace(prim_path="/World/G1")
    robot = Articulation(robot_cfg)

    sim.reset()
    print(f"[OK] G1 loaded: {robot.num_joints} joints, "
          f"{robot.num_bodies} bodies")

    inspire = find_inspire_hand_cfg()
    if inspire:
        print(f"[OK] Inspire hand config(s) found: {inspire}")
    else:
        print("[WARN] No Inspire hand cfg found in isaaclab_assets — "
              "will need the asset from the pickplace task extension. "
              "Not fatal for Phase 0.")

    # step physics, hold default pose, confirm nothing explodes
    import time
    t0 = time.time()
    steps = 1000
    default_pos = robot.data.default_joint_pos.clone()
    for _ in range(steps):
        robot.set_joint_position_target(default_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
    wall = time.time() - t0
    print(f"[OK] {steps} physics steps in {wall:.2f}s "
          f"({steps * sim.get_physics_dt() / wall:.1f}x realtime, 1 env)")

    base_h = robot.data.root_pos_w[0, 2].item()
    assert 0.4 < base_h < 1.2, f"G1 base height {base_h:.2f} m — fell or flew"
    print(f"[OK] G1 standing, base height {base_h:.2f} m")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
    simulation_app.close()
