"""Make a free-root-capable variant of the G1+Inspire USD.

The shipped asset's finger joints carry PhysxMimicJointAPI couplings that
fail to resolve when fix_root_link=False (articulation parsed from the
pelvis instead of the baked root_joint) — articulation creation dies with
"failed to find internal joint object for PhysxMimicJointAPI". The grasp
policy commands every hand joint directly, so the couplings are removable.
Writes g1_29dof_inspire_hand_free.usd next to the original.

    python -u g1_loco/fix_usd.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import Usd, UsdPhysics

SRC = "/workspace/assets/G1/g1_29dof_inspire_hand.usd"
DST = "/workspace/assets/G1/g1_29dof_inspire_hand_free.usd"

# flatten FIRST: root_joint and the APIs live in referenced sub-layers, so
# edits on the composed stage are silently discarded by composition
src_stage = Usd.Stage.Open(SRC)
stage = Usd.Stage.Open(src_stage.Flatten())
removed = 0
root_joint_paths = []
art_root_paths = []
pelvis_path = None
for prim in stage.Traverse():
    for schema in list(prim.GetAppliedSchemas()):
        if schema.startswith("PhysxMimicJointAPI"):
            prim.RemoveAppliedSchema(schema)
            removed += 1
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        art_root_paths.append(prim.GetPath())
    if prim.GetName() == "root_joint":
        root_joint_paths.append(prim.GetPath())
    if prim.GetName() == "pelvis" and pelvis_path is None:
        pelvis_path = prim.GetPath()

# floating-base layout: articulation root on the pelvis link, no world joint
for p in art_root_paths:
    stage.GetPrimAtPath(p).RemoveAPI(UsdPhysics.ArticulationRootAPI)
for p in root_joint_paths:
    stage.RemovePrim(p)
assert pelvis_path is not None, "pelvis link not found"
UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(pelvis_path))

stage.Export(DST)
print(f"FIX_USD_DONE mimic_removed={removed} old_roots={art_root_paths} "
      f"root_joints_removed={root_joint_paths} new_root={pelvis_path} -> {DST}")

import os, sys
sys.stdout.flush()
os._exit(0)
