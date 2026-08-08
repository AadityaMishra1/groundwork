"""Stamp a `leg_stiff` scalar into existing reference-bank npz files.

Part of the bank stiffness lock (see pod/g1_grasp/bank_guard.py): the gc
banks' joint_targets are gravity-compensated against the make_ref_traj.py
KP_EFF table (x4 legs), but the files predating the lock carry only the
legacy `leg_stiff_mult` key (gc/gc2) or nothing (the pre-gc bank). This
stamps the canonical key so grasp_env's load-time assert can verify the
replay stiffness instead of assuming it.

Safety: writes to <file>.tmp.npz then os.replace() — the original is never
half-written; a differing EXISTING stamp is refused without --force (a
re-stamp that changes the recorded physics assumption must be deliberate).
If `leg_stiff_mult` exists and disagrees with --value, that is refused too:
the legacy key is evidence, not noise.

    python tools/stamp_leg_stiff.py refs/ref_traj_bank_gc.npz \
        refs/ref_traj_bank_gc2.npz            # stamps leg_stiff=4.0
"""

import argparse
import os

import numpy as np


def stamp(path, value, force=False):
    # allow_pickle: these banks are produced by this project's own
    # make_ref_traj/apply_grasp_offset tooling (trusted artifacts, same
    # trust model as every torch.load bank in grasp_env.py); joint_names
    # may be an object array, which plain np.load refuses.
    z = np.load(path, allow_pickle=True)
    data = {k: z[k] for k in z.files}
    if "leg_stiff" in data:
        got = float(data["leg_stiff"])
        if abs(got - value) <= 1e-6:
            print(f"{path}: already stamped leg_stiff={got:g} — no-op")
            return False
        if not force:
            raise RuntimeError(
                f"{path}: existing leg_stiff={got:g} != requested "
                f"{value:g}; re-stamping changes the recorded physics "
                f"assumption — pass --force only if that is deliberate")
        print(f"{path}: FORCED re-stamp {got:g} -> {value:g}")
    if "leg_stiff_mult" in data:
        legacy = float(data["leg_stiff_mult"])
        if abs(legacy - value) > 1e-6 and not force:
            raise RuntimeError(
                f"{path}: legacy leg_stiff_mult={legacy:g} disagrees with "
                f"requested {value:g} — the legacy key is evidence of what "
                f"the bank was baked against; refusing without --force")
    data["leg_stiff"] = np.float64(value)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)
    print(f"{path}: stamped leg_stiff={value:g} "
          f"({len(data)} keys preserved)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--value", type=float, default=4.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for p in args.files:
        stamp(p, args.value, force=args.force)


if __name__ == "__main__":
    main()
