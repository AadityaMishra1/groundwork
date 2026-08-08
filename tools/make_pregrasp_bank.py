"""P0 deliverable: the certified squat-pregrasp pose bank (Mac, CPU, $0).

Sweeps object placements across the certified corridor, solves a whole-body
pregrasp pose for each (wbik), certifies it statically (statics.equilibrium:
contact-cone balance + G1 spec torque), and saves every CERTIFIED pose with
its certificate. This bank seeds the squat-pregrasp slice of the chain start
mixture in Phase 3 (docs/PLAN_2026-07-25.md).

Two-pass solve: cold seeds first, then CONTINUATION — misses re-solved
warm-started from their nearest certified neighbors. The first run's 13
misses formed an interior hole (r 0.32-0.44, mid bearings) between certified
corners, which is a local-minimum artifact, not a workspace boundary; the
fill pass exists so the bank never records a solver failure as infeasibility.

Importing `statics` re-runs its control validation (standing keyframe + deep
crouch) on every bank build — intentional, per the standing law that a probe
must reproduce a known answer before its real verdicts are believed.

Cross-model caveat (recorded in the manifest): the local Menagerie model
carries the 3-finger G1 hand, not the Inspire five-finger. Legs/waist/arm
kinematics and the palm frame (right_wrist_yaw_link) are exact; hand length
below the wrist is approximate. A 5-minute Isaac-side verification of a
sample of these poses is a Phase 1 gate item.

Measured result (2026-07-25): the certified family collapses to a SINGLE
dwell — pelvis 0.387-0.394, torso upright, worst joint the right knee at
49-50% of spec — across the whole corridor. One squat height serves every
target. NOTE for Phase 2: the m1f3 walker's hold quality at ~0.39-0.45 is
unmeasured (0.45 was its weak transit band); sweep before picking the dwell.

    .venv/bin/python -u tools/make_pregrasp_bank.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wbik as W                                    # noqa: E402
from statics import equilibrium, all_foot_geoms, pose_from_x  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "refs")
OBJ_H = 0.15
# corridor: certified band, slightly overscanned at the edges
RS = np.arange(0.28, 0.541, 0.04)                   # 7 ranges
THS = np.arange(-0.85, -0.099, 0.15)                # 6 bearings
STANCES = ("A_squat_upright", "B_squat_freetorso")
SEEDS = 4
FILL_ROUNDS = 3


def attempt(tgt, seed_list):
    """Solve + certify one target; first certified stance wins."""
    for stance in STANCES:
        b = W.best_pose(stance, tgt, seed_list)
        if b is None:
            continue
        e, x, rp = b
        pose_from_x(x)
        ok, info = equilibrium(all_foot_geoms())
        if ok:
            return dict(stance=stance, err=float(e), x=x, rp=rp, info=info)
    return None


def describe(key, entry):
    rp, info = entry["rp"], entry["info"]
    if info["worst"]:  # already sorted worst-first
        wf, wn = info["worst"][0][0], info["worst"][0][1]
    else:
        wf, wn = 0.0, "-"
    return (f"  r={key[0]:.2f} th={np.degrees(key[1]):+05.1f}  CERTIFIED "
            f"[{entry['stance']}] pelvis={rp['pelvis_z']:.3f} "
            f"torso={np.degrees(rp['torso_pitch']):+05.1f}deg "
            f"palm_err={entry['err']:.3f} worst={wn} {wf*100:.0f}%"), wf, wn


def main():
    rng = np.random.default_rng(1)
    S = W.seeds(rng, SEEDS)
    grid = [(float(r), float(th)) for r in RS for th in THS]
    cert = {}

    print("\n### PASS 1 — cold seeds ###", flush=True)
    for key in grid:
        e = attempt(W.pregrasp_target(key[0], key[1], OBJ_H), S)
        if e is None:
            print(f"  r={key[0]:.2f} th={np.degrees(key[1]):+05.1f}  -- miss",
                  flush=True)
            continue
        cert[key] = e
        print(describe(key, e)[0], flush=True)

    for rnd in range(1, FILL_ROUNDS + 1):
        misses = [k for k in grid if k not in cert]
        if not misses or not cert:
            break
        print(f"\n### FILL ROUND {rnd} — continuation from certified "
              f"neighbors ({len(misses)} misses) ###", flush=True)
        filled = 0
        for key in misses:
            # nearest certified neighbors in (r, th); th step 0.15 rad vs
            # r step 0.04 m — weight so one grid step costs the same
            neigh = sorted(cert, key=lambda c: ((c[0] - key[0]) / 0.04) ** 2
                           + ((c[1] - key[1]) / 0.15) ** 2)[:3]
            seeds2 = [cert[c]["x"].copy() for c in neigh] + S[:2]
            e = attempt(W.pregrasp_target(key[0], key[1], OBJ_H), seeds2)
            if e is not None:
                cert[key] = e
                filled += 1
                print(describe(key, e)[0], flush=True)
        print(f"  fill round {rnd}: recovered {filled}/{len(misses)}", flush=True)
        if filled == 0:
            break

    rows, manifest = [], []
    for key in grid:
        if key not in cert:
            continue
        entry = cert[key]
        rp, info = entry["rp"], entry["info"]
        _, wf, wn = describe(key, entry)
        pose_from_x(entry["x"])                      # leave D at this pose
        rows.append(dict(
            r=key[0], th=key[1], stance=entry["stance"],
            x=entry["x"].copy(), qpos=W.D.qpos.copy(),
            palm=rp["palm"].copy(),
            target=W.pregrasp_target(key[0], key[1], OBJ_H),
            pelvis_z=float(rp["pelvis_z"]),
            torso_pitch=float(rp["torso_pitch"])))
        manifest.append(dict(
            r=key[0], th_deg=round(float(np.degrees(key[1])), 1),
            stance=entry["stance"],
            pelvis_z=round(float(rp["pelvis_z"]), 3),
            torso_pitch_deg=round(float(np.degrees(rp["torso_pitch"])), 1),
            palm_err_m=round(entry["err"], 4),
            worst_joint=wn, worst_torque_frac=round(float(wf), 3),
            cone_frac=round(float(info["cone"]), 3)))

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, "pregrasp_bank.npz"),
             x=np.stack([e["x"] for e in rows]),
             qpos=np.stack([e["qpos"] for e in rows]),
             palm=np.stack([e["palm"] for e in rows]),
             target=np.stack([e["target"] for e in rows]),
             r=np.array([e["r"] for e in rows]),
             th=np.array([e["th"] for e in rows]),
             pelvis_z=np.array([e["pelvis_z"] for e in rows]),
             stance=np.array([e["stance"] for e in rows]),
             body_joint_names=np.array(W.BODY_JOINTS),
             obj_h=OBJ_H)
    still = [k for k in grid if k not in cert]
    with open(os.path.join(OUT_DIR, "pregrasp_bank_manifest.json"), "w") as f:
        json.dump(dict(
            date="2026-07-25", object_height_m=OBJ_H,
            model="menagerie g1_with_hands (3-finger hand — palm frame exact, "
                  "hand length approximate vs Inspire; Isaac re-check is a "
                  "Phase 1 gate item)",
            certificate="statics.equilibrium: cone-constrained contact balance "
                        "+ G1 datasheet torque; controls validated on import",
            n_grid=len(grid), n_certified=len(rows),
            unresolved=[dict(r=k[0], th_deg=round(float(np.degrees(k[1])), 1))
                        for k in still],
            poses=manifest), f, indent=1)
    print(f"\nBANK: {len(rows)}/{len(grid)} certified "
          f"-> {OUT_DIR}/pregrasp_bank.npz", flush=True)
    if still:
        print("UNRESOLVED (solver, after continuation): "
              + ", ".join(f"(r={k[0]:.2f},th={np.degrees(k[1]):+.0f})"
                          for k in still), flush=True)
    print("BANK_DONE", flush=True)


if __name__ == "__main__":
    main()
