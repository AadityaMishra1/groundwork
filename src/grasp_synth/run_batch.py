"""Batch grasp synthesis: sample objects x pre-grasps, physics-validate in
parallel, save the survivors as the seed dataset for the hand prior and the
reverse curriculum.

Usage:
    python -m src.grasp_synth.run_batch --objects 40 --samples 150 --out data/grasps
"""

import argparse
import json
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .sampler import sample_pregrasp
from .scene import CylinderSpec, build_scene, grasp_clearance
from .validate import run_trial


def _worker(args):
    seed, n_samples = args
    rng = np.random.default_rng(seed)
    obj = CylinderSpec.sample(rng)
    model = build_scene(obj)  # one compile per object, reused across samples
    clearance = grasp_clearance(model)
    hits, reasons = [], Counter()
    for _ in range(n_samples):
        pg = sample_pregrasp(rng, obj, clearance=clearance)
        res = run_trial(obj, pg.palm_pos, pg.palm_quat, pg.closure, model=model)
        reasons[res.reason] += 1
        if res.success:
            hits.append({
                "object": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
                           for k, v in vars(obj).items()},
                "pregrasp": pg.to_dict(),
                "lift_height": res.lift_height,
                "fingers": res.fingers_in_contact,
            })
    return hits, reasons, seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", type=int, default=40)
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--out", type=Path, default=Path("data/grasps"))
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [(args.seed + i, args.samples) for i in range(args.objects)]
    t0 = time.time()
    all_hits, total_reasons = [], Counter()
    with mp.Pool(args.workers) as pool:
        for hits, reasons, seed in pool.imap_unordered(_worker, jobs):
            all_hits += hits
            total_reasons += reasons
            print(f"  object seed {seed}: {len(hits)}/{args.samples} grasps held")

    wall = time.time() - t0
    n_trials = args.objects * args.samples
    summary = {
        "trials": n_trials,
        "stable_grasps": len(all_hits),
        "acceptance_rate": len(all_hits) / n_trials,
        "outcomes": dict(total_reasons),
        "wall_seconds": wall,
        "trials_per_second": n_trials / wall,
    }
    (args.out / "grasps.json").write_text(json.dumps(all_hits, indent=1))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
