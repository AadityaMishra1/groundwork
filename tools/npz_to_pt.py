#!/usr/bin/env python3
"""Convert the shipped .npz banks in refs/ into the .pt files the env loads.

The banks are committed as .npz (portable, inspectable, diffable); the Isaac
Lab side loads torch .pt. This script is the bridge. It previously existed
only as an ssh one-liner inside a docstring, which meant the committed data
could not actually be loaded by the committed code.

    python tools/npz_to_pt.py                      # -> $GW_BANK_DIR (default /workspace)
    python tools/npz_to_pt.py --out-dir ./banks    # anywhere else

Array keys become float32 tensors; string arrays (e.g. joint_names) stay as
python lists of str, which is what the loaders expect for name-based remap.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

# refs/<name>.npz  ->  <out_dir>/<name>.pt
BANKS = [
    "pregrasp_bank",
    "start_bank_certified",
    "grasp_offset_samples",
    "held_states",
    "ref_traj_bank",
    "ref_traj_bank_gc",
    "ref_traj_bank_gc2",
]


def convert(src: Path, dst: Path) -> dict:
    d = np.load(src, allow_pickle=False)
    out = {}
    for k in d.keys():
        arr = d[k]
        if arr.dtype.kind in ("U", "S"):          # joint_names and friends
            out[k] = [str(s) for s in arr]
        elif arr.dtype.kind in ("f", "i", "u", "b"):
            out[k] = torch.tensor(arr, dtype=torch.float32)
        else:
            raise TypeError(f"{src.name}:{k} has unsupported dtype {arr.dtype}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.environ.get("GW_BANK_DIR", "/workspace"),
                    help="where the .pt files go (default: $GW_BANK_DIR or /workspace)")
    ap.add_argument("--refs-dir", default=str(Path(__file__).resolve().parent.parent / "refs"))
    args = ap.parse_args()

    refs, out_dir = Path(args.refs_dir), Path(args.out_dir)
    missing = []
    for name in BANKS:
        src = refs / f"{name}.npz"
        if not src.exists():
            missing.append(name)
            continue
        payload = convert(src, out_dir / f"{name}.pt")
        shapes = {k: (tuple(v.shape) if torch.is_tensor(v) else f"list[{len(v)}]")
                  for k, v in payload.items()}
        print(f"{name}.npz -> {out_dir / (name + '.pt')}  {shapes}")

    if missing:
        print(f"\nnot present in {refs}: {', '.join(missing)}")


if __name__ == "__main__":
    main()
