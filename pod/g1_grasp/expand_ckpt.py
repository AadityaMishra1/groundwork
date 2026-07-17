"""Expand a 69-obs rsl_rl checkpoint to the 90-obs tactile observation space.

The new dims (5 tip-to-object vectors, 5 contact flags, grasped flag) are
appended at the END of the obs vector, so the old policy is preserved exactly
by zero-padding the actor/critic input-layer columns: with zero weights on the
new inputs the network computes identical outputs, and fine-tuning wires the
new senses in. Optimizer state is dropped (shapes changed).

    python -u g1_grasp/expand_ckpt.py --ckpt <model_16295.pt> --out /workspace/tactile_init.pt
"""

import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--out", default="/workspace/tactile_init.pt")
parser.add_argument("--old_dim", type=int, default=69)
parser.add_argument("--new_dim", type=int, default=90)
args = parser.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"]

expanded = []
for key in ("actor.0.weight", "critic.0.weight"):
    w = sd[key]
    assert w.shape[1] == args.old_dim, f"{key}: expected in-dim {args.old_dim}, got {w.shape[1]}"
    pad = torch.zeros(w.shape[0], args.new_dim - args.old_dim, dtype=w.dtype)
    sd[key] = torch.cat([w, pad], dim=1)
    expanded.append(f"{key}: {tuple(w.shape)} -> {tuple(sd[key].shape)}")

# pad Adam moments for the expanded input layers so the runner's resume path
# (which load_state_dict's the optimizer) doesn't crash on shape mismatch
opt = ckpt.get("optimizer_state_dict")
if opt is not None:
    for st in opt.get("state", {}).values():
        for k, v in st.items():
            if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == args.old_dim:
                st[k] = torch.cat(
                    [v, torch.zeros(v.shape[0], args.new_dim - args.old_dim,
                                    dtype=v.dtype)], dim=1)
                expanded.append(f"optimizer.{k}: padded {tuple(v.shape)}")
torch.save(ckpt, args.out)
print("\n".join(expanded))
print(f"EXPAND_DONE saved {args.out}")
