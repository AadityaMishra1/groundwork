"""Behavior-clone recorded demos into the actor of an existing checkpoint.

Loads an rsl_rl checkpoint, MSE-trains the actor mean on (obs, action) demo
pairs, keeps the critic, shrinks exploration std, and writes a checkpoint
the OnPolicyRunner can resume from.

    python -u g1_grasp/bc_train.py --ckpt <model.pt> --demos /workspace/bc_demos.pt \
        --out /workspace/bc_init.pt
"""

import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--demos", default=f"{_WS}/bc_demos.pt")
parser.add_argument("--out", default=f"{_WS}/bc_init.pt")
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--lr", type=float, default=3e-4)
args = parser.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"

from rsl_rl.modules import ActorCritic

import os as _os
_WS = _os.environ.get("GW_ROOT", "/workspace")

ac = ActorCritic(69, 69, 20,
                 actor_hidden_dims=[512, 256, 128],
                 critic_hidden_dims=[512, 256, 128],
                 activation="elu", init_noise_std=1.0).to(dev)
ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
ac.load_state_dict(ckpt["model_state_dict"])

d = torch.load(args.demos, weights_only=True)
obs = d["obs"].reshape(-1, 69).to(dev)
act = d["act"].reshape(-1, 20).to(dev)
print(f"demo pairs: {len(obs)} from {len(d['obs'])} episodes")

opt = torch.optim.Adam(ac.actor.parameters(), lr=args.lr)
n = len(obs)
for ep in range(args.epochs):
    perm = torch.randperm(n, device=dev)
    tot = 0.0
    for i in range(0, n, 4096):
        idx = perm[i:i + 4096]
        loss = torch.nn.functional.mse_loss(ac.actor(obs[idx]), act[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += float(loss) * len(idx)
    if ep % 10 == 0 or ep == args.epochs - 1:
        print(f"epoch {ep}: mse {tot / n:.5f}")

# shrink exploration so PPO doesn't immediately blast away the cloned skill
with torch.no_grad():
    if hasattr(ac, "std"):
        ac.std.copy_(torch.full_like(ac.std, 0.3))
    elif hasattr(ac, "log_std"):
        ac.log_std.copy_(torch.full_like(ac.log_std, float(torch.log(torch.tensor(0.3)))))

ckpt["model_state_dict"] = ac.state_dict()
ckpt.pop("optimizer_state_dict", None)  # fresh optimizer at resume
torch.save(ckpt, args.out)
print(f"BC_DONE saved {args.out}")
