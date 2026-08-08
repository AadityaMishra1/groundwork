"""OBS-SHIFT probe (PM Amendment B, 2026-07-31): decide freeze vs no-freeze
for the empirical normalizer across the FAR_FRAC 0.0 -> 0.25 transition.

Why this must run before the walk lane boots: train_grasp.py's own
GRASP_EMPNORM_FREEZE comment names THIS transition ("FAR_FRAC 0->0.4") as
the documented case where a resume drags the running normalizer within one
iteration and wrecks the deterministic mean for 1500-3000 iterations. On a
confirm-the-best plan with ~$90 left, that many invalid evals is fatal.
But freezing has the opposite failure mode: far-start states carry
base->object distance features the frozen statistics have never seen, so a
frozen normalizer may squash or blow up exactly the dimensions the walk
policy needs to read.

Measurement: collect observations under FAR_FRAC=0 (the banked artifact's
training distribution) and under FAR_FRAC=0.25 (the walk lane's), then
express the far-distribution's per-dimension mean in units of the BANKED
CHECKPOINT'S OWN stored normalizer sigma. That z-score is exactly what the
frozen normalizer would feed the network.

    PYTHONPATH=. python -u g1_grasp/probe_obsshift.py \
        --checkpoint /workspace/banked/GRASP_ARTIFACT_laneYb_25200.pt \
        --num_envs 256 --resets 12 --headless
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("GRASP_FORCE_STAGE", "3")
os.environ.setdefault("GRASP_NO_VIDEO", "1")

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--resets", type=int, default=12)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--far_frac", type=float, default=0.25)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import g1_grasp  # noqa: F401, E402


def collect(env, raw, resets, steps):
    """Mean/std of observations over `resets` resets x `steps` steps."""
    acc = []
    zero = torch.zeros(raw.num_envs, raw.cfg.action_space, device=raw.device)
    with torch.inference_mode():
        for _ in range(resets):
            obs, _ = env.reset()
            o = obs["policy"] if isinstance(obs, dict) else obs
            acc.append(o.clone())
            for _ in range(steps):
                out = env.step(zero)
                o = out[0]
                o = o["policy"] if isinstance(o, dict) else o
                acc.append(o.clone())
    cat = torch.cat(acc, dim=0)
    return cat.mean(dim=0), cat.std(dim=0)


def main():
    # weights_only=True first (the checkpoint holds tensors + plain dicts);
    # fall back only for this project's own checkpoints, matching
    # tools/make_std_reset_init.py's precedent.
    try:
        ck = torch.load(args.checkpoint, map_location="cpu",
                        weights_only=True)
    except Exception as e:
        print(f"weights_only=True failed ({type(e).__name__}); "
              "trusted-provenance fallback")
        ck = torch.load(args.checkpoint, map_location="cpu",
                        weights_only=False)
    # locate the stored empirical-normalizer statistics
    mean_k = var_k = None
    for k in ck:
        if not isinstance(ck[k], dict):
            continue
        for kk in ck[k]:
            if kk.endswith("_mean") and mean_k is None:
                mean_k, mean_v = (k, kk), ck[k][kk]
            if kk.endswith("_var") and var_k is None:
                var_k, var_v = (k, kk), ck[k][kk]
    if mean_k is None:
        print("OBSSHIFT_FAIL no normalizer statistics in checkpoint "
              f"(keys: {list(ck)[:8]})")
        return
    stored_mean = mean_v.float().flatten()
    stored_std = var_v.float().flatten().clamp_min(1e-8).sqrt()
    print(f"OBSSHIFT stored normalizer from {mean_k}/{var_k}, "
          f"dim={stored_mean.numel()}")

    results = {}
    for tag, ff in (("near", 0.0), ("far", args.far_frac)):
        os.environ["GRASP_FAR_FRAC"] = str(ff)
        cfg = parse_env_cfg("G1-Grasp-v0", num_envs=args.num_envs)
        env = gym.make("G1-Grasp-v0", cfg=cfg)
        raw = env.unwrapped
        raw.far_frac = ff          # env read it at __init__; set both
        m, s = collect(env, raw, args.resets, args.steps)
        results[tag] = (m.cpu().float(), s.cpu().float())
        print(f"OBSSHIFT collected {tag} FAR_FRAC={ff} "
              f"(far_frac in env = {raw.far_frac})")
        env.close()

    near_m, _ = results["near"]
    far_m, far_s = results["far"]
    n = min(stored_mean.numel(), far_m.numel())
    sm, ss = stored_mean[:n], stored_std[:n]
    fm, nm = far_m[:n], near_m[:n]

    z_far = ((fm - sm) / ss).abs()
    z_near = ((nm - sm) / ss).abs()
    drift = ((fm - nm) / ss).abs()

    def q(t, p):
        return float(t.quantile(p))

    print(f"OBSSHIFT z(far vs stored)  p50/p90/max = "
          f"{q(z_far,.5):.2f}/{q(z_far,.9):.2f}/{float(z_far.max()):.2f}")
    print(f"OBSSHIFT z(near vs stored) p50/p90/max = "
          f"{q(z_near,.5):.2f}/{q(z_near,.9):.2f}/{float(z_near.max()):.2f}")
    print(f"OBSSHIFT far-vs-near drift p50/p90/max = "
          f"{q(drift,.5):.2f}/{q(drift,.9):.2f}/{float(drift.max()):.2f} sigma")
    worst = int(drift.argmax())
    print(f"OBSSHIFT worst dim {worst}: stored mean {float(sm[worst]):+.3f} "
          f"sigma {float(ss[worst]):.3f} | near {float(nm[worst]):+.3f} "
          f"| far {float(fm[worst]):+.3f}")
    n_big = int((drift > 1.0).sum())
    print(f"OBSSHIFT dims shifted >1 sigma: {n_big}/{n} "
          f"({n_big/n:.1%}); >3 sigma: {int((drift > 3.0).sum())}/{n}")
    print("OBSSHIFT_VERDICT "
          + ("FREEZE_UNSAFE (far states sit far outside stored stats; a "
             "frozen normalizer would misrepresent them)"
             if float(drift.max()) > 3.0 else
             "FREEZE_SAFE (far states lie within the stored statistics' "
             "support; freezing preserves eval comparability)"))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
