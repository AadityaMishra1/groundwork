"""Measure the REAL in-grasp palm<->object relative pose from the lead grasp
policy's own holds (Isaac physics, Inspire hand — not the Mac MuJoCo model).

Why: the gc reference bank's aloft object placements were derived from the
MuJoCo palm offset; in Isaac the Inspire hand holds the cylinder somewhere
else, so aloft restores show engine grasp 0.000 at t=1 and the object
free-falls. The fix starts by measuring where a grasped object ACTUALLY sits
relative to the palm body (right_wrist_yaw_link) during engine-verified
in-air holds.

Read-only standalone probe: no grasp_env changes, no training impact. Run it
under the lead lane's exact env vars (obs dims must match the checkpoint):

    GRASP_FULLBODY=1 GRASP_FREE=1 GRASP_LEG_STIFF=4.0 GRASP_EP_S=30 \
    GRASP_FORCE_STAGE=3 GRASP_FAR_FRAC=0.0 GRASP_FLUNG_REF=robot \
    GRASP_POSTURE_GATE=1 GRASP_GATE_PELVIS=0.35 GRASP_GATE_UP=0.5 \
    GRASP_GATE_PELVIS_INC=0.32 GRASP_ARRIVAL_FRAC=0.15 GRASP_EMPNORM=1 \
    GRASP_SPAWN_Z=0.80 GRASP_START_BANK=/workspace/start_bank_certified.pt \
    GRASP_NO_VIDEO=1 \
    python -u g1_grasp/probe_grasp_offset.py \
        --checkpoint <lead model.pt> --num_envs 16 --steps 6000 --headless

Samples every frame where engine grasp holds (>=3 finger groups + thumb via
the filtered contact sensors) AND obj_z > --min_z (default 0.15 so ONE run
supports both the 0.30 in-air bar and the relaxed 0.15 fallback — threshold
applied at aggregation time). Records the object pose in the palm body frame
(p_rel = R_palm^T (p_obj - p_palm), q_rel = q_palm^-1 * q_obj, w>=0), plus
obj_z / env / episode ids so the aggregator can cluster grasp modes.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# pod root on sys.path for cli_args (this file lives in g1_grasp/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cli_args  # noqa: E402  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=6000)
parser.add_argument("--min_z", type=float, default=0.15)
parser.add_argument("--out", type=str, default=f"{_WS}/grasp_offset_samples.pt")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.checkpoint:
    parser.error("--checkpoint (lead grasp policy .pt) is required")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import g1_grasp  # noqa: F401, E402

import os as _os
_WS = _os.environ.get("GW_ROOT", "/workspace")


def quat_conj(q):
    return torch.cat([q[:, :1], -q[:, 1:]], dim=-1)


def quat_mul(a, b):
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                        w1*x2 + x1*w2 + y1*z2 - z1*y2,
                        w1*y2 - x1*z2 + y1*w2 + z1*x2,
                        w1*z2 + x1*y2 - y1*x2 + z1*w2], dim=-1)


def quat_rot_inv(q, v):
    """Rotate world vector v into the frame of quat q (wxyz, body->world)."""
    return quat_mul(quat_mul(quat_conj(q),
                             torch.cat([torch.zeros_like(v[:, :1]), v], -1)),
                    q)[:, 1:]


def main():
    task = "G1-Grasp-v0"
    env_cfg = parse_env_cfg(task, num_envs=args.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task, args)
    env = gym.make(task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw = env.unwrapped
    dev = raw.device
    n = raw.num_envs
    org = raw.scene.env_origins

    P, Q, Z, E, EP, TS = [], [], [], [], [], []
    ep_ctr = torch.zeros(n, dtype=torch.long, device=dev)
    prev_len = torch.zeros(n, dtype=torch.long, device=dev)
    obs, _ = env.get_observations(), None
    with torch.inference_mode():
        for step in range(args.steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            # episode bookkeeping: length drop == the env restarted
            cur = raw.episode_length_buf
            ep_ctr = ep_ctr + (cur < prev_len).long()
            prev_len = cur.clone()

            nf, th = raw._finger_contacts()
            grasped = (nf >= 3) & th
            obj_z = raw.obj.data.root_pos_w[:, 2] - org[:, 2]
            m = grasped & (obj_z > args.min_z)
            if m.any():
                idx = m.nonzero(as_tuple=False).squeeze(-1)
                pp = raw.robot.data.body_pos_w[idx, raw.palm_id]
                pq = raw.robot.data.body_quat_w[idx, raw.palm_id]
                op = raw.obj.data.root_pos_w[idx]
                oq = raw.obj.data.root_quat_w[idx]
                p_rel = quat_rot_inv(pq, op - pp)
                q_rel = quat_mul(quat_conj(pq), oq)
                q_rel = torch.where(q_rel[:, :1] < 0, -q_rel, q_rel)
                P.append(p_rel.cpu())
                Q.append(q_rel.cpu())
                Z.append(obj_z[idx].cpu())
                E.append(idx.cpu())
                EP.append(ep_ctr[idx].cpu())
                TS.append(torch.full((len(idx),), step, dtype=torch.long))
            if step % 500 == 0 or step == args.steps - 1:
                tot = int(sum(len(x) for x in P))
                eps = len(set(zip(torch.cat(E).tolist(),
                                  torch.cat(EP).tolist()))) if P else 0
                print(f"OFFSET t={step:5d} grasped={grasped.float().mean():.2f} "
                      f"objz_p50={obj_z.median():.3f} samples={tot} "
                      f"episodes={eps}", flush=True)

    if not P:
        print("OFFSET_FAIL zero qualifying frames", flush=True)
        return
    p = torch.cat(P); q = torch.cat(Q); z = torch.cat(Z)
    e = torch.cat(E); ep = torch.cat(EP); ts = torch.cat(TS)
    torch.save({"p_rel": p, "q_rel": q, "obj_z": z, "env": e, "ep": ep,
                "step": ts, "min_z": args.min_z, "ckpt": args.checkpoint},
               args.out)
    for band, lo in (("z>0.30", 0.30), ("z>0.15", 0.15)):
        m = z > lo
        if int(m.sum()) == 0:
            print(f"OFFSET_AGG {band}: 0 samples", flush=True)
            continue
        pm = p[m]
        eps = len(set(zip(e[m].tolist(), ep[m].tolist())))
        med = pm.median(dim=0).values
        p25 = pm.quantile(0.25, dim=0)
        p75 = pm.quantile(0.75, dim=0)
        print(f"OFFSET_AGG {band}: n={int(m.sum())} episodes={eps} "
              f"median=({med[0]:+.4f},{med[1]:+.4f},{med[2]:+.4f}) "
              f"p25=({p25[0]:+.4f},{p25[1]:+.4f},{p25[2]:+.4f}) "
              f"p75=({p75[0]:+.4f},{p75[1]:+.4f},{p75[2]:+.4f})", flush=True)
    print(f"OFFSET_SAVED n={len(p)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
