"""WAISTFOLD certificate: a detector that must FLAG the laneW3 crab-sprawl
thigh-pin episodes as illegal AND PASS the self-authored reference squat-lift
frames as legal, before any gate term built from it is allowed near training.

PM condition (Waist-Fold Autopsy, 2026-07-30): "No gate term ships until it
provably fails the crab." The existing posture gate reads only the root link
(grasp_env.py ~1316-1328) and exempts the whole hand chain from the floor
check (c5 allowlist, grasp_env.py ~380-384) — the crab satisfies every term.

Detector quantities, all measured DURING strict windows (obj>=0.40m, >=3
finger groups + thumb — the exact demo_grasp strict_now predicate):
    torso_up   z of the TORSO link's up axis (1 - 2(qx^2+qy^2)), not the root
    hand_z     min height of any wrist/hand/palm/finger link center
    objleg     min distance from object center to any hip/knee link center
               (thigh-pin diagnostic; reported, not a flag term)
Episode flags (pre-registered before either dataset was run):
    FOLD   min torso_up < 0.60  (upright squat torso stays well above)
    PLANT  hand_z < 0.05 for >= 10 strict steps (a planted palm skims 0.02;
           a palm carrying an object at 0.40m is nowhere near the floor)
    ILLEGAL = FOLD | PLANT

--mode policy: roll out a checkpoint (demo_grasp env stack, FORCE_STAGE=3),
    collect strict episodes, report flag rate. Crab bar: >= 0.95 flagged.
--mode ref: REF_TRACK restore sweep over the gc2 bank (probe_aloft_restore
    machinery), zero actions, detector read at the restored frames. Legit
    bar: <= 0.05 flagged.

    # policy mode (eval env line, GRASP_NO_VIDEO=1):
    python -u g1_grasp/probe_waistfold.py --mode policy \
        --ckpt <model_23200.pt> --episodes 60 --headless
    # ref mode:
    python -u g1_grasp/probe_waistfold.py --mode ref --rounds 40 --headless
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["policy", "ref"], required=True)
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--episodes", type=int, default=60)
parser.add_argument("--rounds", type=int, default=40)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--fold_bar", type=float, default=0.60)
parser.add_argument("--plant_z", type=float, default=0.05)
parser.add_argument("--plant_steps", type=int, default=10)

# env knobs forced BEFORE g1_grasp import (module reads at import time)
os.environ["GRASP_FORCE_STAGE"] = "3"
os.environ["GRASP_REF_TRACK"] = "1"
os.environ["GRASP_NO_VIDEO"] = "1"

_pre = parser.parse_known_args()[0]
if _pre.mode == "ref":
    os.environ["GRASP_REF_FRAC"] = "1.0"
    os.environ["GRASP_REF_T0MIN"] = "0"
else:
    os.environ["GRASP_REF_FRAC"] = "0.0"

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import g1_grasp  # noqa: F401, E402


def q(v, p):
    t = torch.tensor(v)
    return float(t.quantile(p))


def main():
    env_cfg = parse_env_cfg("G1-Grasp-v0", num_envs=args.num_envs)
    env = gym.make("G1-Grasp-v0", cfg=env_cfg)
    raw = env.unwrapped
    dev = raw.device
    n = raw.num_envs
    org = raw.scene.env_origins
    names = list(raw.robot.data.body_names)
    torso = [i for i, nm in enumerate(names) if "torso" in nm]
    assert len(torso) == 1, f"torso link ambiguous: {torso}"
    torso = torso[0]
    hand_ids = [i for i, nm in enumerate(names)
                if any(a in nm for a in ("wrist", "hand", "palm", "thumb",
                                         "index", "middle", "ring", "little",
                                         "pinky"))]
    leg_ids = [i for i, nm in enumerate(names)
               if ("hip" in nm or "knee" in nm)]
    print(f"WAISTFOLD mode={args.mode} torso={names[torso]} "
          f"hands={len(hand_ids)} legs={len(leg_ids)} "
          f"bars: fold<{args.fold_bar} plant<{args.plant_z}m"
          f"x{args.plant_steps}", flush=True)

    def torso_up():
        tq = raw.robot.data.body_quat_w[:, torso]
        return 1.0 - 2.0 * (tq[:, 1] ** 2 + tq[:, 2] ** 2)

    def hand_z():
        return (raw.robot.data.body_pos_w[:, hand_ids, 2]
                - org[:, 2:3]).min(dim=1).values

    def objleg():
        d = (raw.robot.data.body_pos_w[:, leg_ids]
             - raw.obj.data.root_pos_w[:, None, :]).norm(dim=-1)
        return d.min(dim=1).values

    ep_fold, ep_plant, ep_up, ep_hz, ep_ol = [], [], [], [], []

    if args.mode == "ref":
        acts = torch.zeros(n, raw.cfg.action_space, device=dev)
        with torch.inference_mode():
            for rnd in range(args.rounds):
                env.reset()
                live = raw.ref_t0 >= 0
                env.step(acts)  # settle one physics step on the restore
                up, hz, ol = torso_up(), hand_z(), objleg()
                # scope = the SHIPPED gate's scope (obj_z > 0.22, the
                # lifted-context plant condition in grasp_env). First cert
                # run judged descent frames (28.4% false plants from
                # legitimately floor-adjacent hands); a 0.40 scope left
                # n=2 — no power. 0.22 matches the training gate exactly:
                # every frame the gate can zero must be a legal frame here.
                objz_r = raw.obj.data.root_pos_w[:, 2] - org[:, 2]
                live = live & (objz_r > 0.22)
                for i in range(n):
                    if not bool(live[i]):
                        continue
                    fold = bool(up[i] < args.fold_bar)
                    # single-frame read: plant = instantaneous skim
                    plant = bool(hz[i] < args.plant_z)
                    ep_fold.append(fold)
                    ep_plant.append(plant)
                    ep_up.append(float(up[i]))
                    ep_hz.append(float(hz[i]))
                    ep_ol.append(float(ol[i]))
                if rnd % 10 == 0:
                    print(f"REF rnd={rnd} samples={len(ep_up)}", flush=True)
    else:
        assert args.ckpt, "--ckpt required in policy mode"
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        agent_cfg = cli_args.parse_rsl_rl_cfg("G1-Grasp-v0", args)
        env = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                                device=agent_cfg.device)
        runner.load(args.ckpt)
        policy = runner.get_inference_policy(device=dev)

        strict_streak = torch.zeros(n, device=dev)
        strict_seen = torch.zeros(n, dtype=torch.bool, device=dev)
        min_up = torch.full((n,), 10.0, device=dev)
        min_hz = torch.full((n,), 10.0, device=dev)
        min_ol = torch.full((n,), 10.0, device=dev)
        plant_ct = torch.zeros(n, device=dev)
        done_eps = 0
        obs = env.get_observations()
        if isinstance(obs, tuple):
            obs = obs[0]
        with torch.inference_mode():
            while done_eps < args.episodes and app.is_running():
                obs, _, dones, _ = env.step(policy(obs))
                objz = raw.obj.data.root_pos_w[:, 2] - org[:, 2]
                nf, th = raw._finger_contacts()
                strict_now = (objz > 0.40) & (nf >= 3) & th
                strict_streak = torch.where(
                    strict_now, strict_streak + 1,
                    torch.zeros_like(strict_streak))
                strict_seen |= strict_streak >= 150
                if strict_now.any():
                    up, hz, ol = torso_up(), hand_z(), objleg()
                    m = strict_now
                    min_up[m] = torch.minimum(min_up[m], up[m])
                    min_hz[m] = torch.minimum(min_hz[m], hz[m])
                    min_ol[m] = torch.minimum(min_ol[m], ol[m])
                    plant_ct[m] += (hz[m] < args.plant_z).float()
                fin = dones.bool() if not isinstance(dones, tuple) else dones[0].bool()
                if fin.any():
                    for i in fin.nonzero(as_tuple=True)[0].tolist():
                        if bool(strict_seen[i]):
                            fold = bool(min_up[i] < args.fold_bar)
                            plant = bool(plant_ct[i] >= args.plant_steps)
                            ep_fold.append(fold)
                            ep_plant.append(plant)
                            ep_up.append(float(min_up[i]))
                            ep_hz.append(float(min_hz[i]))
                            ep_ol.append(float(min_ol[i]))
                            done_eps += 1
                        strict_seen[i] = False
                        strict_streak[i] = 0.0
                        min_up[i], min_hz[i], min_ol[i] = 10.0, 10.0, 10.0
                        plant_ct[i] = 0.0
                    if done_eps and done_eps % 10 == 0:
                        print(f"POLICY strict_eps={done_eps}", flush=True)

    tot = len(ep_up)
    ill = sum(f or p for f, p in zip(ep_fold, ep_plant))
    fr = ill / max(tot, 1)
    print(f"WAISTFOLD_DIST n={tot} torso_up p10/50/90="
          f"{q(ep_up,.1):.3f}/{q(ep_up,.5):.3f}/{q(ep_up,.9):.3f} "
          f"hand_z p10/50/90={q(ep_hz,.1):.3f}/{q(ep_hz,.5):.3f}/"
          f"{q(ep_hz,.9):.3f} objleg p10/50/90={q(ep_ol,.1):.3f}/"
          f"{q(ep_ol,.5):.3f}/{q(ep_ol,.9):.3f}", flush=True)
    print(f"WAISTFOLD_FLAGS fold={sum(ep_fold)}/{tot} "
          f"plant={sum(ep_plant)}/{tot} illegal={ill}/{tot} "
          f"rate={fr:.3f}", flush=True)
    if args.mode == "policy":
        ok = fr >= 0.95
        print(f"WAISTFOLD_{'PASS' if ok else 'FAIL'} policy(crab) "
              f"flag rate {fr:.3f} (need >=0.95)", flush=True)
    else:
        ok = fr <= 0.05
        print(f"WAISTFOLD_{'PASS' if ok else 'FAIL'} ref(legit) "
              f"flag rate {fr:.3f} (need <=0.05)", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
