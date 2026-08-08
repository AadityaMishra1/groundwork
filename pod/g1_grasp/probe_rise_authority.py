"""RISE-AUTHORITY probe: is standing up from the banked policy's hold
posture a LEARNING problem or a PHYSICS wall?

Context (2026-07-31). The banked artifact holds the cylinder at 0.40 m in a
low lunge, pelvis ~0.41 m, and never rises. A rise reward was added, proved
to activate (tools/cert_rise_activation.py), and the policy DECLINED it —
recorded as refuted. But nobody ever asked the prior question: can this
posture rise at all? Earlier work (probe_lift_authority, 2026-07-29) found
deep-fold grasps pin the knee at its 139 N.m cap with an object ceiling of
0.39 m, while the certified squat band has ~2x headroom. Which side of that
line the banked lunge sits on has never been measured.

Method — no scripting, no new controller. Run the banked policy; during
CERTIFIED strict holds (object >= 0.40 m, >=3 finger groups + thumb),
record the torque the actuators are already spending, per joint, as a
fraction of that joint's effort limit. Interpretation:
    knees near the cap  -> no authority remains to extend; rising is a
                           physics wall and NO reward can buy it. The fix
                           is a different grasp posture, not a new term.
    knees with headroom -> the plant can rise; the policy has not learned
                           to. That is an exploration/curriculum problem,
                           and staged starts (drop the policy into a
                           half-risen holding state) is the tool, not a
                           bigger ramp.

    PYTHONPATH=. python -u g1_grasp/probe_rise_authority.py \
        --ckpt /workspace/banked/GRASP_ARTIFACT_laneYb_25200.pt \
        --episodes 40 --headless
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("GRASP_FORCE_STAGE", "3")
os.environ.setdefault("GRASP_NO_VIDEO", "1")

from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # isort: skip  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--episodes", type=int, default=40)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import g1_grasp  # noqa: F401, E402

# measured effort limits (probe_lift_authority, 2026-07-29): the USD's
# reported limits are dummies (1e9), so these are the real actuator caps.
LIMITS = {"knee": 139.0, "hip": 88.0}


def main():
    cfg = parse_env_cfg("G1-Grasp-v0", num_envs=32)
    env = gym.make("G1-Grasp-v0", cfg=cfg)
    raw = env.unwrapped
    dev = raw.device
    names = list(raw.robot.data.joint_names)
    groups = {k: [i for i, nm in enumerate(names) if k in nm.lower()]
              for k in LIMITS}
    print("RISEAUTH joints: "
          + "; ".join(f"{k}={[names[i] for i in v]}" for k, v in groups.items()),
          flush=True)

    env = RslRlVecEnvWrapper(env)
    agent_cfg = cli_args.parse_rsl_rl_cfg("G1-Grasp-v0", args)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=agent_cfg.device)
    runner.load(args.ckpt)
    policy = runner.get_inference_policy(device=dev)

    frac = {k: [] for k in LIMITS}
    pelv, done_eps, steps_held = [], 0, 0
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    with torch.inference_mode():
        while done_eps < args.episodes and app.is_running():
            obs, _, dones, _ = env.step(policy(obs))
            objz = (raw.obj.data.root_pos_w[:, 2]
                    - raw.scene.env_origins[:, 2])
            nf, th = raw._finger_contacts()
            hold = (objz > 0.40) & (nf >= 3) & th
            if hold.any():
                steps_held += int(hold.sum())
                tau = raw.robot.data.applied_torque
                for k, idx in groups.items():
                    if not idx:
                        continue
                    t = tau[:, idx].abs().max(dim=1).values[hold]
                    frac[k] += (t / LIMITS[k]).tolist()
                pelv += (raw.robot.data.root_pos_w[hold, 2]
                         - raw.scene.env_origins[hold, 2]).tolist()
            d = dones.bool() if not isinstance(dones, tuple) else dones[0].bool()
            done_eps += int(d.sum())

    if steps_held == 0:
        print("RISEAUTH_FAIL no strict holds observed — cannot measure")
        return

    def q(v, p):
        return float(torch.tensor(v).quantile(p))

    print(f"RISEAUTH held-steps={steps_held} pelvis_z p50={q(pelv,.5):.3f}",
          flush=True)
    verdict_capped = False
    for k in LIMITS:
        if not frac[k]:
            continue
        f = frac[k]
        print(f"RISEAUTH {k}: torque/limit p50={q(f,.5):.3f} "
              f"p90={q(f,.9):.3f} max={max(f):.3f} (limit {LIMITS[k]} N.m)")
        if q(f, .9) > 0.85:
            verdict_capped = True
    hr = 1.0 - q(frac["knee"], .9) if frac["knee"] else 0.0
    print(f"RISEAUTH knee headroom at p90 = {hr:.1%} of cap")
    if verdict_capped:
        print("RISEAUTH_VERDICT PHYSICS_WALL — actuators are already near "
              "cap holding this posture; no reward can buy a rise. Fix is a "
              "different grasp posture, not another term.")
    else:
        print("RISEAUTH_VERDICT HEADROOM_EXISTS — the plant can extend from "
              "this posture; not rising is a LEARNING gap. Staged starts "
              "(half-risen holding states) are the tool, not a bigger ramp.")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
