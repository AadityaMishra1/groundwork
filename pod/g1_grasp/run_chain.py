"""Composition eval: one uncut walk -> kneel -> settle -> grasp -> lift episode.

Drives the existing G1-Grasp-v0 env with TWO frozen policies and a per-env
phase state machine: the walker (embedded in the env via GRASP_LOCO_CKPT,
owns legs/waist/left arm) and the grasp policy (--checkpoint, owns right arm
+ hand). No training, no reward — just the strict protocol verdict
(0.40 m + 3 s continuous hold, >=3 finger groups + thumb) plus a chain
taxonomy that says WHERE the chain broke.

    GRASP_LOCO_CKPT=/workspace/g1_locokneel_m1f3_final_model_5999.pt \
    python -u g1_grasp/run_chain.py \
        --checkpoint /workspace/g1_grasp_r6b_28pct_strict_model_16998.pt \
        --num_envs 64 --episodes 200 --max_steps 900 --headless

The runner mutates raw.loco_vel_cmd / raw.loco_hcmd_t / raw.park_now /
raw.suppress_grasp in place each step; grasp_env.py owns their semantics.
"""

import argparse
import os

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
# NOTE: --checkpoint (the grasp policy) is already defined by
# cli_args.add_rsl_rl_args, so re-adding it here is an argparse conflict —
# it is required after parsing instead.
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=200)
# WALK target is a STANDOFF POSE, not a distance: the grasp policy only ever
# saw the object in the front-right corridor of the base frame (grasp_env
# _reset_idx: r 0.20-0.35, bearing -1.05..-0.45 rad). Walking to a scalar
# distance and facing the object head-on parks it at bearing ~0 — outside the
# seated-reach workspace, where the grasp policy has never been.
parser.add_argument("--target_r", type=float, default=0.27, help="corridor centre range")
parser.add_argument("--target_th", type=float, default=-0.75, help="corridor centre bearing")
parser.add_argument("--tol_r", type=float, default=0.06)
parser.add_argument("--tol_th", type=float, default=0.20)
parser.add_argument("--settle_steps", type=int, default=100)
parser.add_argument("--max_steps", type=int, default=600,
                    help="episode budget in control steps (sets episode_length_s)")
parser.add_argument("--walk_speed", type=float, default=0.6)
# the walker's height command was only ever trained on (0.24, 0.72) —
# commanding the nominal 0.75 stand is out of distribution
parser.add_argument("--stand_h", type=float, default=0.72)
parser.add_argument("--kneel_h", type=float, default=0.32)
parser.add_argument("--descend_z", type=float, default=0.30,
                    help="pelvis z below which the kneel counts as arrived")
parser.add_argument("--yaw_gain", type=float, default=1.0)
# EVAL_PROTOCOL: object spawns 1.0-4.0 m away at any bearing. The env's own
# full-task spawn is the in-reach grasp corridor itself, i.e. the standoff is
# already satisfied at step 0 and there is no walk to evaluate — so the runner
# re-places the object at episode t=0. 0 = keep the env's spawn.
parser.add_argument("--spawn_dist", type=float, default=1.5)
parser.add_argument("--spawn_bearing", type=float, default=0.6,
                    help="half-width of the spawn bearing arc, rad")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.checkpoint:
    parser.error("--checkpoint (grasp policy .pt) is required")
if not os.environ.get("GRASP_LOCO_CKPT"):
    parser.error("GRASP_LOCO_CKPT=<locokneel model.pt> must be set — the chain "
                 "has no walker without it")

# free root + honest gravity, full-task starts: the composition physics
os.environ["GRASP_FREE"] = "1"
os.environ["GRASP_FORCE_STAGE"] = "3"
# the phase machine owns settling and parking; the env's own settle window
# and latch run on episode_length_buf and would fight it
os.environ["GRASP_LOCO_SETTLE"] = "0"
os.environ["GRASP_LOCO_PARK"] = "0"
# objects legitimately start 1.2-1.8 m out here — the training-era flung
# check (>1.2 m from origin) would end every episode at step 1
os.environ["GRASP_FLUNG_REF"] = "robot"
# the walker reads joint_pos RELATIVE to default_joint_pos and writes targets
# as default + 0.5*action — its reference must be the stand pose it trained
# on, not the kneel the grasp env bakes in. Also spawn standing (protocol).
os.environ.setdefault("GRASP_LEG_BLEND", "1.0")
os.environ.setdefault("GRASP_SPAWN_Z", "0.80")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_tasks  # noqa: F401
import g1_grasp  # noqa: F401

import cli_args

WALK, KNEEL, SETTLE, GRASP = 0, 1, 2, 3
# trained command envelope (loco_env_cfg.commands.base_velocity.ranges)
VX_RANGE, VY_RANGE, WZ_RANGE = (-0.3, 0.7), (-0.3, 0.3), (-0.6, 0.6)
# the grasp policy's trained object corridor, base frame (grasp_env _reset_idx)
CORRIDOR_R, CORRIDOR_TH = (0.20, 0.35), (-1.05, -0.45)
# range error at which the approach runs at full --walk_speed
APPROACH_BAND = 0.30

# the chain-control interface grasp_env.py must expose
CONTRACT = {
    "loco_vel_cmd": "(num_envs, 3) float — walker vx/vy/yaw_rate command",
    "loco_hcmd_t": "(num_envs,) float — per-env walker height command",
    "park_now": "(num_envs,) bool — latch the walker's leg targets",
    "suppress_grasp": "(num_envs,) bool — force the grasp action to zero",
    "arm_follow_t": "(num_envs,) bool — right arm follows the walker (swing) "
                    "vs holds the grasp start pose; True for WALK/KNEEL, "
                    "False from SETTLE on (dead-arm stall vs cold-handoff)",
}


def main():
    task = "G1-Grasp-v0"
    env_cfg = parse_env_cfg(task, num_envs=args.num_envs)
    # the walk needs a budget the 12 s grasp episode does not have
    env_cfg.episode_length_s = args.max_steps * env_cfg.sim.dt * env_cfg.decimation
    agent_cfg = cli_args.parse_rsl_rl_cfg(task, args)

    env = gym.make(task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw = env.unwrapped
    missing = [f"{k}: {v}" for k, v in CONTRACT.items() if not hasattr(raw, k)]
    if missing:
        raise RuntimeError(
            "grasp_env.py is missing the chain-control interface run_chain.py "
            "drives. Required attributes:\n  " + "\n  ".join(missing))
    if getattr(raw, "loco", None) is None:
        raise RuntimeError("the env did not load a walker — check GRASP_LOCO_CKPT")

    dev = raw.device
    n = raw.num_envs
    org = raw.scene.env_origins
    obj_z0 = float(raw.cfg.object_cfg.init_state.pos[2])

    def place_objects(ids):
        """Re-place the object at a walking distance (episode t=0 write)."""
        if args.spawn_dist <= 0.0:
            return
        m = len(ids)
        th = (2.0 * torch.rand(m, device=dev) - 1.0) * args.spawn_bearing
        d = args.spawn_dist * (0.8 + 0.4 * torch.rand(m, device=dev))
        pos = torch.stack([d * torch.cos(th), d * torch.sin(th),
                           torch.full((m,), obj_z0, device=dev)], dim=-1)
        quat = torch.zeros(m, 4, device=dev)
        quat[:, 0] = 1.0
        pose = torch.cat([pos + org[ids], quat], dim=-1)
        raw.obj.write_root_pose_to_sim(pose, ids)
        raw.obj.write_root_velocity_to_sim(torch.zeros(m, 6, device=dev), ids)

    phase = torch.zeros(n, dtype=torch.long, device=dev)
    ep_step = torch.zeros(n, dtype=torch.long, device=dev)
    t_kneel = torch.zeros(n, dtype=torch.long, device=dev)   # step WALK ended
    t_settle = torch.zeros(n, dtype=torch.long, device=dev)  # step KNEEL ended
    # per-episode outcome accumulators
    hit_kneel = torch.zeros(n, dtype=torch.bool, device=dev)
    hit_settle = torch.zeros(n, dtype=torch.bool, device=dev)
    hit_grasp = torch.zeros(n, dtype=torch.bool, device=dev)
    got_near = torch.zeros(n, dtype=torch.bool, device=dev)     # palm within 12 cm
    got_contact = torch.zeros(n, dtype=torch.bool, device=dev)
    got_grasp = torch.zeros(n, dtype=torch.bool, device=dev)    # >=3 groups + thumb
    got_lift = torch.zeros(n, dtype=torch.bool, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    strict_streak = torch.zeros(n, device=dev)
    strict_ok = torch.zeros(n, dtype=torch.bool, device=dev)
    loose_ok = torch.zeros(n, dtype=torch.bool, device=dev)

    done_count = strict_count = loose_count = 0
    n_kneel = n_settle = n_grasp = 0
    tax = {k: 0 for k in ["never_near", "near_no_contact", "contact_no_grasp",
                          "grasp_no_lift", "lift_no_hold", "success", "fell"]}
    walk_lens, descend_lens, grasp_starts = [], [], []
    # standoff geometry captured at the two handoff moments
    so_kneel_r, so_kneel_th, so_grasp_r, so_grasp_th, arr_palm = [], [], [], [], []
    peak_zs = []
    max_objz = torch.zeros(n, device=dev)

    place_objects(torch.arange(n, device=dev))
    obs, _ = env.get_observations(), None
    gstep = 0
    with torch.inference_mode():
        while done_count < args.episodes and simulation_app.is_running():
            gstep += 1
            # --- state (post previous step / post reset) ---
            base = raw.robot.data.root_pos_w
            objw = raw.obj.data.root_pos_w
            d_w = objw[:, :2] - base[:, :2]
            dist = d_w.norm(dim=-1)
            pelvis_z = base[:, 2] - org[:, 2]
            q = raw.robot.data.root_quat_w
            yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                              1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            c, s = torch.cos(yaw), torch.sin(yaw)
            # world -> base frame (yaw only: velocity commands live in the
            # heading frame, a pitched torso must not tilt them)
            dx = c * d_w[:, 0] + s * d_w[:, 1]
            dy = -s * d_w[:, 0] + c * d_w[:, 1]
            bearing = torch.atan2(dy, dx)

            # standoff error: kept for the CHAIN_STANDOFF diagnostics
            e_r = dist - args.target_r
            e_th = bearing - args.target_th
            e_th = torch.atan2(torch.sin(e_th), torch.cos(e_th))

            # POSITION SERVO (v6): the m1f3 walker has NO usable yaw
            # authority — v5 measured wz_cmd -0.60 -> yawrate +0.03 and a
            # +0.27 rad/s UNCOMMANDED left drift; turning was never trained
            # (kneel-era rewards) nor probed (probe_walk commands pure vx).
            # The corridor bearing is a property of POSITION, not heading:
            # stand at the world point that puts the object at
            # (target_r, target_th) in the base frame. No wz ever commanded;
            # yaw drift just moves the target point and the servo recomputes.
            ox = args.target_r * torch.cos(torch.tensor(args.target_th, device=dev))
            oy = args.target_r * torch.sin(torch.tensor(args.target_th, device=dev))
            tx = objw[:, 0] - (c * ox - s * oy)
            ty = objw[:, 1] - (s * ox + c * oy)
            ex_w, ey_w = tx - base[:, 0], ty - base[:, 1]
            e_b = torch.stack([c * ex_w + s * ey_w,
                               -s * ex_w + c * ey_w], dim=-1)
            e_norm = e_b.norm(dim=-1)

            # --- per-env phase advance (never a global barrier) ---
            # region acceptance: object inside the trained grasp corridor
            # (with margin), not at a point — the corridor is 15 cm deep
            adv = ((phase == WALK)
                   & (dist > CORRIDOR_R[0] + 0.03) & (dist < CORRIDOR_R[1] - 0.01)
                   & (bearing > CORRIDOR_TH[0] + 0.10)
                   & (bearing < CORRIDOR_TH[1] - 0.05))
            if adv.any():
                phase[adv] = KNEEL
                t_kneel[adv] = ep_step[adv]
                hit_kneel |= adv
                so_kneel_r += dist[adv].tolist()
                so_kneel_th += bearing[adv].tolist()
            adv = (phase == KNEEL) & (pelvis_z < args.descend_z)
            if adv.any():
                phase[adv] = SETTLE
                t_settle[adv] = ep_step[adv]
                hit_settle |= adv
            adv = (phase == SETTLE) & (ep_step - t_settle >= args.settle_steps)
            if adv.any():
                phase[adv] = GRASP
                hit_grasp |= adv
                palm = raw.robot.data.body_pos_w[:, raw.palm_id]
                grasp_starts += ep_step[adv].tolist()
                # the base shifts during the kneel + settle — this, not the
                # kneel-time pose, is what the grasp policy inherits
                so_grasp_r += dist[adv].tolist()
                so_grasp_th += bearing[adv].tolist()
                arr_palm += (palm - objw).norm(dim=-1)[adv].tolist()

            # --- write the walker / grasp gating for this step ---
            walking = phase == WALK
            # drive straight at the standoff POINT; never command yaw
            u = e_b / e_norm.clamp(min=1e-6).unsqueeze(-1)
            speed = args.walk_speed * (e_norm / APPROACH_BAND).clamp(max=1.0)
            # creep near the OBJECT regardless of target distance: 88/201 v6
            # episodes died to kicked cylinders — stride forces at 0.35 m/s
            # tip a 45 mm cylinder the moment a swing foot grazes it
            speed = speed * ((dist - 0.30) / 0.30).clamp(min=0.2, max=1.0)
            vel = torch.zeros_like(raw.loco_vel_cmd)
            vel[:, 0] = (speed * u[:, 0]).clamp(*VX_RANGE)
            vel[:, 1] = (speed * u[:, 1]).clamp(*VY_RANGE)
            vel[:, 2] = 0.0
            vel[~walking] = 0.0
            raw.loco_vel_cmd[:] = vel
            hcmd = torch.full_like(raw.loco_hcmd_t, args.kneel_h)
            hcmd[walking] = args.stand_h
            raw.loco_hcmd_t[:] = hcmd
            raw.suppress_grasp[:] = phase < GRASP
            raw.park_now[:] = phase == GRASP
            # right arm swings with the walker until the descent completes;
            # from SETTLE on it parks at the grasp policy's start pose
            raw.arm_follow_t[:] = phase <= KNEEL

            # direct observation — v4b showed |eth| DIVERGING 0.65 -> 1.7:
            # the yaw loop turns robots away from the object. Signed bearing
            # + commanded wz + ACTUAL yaw rate in one line pins whether the
            # walker's yaw convention is flipped or it ignores wz entirely.
            if gstep % 100 == 1:
                v = raw.robot.data.root_lin_vel_w[:, :2].norm(dim=-1)
                vb = raw.robot.data.root_lin_vel_b
                tc = raw.term_counts  # cumulative held/flung/tipped/timeout
                print(f"CHAIN_DBG t={gstep} done={done_count} "
                      f"W/K/S/G={int((phase==WALK).sum())}/{int((phase==KNEEL).sum())}"
                      f"/{int((phase==SETTLE).sum())}/{int((phase==GRASP).sum())} "
                      f"dist_p25/50/75={dist.quantile(0.25):.2f}/{dist.median():.2f}"
                      f"/{dist.quantile(0.75):.2f} enorm_p50={e_norm.median():.2f} "
                      f"speed_p50={v.median():.2f} bearing_p50={bearing.median():+.2f} "
                      f"vyerr_p50={(vel[:, 1] - vb[:, 1]).abs().median():.2f} "
                      f"term(h/f/t/to)={tc[0]}/{tc[1]}/{tc[2]}/{tc[3]}",
                      flush=True)

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            ep_step += 1

            # --- outcome accumulation (post step) ---
            palm = raw.robot.data.body_pos_w[:, raw.palm_id]
            objw = raw.obj.data.root_pos_w
            objz = objw[:, 2] - org[:, 2]
            nf, th = raw._finger_contacts()
            grasped = (nf >= 3) & th
            got_near |= (palm - objw).norm(dim=-1) < 0.12
            got_contact |= nf >= 1
            got_grasp |= grasped
            got_lift |= objz > 0.22
            max_objz = torch.maximum(max_objz, objz)
            loose_ok |= raw.hold_counter >= 50
            strict_now = (objz > 0.40) & grasped
            strict_streak = torch.where(strict_now, strict_streak + 1,
                                        torch.zeros_like(strict_streak))
            strict_ok |= strict_streak >= 150
            q = raw.robot.data.root_quat_w
            up = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
            fell |= ((raw.robot.data.root_pos_w[:, 2] - org[:, 2]) < 0.10) | (up < 0.0)

            finished = dones.bool()
            if finished.any():
                idx = finished.nonzero(as_tuple=False).squeeze(-1)
                done_count += len(idx)
                strict_count += int(strict_ok[idx].sum())
                loose_count += int(loose_ok[idx].sum())
                n_kneel += int(hit_kneel[idx].sum())
                n_settle += int(hit_settle[idx].sum())
                n_grasp += int(hit_grasp[idx].sum())
                for i in idx.tolist():
                    if strict_ok[i]:
                        tax["success"] += 1
                    elif fell[i]:
                        tax["fell"] += 1
                    elif got_lift[i]:
                        tax["lift_no_hold"] += 1
                    elif got_grasp[i]:
                        tax["grasp_no_lift"] += 1
                    elif got_contact[i]:
                        tax["contact_no_grasp"] += 1
                    elif got_near[i]:
                        tax["near_no_contact"] += 1
                    else:
                        tax["never_near"] += 1
                    peak_zs.append(float(max_objz[i]))
                    if hit_kneel[i]:
                        walk_lens.append(int(t_kneel[i]))
                    if hit_settle[i]:
                        descend_lens.append(int(t_settle[i] - t_kneel[i]))
                for buf in (hit_kneel, hit_settle, hit_grasp, got_near,
                            got_contact, got_grasp, got_lift, fell, strict_ok,
                            loose_ok):
                    buf[idx] = False
                for buf in (phase, ep_step, t_kneel, t_settle):
                    buf[idx] = 0
                strict_streak[idx] = 0.0
                max_objz[idx] = 0.0
                # the env already auto-reset these envs; this is their t=0
                # (obs for their first step still carries the env's own spawn)
                place_objects(idx)

    p50 = lambda xs: float(torch.tensor(xs, dtype=torch.float).median()) if xs else -1.0
    print(f"CHAIN_RESULT episodes={done_count} successes={strict_count} "
          f"rate={strict_count / max(1, done_count):.3f} "
          f"loose={loose_count} (strict = 0.40 m + 3 s continuous hold)")
    print(f"CHAIN_PHASE reached_kneel={n_kneel} reached_settle={n_settle} "
          f"reached_grasp={n_grasp} of {done_count}")
    print("CHAIN_TAXONOMY " + " ".join(f"{k}={v}" for k, v in tax.items()))
    print(f"CHAIN_STEPS walk_p50={p50(walk_lens):.0f} "
          f"descend_p50={p50(descend_lens):.0f} "
          f"to_grasp_p50={p50(grasp_starts):.0f} "
          f"(n={len(walk_lens)}/{len(descend_lens)}/{len(grasp_starts)})")
    # standoff quality — the most likely chain failure point: a pose outside
    # the trained corridor hands the grasp policy a state it has never seen
    for tag, rs, ths, extra in (("kneel", so_kneel_r, so_kneel_th, ""),
                                ("grasp", so_grasp_r, so_grasp_th,
                                 f" palm_to_obj_p50={p50(arr_palm):.3f}")):
        if not rs:
            continue
        r_t, th_t = torch.tensor(rs), torch.tensor(ths)
        ok = ((r_t >= CORRIDOR_R[0]) & (r_t <= CORRIDOR_R[1])
              & (th_t >= CORRIDOR_TH[0]) & (th_t <= CORRIDOR_TH[1]))
        print(f"CHAIN_STANDOFF at={tag} r_p50={p50(rs):.3f} "
              f"th_p50={p50(ths):+.2f} in_corridor={int(ok.sum())}/{len(rs)}"
              f"{extra}")
    if peak_zs:
        pz = torch.tensor(peak_zs)
        qz = lambda p: float(pz.quantile(p))
        print(f"CHAIN_PEAK_Z n={len(pz)} p25/50/75/95="
              f"{qz(.25):.3f}/{qz(.5):.3f}/{qz(.75):.3f}/{qz(.95):.3f} "
              f"frac>=0.40={float((pz >= 0.40).float().mean()):.3f}")


if __name__ == "__main__":
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
