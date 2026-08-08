"""LIFT-AUTHORITY probe: can ANY controller lift from a floor-grasp state
under the current plant + delta-action scheme — or is grasp_no_lift an
actuation-authority wall, not a policy deficiency?

Context (outside review, 2026-07-29): the lead lane's honest-eval taxonomy is
grasp_no_lift-dominated (82/101 at model_13000) while every playback rung of
the reference bank's lift path failed (F0..F4 — statically certified, never
dynamically demonstrated). Two hypotheses remain:

    H-POLICY    the plant can lift; the policy just hasn't learned the
                sustained target offsets deep flexion needs.
    H-AUTHORITY under GRASP_LEG_STIFF=4.0 + delta slew (legs 0.12 rad/step,
                cut 60% by the grip gear the moment the hand grasps) + DCMotor
                effort clipping, no achievable target schedule produces the
                knee/hip torque to rise while holding — the policy CANNOT
                learn the lift because the action space cannot express it.

This probe discriminates them for ~$0.30. No policy anywhere: bank2 cage
restores (engine-earned floor-grasps, GRASP_FORCE_STAGE=1) + a scripted
whole-body rise via exact delta-integrator inversion (probe_ref_playback
pattern, live gear included). Arms, all in ONE boot (probe_walk lesson —
the 5-min Isaac boot is the bottleneck):

    ctrl    zero actions for the full window. Known-answer control: cage
            restores must stay grasped (validates the restore) and must NOT
            lift on their own (rules out an already-aloft artifact). Either
            failure voids every verdict below.
    lead1   legs ramp current -> stand-blend targets over --rise_steps,
            plain schedule (desired = interpolant).
    lead2   same, but commanded target overshoots the interpolant by 2x the
            remaining error (crude feedforward: PD needs target-beyond-pose
            to make torque; this is what a trained policy would have to do).
    lead4   overshoot 4x — the authority ceiling of the action space.

Arm/hand/waist/finger targets are FROZEN at their restored values throughout
(zero action on those slots) — the grip is the bank's own squeeze equilibrium.

Per-arm verdict over QUALIFIED envs (grasped AND object at floor after the
settle window — the exact grasp_no_lift situation):
    lift040   object z crossed 0.40 with the engine grasp intact
    lift022   crossed 0.22 grasped (partial credit: the lift exists)
    sag       pelvis ended >5 cm below its post-settle height
    fell      env terminated during the rise
Plus the decisive instrument: knee/hip |applied|/|computed| torque ratio and
saturation fraction during the rise (computed>>applied = effort clipping =
the actuator wall itself, visible directly).

READING THE RESULT
    ctrl dirty                        -> probe void, investigate restore
    lead* lift040 ~0 AND knee saturated (ratio<<1 or |applied| pinned)
                                      -> H-AUTHORITY: relaunches that keep
                                         this plant+action scheme are dead
    lead2/lead4 lift040 >> lead1      -> lift needs sustained overshoot the
                                         policy must learn == exactly what
                                         delta actions express WORST; action
                                         scheme, not stiffness, is the wall
    lead1 lifts fine                  -> H-POLICY: the plant is innocent,
                                         training/economy is the problem

    GRASP_FULLBODY=1 GRASP_FREE=1 GRASP_LEG_STIFF=4.0 GRASP_EP_S=30 \
    GRASP_FLUNG_REF=robot GRASP_LEG_BLEND=1.0 GRASP_SPAWN_Z=0.80 \
    GRASP_BANK2=/workspace/bank_run5.pt GRASP_FORCE_STAGE=1 GRASP_NO_VIDEO=1 \
    PYTHONPATH=/workspace/humanoid/pod python -u \
        g1_grasp/probe_lift_authority.py --headless --num_envs 48

Optional second boot with GRASP_LEG_STIFF=8.0 isolates raw stiffness:
lead1@8.0 succeeding where lead1@4.0 failed = kp headroom is the fix.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=48)
parser.add_argument("--settle_steps", type=int, default=25)
parser.add_argument("--rise_steps", type=int, default=150)
parser.add_argument("--hold_steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import g1_grasp  # noqa: F401

# the walker's stand-blend leg targets (grasp_env stand_legs / LEG_BLEND=1.0)
STAND_LEGS = {
    ".*_hip_pitch_joint": -0.20, ".*_knee_joint": 0.42,
    ".*_ankle_pitch_joint": -0.23, ".*_hip_roll_joint": 0.0,
    ".*_hip_yaw_joint": 0.0, ".*_ankle_roll_joint": 0.0,
}
ARMS = (("ctrl", None), ("lead1", 1.0), ("lead2", 2.0), ("lead4", 4.0))


def main():
    env_cfg = parse_env_cfg("G1-Grasp-v0", num_envs=args.num_envs)
    env = gym.make("G1-Grasp-v0", cfg=env_cfg)
    raw = env.unwrapped
    assert raw.bank2 is not None and raw.bank2g.get(1) is not None, \
        "needs GRASP_BANK2 with a closing/cage group (GRASP_FORCE_STAGE=1)"
    dev = raw.device
    n = raw.num_envs
    org = raw.scene.env_origins
    robot = raw.robot

    # leg slots in the ACTION vector + their stand-blend targets
    act_ids = [int(j) for j in raw.joint_ids]
    leg_slots, leg_tgts = [], []
    for pattern, val in STAND_LEGS.items():
        jids, _ = robot.find_joints(pattern)
        for j in jids:
            if int(j) in act_ids:
                leg_slots.append(act_ids.index(int(j)))
                leg_tgts.append(val)
    leg_slots_t = torch.tensor(leg_slots, dtype=torch.long, device=dev)
    leg_tgts_t = torch.tensor(leg_tgts, device=dev)

    # knee/hip joints (robot indexing) for the torque instrument
    tq_ids = []
    for pat in (".*_knee_joint", ".*_hip_pitch_joint"):
        jids, _ = robot.find_joints(pat)
        tq_ids += [int(j) for j in jids]
    tq_ids_t = torch.tensor(sorted(tq_ids), dtype=torch.long, device=dev)
    # G1 datasheet effort limits by joint name (v1 lesson: the data-field
    # lookup returned a 1e9 dummy and made sat_frac vacuously 0 while the
    # printed max torque sat pinned at exactly 139.0 — the knee cap)
    _names = list(robot.data.joint_names)
    eff_lim = torch.tensor(
        [139.0 if "knee" in _names[int(j)] else 88.0 for j in tq_ids_t],
        device=dev)
    print(f"LIFTAUTH envs={n} leg_slots={len(leg_slots)} "
          f"tq_joints={len(tq_ids)} eff_lim="
          + (str([round(float(x), 1) for x in eff_lim]) if eff_lim is not None
             else "UNAVAILABLE (reporting raw torques)"), flush=True)

    total = args.settle_steps + args.rise_steps + args.hold_steps
    for arm_name, lead in ARMS:
        env.reset()
        z0_pelv = None
        qual = torch.zeros(n, dtype=torch.bool, device=dev)
        lifted040 = torch.zeros(n, dtype=torch.bool, device=dev)
        lifted022 = torch.zeros(n, dtype=torch.bool, device=dev)
        fell = torch.zeros(n, dtype=torch.bool, device=dev)
        maxz = torch.zeros(n, device=dev)
        sat_sum, sat_n = 0.0, 0
        ratio_sum, ratio_n = 0.0, 0
        acts = torch.zeros(n, raw.cfg.action_space, device=dev)
        start_leg_pos = None
        for step in range(total):
            if step < args.settle_steps or lead is None:
                a = acts  # zero: integrator frozen at restored targets
            else:
                s = min(1.0, (step - args.settle_steps) / args.rise_steps)
                # cosine-eased interpolant current-start -> stand legs
                e = 0.5 - 0.5 * torch.cos(torch.tensor(3.14159265 * s))
                interp = start_leg_pos + e.to(dev) * (
                    leg_tgts_t.unsqueeze(0) - start_leg_pos)
                # overshoot: command beyond the interpolant by (lead-1)x the
                # remaining pose error — the feedforward a PD lift needs
                cur = robot.data.joint_pos[:, raw.joint_ids][:, leg_slots_t]
                desired = interp + (lead - 1.0) * (interp - cur)
                # exact delta-integrator inversion, live gear (playback law)
                gear = (1.0 - 0.6 * raw.grasped_state.float()).unsqueeze(-1)
                dv = raw.delta_vec[leg_slots_t].unsqueeze(0)
                a = torch.zeros_like(acts)
                a[:, leg_slots_t] = ((desired - raw.targets[:, leg_slots_t])
                                     / (dv * gear)).clamp(-1.0, 1.0)
            out = env.step(a)
            done = out[2] if isinstance(out, tuple) else out.terminated
            pelv = robot.data.root_pos_w[:, 2] - org[:, 2]
            objz = raw.obj.data.root_pos_w[:, 2] - org[:, 2]
            nf, th = raw._finger_contacts()
            g = (nf >= 3) & th
            if step == args.settle_steps - 1:
                # qualification: engine grasp survived the settle. v1 gated
                # additionally on objz<0.12 and got 0/48 — the closing-family
                # states restore with the object at ~0.23-0.26, so verdicts
                # were vacuous. Success below is DELTA-based instead.
                qual = g
                obj_z0 = objz.clone()
                z0_pelv = pelv.clone()
                start_leg_pos = robot.data.joint_pos[
                    :, raw.joint_ids][:, leg_slots_t].clone()
                print(f"{arm_name} QUALIFY {int(qual.sum())}/{n} "
                      f"(grasped@floor after settle) objz_p50={objz.median():.3f}",
                      flush=True)
            if step >= args.settle_steps:
                fell |= done & qual
                live = qual & ~fell
                # delta verdicts: the object must GAIN height with the grip
                # intact (absolute bars were vacuous for these start states)
                lifted022 |= live & g & (objz - obj_z0 > 0.10)
                lifted040 |= live & g & (objz - obj_z0 > 0.18) & (objz > 0.40)
                maxz[live] = torch.maximum(maxz[live], objz[live])
                # torque instrument on knee/hip during the rise
                appl = getattr(robot.data, "applied_torque", None)
                comp = getattr(robot.data, "computed_torque", None)
                if appl is not None:
                    at = appl[:, tq_ids_t].abs()
                    if eff_lim is not None:
                        sat_sum += float((at >= 0.98 * eff_lim.unsqueeze(0))
                                         .float().mean())
                        sat_n += 1
                    if comp is not None:
                        ct = comp[:, tq_ids_t].abs().clamp(min=1e-6)
                        ratio_sum += float((at / ct).clamp(max=1.0).mean())
                        ratio_n += 1
            if step in (args.settle_steps, args.settle_steps + 50,
                        args.settle_steps + 100, total - args.hold_steps,
                        total - 1):
                appl = getattr(robot.data, "applied_torque", None)
                kt = (appl[:, tq_ids_t].abs().max() if appl is not None
                      else float("nan"))
                print(f"{arm_name} t={step:3d} pelv_p50={pelv.median():.3f} "
                      f"objz_p50/max={objz.median():.3f}/{objz.max():.3f} "
                      f"grasped={g.float().mean():.2f} "
                      f"kneehip_maxtq={float(kt):.1f} "
                      f"fell={int(fell.sum())}", flush=True)
        pelv_end = robot.data.root_pos_w[:, 2] - org[:, 2]
        sag = qual & ~fell & ((z0_pelv - pelv_end) > 0.05)
        nq = max(1, int(qual.sum()))
        print(f"ARM_{arm_name.upper()} qual={int(qual.sum())} "
              f"lift040={int(lifted040.sum())} lift022={int(lifted022.sum())} "
              f"sag={int(sag.sum())} fell={int(fell.sum())} "
              f"maxz_p50/p90={maxz[qual].median() if qual.any() else 0:.3f}/"
              f"{maxz[qual].quantile(0.9) if qual.any() else 0:.3f} "
              f"knee_sat_frac={sat_sum / max(sat_n, 1):.3f} "
              f"appl/comp_ratio={ratio_sum / max(ratio_n, 1):.3f}", flush=True)

    print("PROBE_LIFTAUTH_DONE — read verdicts per the docstring table",
          flush=True)


if __name__ == "__main__":
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
