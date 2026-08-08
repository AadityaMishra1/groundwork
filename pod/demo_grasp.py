"""Deterministic close-up eval of the grasp policy: N episodes, measured
success rate (object airborne >=2s with >=3 finger groups + thumb), and a
close-up video of the first few episodes.

    python -u demo_grasp.py --ckpt <model.pt> --headless --episodes 200
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--stochastic", action="store_true",
                    help="sample actions like training rollouts — separates skill-exists-mean-lags from skill-absent")
parser.add_argument("--video_dir", type=str, default="/workspace/demo_videos")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# GRASP_NO_VIDEO=1: metrics-only eval for hosts without working Vulkan —
# with cameras enabled a failed renderer also kills the PhysX GPU solver
# (falls back to CPU: glacial and not physics-comparable to training)
import os as _os
NO_VIDEO = _os.environ.get("GRASP_NO_VIDEO", "0") == "1"
args.video = not NO_VIDEO
args.enable_cameras = not NO_VIDEO

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


def main():
    task = "G1-Grasp-v0"
    env_cfg = parse_env_cfg(task, num_envs=32)
    # close-up viewer: look at env 0's robot hand area
    env_cfg.viewer.eye = (1.3, -0.8, 0.7)
    env_cfg.viewer.lookat = (0.3, -0.1, 0.2)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task, args)

    env = gym.make(task, cfg=env_cfg,
                   render_mode=None if NO_VIDEO else "rgb_array")
    if not NO_VIDEO:
        env = gym.wrappers.RecordVideo(
            env, video_folder=args.video_dir, step_trigger=lambda s: s == 0,
            # must finalize INSIDE the eval or no file is written (the 800
            # lesson); default 150 stays safe for tiny evals, GRASP_VIDEO_LEN
            # raises it for full-length showcase clips on multi-episode runs
            video_length=int(_os.environ.get("GRASP_VIDEO_LEN", "150")),
            name_prefix="grasp_demo", disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    if args.stochastic:
        # this rsl_rl: alg.policy (not .actor_critic); act() applies the
        # obs normalizer internally; eval_mode freezes its running stats
        runner.eval_mode()
        _ac = (runner.alg.policy if hasattr(runner.alg, "policy")
               else runner.alg.actor_critic)
        policy = lambda o: _ac.act(o).detach()
        print("EVAL_MODE stochastic", flush=True)

    raw = env.unwrapped
    n_envs = raw.num_envs
    done_count, success_count = 0, 0
    ep_success = torch.zeros(n_envs, dtype=torch.bool, device=raw.device)
    # failure taxonomy accumulators (per current episode)
    dev = raw.device
    got_near = torch.zeros(n_envs, dtype=torch.bool, device=dev)      # palm within 12cm
    got_contact = torch.zeros(n_envs, dtype=torch.bool, device=dev)   # any finger force
    got_grasp = torch.zeros(n_envs, dtype=torch.bool, device=dev)     # >=3 groups+thumb
    got_lift = torch.zeros(n_envs, dtype=torch.bool, device=dev)      # obj above 0.22
    max_objz = torch.zeros(n_envs, device=dev)
    ejected = torch.zeros(n_envs, dtype=torch.bool, device=dev)       # airborne w/o contact
    tax = {k: 0 for k in ["never_near", "near_no_contact", "contact_no_grasp",
                          "grasp_no_lift", "lift_no_hold", "success"]}
    eject_count = 0
    # hold-stage diagnosis: flicker vs genuine loss. cur/max = consecutive
    # (lifted & grasped) streak; tot = accumulated held steps; frag = number
    # of separate held fragments; loss_speed = |obj vel| when a streak breaks
    cur_streak = torch.zeros(n_envs, device=dev)
    max_streak = torch.zeros(n_envs, device=dev)
    tot_held = torch.zeros(n_envs, device=dev)
    frags = torch.zeros(n_envs, device=dev)
    loss_speeds = []
    hold_stats = []  # (max_streak, tot_held, frags) per lift-reaching episode
    peak_zs = []  # per-episode max object height — reachability diagnostic
    # strict protocol bar (docs/EVAL_PROTOCOL.md): object >= 0.40 m with the
    # grasp held for 3 continuous seconds (150 steps at 50 Hz)
    strict_streak = torch.zeros(n_envs, device=dev)
    strict_success = torch.zeros(n_envs, dtype=torch.bool, device=dev)
    strict_count = 0
    # never_near forensics: fell-fast vs stood-frozen (mechanism diagnostic)
    ep_len = torch.zeros(n_envs, device=dev)
    last_pz = torch.ones(n_envs, device=dev)
    last_up = torch.ones(n_envs, device=dev)
    nn_lens, nn_fell = [], []
    # CLAUSE-5 instrument: EVAL_PROTOCOL forbids ground contact above the
    # knee — never enforced by any harness until now (the 35.5% asterisk).
    # Proxy without new sensors: running min height of every above-knee
    # link's center over the episode; a link skimming the floor reads a few
    # cm. foot/ankle/knee/shin/calf links are exempt. Both a 0.03 and 0.05
    # bar are reported with the min-z quantiles (filter law: print the
    # distribution every cut is made on).
    body_names = list(raw.robot.data.body_names)
    # clause-5 wording is "has not fallen (no ground contact by any link
    # above the knee)". The FALLEN-proxy set excludes the wrist/hand chain:
    # palms work at floor level in every legitimate floor grasp (certified
    # pregrasp poses put them there by construction), so counting them
    # would make legal pickups unearnable. Hand-floor proximity is reported
    # separately so the letter-strict reading stays measurable.
    _C5_ALLOWED = ("foot", "ankle", "knee", "shin", "calf",
                   "wrist", "hand", "palm", "thumb", "index", "middle",
                   "ring", "little", "pinky")
    c5_ids = [i for i, nm in enumerate(body_names)
              if not any(a in nm for a in _C5_ALLOWED)]
    hand_ids = [i for i, nm in enumerate(body_names)
                if any(a in nm for a in ("wrist", "hand", "palm"))]
    print(f"CLAUSE5 links checked ({len(c5_ids)}/{len(body_names)}): "
          + ",".join(body_names[i] for i in c5_ids), flush=True)
    print(f"CLAUSE5 hand-chain (proximity-only): "
          + ",".join(body_names[i] for i in hand_ids), flush=True)
    hand_min = torch.full((n_envs,), 10.0, device=dev)
    hand_success_minz = []
    c5_min = torch.full((n_envs,), 10.0, device=dev)
    c5_success_minz = []   # per strict-success episode: min above-knee z
    c5_all_minz = []       # per finished episode: min above-knee z
    strict_c5_03 = 0       # strict AND clean at 0.03 m
    strict_c5_05 = 0       # strict AND clean at 0.05 m
    posture_pz, posture_up = [], []   # pelvis z / uprightness during strict
    pg_up = []                        # uprightness during FLOOR grasps
    # WAISTFOLD detector (WAISTFOLD_AUTOPSY_2026-07-30): the crab-sprawl
    # thigh-pin passed every root-link instrument. Per strict episode:
    # FOLD = min torso-link up-z < 0.60; PLANT = a hand link below 0.05m
    # while >0.25m from the object for >=10 strict steps. STRICT_LEGAL is
    # the strict count with neither flag — quote BOTH numbers, always.
    _wf_torso = [i for i, nm in enumerate(body_names) if "torso" in nm][0]
    _wf_hands = [i for i, nm in enumerate(body_names)
                 if any(a in nm for a in ("wrist", "hand", "palm", "thumb",
                                          "index", "middle", "ring",
                                          "little", "pinky"))]
    wf_min_up = torch.full((n_envs,), 10.0, device=dev)
    wf_plant_ct = torch.zeros(n_envs, device=dev)
    wf_fold_n, wf_plant_n, strict_legal = 0, 0, 0
    # PM amendment (lane X sign-off): objleg = min object-to-leg-link
    # distance over strict frames. MEASUREMENT ONLY — no gate/reward term
    # (no two-sided certificate). Watches for the upright-pin migration:
    # STRICT_LEGAL rising while objleg sits under ~0.10 is the tell.
    _wf_legs = [i for i, nm in enumerate(body_names)
                if ("hip" in nm or "knee" in nm)]
    wf_min_ol = torch.full((n_envs,), 10.0, device=dev)
    wf_strict_objleg = []
    # APPROACH SPEED (PM instrument, 2026-07-31): the walk economics hinge
    # on gait speed — 127 vs 507 steps to traverse is the difference between
    # the destination sitting inside the gamma=0.998 horizon and on its
    # boundary (tools/cert_walk_activation.py). Printed at every read so it
    # is visible at gate time, not reconstructed at post-mortem (the lesson
    # pelvis_z taught expensively). Closing speed is measured only while the
    # robot is genuinely far (>0.5 m), so crouch/grasp motion never pollutes it.
    _appr_prev = None
    appr_speeds = []

    obs, _ = env.get_observations(), None
    with torch.inference_mode():
        while done_count < args.episodes and simulation_app.is_running():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            ep_len += 1.0
            ep_success |= raw.hold_counter >= 50
            palm = raw.robot.data.body_pos_w[:, raw.palm_id]
            objw = raw.obj.data.root_pos_w
            objz = objw[:, 2] - raw.scene.env_origins[:, 2]
            nf, th = raw._finger_contacts()
            _base_d = (raw.robot.data.root_pos_w[:, :2]
                       - objw[:, :2]).norm(dim=-1)
            if _appr_prev is not None and len(appr_speeds) < 40000:
                _far = _base_d > 0.5
                _closed = (_appr_prev - _base_d) * 50.0   # m/s at 50 Hz
                _m = _far & (_closed.abs() < 3.0)         # reject respawns
                if _m.any():
                    appr_speeds += _closed[_m].tolist()
            _appr_prev = _base_d.clone()
            got_near |= (palm - objw).norm(dim=-1) < 0.12
            got_contact |= nf >= 1
            got_grasp |= (nf >= 3) & th
            got_lift |= objz > 0.22
            max_objz = torch.maximum(max_objz, objz)
            ejected |= (objz > 0.30) & (nf == 0)
            held_now = (objz > 0.22) & (nf >= 3) & th
            broke = (~held_now) & (cur_streak > 0)
            if broke.any():
                bi = broke.nonzero(as_tuple=True)[0]
                loss_speeds += raw.obj.data.root_lin_vel_w[bi].norm(dim=-1).tolist()
                frags[bi] += 1
            cur_streak = torch.where(held_now, cur_streak + 1,
                                     torch.zeros_like(cur_streak))
            max_streak = torch.maximum(max_streak, cur_streak)
            tot_held += held_now.float()
            strict_now = (objz > 0.40) & (nf >= 3) & th
            strict_streak = torch.where(strict_now, strict_streak + 1,
                                        torch.zeros_like(strict_streak))
            strict_success |= strict_streak >= 150
            # posture during FLOOR grasps (grasped, not yet lifted) — the
            # learned grasp posture's up-distribution calibrates the gate's
            # uprightness threshold (streak-guillotine fix: the threshold
            # must sit interior to this distribution, not inside it)
            g_now = (nf >= 3) & th & (objz <= 0.22)
            if g_now.any() and len(pg_up) < 20000:
                _q_g = raw.robot.data.root_quat_w[g_now]
                pg_up += (1.0 - 2.0 * (_q_g[:, 1] ** 2
                                       + _q_g[:, 2] ** 2)).tolist()
            if strict_now.any():
                _twq = raw.robot.data.body_quat_w[:, _wf_torso]
                _t_up = 1.0 - 2.0 * (_twq[:, 1] ** 2 + _twq[:, 2] ** 2)
                wf_min_up = torch.where(strict_now,
                                        torch.minimum(wf_min_up, _t_up),
                                        wf_min_up)
                _hp = raw.robot.data.body_pos_w[:, _wf_hands]
                _hp_low = (_hp[..., 2]
                           - raw.scene.env_origins[:, 2:3]) < 0.05
                _hp_far = (_hp - raw.obj.data.root_pos_w[:, None, :]
                           ).norm(dim=-1) > 0.25
                wf_plant_ct += ((_hp_low & _hp_far).any(dim=1)
                                & strict_now).float()
                _ol = (raw.robot.data.body_pos_w[:, _wf_legs]
                       - raw.obj.data.root_pos_w[:, None, :]
                       ).norm(dim=-1).min(dim=1).values
                wf_min_ol = torch.where(strict_now,
                                        torch.minimum(wf_min_ol, _ol),
                                        wf_min_ol)
            # posture stats DURING strict streaks — informs the 9S-d
            # upright-gated success predicate with data, not guesses
            if strict_now.any() and len(posture_pz) < 20000:
                _pzq2 = raw.robot.data.root_quat_w[strict_now]
                posture_up += (1.0 - 2.0 * (_pzq2[:, 1] ** 2
                                            + _pzq2[:, 2] ** 2)).tolist()
                posture_pz += (raw.robot.data.root_pos_w[strict_now, 2]
                               - raw.scene.env_origins[strict_now, 2]).tolist()
            finished = dones.bool()
            if finished.any():
                idx = finished.nonzero(as_tuple=False).squeeze(-1)
                done_count += len(idx)
                success_count += int(ep_success[idx].sum())
                for i in idx.tolist():
                    if ep_success[i]:
                        tax["success"] += 1
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
                        # terminal forensics for the dominant failure bucket:
                        # last_pz/last_up hold the PRE-terminal state (post-
                        # step reads of a done env are the respawn — reset-
                        # boundary rule), ep_len our own pre-reset counter
                        nn_lens.append(float(ep_len[i]))
                        nn_fell.append(bool((last_pz[i] < 0.15)
                                            | (last_up[i] < 0.3)))
                    eject_count += int(ejected[i])
                strict_count += int(strict_success[idx].sum())
                for i in idx.tolist():
                    if got_lift[i]:
                        hold_stats.append((float(max_streak[i]),
                                           float(tot_held[i]), float(frags[i])))
                    peak_zs.append(float(max_objz[i]))
                    # clause-5 accounting MUST precede the strict_success
                    # buffer reset below
                    c5_all_minz.append(float(c5_min[i]))
                    if strict_success[i]:
                        c5_success_minz.append(float(c5_min[i]))
                        hand_success_minz.append(float(hand_min[i]))
                        strict_c5_03 += int(c5_min[i] > 0.03)
                        strict_c5_05 += int(c5_min[i] > 0.05)
                        _fold = bool(wf_min_up[i] < 0.60)
                        _plant = bool(wf_plant_ct[i] >= 10)
                        wf_fold_n += int(_fold)
                        wf_plant_n += int(_plant)
                        strict_legal += int(not (_fold or _plant))
                        wf_strict_objleg.append(float(wf_min_ol[i]))
                for buf in (ep_success, got_near, got_contact, got_grasp,
                            got_lift, ejected, strict_success):
                    buf[idx] = False
                max_objz[idx] = 0.0
                for buf in (cur_streak, max_streak, tot_held, frags, strict_streak):
                    buf[idx] = 0.0
                ep_len[idx] = 0.0
                c5_min[idx] = 10.0
                hand_min[idx] = 10.0
                wf_min_up[idx] = 10.0
                wf_plant_ct[idx] = 0.0
                wf_min_ol[idx] = 10.0
            # refresh pre-terminal snapshots AFTER finished-processing: for
            # done envs these now hold the respawn state, which is correct
            # for the NEW episode's future termination read
            _pzq = raw.robot.data.root_quat_w
            last_up = 1.0 - 2.0 * (_pzq[:, 1] ** 2 + _pzq[:, 2] ** 2)
            last_pz = (raw.robot.data.root_pos_w[:, 2]
                       - raw.scene.env_origins[:, 2])
            c5_min = torch.minimum(
                c5_min,
                (raw.robot.data.body_pos_w[:, c5_ids, 2]
                 - raw.scene.env_origins[:, 2:3]).min(dim=1).values)
            hand_min = torch.minimum(
                hand_min,
                (raw.robot.data.body_pos_w[:, hand_ids, 2]
                 - raw.scene.env_origins[:, 2:3]).min(dim=1).values)

    rate = success_count / max(1, done_count)
    print(f"EVAL_RESULT episodes={done_count} successes={success_count} "
          f"success_rate={rate:.3f}")
    print("TAXONOMY " + " ".join(f"{k}={v}" for k, v in tax.items())
          + f" ejected={eject_count}")
    print(f"STRICT_RESULT successes={strict_count}/{done_count} "
          f"rate={strict_count / max(1, done_count):.3f} "
          f"(0.40m + 3s continuous hold)")
    print(f"STRICT_LEGAL {strict_legal}/{strict_count} "
          f"(fold={wf_fold_n} plant={wf_plant_n}; strict-by-metric minus "
          f"waistfold flags — a strict number may NEVER be quoted without "
          f"this one)")
    if appr_speeds:
        _as = torch.tensor(appr_speeds)
        _pos = _as[_as > 0.02]
        print(f"APPROACH_SPEED n={len(appr_speeds)} closing m/s while >0.5m "
              f"p25/p50/p75={_as.quantile(.25):+.3f}/{_as.quantile(.5):+.3f}/"
              f"{_as.quantile(.75):+.3f} | frac_advancing="
              f"{len(_pos)/len(_as):.3f} | median_when_advancing="
              f"{(_pos.median() if len(_pos) else float('nan')):.3f} "
              f"(walk economics need >=0.25 m/s to keep the destination "
              f"inside the gamma=0.998 horizon)")
    if wf_strict_objleg:
        _olt = torch.tensor(wf_strict_objleg)
        print(f"STRICT_OBJLEG n={len(wf_strict_objleg)} min obj-to-leg-link "
              f"dist p10/50/90={_olt.quantile(.1):.3f}/"
              f"{_olt.quantile(.5):.3f}/{_olt.quantile(.9):.3f} "
              f"(migration tell: STRICT_LEGAL up while this sits <0.10)")
    if nn_lens:
        nl = torch.tensor(nn_lens)
        qn = lambda p: float(nl.quantile(p))
        fell_frac = sum(nn_fell) / len(nn_fell)
        print(f"NEARFAIL n={len(nl)} len_p25/50/75="
              f"{qn(.25):.0f}/{qn(.5):.0f}/{qn(.75):.0f} steps "
              f"fell_frac={fell_frac:.2f} "
              f"(short+fell=stand collapse; long+upright=frozen standing)")
    if c5_all_minz:
        ca = torch.tensor(c5_all_minz)
        qa = lambda p: float(ca.quantile(p))
        print(f"CLAUSE5_ALL n={len(ca)} above-knee minz p10/25/50/75="
              f"{qa(.10):.3f}/{qa(.25):.3f}/{qa(.50):.3f}/{qa(.75):.3f}")
    if c5_success_minz:
        cs = torch.tensor(c5_success_minz)
        qs = lambda p: float(cs.quantile(p))
        print(f"CLAUSE5_STRICT n={len(cs)} minz p10/25/50/75="
              f"{qs(.10):.3f}/{qs(.25):.3f}/{qs(.50):.3f}/{qs(.75):.3f} | "
              f"STRICT_C5_CLEAN @0.03={strict_c5_03}/{done_count} "
              f"@0.05={strict_c5_05}/{done_count}")
        if hand_success_minz:
            hs2 = torch.tensor(hand_success_minz)
            qh = lambda p: float(hs2.quantile(p))
            print(f"CLAUSE5_HANDPROX strict-episode hand minz p10/50/90="
                  f"{qh(.10):.3f}/{qh(.50):.3f}/{qh(.90):.3f} "
                  f"(proximity only — letter-strict reading, not enforced)")
    else:
        print(f"CLAUSE5_STRICT n=0 | STRICT_C5_CLEAN @0.03=0/{done_count} "
              f"@0.05=0/{done_count}")
    if pg_up:
        pgu = torch.tensor(pg_up)
        qg = lambda p: float(pgu.quantile(p))
        print(f"POSTURE_GRASP n={len(pgu)} up p5/10/25/50/75="
              f"{qg(.05):.2f}/{qg(.10):.2f}/{qg(.25):.2f}/{qg(.50):.2f}/"
              f"{qg(.75):.2f}", flush=True)
    if posture_pz:
        pp = torch.tensor(posture_pz); pu = torch.tensor(posture_up)
        qp = lambda t, p: float(t.quantile(p))
        print(f"POSTURE_STRICT n={len(pp)} pelvis_z p10/25/50/75="
              f"{qp(pp,.10):.3f}/{qp(pp,.25):.3f}/{qp(pp,.50):.3f}/"
              f"{qp(pp,.75):.3f} up p10/25/50/75="
              f"{qp(pu,.10):.2f}/{qp(pu,.25):.2f}/{qp(pu,.50):.2f}/"
              f"{qp(pu,.75):.2f}")
    if peak_zs:
        pz = torch.tensor(peak_zs)
        q0 = lambda t, p: float(t.float().quantile(p))
        print(f"PEAK_Z n={len(pz)} p25/50/75/95="
              f"{q0(pz,.25):.3f}/{q0(pz,.5):.3f}/{q0(pz,.75):.3f}/{q0(pz,.95):.3f} "
              f"frac>=0.40={float((pz>=0.40).float().mean()):.3f} "
              f"frac>=0.42={float((pz>=0.42).float().mean()):.3f}")
    if hold_stats:
        hs = torch.tensor(hold_stats)  # (N, 3): max_streak, tot_held, frags
        ms, tot, fr = hs[:, 0], hs[:, 1], hs[:, 2]
        # flicker signature: tot >> max_streak with many fragments
        q = lambda t, p: float(t.float().quantile(p))
        # a "carry" = >=5 consecutive held frames (0.1s) — separates real
        # lifts from throws that merely crossed the height threshold
        carried = int((ms >= 5).sum())
        print(f"CARRY n_lift_reached={len(hs)} carried={carried}")
        print(f"HOLD_DIAG n={len(hs)} "
              f"max_streak p25/50/75={q(ms,.25):.0f}/{q(ms,.5):.0f}/{q(ms,.75):.0f} "
              f"tot_held p25/50/75={q(tot,.25):.0f}/{q(tot,.5):.0f}/{q(tot,.75):.0f} "
              f"frags p50={q(fr,.5):.0f} "
              f"frac(tot>=50 but max<50)={float(((tot>=50)&(ms<50)).float().mean()):.3f}")
        if loss_speeds:
            ls = torch.tensor(loss_speeds)
            print(f"LOSS_SPEED n={len(ls)} p25/50/75="
                  f"{q(ls,.25):.2f}/{q(ls,.5):.2f}/{q(ls,.75):.2f} m/s "
                  f"frac>0.5={float((ls>0.5).float().mean()):.3f}")


if __name__ == "__main__":
    import os, sys
    main()
    sys.stdout.flush()
    os._exit(0)
