"""REF-PLAYBACK certificate: the reference bank must survive its own physics.

Before ANY training consumes /workspace/ref_traj_bank.npz through REF_TRACK
(RSI restores + tracking income), the bank has to prove that its states
restore cleanly and that its recorded targets, replayed OPEN LOOP with no
policy, re-earn the grasp under the training physics. This is the
certificates-before-training directive plus the reset-boundary law in probe
form: if a zero-intelligence playback cannot reproduce the demo from the
bank's own frames, the bank is teaching the policy a fiction and no reward
tuning downstream can be trusted (the pattern that already killed five
instruments here — validate banks with dumb probes BEFORE training).

Mechanics: the env runs with GRASP_REF_TRACK=1 GRASP_REF_FRAC=1.0
GRASP_FORCE_STAGE=3 (forced below, before g1_grasp is imported — the module
reads GRASP_REF_TRACK at import time to size the observation space), so
EVERY reset restores onto a uniform (ref, t0). Each step the probe reads the
bank's absolute targets at the env's own clock (t = ref_t0 +
episode_length_buf, the exact _get_rewards formula) and INVERTS the env's
delta integrator

    targets <- clamp(targets + a * delta_vec * gear, lo, hi)
    gear     = 1 - 0.6 * grasped_state          (read live from the env)
    =>  a    = (desired - targets) / (delta_vec * gear), clamped to [-1, 1]

Where the required jump exceeds delta_vec*gear the action SATURATES and the
playback approaches the recorded target at the slew limit — rate-limited,
never teleported, exactly the best a trained policy could do. Saturation
fraction is printed (expected mainly at the open->CLOSE_FING finger flip).

Fingers are not in the bank (body-joints-only by design): they are driven to
the env's default targets while phase <= 2 and to the CLOSE_FING grip
(raw.ref_grip_close, the mine_demos.py convention the env reuses) once
phase > 2 — the phase flip is the recorded grasp moment.

Run under the training env config, e.g.:

    GRASP_FULLBODY=1 GRASP_FREE=1 GRASP_LEG_STIFF=4.0 GRASP_EP_S=30 \
    GRASP_FLUNG_REF=robot GRASP_POSTURE_GATE=1 \
    GRASP_REF_BANK=/workspace/ref_traj_bank.npz \
    python -u g1_grasp/probe_ref_playback.py --headless

(GRASP_POSTURE_GATE=1 makes the clause-5 check live — without it c5
terminations cannot fire and that PASS conjunct is vacuous.)

PASS = lift040 (object peak z >= 0.40 m during playback) on >= 50% of the
refs actually played AND zero clause-5 terminations across the probe.
Per-ref verdicts (grasp = engine-verified >=3 fingers + thumb via the
filtered contact sensors, max object z, falls) print before the summary.
"""

import argparse
import os

# forced BEFORE g1_grasp import — see docstring
os.environ["GRASP_REF_TRACK"] = "1"
os.environ["GRASP_REF_FRAC"] = "1.0"
os.environ["GRASP_FORCE_STAGE"] = "3"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=0, help="0 = the bank's T")
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# GRASP_NO_VIDEO unset => record the playback (failure-forensics footage —
# the footage law applies to debugging too)
_NOVID = os.environ.get("GRASP_NO_VIDEO", "0") == "1"
args.video = not _NOVID
args.enable_cameras = not _NOVID
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # noqa: F401
import g1_grasp  # noqa: F401


def main():
    # the probe reads the SAME file the env loads — no policy anywhere.
    # allow_pickle: the bank is produced by this project's own harvester on
    # the pod (trusted path, same trust model as every torch.load bank
    # already in grasp_env.py); joint_names may be an object array.
    bank_path = os.environ.get("GRASP_REF_BANK", "/workspace/ref_traj_bank.npz")
    bank = np.load(bank_path, allow_pickle=True)
    R = int(bank["joint_pos"].shape[0])
    T = int(bank["joint_pos"].shape[1])
    steps = args.steps if args.steps > 0 else T

    env_cfg = parse_env_cfg("G1-Grasp-v0", num_envs=args.num_envs)
    env = gym.make("G1-Grasp-v0", cfg=env_cfg,
                   render_mode=None if _NOVID else "rgb_array")
    if not _NOVID:
        env = gym.wrappers.RecordVideo(
            env, video_folder="/workspace/demo_videos",
            step_trigger=lambda s: s == 0,
            # must finalize inside the run (the 800 lesson): capture less
            # than the probe's total steps
            video_length=int(os.environ.get("GRASP_VIDEO_LEN", "200")),
            name_prefix="playback_forensics", disable_logger=True)
    raw = env.unwrapped
    assert raw.ref_track, "GRASP_REF_TRACK=1 required"
    assert raw.ref_frac >= 1.0, "GRASP_REF_FRAC=1.0 required"
    assert raw.far_frac == 0.0, "GRASP_FAR_FRAC must be 0 (far spawns are not ref starts)"
    dev = raw.device
    n = raw.num_envs
    env.reset()

    tgt_bank = torch.as_tensor(bank["joint_targets"], dtype=torch.float32,
                               device=dev)
    phase = torch.as_tensor(bank["phase"], dtype=torch.float32, device=dev)
    # bank column -> action slot, through the env's own remap
    # (raw.ref_joint_ids holds the ROBOT joint index of each bank column).
    # Bank joints that are not action joints (legs in non-FULLBODY mode)
    # cannot be commanded — run GRASP_FULLBODY=1 for a whole-body cert.
    act_ids = [int(j) for j in raw.joint_ids]
    cols, slots = [], []
    for c, j in enumerate(raw.ref_joint_ids.tolist()):
        if j in act_ids:
            cols.append(c)
            slots.append(act_ids.index(j))
    cols_t = torch.tensor(cols, dtype=torch.long, device=dev)
    slots_t = torch.tensor(slots, dtype=torch.long, device=dev)
    grip_slots = torch.tensor(
        [act_ids.index(int(j)) for j in raw.ref_grip_ids.tolist()],
        dtype=torch.long, device=dev)
    print(f"PLAYBACK bank={bank_path} R={R} T={T} steps={steps} "
          f"mapped={len(cols)}/{tgt_bank.shape[2]} bank joints -> "
          f"{len(act_ids)} action joints", flush=True)

    # per-ref ledgers (an env can recycle onto a new ref mid-probe after a
    # termination; every sample is attributed to the ref live at that step)
    ref_grasp = torch.zeros(R, dtype=torch.bool, device=dev)
    ref_maxz = torch.zeros(R, device=dev)
    ref_falls = torch.zeros(R, dtype=torch.long, device=dev)
    ref_played = torch.zeros(R, dtype=torch.bool, device=dev)
    # path-mode ledger: closest palm-object approach during the P2 close
    # window — the honest full-path criterion (grasp itself is closed-loop
    # RL's job; open-loop playback certifies the PATH + ARRIVAL)
    ref_palmmin = torch.full((R,), 10.0, device=dev)
    gate_ok_sum, gate_n = 0.0, 0
    # term_counts get zeroed by the env's own [curriculum] print every 2400
    # common steps — accumulate deltas so the zeroing cannot eat events
    c5_total, succ_total = 0, 0
    prev_c5, prev_succ = raw.term_counts[5], raw.term_counts[4]
    resets_total = 0
    marks = ({0, 1, 2, 5, 10, 25, 50}
             | {steps // 4, steps // 2, (3 * steps) // 4, steps - 1})

    for step in range(steps):
        live = raw.ref_t0 >= 0
        rid = raw.ref_id.clone()
        ref_played[rid[live]] = True
        # desired ABSOLUTE targets at the env clock, then invert the delta
        # integrator (see docstring; gear read live so inversion is exact)
        # fallback-ladder knobs (fork spec F1/F2): SLOW repeats each frame
        # S times (bank untouched); LEAD commands targets from t+N frames
        # ahead, cancelling PD tracking lag under the x4 plant (the F0
        # failure mode: playbacks died at ~2/3 of descend, trailing pose
        # tipping at depth)
        _slow = int(os.environ.get("GRASP_REF_SLOW", "1"))
        _lead = int(os.environ.get("GRASP_REF_LEAD", "0"))
        ti = ((raw.ref_t0 + raw.episode_length_buf) // _slow
              + _lead).clamp(min=0, max=T - 1)
        des = raw.targets.clone()
        des[:, slots_t] = tgt_bank[rid, ti][:, cols_t]
        closed = (phase[rid, ti] > 2.0).unsqueeze(-1)
        des[:, grip_slots] = torch.where(
            closed, raw.ref_grip_close.unsqueeze(0),
            raw.default_targets[:, grip_slots])
        gear = (1.0 - 0.6 * raw.grasped_state.float()).unsqueeze(-1)
        a = ((des - raw.targets)
             / (raw.delta_vec.unsqueeze(0) * gear)).clamp(-1.0, 1.0)
        a = torch.where(live.unsqueeze(-1), a, torch.zeros_like(a))
        sat = float((a.abs() >= 0.999).float().mean())
        out = env.step(a)
        term = out[2]
        # a termination mid-playback is a fall/fling/c5 of the ref that was
        # being played BEFORE the reset (success-term is off by default)
        fell = term & ~raw.succ_done
        if bool(fell.any()):
            ref_falls.index_add_(
                0, rid[fell],
                torch.ones(int(fell.sum()), dtype=torch.long, device=dev))
        resets_total += int(term.sum())
        # post-step verdict data, attributed to the CURRENT ref (envs that
        # just reset have restored onto a fresh ref — state belongs to it)
        live2 = raw.ref_t0 >= 0
        rid2 = raw.ref_id
        nf, th = raw._finger_contacts()
        g = (nf >= 3) & th & live2
        ref_grasp[rid2[g]] = True
        # palm-object proximity during the P2 close window
        ti2 = ((raw.ref_t0 + raw.episode_length_buf)
               // _slow).clamp(min=0, max=T - 1)
        in_p2 = (phase[rid2, ti2] == 2.0) & live2
        if bool(in_p2.any()):
            palm_d2 = (raw.robot.data.body_pos_w[:, raw.palm_id]
                       - raw.obj.data.root_pos_w).norm(dim=-1)
            rows = in_p2.nonzero(as_tuple=False).squeeze(-1)
            ref_palmmin[rid2[rows]] = torch.minimum(
                ref_palmmin[rid2[rows]], palm_d2[rows])
        objz = raw.obj.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
        ref_maxz.scatter_reduce_(0, rid2[live2], objz[live2], reduce="amax")
        # posture-gate conjuncts computed directly (thresholds exist gate-on
        # or gate-off; c5 TERMINATIONS need GRASP_POSTURE_GATE=1)
        q = raw.robot.data.root_quat_w
        up = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        pelv = raw.robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
        minz = (raw.robot.data.body_pos_w[:, raw.c5_bodies, 2]
                - raw.scene.env_origins[:, 2:3]).min(dim=1).values
        ok = (up > raw.gate_up) & (pelv > raw.gate_pelv) & (minz > raw.gate_minz)
        gate_ok_sum += float(ok.float().mean())
        gate_n += 1
        c = raw.term_counts[5]
        c5_total += (c - prev_c5) if c >= prev_c5 else c
        prev_c5 = c
        c = raw.term_counts[4]
        succ_total += (c - prev_succ) if c >= prev_succ else c
        prev_succ = c
        if step in marks:
            print(f"PLAY t={step:4d} grasped={((nf >= 3) & th).float().mean():.2f} "
                  f"objz_p50/max={objz.median():.3f}/{objz.max():.3f} "
                  f"sat={sat:.2f} gate_ok={ok.float().mean():.2f} "
                  f"resets={resets_total} c5={c5_total}", flush=True)

    ok_ref = ref_grasp & (ref_maxz >= 0.40) & (ref_falls == 0)
    played = int(ref_played.sum())
    for r in range(R):
        if not bool(ref_played[r]):
            print(f"REF {r:3d} unplayed (uniform (ref,t0) sampling over "
                  f"{n} envs — bump --num_envs for coverage)", flush=True)
            continue
        v = "OK  " if bool(ok_ref[r]) else "FAIL"
        print(f"REF {r:3d} {v} grasp={int(ref_grasp[r])} "
              f"maxz={float(ref_maxz[r]):.3f} falls={int(ref_falls[r])}",
              flush=True)
    grasp_n = int((ref_grasp & ref_played).sum())
    lift_n = int(((ref_maxz >= 0.40) & ref_played).sum())
    ok_n = int((ok_ref & ref_played).sum())
    gp = gate_ok_sum / max(gate_n, 1)
    print(f"PLAYBACK_SUMMARY ok={ok_n}/{R} grasp={grasp_n} lift040={lift_n} "
          f"played={played} gatepass_frac={gp:.2f} c5={c5_total} "
          f"succ_term={succ_total} resets={resets_total}", flush=True)
    # PASS bar over the PLAYED refs: unplayed refs are uncertifiable, not
    # failed (see coverage note above); c5 must be zero everywhere
    ok_pass = played > 0 and lift_n * 2 >= played and c5_total == 0
    if os.environ.get("GRASP_REF_T0ZERO", "0") == "1":
        # PATH MODE (t0=0 full-path certificate): a ref passes when the
        # whole 250-step playback runs without a single fall AND the palm
        # arrives within 6 cm of the object during the close window.
        # Engine-grasp is NOT required — open-loop PD cannot nail cm-scale
        # contact (3.4 cm schedule error) and does not need to: the exp
        # kernel lets the learned, feedback-equipped policy deviate to
        # close while keeping tracking income.
        path_ok = ref_played & (ref_falls == 0) & (ref_palmmin <= 0.06)
        pn = int(path_ok.sum())
        pm = ref_palmmin[ref_played]
        print(f"PATHMODE ok={pn}/{played} palmmin_p10/50/90="
              f"{pm.quantile(.1):.3f}/{pm.median():.3f}/"
              f"{pm.quantile(.9):.3f}", flush=True)
        ok2 = pn >= (played + 1) // 2 and c5_total == 0
        print(f"PROBE_PATH_{'PASS' if ok2 else 'FAIL'} "
              f"(path_ok={pn} need>={(played + 1) // 2}, c5={c5_total})",
              flush=True)
    print(f"PROBE_PLAYBACK_{'PASS' if ok_pass else 'FAIL'} "
          f"(lift040={lift_n} need>={(played + 1) // 2} of {played} played, "
          f"c5={c5_total} need 0)", flush=True)


if __name__ == "__main__":
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
