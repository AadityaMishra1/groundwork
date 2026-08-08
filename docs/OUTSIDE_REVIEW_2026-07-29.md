# Outside review — coordination memo (2026-07-29)

Written by a second session doing an independent outside review at the user's
request. Full docs + full code were read. This memo exists so the training
session and the review session don't collide: **the training session owns the
pods' process management and all currently-modified files** (grasp_env.py,
agents.py, train_grasp.py, demo_grasp.py and its new probes). The review
session added exactly two new files and touched nothing else:

- `pod/g1_grasp/probe_lift_authority.py` (new, additive)
- this memo

## 1. A probe is RUNNING on pod A right now

`probe_lift_authority.py` was launched on pod A (alongside FS-2; 48 envs,
~8 GB, ample headroom), logging to `/workspace/probe_lift_authority.log`.
No policy, no checkpoint — zero-intelligence scripted rise from bank_run5
cage restores (GRASP_FORCE_STAGE=1), delta-integrator inversion with live
gear, LEG_STIFF=4.0.

**What it discriminates** (the one question the pre-registered relaunch does
not answer): whether grasp_no_lift is H-POLICY (plant can lift; policy hasn't
learned it) or H-AUTHORITY (under ×4 stiffness + delta slew + the grip gear
that cuts leg slew 60% the moment the hand grasps + DCMotor effort clipping,
the action space cannot express the lift at all). Four arms in one boot:

- `ctrl` — zero action, known-answer control (cage restores must stay
  grasped and must NOT lift on their own; if dirty, all verdicts void)
- `lead1` — legs ramp to stand-blend targets, plain schedule
- `lead2` / `lead4` — same with 2×/4× target overshoot of the remaining
  pose error (the sustained feedforward a PD lift needs)

Readout table is in the probe docstring. Headlines:
- lead* lift040 ≈ 0 AND knee `appl/comp_ratio` << 1 → **H-AUTHORITY**: a
  relaunch that keeps this plant + action scheme is dead regardless of
  grace-anneal; the relaunch spec needs an authority change first.
- lead2/lead4 >> lead1 → the lift needs sustained overshoot, which is what
  delta actions express worst; the action scheme is the wall, not kp.
- lead1 lifts fine → H-POLICY; plant innocent, economy/training is the
  problem, relaunch as planned.

If cage QUALIFY count is low (<10), rerun with
`GRASP_BANK2=/workspace/chain_bank_grasp5.pt` (harvest_success cage family =
grasped & floor & slow, exactly the qualification condition).

### v1 result (2026-07-29, log: probe_lift_authority.log) — partial but major

v1's qualification gate (objz<0.12) was wrong for these bank states (they
restore with the object at ~0.23-0.26), so the ARM_ verdict counters are
VACUOUS ZEROS — ignore them. The per-step telemetry is valid and shows:

1. **Worst knee torque pins at exactly 139.0 N·m (the datasheet cap) in
   EVERY arm — including the zero-action control.** Merely holding the
   restored deep-fold grasp pose (pelvis ~0.16-0.19, the m2d-era family)
   drives the knee into hard effort clipping before any lift is attempted.
   The certified squat band (pelvis ~0.39) was certified at ~50% of that
   limit — the two posture families are on opposite sides of the authority
   wall, and the current lane's measured grasp posture (pelvis dipping to
   ~0.32-0.35 per its own audits) sits between them, closer to the wall.
2. Grasp fraction collapses ~0.5-0.6 -> ~0.17 during every scripted rise:
   the rise itself breaks the grip (the arm holds world-frame targets while
   the torso moves — a policy must actively servo the arm during the rise;
   RSI aloft states alone do not teach that coordination).

### v2 result (log: probe_lift_authority_v2.log) — the verdict

qual 31-34/48 per arm (grasped after settle). Headline numbers:

    ARM_CTRL  qual=31 lift040=0 lift022=18 fell=23 maxz_p90=0.389
    ARM_LEAD1 qual=31 lift040=0 lift022=13 fell=31 maxz_p90=0.388
    ARM_LEAD2 qual=34 lift040=0 lift022=17 fell=34 maxz_p90=0.390
    ARM_LEAD4 qual=31 lift040=0 lift022=18 fell=31 maxz_p90=0.385

Honest caveats first: the CTRL arm is DIRTY per the probe's own bar — the
restored states drift upward under their stored targets and 23/31 terminate
within 5 s untouched — so the clean H-POLICY/H-AUTHORITY discrimination was
NOT achieved. `fell` also inherits the env's merged tip/fall/drop counting
and does not separate robot-fell from object-dropped.

What IS established, and it is decisive enough to act on:

1. **lift040 = 0 across 127 qualified attempts under four schedules, and
   maxz_p90 bunches at 0.385-0.390 in every arm — a hard ceiling just under
   the 0.40 strict bar.** From the m2d-family floor-grasp posture the arm
   alone can raise the object to ~0.39 and no further; crossing 0.40
   requires the pelvis to rise.
2. **Every schedule that extended the legs killed the episode instead**
   (fell 31/31 in lead1, 34/34 in lead2, 31/31 in lead4 — 100% of qualified
   envs). Standing up from the deep fold while holding is the wall — for
   scripted controllers here, and evidently for the policy (0 fresh-spawn
   stricts in ten windows).
3. Mean knee/hip saturation during attempts is low (1-2%) but the worst
   joint pins at its 139 N·m cap (v1); the binding constraint is a
   combination of peak knee authority, balance through the rise, and grip
   survival (grasp frac 0.6 -> 0.17 during every rise), not blanket torque
   clipping.

**Implication for the relaunch decision:** the grace-anneal fixes streak
bookkeeping; none of the three mechanisms above are streak bookkeeping. A
relaunch whose spec does not address the rise specifically — grasp posture
pushed toward the certified shallow-squat band (pelvis ~0.39, 2x knee
headroom, object ceiling well above 0.40 from there), leg-slew/gear
exemption during grasped rises, and rise-segment training with live arm
servoing (aloft RSI alone starts AFTER the coordination it needs to teach)
— is spending the last budget on the wrong variable.

## 2. One amendment to the distillation branch spec

"Labels collected only near zero-yaw where the representations coincide" has
a coverage hole: the student never gets teacher supervision exactly where it
needs steering. The teacher's yaw-blindness is exploitable in the other
direction: for a student state at yaw θ, **rotate the world-frame quantities
by −θ before feeding the teacher**. A yaw-blind teacher cannot tell the
difference, and it then labels every state, not just the zero-yaw slice.
This removes the coverage restriction entirely and turns the defect into
data augmentation.

## 3. Gate immutability (user-endorsed)

The model_14400 gate as pre-registered (strict = 0 AND lifts+successes
< 10/100 AND fresh-family stricts < 100/window → one grace-anneal relaunch →
then the near lane demotes) **does not move after the eval starts**, whatever
the number is. This project's documented failure mode is not bad gates; it is
gates recalibrated the night they would have fired (Gate 2, 2026-07-26).

## 4. Findings from the full read not yet in any doc

- Aloft REF starts are exempt from drop-ET: `started_grasped` is set for
  bank2 grasp-family starts but not for phase>2 reference restores
  (grasp_env.py ~1506 vs the ref block), so an aloft start that drops the
  object grinds out the remaining episode instead of recycling — wasted
  experience in the highest-value start family. One-line fix at the next
  boundary.
- The references are statically certified only; the F0–F4 playback ladder
  failure is on the record. Under REF_W=0 the path is not load-bearing, but
  RSI from states on a dynamically-undemonstrated path can chain toward an
  unreachable island — which is what grasp_no_lift-dominated taxonomy looks
  like. The lift-authority probe is the direct test.
- prev_actions in the observation is one step staler than intended
  (emitted before the update in `_get_observations`); p_rate is computed
  correctly. Cosmetic now, worth fixing at a relaunch boundary, never mid-
  lineage.
