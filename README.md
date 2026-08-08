# groundwork

Whole-body RL for a **Unitree G1 with Inspire five-finger hands** picking objects off the floor in Isaac Lab. Trained from scratch — no demonstrations, no teleoperation, no motion capture, no pretrained weights. Motion priors are computed by a trajectory optimizer in this repo.

Includes the training environment, the whole-body IK and contact-force tooling that generates the priors, ~30 diagnostic probes, a pre-registered evaluation protocol with anti-cheat assertions, and the checkpoints the numbers were measured on.

![Floor grasp, five-finger friction grip](media/result_lunge_hold_76pct.gif)

**76.3% legal floor grasp** from near starts (229/300), reproduced in a second, stricter environment at 75.3%. Object is a cylinder (r=45 mm, h=15 cm, 0.5 kg); success is lift to 0.40 m held 3+ continuous seconds, with ≥3 finger links plus thumb opposition in engine-verified contact, object touching hand links only.

| environment | strict | legal | share |
|---|---|---|---|
| official eval | 237/300 | 229 | **76.3%** |
| stricter (extra gate terms) | 233/300 | 226 | 75.3% |

Near starts only. The policy holds in a deep splayed lunge and does not stand. **0% on the full protocol** (1–4 m spawns) — the walk leg was trained but never chained in. Sibling checkpoints 200 iterations apart span 6%–88%; see [Non-stationarity](#non-stationarity).

## What's here

**`pod/g1_grasp/grasp_env.py`** (2.2k lines) — the training environment. Delta action space over 35 actuated DoF, contact-based grasp predicate read from the PhysX contact report, staged reset-state banks, posture and legality gating, per-episode failure taxonomy binned by deepest stage reached (never-near → touched → gripped → lifted → held).

**`tools/wbik.py`, `tools/statics.py`** — 35-DoF whole-body IK with foot-contact, support-polygon and self-collision constraints, then a friction-cone contact-force QP against spec-sheet torque limits. CPU, runs in minutes, no GPU. Both carry known-answer controls that must pass before any verdict is believed.

**`tools/make_ref_traj.py`** — the trajectory optimizer that produces the crouch/lift/stand reference banks in `refs/`.

**`pod/g1_grasp/probe_*.py`** — ~30 probes, each answering one falsifiable question: reachability, lift authority, descent authority, posture-gate exploitability, bank validity, contact-sensor ground truth. Several killed a hypothesis before it cost a training run.

**`tools/cert_*.py`** — activation certificates. A reward term ships only with proof it pays nonzero on the *observed* state distribution and that acting out-earns doing nothing.

**`docs/EVAL_PROTOCOL.md`** — success criteria fixed before the runs. Six simultaneous conditions. Anti-cheat asserted every step: no fixed/D6 joints or attachments between hand and object at any time, no inflated fingertip colliders, torques clamped to spec, no non-physical state writes after t=0, physics config dumped into every report.

**`banked/`** — the two checkpoints the results table was measured on.

## Method

| | |
|---|---|
| Simulator | Isaac Lab 2.3 / PhysX 5, 2048 parallel envs, dt 1/200, decimation 4 (50 Hz control) |
| Policy | RSL-RL PPO, actor/critic MLP [512, 256, 128], γ 0.998, λ 0.95, clip 0.2, entropy 0.0 |
| Actions | delta joint targets, 0.06 rad/step (3 rad/s), 0.12 for the finger chain |
| Observations | proprioception + object pose/velocity + goal, 205 dims, empirically normalized |
| Episode | 12 s, randomized object geometry/mass/friction/pose, ±10% domain randomization |

Physics is corrected from vendor defaults: the G1 asset ships with self-collision off, gravity disabled on links, and the base bolted in space. Every number here runs with self-collision on, full gravity, and a free base. Freeing the base required moving the articulation root to the pelvis and stripping the finger mimic-joints from the USD.

Contact sensing was validated against scripted ground truth before any learned number was trusted — a PhysX contact sensor silently returns an empty force matrix when one sensor prim matches multiple bodies, so every contact reward read zero while training ran anyway:

![Scripted grasp verification](media/03_scripted_grasp_verification.gif)

## Findings

**The action space was the binding constraint, not the reward.** At a 7% plateau the median continuous hold was 0 frames and median object speed at grip loss was 2.8 m/s — the policy threw objects rather than dropping them. Absolute joint targets at 50 Hz meant PPO exploration noise moved finger targets a large fraction of full travel each step, so no grasp survived training; the critic correctly priced holding at zero. Delta actions make the zero action a fixed point under noise. One training run: lift→hold 17% → 97%, median hold 0 → 11 s, grip-loss speed 2.8 → 0.00 m/s, throws 121 → 9 per 400 episodes. Throws are near zero from the first training hour, so the throwing was an artifact of the action space, not a habit to unlearn.

![Action-space signature](media/fig2_actionspace_signature.png)

**Root-link posture gates are exploitable.** A gate computing up-vector, pelvis height and min-z over a body set that excludes the hand chain admits a straight-legged sprawl: pelvis level, torso folded ~70° where no term measures it, one hand planted on the floor as a tripod where the min-z check cannot see it, object pinned between fingers and thigh with enough force to clear the grip qualifier. It scored 91.1% strict with posture telemetry reading "upright."

![Sprawl exploit: 91.1% strict, telemetry reading upright](media/hack_crab_sprawl_91pct.gif)

Telemetry built from a gate's own terms cannot detect an exploit of that gate. Gates must constrain the torso chain, include hand links in floor-contact checks, and require object clearance from non-hand links. Detectors here are certified both ways before use — flag known-bad footage ≥95%, pass known-good states ≤5%; [`probe_waistfold.py`](pod/g1_grasp/probe_waistfold.py) caught 60/60 sprawls with 1 false positive in 654.

**Reference trajectories need dynamic validation, not just static feasibility.** [`probe_descent_authority.py`](pod/g1_grasp/probe_descent_authority.py) drives the delta integrator along a reference by exact inversion — no policy, the ceiling of what any controller could achieve:

```
DESCENT  frames   0-150 : saturation 0.123   pelvis achieved 0.434   ref 0.560
RISE     frames 150-250 : saturation 0.122   pelvis achieved 0.233   ref 0.568
```

Pelvis 0.233 m is on the floor. Saturation at 12% means the rate limit is not binding — the trajectory is a sequence of statically-valid IK poses that cannot be executed under gravity and contact. Five interventions aimed at that motion all returned null. The bank ships in `refs/` so this is checkable:

```bash
GRASP_REF_BANK=refs/ref_traj_bank_gc2.npz python pod/g1_grasp/probe_descent_authority.py
```

The learned lunge may be the correct solution for this plant rather than a defect: a splayed straight-legged stance carries load through geometry rather than joint torque.

**Reset-boundary reads produce confident, precise, wrong numbers.** Episode resets happen inside `step()`: state buffers reflect the fresh reset pose, sensor buffers and done-semantics lag. Six instruments in this project died to it. A harvester captured 435 "grasp states" that were all spawn poses with no grasp — root height identical to 6 decimals, zero velocities — and three training runs died on that bank. Identical values across a harvested dataset are the fingerprint of capturing the reset writer, not the behavior. Guard every cross-buffer read with `(~dones) & (episode_length > ~10)`, detect falls from state rather than dones, and validate every bank with a zero-action restore probe ([`probe_cage.py`](pod/g1_grasp/probe_cage.py)).

**Config that reads as enabled can be inert.** `isaaclab_rl` supplies `actor_obs_normalization: {}`; rsl_rl's deprecation shim fills the new key only when the old one is `None`. `{}` is not `None` and `{}` is falsy, so observation normalization resolved to `nn.Identity` and 205 raw mixed-scale dims hit the first layer for an entire training epoch. Three sibling mechanisms were live-but-inert the same night, including a reward term paying zero on 95.8% of observed frames. Verify from `/proc/<pid>/environ` and the saved `params/agent.yaml`, never the launcher.

**Value clipping and value normalization are a pair.** `use_clipped_value_loss=True` with no value normalization over a 17-term critic at 30:1 scale imbalance. Removing the clip alone exploded action noise 0.50 → 0.92 across two flights — `entropy_coef > 0` on rsl_rl's state-independent learnable log-std is a permanent upward gradient, and ratio clipping bounds the rate of policy change, not the direction of entropy drift.

### Non-stationarity

Checkpoints 200 iterations apart span 6%–88% strict. A known-answer control — one frozen checkpoint evaluated three times — returns 6/9/4 per 100, so the evaluation is reproducible and the spread is genuine policy motion. Every gate in this project was a single-checkpoint read, which is how an unconverged lane passed two of them cleanly. Two-checkpoint gates with the trailing five reported is the fix.

An earlier free-standing result of 35.5% (142/400) at random object positions is **withdrawn**: measured before the detectors above existed, in the posture family they were built to catch, with the protocol's above-knee ground-contact clause unenforced by any harness.

Full writeup: [`docs/POSTMORTEM_failure_catalogue.md`](docs/POSTMORTEM_failure_catalogue.md). Supporting: [sprawl autopsy](docs/WAISTFOLD_AUTOPSY_2026-07-30.md), [independent review](docs/OUTSIDE_REVIEW_2026-07-29.md).

## Reproduce

```bash
uv venv --python 3.11 .venv && uv pip install mujoco numpy pytest pillow imageio

# hand/robot models — sparse clone, ~85 MB instead of the full Menagerie
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie assets/menagerie
git -C assets/menagerie sparse-checkout set shadow_hand unitree_g1 sharpa_wave

.venv/bin/python -m pytest tests/ -q          # local grasp lab, no GPU
```

Evaluation, on a GPU with Isaac Lab installed ([`pod/setup_pod.sh`](pod/setup_pod.sh)):

```bash
GRASP_NO_VIDEO=1 python pod/demo_grasp.py \
  --ckpt banked/GRASP_ARTIFACT_laneYb_25200.pt --episodes 300
```

[`demo_grasp.py`](pod/demo_grasp.py) prints STRICT and STRICT_LEGAL side by side — the letter-strict count, and the count with neither the waist-fold nor the hand-plant flag raised. `banked/GRASP_ARTIFACT_laneYb_25799.pt` is the sibling 599 iterations later; running both shows the spread directly.

## Layout

```
pod/g1_grasp/grasp_env.py    training environment, 2.2k lines
pod/g1_grasp/agents.py       PPO config
pod/g1_grasp/probe_*.py      ~30 probes, one falsifiable question each
pod/g1_loco/                 locomotion + kneel experts, handoff probes
tools/wbik.py, statics.py    35-DoF whole-body IK + friction-cone contact QP
tools/make_ref_traj.py       trajectory optimizer
tools/cert_*.py              reward-term activation certificates
src/grasp_synth/             local MuJoCo grasp lab, no GPU
banked/                      checkpoints the results were measured on
refs/                        reference banks
docs/                        protocol, postmortem, autopsies
media/                       footage, successes and failures both
tests/                       pytest smoke suite
```

Stages: [walking](media/01_walking.gif) · [walk-crouch-stand](media/02_walk_crouch_stand.gif) · [first pickups, 5%](media/04_first_floor_pickups_5pct.gif) · [pre-fix plateau, 7%](media/05_grasp_progress_7pct.gif) · [kneel-era](media/07_kneel_era_grasp.gif) · [descent + tipover](media/08_unified_descent_tipover.gif)
