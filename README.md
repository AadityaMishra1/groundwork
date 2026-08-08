# groundwork

Unitree G1 + Inspire five-finger hands learning to pick objects off the floor. Isaac Lab, PPO from scratch — no demonstrations, no teleoperation, no motion capture.

![Floor grasp](media/result_lunge_hold_76pct.gif)

**76.3% legal floor grasp** from near starts (229/300), 75.3% in a stricter env. Lift to 0.40 m, held 3+ s, ≥3 finger links plus thumb opposition in engine-verified contact, object touching hand links only. Near starts only — 0% on the full 1–4 m protocol, the walk leg was never chained in.

## Run

```bash
uv venv --python 3.11 .venv && uv pip install mujoco numpy pytest pillow imageio
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie assets/menagerie
git -C assets/menagerie sparse-checkout set shadow_hand unitree_g1 sharpa_wave
.venv/bin/python -m pytest tests/ -q
```

Eval, needs Isaac Lab + GPU ([`pod/setup_pod.sh`](pod/setup_pod.sh)):

```bash
GRASP_NO_VIDEO=1 python pod/demo_grasp.py --ckpt banked/GRASP_ARTIFACT_laneYb_25200.pt --episodes 300
```

## Layout

```
pod/g1_grasp/grasp_env.py   env, 2.2k lines — delta actions over 35 DoF, grasp predicate
                            from the PhysX contact report, staged reset banks
pod/g1_grasp/agents.py      PPO — [512,256,128], γ .998, λ .95, clip .2, entropy 0
pod/g1_grasp/probe_*.py     ~30 probes, one falsifiable question each
pod/g1_loco/                walk + kneel experts, handoff probes
tools/wbik.py statics.py    35-DoF whole-body IK (foot contact, support polygon,
                            self-collision) + friction-cone contact-force QP
tools/make_ref_traj.py      trajectory optimizer for the crouch/lift references
tools/cert_*.py             reward-term activation certificates
src/grasp_synth/            MuJoCo grasp lab, CPU only
banked/                     checkpoints the numbers were measured on
docs/EVAL_PROTOCOL.md       success criteria + anti-cheat, fixed before the runs
```

2048 envs, 50 Hz control, 12 s episodes. Vendor asset defaults corrected: self-collision on, gravity on every link, free base.

## Notes

Six defects, each individually enough to stall training: absolute joint targets making grasps unlearnable under exploration noise, root-link posture gates being exploitable, a dynamically-infeasible reference bank, reset-boundary reads, an observation normalizer that resolved to `nn.Identity`, and value-clip/entropy coupling. Writeup: [`docs/POSTMORTEM_failure_catalogue.md`](docs/POSTMORTEM_failure_catalogue.md).

![Sprawl exploit](media/hack_crab_sprawl_91pct.gif)

91.1% strict, posture telemetry reading "upright" — hand planted on the floor, torso folded 70°, object pinned to the thigh. Every gate term was computed on the root link.

Sibling checkpoints 200 iterations apart span 6%–88%; a frozen checkpoint scored three times returns 6/9/4 per 100, so that spread is policy motion, not harness noise. An earlier 35.5% free-standing result is withdrawn — measured before the above detectors existed.
