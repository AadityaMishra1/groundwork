# groundwork

Teaching a low-cost humanoid (Unitree G1 + Inspire five-finger hands) to pick objects up off the floor with a real friction grasp — learned with RL in simulation. No teleoperation, no motion capture, no physics shortcuts: the object is a free rigid body, success requires PhysX-verified multi-finger contact, and every reported number comes from a pre-registered evaluation protocol designed to make cheating impossible ([docs/EVAL_PROTOCOL.md](docs/EVAL_PROTOCOL.md)).

**What this repo is**: the evaluation standard, the proof, and the supporting engineering. **What it deliberately is not**: a reproduction kit. Trained weights, training environments, reward formulations, observation/action-space specifics, curriculum schedules, and RL configurations are proprietary and not distributed.

## Results

![Success progression](media/fig1_success_progression.png)

Isolated grasp skill (kneeling start, object spawned at random position/orientation on the floor, deterministic policy, 400-episode evals):

| Milestone | Strict protocol (0.40 m lift + 3 s continuous verified hold) |
|---|---|
| Baseline PPO | 0% |
| + instrumentation & curriculum fixes | ~7% plateau across 4 training rounds |
| + tactile observations | ~9% |
| + action-space redesign | **28.2%** (35.5% at the 1 s training bar) |

The plateau-breaking finding: under an absolute joint-position action space, PPO's exploration noise shakes every formed grasp apart during training — the policy can never experience stable holding, so it rationally learns to grab-and-release. No reward function can fix a behavior that never occurs. Restructuring the action space so stillness is its fixed point under noise took lift→hold conversion from 17% to 97% in a single run: median hold went from 0 frames to 11+ seconds, and grip-loss object speed from 2.8 m/s (throws) to 0.00 (grips don't break).

The training telemetry makes the mechanism visible — throwing vanishes from the first hour (it was never a habit to untrain, it was the action space), and stable holds climb through a ceiling that had survived four rounds of reward engineering:

![Action-space signature](media/fig2_actionspace_signature.png)

![Outcome shift](media/fig3_outcome_shift.png)

In progress: fully-neural composition — random robot spawn, random object spawn, one uncut episode of walk → kneel → grasp → lift → rise. Locomotion + commanded-height descent on the hand-equipped robot is training now (descent depth annealed by curriculum; payload randomization on the grasping wrist).

## Videos

Chronological proof-of-progress in [`media/`](media/): walking, walk-crouch-stand, scripted grasp verification (physics ground truth for the instrumentation), first-ever RL floor pickups, and pre-redesign grasp progress. Current-generation eval videos pending (rendering blocked by a cloud-provider driver regression; numbers above are from headless physics, which is unaffected).

## Repo layout

- `pod/demo_grasp.py` — evaluation harness: strict-bar scoring, failure taxonomy (never-near / touched / gripped / lifted / held), per-episode hold-streak and grip-loss diagnostics (reference code — requires the proprietary training environment and weights to execute)
- `pod/g1_grasp/` — workspace-feasibility probes and checkpoint-surgery utilities (the training environment itself is withheld)
- `pod/g1_loco/` — USD asset surgery enabling a free-root hand-equipped G1, kneel-transition stability probe (the locomotion training environment is withheld)
- `pod/g1_crouch/` — earlier height-commanded crouch task (phase 1)
- `src/grasp_synth/` — MuJoCo grasp laboratory used for the phase-0 feasibility study (open-loop failure-mode characterization that motivated closed-loop RL)
- `docs/EVAL_PROTOCOL.md` — the pre-registered success/anti-cheat protocol all numbers are scored against

## Honest-physics guarantees

Self-collision enabled; gravity enabled on all robot links; free rigid-body object (no attachments ever); success requires ≥3 finger links + thumb opposition via PhysX contact reporting, hand-only contact; evaluations at full gravity with zero training assists; all asset modifications disclosed in-code.

Simulation: NVIDIA Isaac Lab 2.3 / PhysX, single rented GPU. Total compute spend to date: under $150.
