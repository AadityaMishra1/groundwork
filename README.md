# groundwork

A Unitree G1 with Inspire five-finger hands learns to pick objects off the floor. RL in simulation. No demonstrations, no teleoperation, no motion capture. One rented GPU, under $250.

**35.5% success (142/400), strict protocol, robot balancing on its own legs.**

Success = lift to 0.40 m, hold 3 continuous seconds, with ≥3 finger links plus thumb in engine-verified contact. Random object position and orientation, deterministic policy, 400 episodes, failure taxonomy published. Protocol was fixed before the runs: [docs/EVAL_PROTOCOL.md](docs/EVAL_PROTOCOL.md), scoring in [pod/demo_grasp.py](pod/demo_grasp.py). Weights and training stack are not distributed.

![Strict-protocol floor pickup](media/06_strict_pickup.gif)

One uncut episode at a user-chosen position (earlier pelvis-fixed policy; success varies with placement — the random-position rate for that policy is 28.2%). More episodes: [media/06_strict_pickup.mp4](media/06_strict_pickup.mp4). Choose positions yourself: [pod/demo_pickup.py](pod/demo_pickup.py).

## Numbers

![Success progression](media/fig1_success_progression.png)

| Round | Strict success | Change |
|---|---|---|
| Baseline PPO | 0% | — |
| Instrumentation + curriculum repairs | ~7% | intermediate metrics up, success flat |
| Tactile observations | ~9% | touch→grip 54% → 64% |
| Action-space restructure | 28.2% | lift→hold 17% → 97% |
| Free-standing robot | 35.5% | same protocol, no fixed base |

Same protocol every row.

## The plateau at 7%, and its cause

Among "successful lifts," the median continuous hold was 0 frames. Median object speed at grip loss: 2.8 m/s. The policy threw objects. It did not drop them.

Cause: the policy sent absolute joint-position targets at 50 Hz. PPO exploration noise moved the finger targets a large fraction of full travel each step, so no grasp survived training. The value function learned that holds never persist and priced holding at zero. Grab-and-throw was the optimal policy. A reward cannot reinforce a behavior that exploration destroys before it pays.

Three checks before changing anything: scripted grasps with frozen targets hold indefinitely under the same physics; six reward and physics interventions did not move the plateau; rate-limiting a trained policy stopped the throwing immediately, along with its competence.

Fix: delta actions. The zero action holds the current targets — a fixed point under noise. Results from one training run: lift→hold 17% → 97%. Median hold 0 frames → 11+ s. Grip-loss speed 2.8 → 0.00 m/s. Throws 121 → 9 per 400 episodes.

![Action-space signature](media/fig2_actionspace_signature.png)
![Outcome shift](media/fig3_outcome_shift.png)

Throws are near zero from the first training hour: the throwing was an artifact of the action space, not a habit to unlearn.

## Context

Bench-mounted dexterous RL ([Dactyl](https://openai.com/index/learning-dexterity/), [DexPBT](https://arxiv.org/pdf/2210.13702)) gets reach and support from the mount; here the robot reaches the floor from its own squat. Humanoid ground pickup today ([CLONE](https://arxiv.org/abs/2506.08931), [HumanPlus/ResMimic](https://github.com/YanjieZe/awesome-humanoid-robot-learning)) trains on teleoperation or human motion data; this does not. Compute here is two-plus orders of magnitude below any of them. Tasks and protocols differ — the numbers do not compare directly.

We tried a demonstration pipeline: 1 usable demo per 10,000 attempts. Cut. A mined bank of pre-grasp states (not trajectories) seeded some mid-round resets; the final policies train from random initialization.

All results are simulation. If prior work covers this combination, open an issue and we will cite it.

## Instrumentation

Contact sensing was validated against scripted ground truth before any learned number was trusted (a PhysX contact sensor silently returns an empty force matrix if one sensor prim matches multiple bodies — every contact reward read zero while training ran anyway):

![Scripted grasp verification](media/03_scripted_grasp_verification.gif)

Earlier stages — round-1 walking on the stock handless G1, first full-task pickups (5%), and the pre-fix plateau (~7%: approach works, holds fail):

![Walking](media/01_walking.gif)
![Walk, crouch, stand](media/02_walk_crouch_stand.gif)
![First floor pickups](media/04_first_floor_pickups_5pct.gif)
![Pre-redesign grasping](media/05_grasp_progress_7pct.gif)

## Notes

- The robot asset ships from the vendor with self-collision off, gravity disabled on the links, and the base bolted in space — defaults that silently inflate results for anyone who trains on it unmodified. We turned all three on/off correctly: every number on this page runs with self-collision on, full gravity on every link, and a free base. Freeing the base required moving the articulation root to the pelvis and stripping the finger mimic-joints from the USD.
- Every eval bins episodes by deepest stage reached (never-near → touched → gripped → lifted → held). Interventions target the binding constraint.
- Difficulty curricula act on physics with a fixed anneal schedule, never on the reward. Evaluation always runs at full difficulty.
- Rigid-body simulation flatters grip stability: the 11-second holds will not transfer at that duration. The Inspire hand's physical fingertip force sensors are the planned bridge.

## Status

Done: strict floor grasp, 35.5% free-standing (one object geometry, easy end of protocol range). Commanded walking and kneeling (stand ↔ 0.32 m). In progress: making the kneel stable under a control handoff, for the composed chain. Next: object size/mass randomization, then one continuous episode — walk, kneel, grasp, lift, rise. Judged episodes contain no scripted motion.

Isaac Lab 2.3 / PhysX 5, RSL-RL PPO, 2048–4096 parallel environments, one rented GPU.
