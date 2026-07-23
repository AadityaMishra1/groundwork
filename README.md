# groundwork

A $16k humanoid (Unitree G1, Inspire five-finger hands) learns to pick objects off the floor. Pure reinforcement learning in simulation: no teleoperation, no human demonstrations, no motion capture. One rented GPU. Under $250 total.

A grasp counts only if the physics engine verifies finger contact — three or more finger links plus thumb opposition. Nothing is welded, nothing is faked.

![Strict-protocol floor pickup](media/06_strict_pickup.gif)

One continuous episode: you pick where the object goes; the robot squats, grasps, lifts to 0.40 m, holds for 3 seconds. (This clip shows the earlier pelvis-fixed policy at a favorable position — success varies with placement. Random-position rate: 28.2%. Multi-episode video: [media/06_strict_pickup.mp4](media/06_strict_pickup.mp4). Try positions yourself: [pod/demo_pickup.py](pod/demo_pickup.py).)

## Result

**35.5% success (142/400) under a strict, pre-specified protocol, with the robot balancing on its own legs.** The earlier headline — 28.2% with the pelvis fixed in space — is now beaten by the free-standing version of the task, which is strictly harder.

The protocol, fixed before the runs ([docs/EVAL_PROTOCOL.md](docs/EVAL_PROTOCOL.md), scoring in [pod/demo_grasp.py](pod/demo_grasp.py)):

- Object spawns at a random position and orientation
- Deterministic policy, 400 episodes, full failure taxonomy published
- Success = lift to 0.40 m and hold 3 continuous seconds with a physics-verified five-finger grasp

We do not distribute trained weights or the training stack.

## Progression

![Success progression](media/fig1_success_progression.png)

| Round | Strict success | What changed |
|---|---|---|
| Baseline PPO | 0% | — |
| Instrumentation + curriculum repairs | ~7% | every intermediate metric improved; success didn't |
| Tactile observations | ~9% | touch-to-grip conversion 54% → 64% |
| Action-space restructure | 28.2% | lift-to-hold conversion 17% → 97% |
| **Free-standing robot** | **35.5%** | same protocol, robot on its own legs |

Same protocol in every row; the rows compare directly.

## The one finding worth reading this page for

Four rounds of reward engineering each fixed a target failure. Success stayed near 7% through all four. Then instrumentation isolated a strange fact: among "successful lifts," the median continuous hold was **0 frames**, and the median object speed at grip loss was **2.8 m/s**. The policy wasn't dropping objects. It was throwing them.

The cause was the action space. The policy sent absolute joint targets at 50 Hz, so PPO's exploration noise yanked the fingers a large fraction of full travel every step. No grasp could survive training. The value function learned the true lesson of that world — *holds never last* — and priced holding at zero. Grab-and-throw was optimal.

**A reward cannot reinforce a behavior that exploration destroys before it pays.**

We checked the mechanism three ways before touching the action space: scripted grasps with frozen targets held forever under identical physics; six reward and physics interventions all failed to move the plateau; rate-limiting a trained policy stopped the throwing instantly (and its competence with it).

The fix makes the zero action a fixed point — "do nothing" holds the current pose even under noise. Holding became the easiest behavior to express, and one training run later: lift-to-hold 17% → 97%, median hold 0 frames → 11+ seconds, grip-loss speed 2.8 → 0.00 m/s, throws 121 → 9 per 400 episodes.

![Action-space signature](media/fig2_actionspace_signature.png)

Throws vanish in the first training hour — the throwing was an artifact of the action space, not a habit to unlearn. Holds then climb through the ceiling that survived every reward intervention.

![Outcome shift](media/fig3_outcome_shift.png)

## Where this sits

Different tasks, different protocols — **do not compare the numbers across rows.** The table shows what each system needs.

| System | Platform | Hand cost | Task | Human data | Compute | Result |
|---|---|---|---|---|---|---|
| [Dactyl (OpenAI, 2018)](https://openai.com/index/learning-dexterity/) | Shadow Hand, fixed mount | ~$100k | in-hand cube, bench height | none | ~400 CPU servers + 32 V100s | real robot |
| [DexPBT (NVIDIA, 2023)](https://arxiv.org/pdf/2210.13702) | Allegro on arm | ~$15k | tabletop grasp/reorient | none | 8 datacenter GPUs | sim, high success |
| [Robust Dexterous Grasping (2025)](https://arxiv.org/abs/2504.05287) | arm-mounted hand | — | tabletop grasping | none | — | 97% sim, 94.6% real |
| [CLONE (2025)](https://arxiv.org/abs/2506.08931) | Unitree G1 | — | ground pickup | teleoperation | — | real demos |
| [HumanPlus](https://github.com/YanjieZe/awesome-humanoid-robot-learning) / [ResMimic](https://github.com/YanjieZe/awesome-humanoid-robot-learning) | full humanoid | — | loco-manipulation | human motion | — | real skills |
| **groundwork** | G1 + Inspire | ~$8k | **floor-level five-finger grasp** | **none** | **1 GPU, <$250** | 35.5% strict, sim |

Three things the table shows. Bench-mounted arms get reach and support for free; here the robot reaches the floor from its own squat. Humanoids that pick things off the ground today learn from teleoperation or motion capture; this one learns from physics alone (we tried a demonstration pipeline — 1 usable demo per 10,000 attempts — and cut it; a mined bank of pre-grasp *states*, not trajectories, seeded some mid-round resets, and the final policies train from random initialization). And the compute is two-plus orders of magnitude below every other row.

What the other rows do better: real robots. Everything on this page is simulation. If you know prior work covering this task combination, open an issue — we'll cite it.

## Proof it isn't faked

We validated the contact instrumentation against scripted ground truth before trusting any learned number. Below, a script executes a mined pre-grasp (labeled as scripted):

![Scripted grasp verification](media/03_scripted_grasp_verification.gif)

Round-1 walking on the stock handless G1, before all training moved to the hand-equipped robot:

![Walking](media/01_walking.gif)

![Walk, crouch, stand](media/02_walk_crouch_stand.gif)

First-ever full-task pickups (5% era) and the pre-fix plateau (~7% — approach works, holds fail):

![First floor pickups](media/04_first_floor_pickups_5pct.gif)

![Pre-redesign grasping](media/05_grasp_progress_7pct.gif)

## Engineering notes

- **Silent failures.** A PhysX contact sensor returns an empty force matrix if one sensor prim matches multiple bodies — silently. Every contact reward read zero while training ran anyway. Now every instrument is validated against scripted ground truth first.
- **Physics honesty, per stage.** The shipped asset has self-collision off, gravity off on links, and a bolted base — each silently inflates results. Now: self-collision on everywhere, free base with full gravity in locomotion *and* grasping (the 35.5% number). Freeing the base required USD surgery: articulation root moved to the pelvis, finger mimic-joints stripped.
- **Funnel analysis.** Every eval bins episodes by deepest stage reached (never-near → touched → gripped → lifted → held). Interventions target the binding constraint, not the loudest symptom.
- **Curricula in physics, not rewards.** The two hardest skills trained through annealed physical difficulty on a fixed schedule. Evaluation always runs at full difficulty.
- **Hold-duration caveat.** Rigid-body simulation flatters grip stability; the 11-second holds will not transfer at that duration. The Inspire hand's real fingertip force sensors are the planned bridge.

## Status

- Done: strict floor grasp at 35.5% free-standing (one object geometry, easy end of the protocol range). Commanded walking and kneeling (stand ↔ 0.32 m) on the hand-equipped robot.
- Now: hardening the walk→kneel→grasp handoff — the kneel must stay stable when leg control freezes for the grasp phase.
- Next: object size and mass randomization over the protocol ranges, then one continuous episode: walk, kneel, grasp, lift, rise. Judged episodes contain no scripted motion.

Stack: Isaac Lab 2.3 / PhysX 5, RSL-RL PPO, 2048–4096 parallel environments, one rented GPU.
