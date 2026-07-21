# groundwork

RL-trained five-finger floor grasping on a low-cost humanoid (Unitree G1 + Inspire hands), in simulation. No teleoperation, no demonstration trajectories, no grasp welds, no proximity-faked success — every claimed grasp is PhysX-verified finger contact under unmodified object physics (per-stage physics status stated precisely in the engineering notes below). One rented GPU at a time, under $250 of rental total.

![Strict-protocol floor pickup](media/06_strict_pickup.gif)

One uncut episode at a user-chosen object placement: deep squat, five-finger grasp, lift to 0.40 m, 3-second verified hold — the strict protocol's full bar. (Placement chosen from the policy's competent region — success is placement-dependent; the 28.2% figure below is the honest global rate over randomized placements.) Same policy, same physics as every number below ([media/06_strict_pickup.mp4](media/06_strict_pickup.mp4) for the multi-episode cut; harness: [pod/demo_pickup.py](pod/demo_pickup.py) — pick the object's position, size, and mass, get a scored episode and video).

**Current headline result: 28.2% success under a strict pre-specified protocol** — random object spawn position and orientation, full gravity, deterministic policy, success only if the object is lifted to 0.40 m and held for 3 continuous seconds with PhysX-verified multi-finger contact (≥3 finger links + thumb opposition, hand-only). 400-episode evaluations, failure taxonomy published with every number. Protocol: [docs/EVAL_PROTOCOL.md](docs/EVAL_PROTOCOL.md). Scoring code: [pod/demo_grasp.py](pod/demo_grasp.py).

Trained weights and the training stack (environments, rewards, curricula, RL configs) are not distributed.

## Results

![Success progression](media/fig1_success_progression.png)

| Intervention round | Strict success | What moved |
|---|---|---|
| Baseline PPO | 0% | — |
| Instrumentation + curriculum fixes | ~7% (plateau, 4 rounds) | every intermediate metric except success |
| Tactile observations (per-finger contact + fingertip states) | ~9% | touch→grip conversion 54%→64% |
| **Action-space restructuring** | **28.2%** | **lift→hold conversion 17%→97%** |

## Where this sits in the field

Dexterous RL grasping results have historically ridden on three crutches this project deliberately goes without:

**1. Expensive, kinematically supported hands.** The classic results train arm-mounted hands at table height — the [Shadow Hand (~$100k)](https://www.technowize.com/openai-robot-hand-learns-dexterity-in-handling-objects/) and the [Allegro (~$15k, the research standard)](https://www.roboticscenter.ai/blog/best-robot-hands-dexterous-2025) — where a fixed industrial arm hands the policy a solved reach/support problem. Here the hand is an [Inspire (~$8k)](https://www.roboticscenter.ai/blog/best-robot-hands-dexterous-2025) on a Unitree G1 — a full humanoid platform that costs less than a Shadow Hand *alone* — whose short arms make floor-level objects barely reachable, so the reach/balance problem is part of the task, not the fixture.

**2. Human demonstrations.** Humanoids that pick objects off the ground today do it through whole-body **teleoperation** ([CLONE](https://arxiv.org/abs/2506.08931)) or **imitation of human motion data** ([HumanPlus](https://github.com/YanjieZe/awesome-humanoid-robot-learning), [ResMimic](https://github.com/YanjieZe/awesome-humanoid-robot-learning)) — a human supplies the strategy; the robot learns to reproduce it. This project uses **no demonstration trajectories and no imitation loss**: the grasp strategy in the video above was discovered by RL from physics alone. Full disclosure of what *was* used: a mined bank of physics-verified pre-grasp **states** (not trajectories) seeded a fraction of training resets as a reverse curriculum — and the final action-space-redesign run that produced the headline number trained from scratch. (We tried the demonstration route; our scripted-replay generator produced 1-in-10,000 usable demos and was abandoned.)

**3. Industrial compute.** OpenAI's Dactyl used [a ~400-server CPU cluster plus 32 V100s](https://openai.com/index/learning-dexterity/); DexPBT brought Allegro grasping down to [8 datacenter GPUs](https://arxiv.org/pdf/2210.13702). Total compute for everything on this page: **one rented GPU at a time, under $250 of rental total.**

What the field does better, stated plainly: those systems show **real-robot** results at high success rates; everything here is simulation (with deliberately un-cheated physics — self-collision on, gravity on every link, PhysX-verified contacts — chosen to make eventual transfer credible: the Inspire hand's real fingertip force sensors mean our tactile observations correspond to hardware that exists). The claim is not "better numbers." The claim is that **five-finger floor-level grasping on a low-cost humanoid via demonstration-free RL at hobby compute** is a corner of the capability space we have not seen staked — and 28.2% under a strict adversarial protocol is an honest baseline for it. (Corrections welcome: if prior work covers this combination, file an issue and we will cite it.)

## The finding that broke the plateau

Four training rounds of reward engineering each improved their targeted failure mode while end-to-end success stayed pinned at 7–10%. Per-episode instrumentation (longest continuous verified hold, object velocity at grip loss) localized the invariant: median true-hold streak among "successful lifts" was **0 frames**, and median object speed at grip loss was **2.8 m/s** — the policy wasn't dropping objects, it was throwing them.

Root cause: under an absolute joint-position action space at 50 Hz, PPO's exploration noise displaces finger targets by a large fraction of their travel every control step. Any grasp formed during training is destroyed by exploration itself, so the value function — correctly — learns that holds never persist, making grab-and-release the optimal policy. **No reward function can reinforce a behavior the exploration process makes impossible to experience.** We verified the mechanism from three independent directions (scripted frozen-target grasps hold indefinitely under identical physics; the success plateau was invariant to six reward/friction/mass interventions; rate-limiting a trained policy's actions eliminated throwing instantly but destroyed its competence) before changing the action space so that zero action is a fixed point under noise — holding becomes the easiest behavior to express rather than the hardest.

One training run later: lift→hold conversion 17% → 97%, median hold 0 frames → 11+ seconds, median grip-loss speed 2.8 → 0.00 m/s, throws 121 → 9 per 400 episodes.

![Action-space signature](media/fig2_actionspace_signature.png)

Left: object-throwing terminations are near-zero from the first training hour — throwing was an artifact of the action space, not a habit to untrain. Right: stable holds climb through the ceiling that survived every reward intervention, at full gravity.

![Outcome shift](media/fig3_outcome_shift.png)

## Proof of progression

Scripted grasp verification — before trusting any learned result, contact instrumentation was validated against open-loop physics ground truth (this closeup is a mined, physics-verified pre-grasp being executed by script; it is labeled as scripted):

![Scripted grasp verification](media/03_scripted_grasp_verification.gif)

Earlier pipeline stages (round 1, on the stock handless G1 — before unifying everything onto the hand-equipped embodiment):

![Walking](media/01_walking.gif)

![Walk, crouch, stand](media/02_walk_crouch_stand.gif)

First-ever full-task floor pickups (5% era, randomized spawns):

![First-ever RL floor pickups](media/04_first_floor_pickups_5pct.gif)

Grasp policy mid-progression (pre-redesign, ~7%): approach and contact are learned; holds still fail — this is the plateau the action-space analysis explains:

![Pre-redesign grasping](media/05_grasp_progress_7pct.gif)

## Engineering notes a robotics reviewer will care about

- **Silent-failure hunting**: round-one's null result traced to a PhysX contact-sensor configuration that silently returns empty force matrices when one sensor prim matches multiple bodies per environment — every contact reward and success check read zero while training "ran fine." All instruments are now validated against scripted ground truth before results are believed.
- **Honest physics, stated precisely**: the shipped robot asset comes with self-collision disabled, gravity disabled on robot links, and a fixed base — all three silently inflate results. Current status per stage: self-collision is enabled everywhere; the locomotion stage runs free-root with gravity on every link (via USD articulation surgery — relocating the articulation root from the baked world-joint to the pelvis and removing finger mimic-joint couplings that fail free-root parsing); **the grasp-stage numbers on this page still use a pelvis-fixed kneeling robot with the asset's link-gravity default** — the free-root, full-gravity grasp fine-tune is the current milestone, and no number will be promoted to the headline until it passes there. Object and world physics are unmodified throughout; there are no grasp welds and success requires PhysX-verified finger contact.
- **Funnel analysis over headline metrics**: every evaluation bins each episode by deepest stage reached (never-near / touched / gripped / lifted / held) with per-stage conversion rates, so interventions are chosen against the binding constraint, not vibes.
- **Curricula in physics, not rewards**: the two hardest skills (grasp formation, deep-kneel descent) both cracked via annealed physical difficulty on fixed schedules, evaluated only at full difficulty.

## Status

- Done: isolated floor grasp at 28.2% strict / 35.5% standard (pelvis-fixed kneel, single object geometry at the easy end of the protocol range); holds that survive 11+ seconds once formed — with the honest caveat that rigid-fingertip contact simulation flatters hold stability, one reason the eventual hardware story leans on the Inspire hand's real fingertip force sensors. Velocity-tracking walking on the hand-equipped free-root robot.
- In progress: commanded-height locomotion (stand ↔ deep kneel on command — two training runs collapsed to a single-height policy and were caught by behavioral evaluation; the third, with randomized-depth spawn curriculum, is running), then the free-root grasp fine-tune, object size/mass randomization per the protocol's full ranges, and fully-neural composition — one uncut episode of walk → kneel → grasp → lift → rise, no scripted motion in judged episodes.

Stack: NVIDIA Isaac Lab 2.3 / PhysX 5, RSL-RL PPO, 2048–4096 parallel environments on one rented consumer-class GPU.
