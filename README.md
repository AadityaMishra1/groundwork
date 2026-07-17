# groundwork

RL-trained five-finger floor grasping on a low-cost humanoid (Unitree G1 + Inspire hands), in simulation with fully honest physics. No teleoperation, no human demonstrations, no physics shortcuts. Single rented GPU, <$150 total compute.

![First-ever RL floor pickups](media/04_first_floor_pickups_5pct.gif)

**Current headline result: 28.2% success under a strict pre-registered protocol** — random object spawn position and orientation, full gravity, deterministic policy, success only if the object is lifted to 0.40 m and held for 3 continuous seconds with PhysX-verified multi-finger contact (≥3 finger links + thumb opposition, hand-only). 400-episode evaluations, failure taxonomy published with every number. Protocol: [docs/EVAL_PROTOCOL.md](docs/EVAL_PROTOCOL.md). Scoring code: [pod/demo_grasp.py](pod/demo_grasp.py).

Trained weights and the training stack (environments, rewards, curricula, RL configs) are not distributed.

## Results

![Success progression](media/fig1_success_progression.png)

| Intervention round | Strict success | What moved |
|---|---|---|
| Baseline PPO | 0% | — |
| Instrumentation + curriculum fixes | ~7% (plateau, 4 rounds) | every intermediate metric except success |
| Tactile observations (per-finger contact + fingertip states) | ~9% | touch→grip conversion 54%→64% |
| **Action-space restructuring** | **28.2%** | **lift→hold conversion 17%→97%** |

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

Earlier pipeline stages on the same embodiment — velocity-tracking locomotion, and walk → commanded-height crouch → stand:

![Walking](media/01_walking.gif)

![Walk, crouch, stand](media/02_walk_crouch_stand.gif)

Grasp policy mid-progression (pre-redesign, ~7%): approach and contact are learned; holds still fail — this is the plateau the action-space analysis explains:

![Pre-redesign grasping](media/05_grasp_progress_7pct.gif)

## Engineering notes a robotics reviewer will care about

- **Silent-failure hunting**: round-one's null result traced to a PhysX contact-sensor configuration that silently returns empty force matrices when one sensor prim matches multiple bodies per environment — every contact reward and success check read zero while training "ran fine." All instruments are now validated against scripted ground truth before results are believed.
- **Honest physics, enforced**: the shipped robot asset ships with self-collision disabled, gravity disabled on robot links, and a fixed base. All three silently inflate results. We enable self-collision, gravity on every link, and (for locomotion) performed USD articulation surgery — relocating the articulation root from the baked world-joint to the pelvis and removing finger mimic-joint couplings that fail free-root articulation parsing — to make the manipulation-rigged asset a legitimate floating-base robot.
- **Funnel analysis over headline metrics**: every evaluation bins each episode by deepest stage reached (never-near / touched / gripped / lifted / held) with per-stage conversion rates, so interventions are chosen against the binding constraint, not vibes.
- **Curricula in physics, not rewards**: the two hardest skills (grasp formation, deep-kneel descent) both cracked via annealed physical difficulty on fixed schedules, evaluated only at full difficulty.

## Status

- Done: isolated floor grasp at 28.2% strict / 35.5% standard; grips essentially unbreakable once formed; walking + commanded-height locomotion on the hand-equipped, free-root, full-gravity robot.
- In progress: fully-neural composition — random robot spawn, random object spawn, one uncut episode of walk → kneel → grasp → lift → rise. No scripted motion in judged episodes.

Stack: NVIDIA Isaac Lab 2.3 / PhysX 5, RSL-RL PPO, 2048–4096 parallel environments on one rented consumer-class GPU.
