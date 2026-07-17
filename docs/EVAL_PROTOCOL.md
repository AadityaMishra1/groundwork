# Evaluation Protocol (anti-cheat)

This document defines success. It is written to survive due diligence: any
claimed success rate refers to this protocol, and every rule below exists to
close a known way of faking the result.

## Task

The robot spawns standing. One object spawns on the flat ground plane at a
randomized position. The policy must walk to the object, lower itself, grasp
the object with one hand, and lift it, ending in a stable stand or crouch with
the object held.

## Randomization (per episode, seeded)

| Variable | Range |
|---|---|
| Object spawn distance from robot | 1.0 – 4.0 m |
| Object spawn bearing | 0 – 360° |
| Object pose | standing or lying, random yaw |
| Object radius (cylinder/bottle category) | 25 – 45 mm |
| Object height | 10 – 25 cm |
| Object mass | 0.1 – 0.8 kg |
| Object–finger friction μ | 0.4 – 1.2 |
| Robot initial yaw | random |
| Ground friction | 0.6 – 1.1 |
| Observation noise / joint friction / motor strength | ±10 % domain randomization |

Evaluation seeds are disjoint from training seeds. ≥1000 episodes,
report mean success with 95% bootstrap CI and a failure taxonomy.

## Success (all conditions simultaneously, evaluated at episode end)

1. Object center of mass ≥ 0.40 m above the ground.
2. Held continuously for ≥ 3 s while the robot is quasi-static
   (base linear velocity < 0.25 m/s).
3. During the hold: the object contacts **only** links of the grasping hand
   (fingers/palm). No ground contact, no forearm/torso cradling.
4. ≥ 3 distinct finger links (thumb counts) plus, at grasp closure, thumb
   opposition present — verified from the PhysX contact report, not from
   joint angles.
5. Robot has not fallen (no ground contact by any link above the knee at any
   point in the episode).
6. Time limit: 30 s wall-clock sim time.

## Anti-cheat rules (enforced in the eval harness, asserted every step)

- **No attachment**: the eval harness asserts no fixed/D6 joints or PhysX
  attachments exist between hand and object at any time. Grasp force is
  friction + normal contact only.
- **No contact simplification on the hand**: Inspire hand collision meshes as
  shipped in Isaac Lab's tuned asset; no inflated fingertip colliders beyond
  the asset's contact offset defaults; solver params logged in the report.
- **No privileged actuation**: joint torques clamped to the G1/Inspire spec
  sheet limits; actuator model (PD gains, torque limits) logged.
- **No teleport/reset tricks**: object and robot poses are continuous;
  harness asserts no non-physical state writes after t=0.
- **Physics config disclosure**: dt, substeps, solver iterations, contact
  offsets, and friction combine mode are printed into every eval report.
- Policy input is proprioception + object pose/velocity + goal only
  (state-based milestone; input schema logged per run).

## Reported artifacts per eval

- `report.json` — per-episode outcome, seed, randomization draws, failure tag
- `summary.md` — success rate, CI, taxonomy table, physics config dump
- Highlight videos with contact points rendered (visible five-finger contact)
