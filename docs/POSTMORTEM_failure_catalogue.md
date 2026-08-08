# Six root causes in four weeks: an autopsy of a solo humanoid RL project

**Project:** one policy that walks to an object on the floor, crouches, grasps
it with five fingers under real friction contacts, lifts and holds — Unitree G1
+ Inspire hands, Isaac Lab / PhysX, state-based observations, trained with zero
human demonstration data.

**Outcome:** the chain never closed. One partial skill (76% floor-grasp from
near starts, in simulation, from a single hand-picked checkpoint of a lane that
never converged), one partial walk expert, and no policy that does the task
end-to-end. Roughly $400 of rented GPU time.

This document is the part worth keeping. Every defect below was real,
individually sufficient to stall training, and invisible to the metrics that
were being watched at the time. They are also, as far as I can tell,
underreported — each one cost days to rediscover, and none of them are the
failure modes that RL tutorials warn you about.

Written after the fact, from run logs, probe output, and footage. Where a
number appears, it came from an instrument, not from memory.

---

## 0. The meta-failure, stated first

Every specific bug below is downstream of one habit: **cheap falsification tests
were consistently deprioritized in favour of expensive hopeful ones.**

The single most valuable script in this repo is `probe_descent_authority.py`.
It runs no policy and no training. It drives the action integrator along the
reference trajectory by exact inversion and reports where the robot actually
ends up. It takes minutes, costs nothing, and it invalidated the entire
reference bank the project had been training against for weeks.

It was written *after* the money was gone.

The information needed to write it earlier was already on the record: project
notes stated the reference was "statically certified, never dynamically
demonstrated," and the F0–F4 open-loop playback ladder had already failed. The
conclusion was available and not drawn.

If you take one thing from this document: **when you have a hypothesis that
would invalidate weeks of work, test it first, precisely because it would.**

---

## 1. Reset-boundary reads

*The most recurrent bug class in the project. Six instances.*

In Isaac Lab — manager-based and direct alike — episode resets happen **inside**
`env.step()`. State buffers (root pose, joint positions, object pose) reflect
the freshly-written reset state immediately. Sensor buffers (contact forces) and
`done` semantics lag or conflate. Any instrument that reads across that boundary
measures a chimera: half post-reset, half pre-reset, internally inconsistent.

Instances found:

| # | instrument | what it reported | what was true |
|---|---|---|---|
| 1 | round-1 contact filter | zero contacts | silent buffer zeros |
| 2 | `eval_kneel` | 62/64 episodes "fell" | `fallen \|= dones` counted 20 s truncation as falling |
| 3 | `run_chain` fall detector | no falls | reads root state post-step = post-reset; a falling robot respawns before the check sees it |
| 4 | walk probes | robot not advancing | terminations teleport robots home — it measured resets, not walking |
| 5 | `harvest_success` | 435 harvested "cage grasps" | state buffers = fresh spawn pose, contact flags = stale pre-reset grasp. **All 435 were spawn poses with no grasp** (root z identically 0.320, zero velocities) |
| 6 | aloft reference restores | grip present at t=1 | measured object pose + *synthesized* grip → object free-fell in 0.2 s, 0/1920 passes |

Instance 5 is the expensive one: **three training runs died on the poisoned
bank** while plant and noise hypotheses were falsified one at a time.

Instance 6 is the subtle one, and generalizes: a restore is only valid if it
writes the **full state family that was measured together**. A grip is the
equilibrium under the PD targets that hold it — restoring a measured finger pose
without its measured squeeze preload produces a pose that looks grasped for one
frame and isn't. Mixed provenance in a restored state is the same defect as a
mixed-time read.

**Rules that fall out of this:**

- Any per-step harvester or evaluator must guard
  `(~dones) & (episode_length_buf > ~10)` before trusting any read that
  combines state and sensor buffers.
- Fall detection is state-based (pelvis z, up-vector). Never `dones`.
- **Validate every data bank before training on it**, two ways: statistical
  sanity (spread in root z, nonzero velocities) and a **zero-action restore
  probe** — spawn from the bank, command nothing, and check the claimed
  property is present at t=1–2.
- Identical or uniform values across a harvested dataset are the fingerprint of
  capturing the reset writer rather than the behaviour.

---

## 2. Live-but-inert mechanisms

*Four found in a single night. All four had passed review.*

A mechanism that exists, is believed active, and does nothing is worse than a
missing one — it consumes the attention budget that would have found the real
problem, and it makes every experiment that assumes it a null result for the
wrong reason.

1. **Rise reward.** Ramp onset at pelvis 0.43 m while the observed strict-hold
   pelvis distribution ran 0.374–0.434. Paid zero on **95.8%** of observed
   frames — mean payment 0.0020 against a 6.0 elevation ramp.

2. **Observation normalization (`GRASP_EMPNORM`).** Set since the run-6 era.
   Never active. `isaaclab_rl` supplies `actor_obs_normalization: {}`; rsl_rl's
   deprecation shim fills the new key only when the old one is `None`. `{}` is
   not `None`, and `{}` is falsy — so the normalizer silently resolved to
   `nn.Identity`. **205 raw mixed-scale dimensions went into the first layer for
   the entire whole-body epoch.** Leading suspect for both the checkpoint
   oscillation (§4) and the catastrophic forgetting (§5).

3. **Reference tracking weight (`GRASP_REF_W=0.0`) in all six lanes.** The
   reward for matching the project's own computed crouch/rise reference was
   switched off everywhere, while the bank sat there containing the full motion.

4. **A false certificate.** `probe_empnorm_freeze` passed — while guarding a
   component that was never instantiated.

**Common cause: every one was verified by reading configuration instead of
observing the running system.**

**Rule:** a flag, term, or component ships only with runtime proof it is live —
a boot-time assertion plus the printed identity of the *instantiated object*,
landing in the training log. Verify from `/proc/<pid>/environ` and the saved
`params/agent.yaml`, never from the launcher script.

For reward terms, that bar is not sufficient. A term must additionally prove:

- it pays **nonzero on the observed state distribution**, not on the
  distribution you imagined; and
- **acting out-earns doing nothing.**

The rise term eventually satisfied the first and failed the second. The policy
was correctly declining a bad deal.

---

## 3. Reward hacking, three generations

*Every generation passed the metrics of its era. All three were caught only by
watching video.*

| generation | metric said | video showed | detector built |
|---|---|---|---|
| crab sprawl | 87–91% strict | legs splayed, one hand planted on the floor as a tripod, torso folded ~70°, object pinned against own thigh | torso up-vector + hand-floor contact (`probe_waistfold.py`) |
| thigh pin | 50% legal | upright, hands free — object pressed against own thigh | object-to-leg-link distance (`STRICT_OBJLEG`) |
| hand carry | 76% legal | genuine hand-only carry, object clear of body | — (this is the banked result) |

Each defect was strictly narrower than its predecessor. That progression is what
progress looks like when your adversary is your own optimizer.

### The gate loophole, in detail

The crab sprawl is worth spelling out because the gate looked thorough. All
three of its terms were computed on the **root/pelvis link**:

1. up-vector from `root_quat_w`
2. pelvis height from `root_pos_w`
3. minimum z over a body set that **excluded the hand chain**

The learned exploit satisfied every one. Legs splayed straight, so the pelvis
stayed level at ~0.43 m (up-vector 0.85, height fine). Torso folded ~70° at the
waist — invisible, because **torso pitch was never measured**. One hand planted
on the floor as a tripod — invisible, because hand links were excluded from the
floor-contact check. Object pinned between fingers and its own thigh, with >3 N
finger force clearing the "secure grip" qualifier.

Result: 91.1% strict, 92/101 clean on the legality clause, and posture telemetry
reading **"upright"** — on a pose that is neither upright nor a carry.

**Rule:** telemetry derived from the gate's own terms cannot detect an exploit
of the gate. Any posture or legality gate must constrain the **torso chain**
(waist/torso pitch, or head height), include hand links in floor-contact checks,
and require object clearance from the robot's own non-hand links. Certify new
gate terms with a **scripted-exploit probe** — deliberately try to satisfy the
gate in a folded pose — before training against them.

**And above all: render and watch the footage before quoting any number.** Three
for three, video was the only thing that caught these.

---

## 4. A broken learning pathway, and the fix that made it worse

Two coupled defects in the PPO configuration, present since day one and never
questioned because they are defaults.

**The value pathway.** `use_clipped_value_loss=True` with **no value
normalization**, a single critic spanning **17 reward terms at ~30:1 scale
imbalance**, and γ=0.99 against a 150-step success streak. Value clipping is
documented to hurt regardless of threshold and to interact badly with
unnormalized values; global advantage normalization does not fix inter-term
ratios. Every previous run had been tuning rewards and curricula *on top of a
critic that could not represent the return* — each patch fighting a target that
moved for reasons unrelated to the patch.

**The entropy term, and a lesson about coupled fixes.** Removing value clipping
alone then triggered a second failure: action-noise std exploded 0.50 → 0.92 in
both subsequent flights (fixed *and* adaptive LR), destroying every grip through
delta-action noise integration.

Cause: `entropy_coef=0.001` on rsl_rl's **state-independent learnable log-std**
is a permanent upward gradient. Ratio clipping and adaptive-KL bound the *rate*
of policy change, not the *direction* of entropy drift. Unclipped and
unnormalized value loss at target magnitude ~100 amplified advantage noise
enough to expose it.

Two rules:

- **`entropy_coef=0` is the dexterous-manipulation standard** (DexPBT, rl_games,
  SAPG, the IsaacGymEnvs dexterous configs). At init std 0.5 the bonus is
  redundant, and on a state-independent log-std it is actively harmful.
- **Value-clip removal and value normalization are a documented pair.** Never
  ship one without the other — reward scaling or PopArt must land in the same
  change.

Worth recording what was **refuted**, since negative results save the next
person money: slew-gear non-Markovianity, 4× stiffness compliance, and delta
actions themselves were all investigated and cleared. Delta actions in
particular are *necessary* on this robot and task class per prior work — the
action space was never the problem.

---

## 5. Statically valid ≠ dynamically executable

*The one that ended the project.*

The crouch-and-stand reference trajectory was generated by quasi-static IK and
trajectory optimization. It was a sequence of individually valid poses. It was
never checked against gravity, momentum, or contact.

`probe_descent_authority.py`, driving the integrator by exact inversion — no
policy, the ceiling of what *any* controller could achieve in this action space:

```
DESCENT  frames   0-150 : saturation 0.123   pelvis achieved 0.434   ref 0.560
RISE     frames 150-250 : saturation 0.122   pelvis achieved 0.233   ref 0.568
```

Pelvis 0.233 m is on the floor. **The robot collapses.** Slew saturation is only
~12%, so the action rate limit is not the binding constraint — the trajectory
itself cannot be executed by anything.

This explains an entire day of nulls. Five separate interventions — rise income,
full-body reference tracking, base-only tracking, a raised success gate, and a
backward curriculum — were all aimed at a motion the robot physically cannot
perform. A sixth run, a posture expert with *every competing reward removed*,
tracked the rise fine and got monotonically worse as the curriculum introduced
the descent (track_err 0.336 → 0.512). That is exactly the signature you would
predict, and it was read at the time as a tuning problem.

**Corollary worth sitting with: the "lunge" may never have been a defect.** The
policy kept converging on a splayed, straight-legged stance that carries load
through geometry rather than joint torque, and five interventions tried to
eliminate it. It is plausible the policy explored, found the demonstrated motion
unusable, and invented the thing that actually works under this plant. We spent
weeks fighting the correct answer.

**Rule: never train against a reference that has not survived open-loop
playback.** This holds whether the reference comes from mocap, from retargeted
video, or from your own optimizer. If you generate your own, generate it with a
method that respects contact dynamics — differential dynamic programming through
contact, or kinodynamic trajectory optimization — not quasi-static IK.

---

## 6. Non-stationarity, and why the gates never caught anything

Checkpoints **200 iterations apart ranged 6% to 88% strict success.**

The obvious suspicion is that the evaluation is noisy. It isn't: a known-answer
control — the same frozen checkpoint evaluated three times — returned 6/9/4 per
100, tight enough to establish that evaluation is reproducible. The swings are
genuine policy motion.

Every gate in this project was a **single-checkpoint read**. That is how a lane
that never converged passed two gates cleanly, and how a 76% number came out of
a lane whose siblings span 6%–88%.

**Rule:** no bar counts unless **two consecutive checkpoints** clear it, with
the trailing five reported alongside. A single checkpoint clearing a gate on a
non-stationary lane is a sample, not a result.

Related discipline, learned the hard way: **gates do not move after the
evaluation starts.** This project's documented failure mode was never bad gate
thresholds — it was gates recalibrated on the night they would have fired.

---

## 7. Catastrophic forgetting under curriculum change

Adding far starts (0.8–1.8 m) to the banked grasp policy destroyed grasping
within a handful of updates: 81/100 → 0/101, with `contact_no_grasp=100`.
Reducing the excursion to 0.6–0.8 m destroyed *reaching* within 200 iterations.

Diagnosis: distance features are a 2–4× raw excursion into an **unnormalized**
first layer (§2, defect 2), so the damage lands on shared weights rather than
staying local to the new regime.

This is listed separately because the same symptom has a benign-looking
explanation ("the curriculum is too aggressive, anneal it slower") that leads
nowhere, and a mechanical explanation that is a one-line fix. Check whether your
normalizer is actually instantiated before you tune a curriculum.

---

## 8. Laws adopted

Ordered by how much they cost to learn.

1. **No milestone without footage.** Render it and watch the frames before any
   number leaves the room. Caught all three reward-hacking generations; nothing
   else did.
2. **Never train against an un-playback-tested reference.** §5 is the price.
3. **Runtime proof, never config reading.** A mechanism is live only when the
   running process says so. §2 is the price.
4. **Validate every bank with a zero-action probe before training on it.** §1,
   instance 5, is the price.
5. **Activation certificates for reward terms:** proof of nonzero payment on the
   observed state distribution, *and* proof that acting out-earns doing nothing.
6. **Two-checkpoint gates**, trailing five reported.
7. **Kill processes by exact PID** read from `ps`. A `pkill` pattern once killed
   a gate evaluation as collateral; earlier, one left a zombie trainer running
   for nine hours.
8. **Test the hypothesis that would invalidate your work first.** §0.

---

## 9. What I would do differently

**Build fewer bespoke components.** Custom reward economy, custom trajectory
optimizer, custom delta-action integrator, custom slew/gear scheme, custom
curriculum, custom cheat detectors. Six root causes in four weeks is not bad
luck — it is the expected bug yield of a system where every part is hand-rolled
and none has a reference implementation to differ from. In RL a bug does not
throw; it just makes the number worse, so the only way to find it is to spend a
training run.

**Build the evidence, not the artifact.** Four weeks went into producing a
policy. Zero went into producing a *comparison*. A 76% success number in
isolation answers no question anyone was asking. The same number next to a
matched run — same task, same environment, same evaluation, one variable changed
— would have been a result regardless of which way it came out.

**Take the diagnostic budget seriously as a budget.** Compute was never the
binding constraint here. Roughly $400 bought bug discovery in our own stack. The
probes that actually resolved things — `probe_descent_authority`,
`probe_waistfold`, `probe_cage`, the known-answer evaluation control — were
cheap, ran locally or on idle capacity, and each one either killed a hypothesis
or killed a component. They should have been the default first move, not the
thing written after a lane failed.

---

*Instruments referenced are in `pod/g1_grasp/` and `tools/`. Footage for every
claim above is in `media/`, including the failures.*
