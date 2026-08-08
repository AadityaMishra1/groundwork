# Waist-Fold Autopsy — 2026-07-30

PM-mandated post-mortem of the laneW3 gate failure. The lane produced
scaffold-near strict rates of 87.4% / 91.1% (rerun) / 88.7% (model_23400
reproduction) that passed the CLAUSE5 audit and posture telemetry — and two
independent films (demo_pickup standing-start; demo_grasp, the exact eval
distribution) show every strict episode is a **crab-sprawl thigh-pin**: legs
splayed straight, torso folded ~70° at the waist, one hand planted on the
floor as a tripod, object clamped between the gripping hand and the robot's
own thigh. Zero upright holds, zero palm-only carries. All numbers from this
lane are labeled **strict-by-metric, posture-invalid** in perpetuity.

Films: `media_local/laneW3_23200_evaldist.mp4` (eval distribution),
`media_local/laneW3_23200_87pct_v2.mp4` (standing start). Both frame-reviewed.

## Q1 — What the instruments actually measure vs. what was believed

| Instrument | Believed | Actually measures |
|---|---|---|
| Gate `up` (`grasp_env.py:1316-1317`) | "robot is upright" | z of the **pelvis** (root link) up-axis only. The waist fold never appears. Crab pelvis stays level: reads ~0.85. |
| Gate `pelvis` (`:1318-1319`) | "standing/squat height" | root height. Splayed straight legs hold the pelvis at ~0.43 m. Passes 0.35/0.37 bars. |
| Gate/audit `minz` (`:1320-1321`, allowlist `:380-384`) | "no ground contact above the knee" | min-z over `c5_bodies`, which **exclude every wrist/hand/finger link by allowlist**. The planted tripod hand is invisible; elbows/shoulders stay above the bar in the crab. |
| CLAUSE5 audit (`demo_grasp.py:116-147`) | "clause-5 clean = legitimate hold" | same c5 body set (min-z quantiles) + hand proximity *reported but not enforced*. Reads 92/101 clean on the crab. |
| Secure grip (`:1530`, `_grip_force` `:1185-1198`) | "load-bearing grasp" | summed finger–object contact force > 3 N. A thigh-pin produces genuine finger force — load path goes through the robot's own leg. |
| POSTURE_STRICT telemetry | "upright during holds" | root quat + root height again — same blind spot as the gate, so it *cannot* disagree with it. |

## Q2 — Why torso orientation and hand-floor contact are invisible

- No term anywhere reads a torso/waist link orientation. Every uprightness
  quantity in env, gate, audit, and telemetry derives from `root_quat_w`.
- The hand-chain exclusion was a **deliberate design decision**
  (`demo_grasp.py:124-129`): palms legitimately work at floor level during
  floor grasps, so counting them would make legal pickups unearnable. The
  unexamined consequence: a hand planted as a load-bearing strut a meter from
  the object is equally legal.
- Nothing measures object contact with (or proximity to) the robot's own
  non-hand links, so bracing the object against the thigh is undetectable.

## Q3 — Is the thigh-pin's >3 N by design or accident?

Accident of physics, not sensor error. The filtered finger sensors report
real finger–object force; pressing the object into the thigh produces ≥3 N
through the fingers. The qualifier's unstated assumption — that finger force
implies the *grip* carries the load — is what fails. The complement check
(object clear of own body) was never implemented.

## Why the policy chose this (mechanism, from prior probes)

`probe_lift_authority` (2026-07-29): from deep-fold grasps, scripted leg
extension dies 100% of the time — knee pinned at the 139 N·m cap, object
ceiling 0.39 m. Rising from the fold is physically punished, so the policy
rationally never tries; the gate loophole then pays full ECON_V2 income
(secure-scaled contact, elevation ramp, stream, bonus) in the sprawl. The
certified squat band (pelvis 0.39–0.59) has ~2× torque headroom — the rise
is achievable, but only from a posture the economy never forced.

## Q4 — Certified detection (`pod/g1_grasp/probe_waistfold.py`)

Pre-registered detector, measured during strict windows only:
- **FOLD**: min torso-link up-z < 0.60 (torso link, not root)
- **PLANT**: any wrist/hand/finger link center < 0.05 m for ≥10 strict steps
- ILLEGAL = FOLD ∨ PLANT; obj-to-leg-link distance reported as diagnostic.

Certification requires BOTH: ≥95% of model_23200 strict episodes flagged
(policy mode) and ≤5% of reference squat-lift restore frames flagged (ref
mode, gc2 bank sweep). **No gate term ships until this probe passes both.**

### Probe receipts (2026-07-30, pod A)

Policy mode (model_23200, 60 strict episodes):

    WAISTFOLD_DIST n=60 torso_up p10/50/90=0.172/0.236/0.315 hand_z p10/50/90=0.006/0.007/0.008 objleg p10/50/90=0.093/0.096/0.100
    WAISTFOLD_FLAGS fold=60/60 plant=60/60 illegal=60/60 rate=1.000
    WAISTFOLD_PASS policy(crab) flag rate 1.000 (need >=0.95)

Both terms flag 100% independently; objleg (median 9.6 cm) independently
confirms the thigh-pin.

Ref mode (gc2 bank restore sweep, gate scope obj_z > 0.22):

    WAISTFOLD_DIST n=654 torso_up p10/50/90=0.962/0.998/0.998 hand_z p10/50/90=0.241/0.364/0.365 objleg p10/50/90=0.163/0.171/0.179
    WAISTFOLD_FLAGS fold=0/654 plant=1/654 illegal=1/654 rate=0.002
    WAISTFOLD_PASS ref(legit) flag rate 0.002 (need <=0.05)

Calibration note: two earlier ref runs are part of the record. Unscoped
(all frames): 28.4% plant false-positives — legit descent frames have
floor-adjacent hands, which is why the SHIPPED gate term is scoped to
obj_z > 0.22. Scoped at 0.40: n=2 (no power). Final scope = the shipped
gate's own condition, so the certificate covers exactly what training
enforces. Separation margin: legit torso_up p10 0.962 vs crab p90 0.315;
bar 0.60 (spec formula clamp(ref_p05−0.05, 0.45, 0.60)).

## Rulebook consequences

- Numbers derived from gate-based telemetry inherit the gate's blind spots;
  they can corroborate footage, never substitute for it. (Second confirmed
  instance of the footage law catching what every number missed.)
- Any future gate/legality term must be certified by a scripted-exploit or
  recorded-exploit probe (this file's pattern) before training against it.
- Trainers stopped 2026-07-30 by exact PID (pod A 1856471 SIGKILL receipt;
  pod B 1618798 + rolling_eval 1623264). Checkpoints preserved through
  model_24000 (laneW3) for autopsy use.
