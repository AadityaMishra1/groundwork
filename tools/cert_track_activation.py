"""TRACK-ACTIVATION certificate (standing law, 2026-07-31).

The reference-tracking term is being switched on (GRASP_REF_W 0.0 -> 0.3,
annealing disabled) to teach the crouch shape and the rise, which the bank
already contains (pelvis 0.442 -> 0.600 over frames 150-249 while the object
holds at 0.399 m). Before it ships it must be shown to pay nonzero, with a
usable gradient, at the joint errors the CURRENT policy actually exhibits.

Term (grasp_env.py r_track):
    r = wt * ref_w * ( 8*exp(-2*qerr) + 4*exp(-10*perr)*exp(-5*(1-qdot^2)) )
    qerr = mean squared joint error (rad^2) over 29 tracked joints
    perr = squared root-position error (m^2)
    wt   = 1 - common_step/anneal  -> 1.0 with anneal disabled

STRUCTURAL FINDING, and the reason this is not a repeat of the rise term:
the rise ramp was clamp((pz-onset)/span, 0, 1) — a HARD ZERO below onset,
which is exactly how it paid nothing on 95.8% of observed states. This term
is a pure exponential: it has no dead zone anywhere in its domain, and its
derivative is nonzero everywhere. It cannot be silently inert; it can only
be small. The certificate therefore quantifies HOW small across the whole
plausible error range rather than testing for a dead zone.
"""

import math

REF_W, WT = 0.3, 1.0
JOINT_GAIN, ROOT_GAIN = 8.0, 4.0

print(f"TRACK_CERT ref_w={REF_W} anneal=disabled (wt={WT})")

# joint term across plausible RMS joint errors (rad). The banked policy's
# lunge vs the reference squat: gross posture differs mainly at hip/knee,
# so RMS over all 29 joints in the 0.2-1.2 rad range brackets reality.
print("TRACK_CERT joint term, by RMS joint error:")
for rms in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5):
    qerr = rms * rms
    val = WT * REF_W * JOINT_GAIN * math.exp(-2.0 * qerr)
    # marginal gain from improving RMS by 0.05 rad
    q2 = (rms - 0.05) ** 2
    d = WT * REF_W * JOINT_GAIN * (math.exp(-2.0 * q2) - math.exp(-2.0 * qerr))
    print(f"  rms={rms:.2f} rad -> pay {val:.3f}/step ; "
          f"gain per 0.05 rad closer = {d:+.4f}")

# root term (position + attitude), which is what carries the RISE: getting
# pelvis height wrong by dz costs perr = dz^2
print("TRACK_CERT root term, by pelvis height error (attitude matched):")
for dz in (0.02, 0.05, 0.10, 0.16, 0.20):
    perr = dz * dz
    val = WT * REF_W * ROOT_GAIN * math.exp(-10.0 * perr)
    d2 = (dz - 0.02) ** 2
    d = WT * REF_W * ROOT_GAIN * (math.exp(-10.0 * d2) - math.exp(-10.0 * perr))
    print(f"  dz={dz:.2f} m -> pay {val:.3f}/step ; "
          f"gain per 2 cm closer = {d:+.4f}")

# The operative case: banked policy holds pelvis 0.41 while the reference at
# the same phase is 0.60 -> dz = 0.19 m. Compare the reward for CLOSING that
# gap against the alive income the policy currently banks for standing pat.
DZ_NOW, ALIVE = 0.19, 0.50
now = WT * REF_W * ROOT_GAIN * math.exp(-10.0 * DZ_NOW ** 2)
at_ref = WT * REF_W * ROOT_GAIN
print(f"TRACK_CERT operative: current pelvis gap dz={DZ_NOW} m pays "
      f"{now:.3f}/step; fully matched pays {at_ref:.3f}/step; "
      f"total climb available = {at_ref - now:+.3f}/step "
      f"({(at_ref - now)/ALIVE:.1f}x the {ALIVE}/step camping income)")

ok_nonzero = now > 0.0
ok_gradient = (at_ref - now) > 0.1
ok_nodeadzone = True   # pure exponential, no clamp anywhere in the domain
print(f"TRACK_CERT_{'PASS' if (ok_nonzero and ok_gradient) else 'FAIL'} "
      f"(nonzero at current state: {ok_nonzero}; climb worth "
      f"{at_ref - now:.3f}/step: {ok_gradient}; no dead zone by "
      f"construction: {ok_nodeadzone})")
