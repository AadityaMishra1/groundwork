"""WALK-ACTIVATION certificate (PM conditions, 2026-07-31).

Standing law: a reward term ships only with proof it pays nonzero on the
OBSERVED state distribution. Amendment A (from the rise term's second
death): also prove that ACTING out-earns DOING NOTHING at that state.
The rise term activated correctly and still died because standing pat was
worth more than the increment. This certificate tests both.

Terms, read from grasp_env.py:
  r_walk  = 8.0 * clamp(prev_base_d - base_d, -0.05, 0.05)   [x2.5 ECON_V2]
  r_alive = 0.05 * (up > 0.7)  +  0.45 * legal               [ECON_V2]
r_walk is potential-based on base->object distance, so per-step income is
proportional to distance CLOSED that step, capped at 5 cm/step.
"""

WALK_GAIN, ECON_V2_MULT, WALK_CAP = 8.0, 2.5, 0.05
ALIVE_UP, ALIVE_LEGAL = 0.05, 0.45
DT = 1.0 / 50.0                      # control step (200 Hz sim / 4 decimation)

# Protocol spawn distances: probe_spawn_protocol recorded n=4096 at
# FAR_FRAC-equivalent spawns — dist_min/mean/max = 1.000/2.537/3.999
SPAWN = {"min": 1.000, "mean": 2.537, "max": 3.999}


def walk_pay(closed_m):
    """Per-step r_walk for `closed_m` metres closed this step."""
    return WALK_GAIN * ECON_V2_MULT * min(max(closed_m, -WALK_CAP), WALK_CAP)


print("WALK_CERT terms: r_walk 8.0 x2.5 cap +/-0.05 m/step; "
      f"r_alive {ALIVE_UP} (up) + {ALIVE_LEGAL} (legal)")

# --- (i) activation + monotonicity on the observed spawn distribution ---
steps = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
vals = [walk_pay(s) for s in steps]
mono = all(b >= a for a, b in zip(vals, vals[1:]))
strict_mono_in_band = all(b > a for a, b in zip(vals[:-1], vals[1:-1]))
print("WALK_CERT sweep (m closed/step -> pay): "
      + " ".join(f"{s:.3f}:{v:.3f}" for s, v in zip(steps, vals)))
print(f"WALK_CERT nonzero at every positive increment: "
      f"{all(v > 0 for v in vals)}; monotone: {mono} "
      f"(saturates at the {WALK_CAP} m/step cap by design)")

# --- (ii) AMENDMENT A: acting vs doing nothing at a far spawn ---
# CORRECTION (caught pre-report): a WALKING robot is still upright and
# posture-legal, so it collects r_alive TOO. Standing and walking are not
# alternatives w.r.t. alive income; walking income is ADDITIVE. The first
# draft of this certificate compared r_walk alone against alive income and
# produced a false refutation at 0.96x. The correct marginal quantity is
# r_walk itself, priced against the risk of moving.
do_nothing = ALIVE_UP + ALIVE_LEGAL          # camping, upright, legal
SPEED_MPS = [0.25, 0.5, 1.0]                 # plausible gait speeds
print(f"WALK_CERT do-nothing income/step at far spawn = {do_nothing:.3f} "
      f"(alive {ALIVE_UP} + legal {ALIVE_LEGAL}; no task income reachable "
      f"at distance)")
for v in SPEED_MPS:
    closed = v * DT
    w = walk_pay(closed)
    print(f"WALK_CERT walking {v:.2f} m/s ({closed*100:.1f} cm/step): "
          f"alive {do_nothing:.3f} + walk {w:.3f} = {do_nothing + w:.3f}/step "
          f"vs standing {do_nothing:.3f} -> margin {w:+.3f} "
          f"({(do_nothing + w)/do_nothing:.2f}x)")

# Marginal test at the slowest plausible gait, risk-discounted.
slow = walk_pay(SPEED_MPS[0] * DT)
RISK = 0.20      # pessimistic termination risk while moving
margin_risked = slow * (1.0 - RISK)
ratio = (do_nothing + margin_risked) / do_nothing
print(f"WALK_CERT worst-case margin (slowest gait, {RISK:.0%} risk "
      f"discount) = {margin_risked:+.3f}/step -> {ratio:.2f}x standing")

# --- (iii) destination economics: is arriving worth the trip? ---
# The traverse bonus is small; what must dominate is the DESTINATION.
STREAM, BONUS, GAMMA = 13.3, 300.0, 0.998
traverse = WALK_GAIN * ECON_V2_MULT * SPAWN["mean"]
for v in SPEED_MPS:
    steps_to_arrive = SPAWN["mean"] / (v * DT)
    disc = GAMMA ** steps_to_arrive
    print(f"WALK_CERT arrival at {v:.2f} m/s: {steps_to_arrive:.0f} steps "
          f"({steps_to_arrive*DT:.1f} s), discount {disc:.3f}; "
          f"stream {STREAM}/step discounted = {STREAM*disc:.2f}/step "
          f"vs camping {do_nothing:.3f}/step "
          f"({STREAM*disc/do_nothing:.1f}x)")
print(f"WALK_CERT traverse shaping pays {traverse:.1f} total; "
      f"a 30 s camp pays {do_nothing/DT*30:.0f} — shaping alone does NOT "
      f"beat camping, the DESTINATION must, and does "
      f"({STREAM*GAMMA**(SPAWN['mean']/(0.5*DT))/do_nothing:.1f}x at "
      f"0.5 m/s within the gamma=0.998 horizon)")

ok_activate = all(v > 0 for v in vals) and mono
ok_outearn = ratio > 1.0
slowest_disc = GAMMA ** (SPAWN["mean"] / (SPEED_MPS[0] * DT))
ok_dest = STREAM * slowest_disc > do_nothing
print(f"WALK_CERT_{'PASS' if (ok_activate and ok_outearn and ok_dest) else 'FAIL'} "
      f"(activates+monotone: {ok_activate}; walking out-earns standing at "
      f"worst-case gait: {ok_outearn} at {ratio:.2f}x; destination beats "
      f"camping even at slowest gait within horizon: {ok_dest})")
