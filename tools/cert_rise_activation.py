"""RISE-ACTIVATION certificate (PM condition 1, 2026-07-31).

Known-answer test: replay the RECORDED strict-hold pelvis_z distribution
through the old rise term (onset 0.43) and the proposed one (onset 0.36).
Pass requires: old pays ~0 across the observed band; new pays nonzero and
monotonically increasing across 0.37 -> 0.50.

Standing law this enforces: every reward term ships with proof it pays
nonzero on the OBSERVED state distribution, not the intended one.

Data: POSTURE_STRICT pelvis_z p10/25/50/75 from every laneYb scaffold-near
eval log (12 checkpoints x 4 quantiles = 48 recorded points).
"""

RISE_W = 3.0
OLD_ONSET, NEW_ONSET, TOP = 0.43, 0.36, 0.62


def pay(pz, onset):
    span = TOP - onset
    return RISE_W * min(max((pz - onset) / span, 0.0), 1.0)


# recorded (checkpoint, p10, p25, p50, p75) — laneYb scaffold-near evals
REC = [
    (23600, .406, .415, .424, .432), (23800, .407, .415, .422, .428),
    (24000, .408, .412, .419, .426), (24200, .374, .383, .396, .409),
    (24400, .379, .392, .400, .407), (24600, .383, .388, .397, .405),
    (24800, .391, .399, .415, .434), (25000, .387, .395, .403, .412),
    (25200, .390, .396, .404, .415), (25400, .383, .389, .398, .412),
    (25600, .394, .399, .405, .411), (25799, .398, .407, .416, .423),
]

pts = [v for row in REC for v in row[1:]]
old = [pay(p, OLD_ONSET) for p in pts]
new = [pay(p, NEW_ONSET) for p in pts]
old_zero = sum(1 for v in old if v == 0.0)

print(f"RISE_CERT n={len(pts)} recorded pelvis_z points "
      f"(range {min(pts):.3f}-{max(pts):.3f})")
print(f"RISE_CERT old(onset={OLD_ONSET}): zero-pay {old_zero}/{len(pts)} "
      f"({old_zero/len(pts):.1%}), mean {sum(old)/len(old):.4f}, "
      f"max {max(old):.4f}")
print(f"RISE_CERT new(onset={NEW_ONSET}): zero-pay "
      f"{sum(1 for v in new if v == 0.0)}/{len(pts)}, "
      f"mean {sum(new)/len(new):.4f}, max {max(new):.4f}")

# monotonicity sweep across the legal band
sweep = [0.37 + 0.01 * i for i in range(14)]
vals = [pay(p, NEW_ONSET) for p in sweep]
mono = all(b > a for a, b in zip(vals, vals[1:]))
print("RISE_CERT sweep 0.37->0.50 new: "
      + " ".join(f"{p:.2f}:{v:.3f}" for p, v in zip(sweep, vals)))
print(f"RISE_CERT sweep old: "
      + " ".join(f"{p:.2f}:{pay(p, OLD_ONSET):.3f}" for p in sweep))

ok = (old_zero / len(pts) > 0.70) and all(v > 0 for v in new) and mono
print(f"RISE_CERT_{'PASS' if ok else 'FAIL'} "
      f"(old dead on >70% of observed: {old_zero/len(pts):.1%}; "
      f"new nonzero everywhere: {all(v > 0 for v in new)}; "
      f"monotone across legal band: {mono})")
