"""Convert the certified pregrasp bank (refs/pregrasp_bank.npz, Mac/MuJoCo)
into GRASP_START_BANK format for the unified training run.

Contingency for / complement to probe_handoff's walker-arrival harvest: these
are the statically CERTIFIED squat states (wbik + cone-QP, 42/42), so the
start distribution's "arrived, ready to reach" slice does not depend on the
walker checkpoint or a pod harvest.

Mac side (numpy only — no torch here): emits refs/start_bank_certified.npz.
Pod side: one-liner converts to .pt (see bottom of this docstring).

Body joint values come from the wbik parameter vector (BODY_JOINTS order,
names identical to the Isaac asset). Inspire hand joints are set to 0 (open);
the GRASP_START_BANK loader remaps by joint NAME, so only names present in
the bank matter and every Isaac joint must appear.

    .venv/bin/python tools/make_start_bank.py
    scp refs/start_bank_certified.npz humanoid-pod:/workspace/
    ssh humanoid-pod '/workspace/venv/bin/python -c "
import numpy as np, torch
d = np.load(\"/workspace/start_bank_certified.npz\", allow_pickle=False)
torch.save({\"joint_pos\": torch.tensor(d[\"joint_pos\"], dtype=torch.float32),
            \"root_state\": torch.tensor(d[\"root_state\"], dtype=torch.float32),
            \"joint_names\": [str(s) for s in d[\"joint_names\"]]},
           \"/workspace/start_bank_certified.pt\")
print(\"OK\", d[\"joint_pos\"].shape)"'
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wbik import BODY_JOINTS, rpy2quat  # noqa: E402

REFS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "refs")

# full Inspire joint list (both hands), Isaac names; parked open (0.0)
HAND_JOINTS = []
for side in ("L", "R"):
    for f in ("index", "middle", "ring", "pinky"):
        HAND_JOINTS += [f"{side}_{f}_proximal_joint", f"{side}_{f}_intermediate_joint"]
    HAND_JOINTS += [f"{side}_thumb_proximal_yaw_joint",
                    f"{side}_thumb_proximal_pitch_joint",
                    f"{side}_thumb_intermediate_joint",
                    f"{side}_thumb_distal_joint"]


def main():
    b = np.load(os.path.join(REFS, "pregrasp_bank.npz"), allow_pickle=False)
    X = b["x"]                                   # (n, 6+29) wbik param vectors
    n = len(X)
    names = list(BODY_JOINTS) + HAND_JOINTS
    jp = np.zeros((n, len(names)), dtype=np.float32)
    jp[:, :len(BODY_JOINTS)] = X[:, 6:6 + len(BODY_JOINTS)]
    root = np.zeros((n, 13), dtype=np.float32)
    root[:, 0:3] = X[:, 0:3]
    for i in range(n):
        root[i, 3:7] = rpy2quat(*X[i, 3:6])
    out = os.path.join(REFS, "start_bank_certified.npz")
    np.savez(out, joint_pos=jp, root_state=root,
             joint_names=np.array(names))
    print(f"start bank: {n} certified squat states, {len(names)} joints -> {out}")
    print(f"pelvis z range {root[:, 2].min():.3f}-{root[:, 2].max():.3f}")


if __name__ == "__main__":
    main()
