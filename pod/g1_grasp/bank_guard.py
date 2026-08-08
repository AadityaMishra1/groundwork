"""Bank stiffness lock (2026-07-29): hard-fail on replaying a
gravity-compensated reference bank under the wrong leg stiffness.

Why this exists: make_ref_traj.py bakes joint_targets = joint_pos +
tau_required / kp_eff against a SPECIFIC actuator stiffness table (KP_EFF,
x4 legs = GRASP_LEG_STIFF 4.0). The bank carries `tau` and `kp_eff` keys and
its docstring claims "the env can re-derive targets under a different
GRASP_LEG_STIFF" — grep-verified FALSE as shipped: grasp_env reads the baked
joint_targets and NOTHING reads tau/kp_eff back. So a bank replayed under a
different LEG_STIFF silently sags (stiffer assumption than reality) or
overshoots (softer), and the failure looks like a policy problem. The lock
turns that silent mismatch into a loud RuntimeError at load time.

Pure CPU, no Isaac, no torch, no numpy import required — callable from
grasp_env's bank-load path AND from a bare unit check on any machine
(the certificate for this lock runs without a simulator on purpose).

Keys: `leg_stiff` is canonical (stamped by tools/stamp_leg_stiff.py and
emitted by tools/apply_grasp_offset.py); `leg_stiff_mult` is the legacy
spelling make_ref_traj.py already writes. Either satisfies the lock.
"""


def check_leg_stiff(bank, path, expected, warn=print):
    """Assert the bank's recorded leg stiffness matches `expected`.

    bank      npz handle or plain dict (anything with .files or .keys()).
    path      for the error/warning text only.
    expected  float(GRASP_LEG_STIFF) the env is about to replay under.
    warn      injection point for the no-key warning (tests capture it).

    Returns the recorded value (float) if a key exists and matches,
    None if no key exists (after a prominent warning). Raises RuntimeError
    on mismatch — never normalize, never rescale: a wrong-stiffness bank is
    a different physics and must not load.
    """
    keys = list(getattr(bank, "files", None) or bank.keys())
    key = ("leg_stiff" if "leg_stiff" in keys
           else "leg_stiff_mult" if "leg_stiff_mult" in keys else None)
    if key is None:
        warn(f"[bank-guard] WARNING: {path} carries NO leg_stiff(_mult) "
             f"key — its gravity-compensated joint_targets ASSUME leg "
             f"stiffness x{expected:g} (the make_ref_traj.py KP_EFF table) "
             f"and this cannot be verified. Stamp the file with "
             f"tools/stamp_leg_stiff.py to lock it.")
        return None
    got = float(bank[key])
    if abs(got - float(expected)) > 1e-6:
        raise RuntimeError(
            f"[bank-guard] {path}: bank {key}={got:g} but GRASP_LEG_STIFF="
            f"{float(expected):g}. The baked joint_targets are gravity-"
            f"compensated against x{got:g} legs; replaying them under "
            f"x{float(expected):g} sags/overshoots silently (nothing reads "
            f"tau/kp_eff back — the re-derive path does not exist as "
            f"shipped). Re-bake the bank or set GRASP_LEG_STIFF={got:g}.")
    return got
