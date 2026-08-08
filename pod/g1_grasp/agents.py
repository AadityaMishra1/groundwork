from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


# GRASP_VPATH_V2=1 (2026-07-30 relaunch spec §A, research-verified): the
# value-pathway package. (a) value-loss clipping OFF — "hurts performance
# regardless of the clipping threshold" and interacts badly with
# unnormalized values, our exact prior combination (2006.05990, 3-0
# verified; the 9S post-mortem saw this mechanism and patched the reward
# around it instead of here); (b) gamma 0.99 -> 0.998 — a 150-step strict
# streak is invisible inside a ~100-step effective horizon (VIRAL/2511.15200,
# same robot + task class, trained at 0.998); (c) rollouts 24 -> 48 steps —
# short fragments force early bootstrapping onto the weakest component;
# (d) fixed LR 3e-4 replacing adaptive-KL 1e-3 (VIRAL: fixed 2e-5; ours is
# a middle point for 4096-env scale). Default off = legacy cfg bit-exact.
_VPATH_V2 = __import__("os").environ.get("GRASP_VPATH_V2", "0") == "1"


@configclass
class G1GraspPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48 if _VPATH_V2 else 24
    max_iterations = 2000
    save_interval = 200
    experiment_name = "g1_grasp"
    # GRASP_EMPNORM=1 (run-6 lane B): the FULLBODY obs is 203 raw dims of
    # wildly mixed scales; running obs-normalization is the standard fix
    # (Andrychowicz 2006.05990 SS3.3). Lane-diff vs lane A isolates it.
    # 2026-07-31: this flag has been INERT for the entire whole-body epoch.
    # isaaclab_rl's RslRlPpoActorCriticCfg supplies actor_obs_normalization
    # as {}; rsl_rl's deprecation shim only fills the new keys when they are
    # None, and {} is not None — then `if actor_obs_normalization:` is False
    # because {} is falsy, so the normalizer became nn.Identity and 205 raw
    # mixed-scale dims went straight into the first layer. Verified three
    # ways: no normalizer buffers in any checkpoint, `is None` -> False on
    # the pod, zero "EmpiricalNormalization" lines in any training log.
    # GRASP_EMPNORM_FIX=1 sets the NEW keys directly, which the shim cannot
    # override. Only safe on a FROM-SCRATCH run: banked weights were trained
    # on raw inputs and would be destroyed by suddenly normalized ones.
    empirical_normalization = (
        __import__("os").environ.get("GRASP_EMPNORM", "0") == "1")
    _EMPNORM_FIX = __import__("os").environ.get("GRASP_EMPNORM_FIX", "0") == "1"
    policy = RslRlPpoActorCriticCfg(
        # TRIPWIRE FIRED 2026-07-26 03:2x UTC (iter 385: grasped_now 0.008 —
        # x4-plant run tracked the x1 run exactly, so the plant was real but
        # not binding). ACTIVE: init_noise_std 0.5 for the FULLBODY lineage —
        # delta actions INTEGRATE noise; 41 DoF at std 1.0 = ~0.33 rad/s
        # setpoint drift/joint, which shakes every restored grip apart before
        # hold income can flow (Andrychowicz 2006.05990 SS3.2: 0.5 best on
        # humanoids). entropy_coef 0.005 -> 0.001: entropy inflates
        # clipped-Gaussian rail saturation (DexPBT uses 0). m2d (20 DoF,
        # parked stiff legs) genuinely worked at 1.0 — the 21 policy-driven
        # leg DoF are the difference. Seated-grasp lineage evals unaffected
        # (checkpoints carry their own std).
        init_noise_std=0.5,
        # set the NEW rsl_rl keys directly (see _EMPNORM_FIX note above) —
        # True actually instantiates EmpiricalNormalization; {} silently
        # does not. Assert-on-boot lives in train_grasp so a future session
        # cannot repeat this by reading config and believing it.
        actor_obs_normalization=_EMPNORM_FIX,
        critic_obs_normalization=_EMPNORM_FIX,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    # Value-clip removal SPLIT OUT of VPATH_V2 (2026-07-30, laneV post-
    # mortem): removing clipping WITHOUT value normalization is the prime
    # suspect for the std explosion (0.50->0.92) that killed both laneV
    # flights — the research paired the two and only half shipped. Clipping
    # stays ON until the std deep-research verdict lands and a certified
    # value-norm implementation exists. GRASP_VPATH_NOCLIP=1 re-enables
    # the removal, deliberately, never as a package side effect.
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=(
            __import__("os").environ.get("GRASP_VPATH_NOCLIP", "0") != "1"),
        clip_param=0.2,
        # entropy 0.001 -> 0 under VPATH_V2 (2026-07-30 std research,
        # 3-0 verified): a positive entropy bonus on a state-independent
        # learnable log-std is a PERMANENT upward gradient that neither
        # ratio clipping nor adaptive-KL can direct downward (they bound
        # rate, not direction — Hsu 2009.10897 failure mode 1). It drove
        # the 0.50->0.92 std explosion that killed both laneV flights.
        # entropy 0 is what DexPBT / rl_games / SAPG / all IsaacGymEnvs
        # dexterous configs ship, and Andrychowicz 2006.05990 shows the
        # bonus is redundant at init std 0.5 — our exact init.
        entropy_coef=0.0 if _VPATH_V2 else 0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        # V2 LR history (2026-07-30): first flight used fixed 3e-4 ("middle
        # ground" toward VIRAL's fixed 2e-5). MEASURED WRONG: without the
        # adaptive-KL brake, action std grew 0.50 -> 0.92 over ~2700 iters
        # and noise integration destroyed every grip (grasped_now 0.95 ->
        # 0.02, flings exploding) — the round-6 mechanism, rediscovered at
        # the optimizer level. Adaptive-KL restored; it is the only
        # configuration in this project's history that held std sane on
        # the 41-DoF integrating action space.
        # GRASP_LR: 1e-3 was set during the std-explosion firefight while an
        # entropy bonus was still fighting the schedule, and never re-tuned
        # after entropy_coef went to 0. laneYb (2026-07-31) oscillated 6% <->
        # 88% strict across 200-iteration windows for 3200 iterations and
        # never converged; a known-answer control (model_25600 read three
        # times: 6/9/4 per 100, agreeing within binomial noise) proved the
        # swings are policy motion, not eval noise — the too-large-step
        # signature. Lane Z drops to 3e-4 with the adaptive-KL brake RETAINED
        # (the brake is what held std sane on the 41-DoF integrating action
        # space; removing it caused the 0.50->0.92 explosion below).
        learning_rate=float(__import__("os").environ.get("GRASP_LR",
                                                         "1.0e-3")),
        schedule="adaptive",
        gamma=0.998 if _VPATH_V2 else 0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# ---------------------------------------------------------------------------
# GRASP_EMPNORM_FREEZE=1 support (2026-07-29). Measured on today's resume
# family: every resume whose CONFIG change shifts the STATE distribution
# (FAR_FRAC 0 -> 0.4; grace-length changes) drags the running empirical
# normalizer within ~1 iteration, wrecking the deterministic mean for
# 1500-3000 iters of re-adaptation; reward-only changes (PBRS) do not move
# the stats and recover in ~600. Freezing a MATURE normalizer across the
# resume prevents the drift entirely: statistics load from the checkpoint
# as usual, normalization is still APPLIED every step, the running update
# just stops. The gate is called from train_grasp.py (after runner.load(),
# so the stats being frozen are the checkpoint's) and by the certificate
# probe (probe_empnorm_freeze.py) — SAME function, so the certs exercise
# the shipped code path. No rsl_rl files are edited (the ADVLOG pattern).
# ---------------------------------------------------------------------------

def find_empnorm_modules(runner):
    """Locate every EmpiricalNormalization an rsl-rl runner owns.

    Duck-typed (has update(), an ``until`` attr and an ``_mean`` buffer)
    instead of isinstance, so it survives the rsl-rl 2.x -> 3.x relocation
    of the normalizers (2.x: runner.obs_normalizer /
    runner.critic_obs_normalizer; 3.x: submodules of the policy network).
    Returns a sorted list of (tag, module), deduplicated by identity.
    """
    import torch as _t

    def _is_norm(m):
        return (hasattr(m, "update") and hasattr(m, "until")
                and hasattr(m, "_mean"))

    found, seen = {}, set()
    _alg = getattr(runner, "alg", None)
    roots = [("runner", runner), ("alg", _alg)]
    for nm in ("policy", "actor_critic", "student"):
        m = getattr(_alg, nm, None)
        if m is not None:
            roots.append((f"alg.{nm}", m))
    for tag, root in roots:
        if root is None:
            continue
        # direct attributes (rsl-rl 2.x runner layout)
        for attr in ("obs_normalizer", "critic_obs_normalizer",
                     "actor_obs_normalizer"):
            m = getattr(root, attr, None)
            if m is not None and _is_norm(m) and id(m) not in seen:
                seen.add(id(m))
                found[f"{tag}.{attr}"] = m
        # submodules (rsl-rl 3.x layout — normalizers inside the network)
        if isinstance(root, _t.nn.Module):
            for sub_tag, sub in root.named_modules():
                if sub_tag and _is_norm(sub) and id(sub) not in seen:
                    seen.add(id(sub))
                    found[f"{tag}.{sub_tag}"] = sub
    return sorted(found.items())


def apply_empnorm_freeze(runner):
    """Freeze the runner's empirical-normalization statistics in place.

    Belt-and-braces, both inert w.r.t. the normalization math itself:
      (a) ``until = 0`` — the module's own "stop adapting after N samples"
          knob (update() short-circuits on ``count >= until``);
      (b) an instance-level ``update`` no-op — version-proof against
          internals AND against learn()'s train_mode() flipping the module
          back to training mode each iteration (the reason a plain .eval()
          call cannot work here).
    forward() still normalizes with the loaded mean/std. Call AFTER
    runner.load(): freezing a fresh runner freezes mean=0/var=1 (i.e. the
    normalization is never learned at all) — legal, loudly flagged, and
    almost never what you want.
    """
    mods = find_empnorm_modules(runner)
    if not mods:
        print("[empnorm-freeze] WARNING: no EmpiricalNormalization found "
              "(empirical_normalization off? GRASP_EMPNORM unset?) — the "
              "flag has NO effect on this runner", flush=True)
        return []
    for tag, m in mods:
        m.until = 0
        m.update = lambda *a, **k: None
        try:
            cnt = int(m.count)
        except Exception:
            cnt = None
        print(f"[empnorm-freeze] FROZEN {tag} count={cnt} "
              f"|mean|={float(m._mean.norm()):.3f}", flush=True)
    return [t for t, _ in mods]
