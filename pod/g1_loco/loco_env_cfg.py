"""Phase 3A: locomotion + kneel on the GRASP embodiment (G1 + Inspire hands).

The round-1 walk/crouch policies were trained on the handless G1 asset — a
different robot, strictly. Composition requires one body for everything, so
this env re-trains velocity walking + commanded base height on
G1_INSPIRE_FTP with the physics debts paid: gravity ENABLED on robot links
(the shipped asset disables it) and self-collisions ON. Height command
extends down to the kneel; below 0.30 m a pose-shaping reward pulls the legs
toward the grasp policy's folded-kneel stance (CROUCH_LEGS in grasp_env.py)
so the bottom posture is the grasp policy's expected stance. The right wrist
carries a randomized payload so rising-while-holding is trained from birth.
Hands are NOT actuated here — hand PD holds defaults; the grasp policy owns
the hand during composition.
"""

from __future__ import annotations

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import (
    CommandTerm,
    CommandTermCfg,
    EventTermCfg as EventTerm,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
)
from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG

import os as _os
_WS = _os.environ.get("GW_ROOT", "/workspace")

# free-root variant: mimic-joint APIs stripped (fix_usd.py) — the shipped
# asset's finger couplings fail articulation parsing when the root is freed
LOCAL_USD = f"{_WS}/assets/G1/g1_29dof_inspire_hand_free.usd"

BODY_JOINTS = [
    ".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint",
    ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
    ".*_elbow_joint", ".*_wrist_roll_joint", ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
]

# grasp stance (keep in sync with grasp_env.py CROUCH_LEGS)
KNEEL_POSE = {
    ".*_hip_pitch_joint": -2.2, ".*_knee_joint": 2.8, ".*_ankle_pitch_joint": -0.87,
    ".*_hip_roll_joint": 0.4, ".*_hip_yaw_joint": 0.0, ".*_ankle_roll_joint": 0.0,
}
# standing legs (matches init_state below)
STAND_POSE = {
    ".*_hip_pitch_joint": -0.20, ".*_knee_joint": 0.42, ".*_ankle_pitch_joint": -0.23,
    ".*_hip_roll_joint": 0.0, ".*_hip_yaw_joint": 0.0, ".*_ankle_roll_joint": 0.0,
}
# pelvis height across the stand->kneel leg interpolation (approx, probed)
SPAWN_Z_STAND, SPAWN_Z_KNEEL = 0.80, 0.28


_SPAWN_IDS = None  # (joint ids, stand targets, kneel targets) cache


def spawn_at_random_height(env, env_ids):
    """M1f reset event: start episodes balanced at a random depth.

    m1e proved the parking exploit: with all episodes starting at standing
    height, depths are only reachable through a fall-risky descent, so the
    policy parks at one safe height forever. Starting on the stand<->kneel
    interpolation (t ~ U[0,1]) makes every depth a place the policy has
    BEEN from step 0 — balance there is trained directly, and in-episode
    command resamples (4-8 s) train the transitions between depths.
    """
    global _SPAWN_IDS
    robot = env.scene["robot"]
    if _SPAWN_IDS is None:
        ids, s_tgt, k_tgt = [], [], []
        for pattern in KNEEL_POSE:
            jids, _ = robot.find_joints(pattern)
            ids += jids
            s_tgt += [STAND_POSE[pattern]] * len(jids)
            k_tgt += [KNEEL_POSE[pattern]] * len(jids)
        _SPAWN_IDS = (torch.tensor(ids, device=env.device),
                      torch.tensor(s_tgt, device=env.device),
                      torch.tensor(k_tgt, device=env.device))
    jids, s_tgt, k_tgt = _SPAWN_IDS
    n = len(env_ids)
    # depth anneal: probe_spawn2 showed a fresh flailing policy dies in ~2
    # steps from ANY spawn — deep starts are only trainable once basic
    # balance exists. Start mostly-standing (the regime the original from-
    # scratch walker survived), widen to the full range by ~iter 1500.
    t_max = min(1.0, 0.25 + env.common_step_counter / 36_000)
    t = torch.rand(n, 1, device=env.device) * t_max
    jp = robot.data.default_joint_pos[env_ids].clone()
    jp[:, jids] = s_tgt + t * (k_tgt - s_tgt) \
        + 0.01 * (torch.rand(n, len(jids), device=env.device) - 0.5)
    root = robot.data.default_root_state[env_ids].clone()
    # +0.02 clearance: the deepest spawn showed a step-1 depenetration
    # launch (z 0.28 -> 0.64) — never spawn touching/overlapping the ground
    root[:, 2] = SPAWN_Z_STAND + t[:, 0] * (SPAWN_Z_KNEEL - SPAWN_Z_STAND) + 0.02
    root[:, :3] += env.scene.env_origins[env_ids] * torch.tensor(
        [1.0, 1.0, 0.0], device=env.device)
    root[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root[:, :7], env_ids)
    robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
    # deposit spawn height for the command term's bootstrap coupling
    # (events run before command resampling in the reset sequence)
    term = env.command_manager.get_term("base_height")
    term.spawn_z[env_ids] = SPAWN_Z_STAND + t[:, 0] * (SPAWN_Z_KNEEL - SPAWN_Z_STAND)
    if hasattr(env, "freeze_left"):  # LOCO_FREEZE: fresh episodes start live
        env.freeze_left[env_ids] = 0


class HeightCommand(CommandTerm):
    """Uniformly resampled base-height target, shape (num_envs, 1)."""

    cfg: "HeightCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        # spawn-coupling mailbox: the spawn event deposits each reset env's
        # spawn height here; the next resample consumes it (-1 = empty)
        self.spawn_z = torch.full((self.num_envs,), -1.0, device=self.device)
        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.height_command

    def _update_metrics(self):
        self.metrics["height_error"] = torch.abs(
            self.height_command[:, 0] - self.robot.data.root_pos_w[:, 2]
        )

    def _resample_command(self, env_ids):
        # U-shaped sampling (M1f): uniform commands let the policy park at
        # one mid height and collect most of the wide-Gaussian pay (m1e
        # collapsed to 0.37 m for ALL commands, 5.6k iters, eval-proven).
        # Weight the extremes so obeying hard commands is where the income
        # is: 40% deep band, 30% stand band, 30% uniform in between.
        n = len(env_ids)
        lo, hi = self.cfg.height_range
        r = torch.rand(n, device=self.device)
        band = torch.rand(n, device=self.device)
        cmd = torch.where(
            band < 0.40, lo + r * (0.28 - lo),            # deep: crouch band 0.24-0.28
            torch.where(band < 0.70, 0.60 + r * (hi - 0.60),  # stand: 0.60-0.72
                        0.35 + r * (0.60 - 0.35)))        # middle: uniform
        # M1f-v3 bootstrap coupling: gated height income requires on-feet AND
        # near-command AT THE SAME TIME — with independent spawn/command
        # sampling a fresh policy almost never experiences that state, the
        # gradient never forms, and sitting becomes the optimal zero (the
        # collapse mechanism behind m1e AND m1f-v2, eval-proven). 70% of
        # RESET commands = the height the env just spawned at, on its feet:
        # obedience pays from step 0. Mid-episode resamples stay random, so
        # transitions are still trained and the command can't be dragged.
        sz = self.spawn_z[env_ids]
        couple = (sz > 0) & (torch.rand(n, device=self.device) < 0.70)
        cmd = torch.where(couple, sz.clamp(lo, hi), cmd)
        self.spawn_z[env_ids] = -1.0
        self.height_command[env_ids, 0] = cmd

    def _update_command(self):
        # kneel means STATIONARY kneel: independent height and velocity
        # commands paid for gliding kneels (m1i skated at 0.4-1.0 m/s to
        # satisfy both). Deep height command forces zero velocity command.
        deep = self.height_command[:, 0] < 0.45
        if deep.any():
            vel = self._env.command_manager.get_term("base_velocity")
            vel.vel_command_b[deep] = 0.0


class FreezableJointPositionAction(mdp.JointPositionAction):
    """M3 handoff hardening: envs can be 'frozen' — incoming actions ignored,
    PD holds the pose captured at freeze time. Trains the policy to occupy
    poses that survive a control handoff (the composition runner parks the
    legs exactly like this while the grasp policy owns the arm)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        env.freeze_left = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.frozen_targets = torch.zeros(env.num_envs, len(self._joint_ids)
                                         if not isinstance(self._joint_ids, slice)
                                         else self._num_joints, device=env.device)
        self._the_env = env

    def apply_actions(self):
        fz = self._the_env.freeze_left > 0
        if fz.any():
            self._the_env.freeze_left[fz] -= 1
            tgt = self.processed_actions.clone()
            tgt[fz] = self._the_env.frozen_targets[fz]
            self._asset.set_joint_position_target(tgt, joint_ids=self._joint_ids)
        else:
            self._asset.set_joint_position_target(
                self.processed_actions, joint_ids=self._joint_ids)


def freeze_at_depth(env, env_ids):
    """Interval event: freeze a random subset of kneeling envs for 0.5-1.5 s.
    Survival income (feet_alive + height tracking) then prices statically
    holdable kneels; dynamically-balanced ones die in the freeze."""
    if not hasattr(env, "freeze_left"):
        return
    # from-scratch warmup: no freezes until basic balance exists (~iter 750
    # at 4096 envs / 24 steps per iter) — a flailing fresh policy dies from
    # any freeze and the drill teaches nothing but noise
    if env.common_step_counter < 18_000:
        return
    robot = env.scene["robot"]
    # v4: eligibility keyed to the COMMAND, which the policy cannot influence.
    # Every state-keyed gate got gamed: depth gate -> camping just above it
    # (m1g); speed gate -> perpetual gliding kneels that stay freeze-immune
    # at 0.4-1.0 m/s (m1i). Kneel commanded => freezes fire, full stop.
    hcmd = env.command_manager.get_term("base_height").height_command[env_ids, 0]
    # v5 STILLNESS-GATED: m1m proved unavoidable freezes strangle from-scratch
    # kneel learning (deep kneel never bootstrapped — depth=death learned
    # first; final eval stood at 0.73 for a 0.32 command). Freeze only envs
    # that were verifiably STILL since the last event check (displacement
    # window, chatter-proof). Unskilled descents are never punished; still
    # kneels — which the drift penalty already forces — get pose-selected.
    # "Never be still" as an escape is priced away by kneel_drift (-9).
    if not hasattr(env, "_freeze_last_xy"):
        env._freeze_last_xy = torch.zeros(env.num_envs, 2, device=env.device)
    xy = (robot.data.root_pos_w[env_ids, :2]
          - env.scene.env_origins[env_ids, :2])
    still = (xy - env._freeze_last_xy[env_ids]).norm(dim=-1) < 0.08
    env._freeze_last_xy[env_ids] = xy
    # v6: freeze only in the CROUCH band. A 0.32-0.37 half-squat is
    # physically unholdable under PD (every generation proved it — glide,
    # skid, or refusal); the deep fold resting on shins is the one static
    # family (probe_stance: 0.217m, up 0.98). Demand holdability only
    # where physics permits it.
    elig = (hcmd < 0.30) & still & (env.freeze_left[env_ids] == 0)
    roll = torch.rand(len(env_ids), device=env.device) < 0.3
    pick = env_ids[elig & roll]
    if len(pick) == 0:
        return
    term = env.action_manager.get_term("joint_pos")
    env.frozen_targets[pick] = robot.data.joint_pos[pick][:, term._joint_ids]
    # v3: 2-6 s freezes. The 0.5-1.5 s drills taught "survive short freezes
    # and recover" — the m1h kneel still bows out over ~1.5 s under a static
    # hold (probe_stance up 0.75 -> 0.20). The grasp phase parks the legs for
    # up to 12 s, so the drill must cover that timescale.
    env.freeze_left[pick] = torch.randint(75, 150, (len(pick),), device=env.device)


@configclass
class HeightCommandCfg(CommandTermCfg):
    class_type: type = HeightCommand
    asset_name: str = "robot"
    height_range: tuple = (0.24, 0.72)
    # descent-anneal fields removed (M1f): learnability now comes from
    # spawn-at-random-height starts, not from easing the command floor —
    # the anneal drifted the command mass toward mid heights and fed the
    # parking exploit.


# reward-warmup ramp: the gate was accidentally dead for the entire prior
# 3888-iteration run, so the critic's value function has zero calibration
# for height/kneel reward. Turning the gate on at full weight on a resumed
# policy was a step-function reward shock — fall rate went 0.2%->97% in 8
# iterations and did not recover. Ramp the newly-live reward terms in
# linearly over the first RAMP_STEPS env steps of THIS process's life so
# the critic can track the change instead of being blindsided by it.
RAMP_STEPS = 8000


def _reward_ramp(env):
    return min(1.0, env.common_step_counter / RAMP_STEPS)


def track_height_exp(env, std: float, command_name: str,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, 0]
    err = torch.square(asset.data.root_pos_w[:, 2] - cmd)
    # feet-only gate: a collapsed heap earns nothing (sit-exploit fix)
    return torch.exp(-err / std**2) * feet_only_contact(env) * _reward_ramp(env)


_FEET_IDS = None  # (feet body ids, other body ids) cache for contact sensor


def feet_only_contact(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """1.0 where BOTH feet carry load and NO other body touches anything.

    The v2/v3 sit-exploit post-mortem: a collapsed heap at 0.15 m earned
    height reward (wide tracking Gaussian), kneel-pose shaping (a sit looks
    like a kneel), and zero penalties. Gating all height/pose income on
    feet-only support makes the heap worth exactly nothing — the only way
    to get paid low is a balanced deep squat on the feet.
    """
    global _FEET_IDS
    sensor = env.scene.sensors["contact_forces"]
    if _FEET_IDS is None:
        feet, _ = sensor.find_bodies(".*_ankle_roll_link")
        # v3: probe_gate.py showed hip/knee links carrying 170-460N of
        # phantom "contact" force even while genuinely standing — the
        # grasp-derived arm pose rests the wrists/hands against the thighs,
        # and that self-collision bleeds onto hip_pitch/knee readings in the
        # contact sensor. hips+knees are not usable collapse indicators.
        # Only the pelvis directly touching the ground means "sitting"; the
        # arms resting on the legs is expected and harmless.
        bad, _ = sensor.find_bodies(["pelvis", "torso_link"])
        _FEET_IDS = (torch.tensor(feet, device=env.device),
                     torch.tensor(bad, device=env.device))
    fids, bids = _FEET_IDS
    f = sensor.data.net_forces_w.norm(dim=-1)
    some_foot = (f[:, fids] > 1.0).any(dim=1)
    not_collapsed = f[:, bids].max(dim=1).values < 5.0
    return (some_foot & not_collapsed).float()


def feet_alive(env):
    """Unconditional on-feet income — see M1f-v3 note at the RewTerm."""
    return feet_only_contact(env)


_KNEEL_IDS = None  # (joint_ids tensor, target tensor) cache


def kneel_drift(env, command_name: str):
    """Root xy speed while a kneel is commanded. Every M1 generation GLIDES
    at the kneel (~0.5 m/s with zero velocity command) — height/upright evals
    never saw it, and under unavoidable freezes gliding became the survival
    strategy (skid into a wedged stop). Composition needs the robot to kneel
    AT the object and stay; price the drift directly."""
    cmd = env.command_manager.get_term(command_name).height_command[:, 0]
    deep = (cmd < 0.45).float()
    speed = env.scene["robot"].data.root_lin_vel_w[:, :2].norm(dim=-1)
    return deep * speed


def kneel_pose_shaping(env, command_name: str,
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Below 0.30 m commanded height, pull legs toward the grasp kneel."""
    global _KNEEL_IDS
    asset = env.scene[asset_cfg.name]
    if _KNEEL_IDS is None:
        ids, targets = [], []
        for pattern, val in KNEEL_POSE.items():
            jids, _ = asset.find_joints(pattern)
            ids += jids
            targets += [val] * len(jids)
        _KNEEL_IDS = (torch.tensor(ids, device=env.device),
                      torch.tensor(targets, device=env.device))
    jids, tgt = _KNEEL_IDS
    cmd = env.command_manager.get_command(command_name)[:, 0]
    err = (asset.data.joint_pos[:, jids] - tgt).square().mean(dim=-1)
    # M1M: shaping now covers EVERY kneel command (was cmd<0.30 only — the
    # 0.32 composition dwell got zero pull toward the foldable crouch, so the
    # walker's kneel was a different, PD-unholdable pose family; frozen from
    # stillness it died 64/64 while the crouch holds statically). Weight
    # scales with depth: full pull at 0.24, fading out by 0.42.
    depth = ((0.42 - cmd) / (0.42 - 0.24)).clamp(0.0, 1.0)
    # feet-only gate: shaping must not pay for a collapsed sit
    return (torch.exp(-2.0 * err) * depth
            * feet_only_contact(env) * _reward_ramp(env))


@configclass
class G1LocoKneelEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # grasp embodiment, honest physics
        self.scene.robot = G1_INSPIRE_FTP_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.usd_path = LOCAL_USD
        self.scene.robot.spawn.rigid_props.disable_gravity = False
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        # the Inspire asset ships as a MANIPULATION rig: root bolted in
        # space and a zero init pose — free the root and stand it up
        self.scene.robot.spawn.articulation_props.fix_root_link = False
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.80)
        self.scene.robot.init_state.joint_pos = {
            ".*_hip_pitch_joint": -0.20, ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23, ".*_elbow_joint": 0.87,
            "left_shoulder_pitch_joint": 0.35, "right_shoulder_pitch_joint": 0.35,
        }
        # actuate the 29 body joints only — hand PD holds defaults
        self.actions.joint_pos.joint_names = BODY_JOINTS
        # the stock flat cfg targets the 23-DoF G1 (torso_joint, elbow_pitch,
        # rubber-hand five/three joints) — remap deviation penalties to the
        # Inspire 29-DoF names or they resolve to zero joints and crash
        self.rewards.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                                  ".*_shoulder_yaw_joint", ".*_elbow_joint",
                                  ".*_wrist_.*_joint"])
        self.rewards.joint_deviation_fingers.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_index_.*", ".*_middle_.*", ".*_ring_.*",
                                  ".*_pinky_.*", ".*_thumb_.*"])
        # fingers are unactuated here (hand PD parks them) — deviation is
        # gravity's fault, not the policy's; near-mute the penalty
        self.rewards.joint_deviation_fingers.weight = -0.005
        self.rewards.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names="waist_.*_joint")

        self.commands.base_height = HeightCommandCfg(resampling_time_range=(4.0, 8.0))
        self.observations.policy.height_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_height"}
        )
        # M1f rebalance: the wide Gaussian (std 0.25) paid 0.9/1.0 for
        # parking 8 cm off command — it funded the m1e one-height collapse.
        # Precision is now the main income; the wide term is only far-field
        # gradient so a fresh policy still feels distant commands.
        self.rewards.track_height = RewTerm(
            func=track_height_exp, weight=2.0,
            params={"std": 0.25, "command_name": "base_height"},
        )
        self.rewards.track_height_fine = RewTerm(
            func=track_height_exp, weight=2.5,
            params={"std": 0.08, "command_name": "base_height"},
        )
        # M1f-v3: sitting was the safe zero (no falls, no gated income
        # anywhere). A small unconditional on-feet bonus makes being on the
        # feet strictly dominate the heap at every height, independent of
        # command accuracy — kills sit-as-optimum without touching height
        # incentives (U-shaped commands still price obedience).
        self.rewards.feet_alive = RewTerm(func=feet_alive, weight=0.3)
        # M1f: start episodes across the whole stand<->kneel depth range
        self.events.spawn_height = EventTerm(
            func=spawn_at_random_height, mode="reset")
        self.rewards.kneel_pose = RewTerm(
            func=kneel_pose_shaping, weight=2.0,
            params={"command_name": "base_height"},
        )
        # randomized right-wrist payload: rising-while-holding trained in
        self.events.hand_payload = EventTerm(
            func=mdp.randomize_rigid_body_mass, mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
                "mass_distribution_params": (0.0, 0.6),
                "operation": "add",
            },
        )
        self.rewards.kneel_drift = RewTerm(
            func=kneel_drift, weight=-9.0,
            params={"command_name": "base_height"},
        )
        # M1g: freeze-hardening for the composition handoff (LOCO_FREEZE=1)
        import os as _os
        if _os.environ.get("LOCO_FREEZE", "0") == "1":
            self.actions.joint_pos.class_type = FreezableJointPositionAction
            self.events.kneel_freeze = EventTerm(
                func=freeze_at_depth, mode="interval",
                interval_range_s=(1.0, 2.5))
        # gentle velocity envelope while the kneel skill forms
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)


@configclass
class G1LocoKneelEnvCfg_PLAY(G1LocoKneelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
