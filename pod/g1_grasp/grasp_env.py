"""Phase 2: G1 + Inspire right hand learns to reach and five-finger grasp a
cylinder from the floor, torso fixed in a crouch. Direct-workflow RL env.

Anti-cheat: the object has only a rigid-body free state; success requires
object lifted with >=3 finger segments + thumb in contact (from the filtered
contact sensors) and nothing else touching it. Reward terms follow
docs/GRASP_SYNTH_FINDINGS.md.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG

LOCAL_USD = "/workspace/assets/G1/g1_29dof_inspire_hand.usd"

# UNIFIED MODE (GRASP_FULLBODY=1, 2026-07-25): one policy owns the whole
# body. Motivated by measurement, not preference: the composed chain's
# approach leg is capped by walker BLINDNESS — v6/v7 runs put arrivals
# in-corridor 12/12 via position servo, yet 88-116 of 200 episodes died to
# the object being kicked, because the frozen walker has no object in its
# observations. No harness can fix a policy that cannot see the thing it
# must not step on. The unified policy sees the object (obj_rel is already
# in this env's obs) and owns its own feet.
FULLBODY = __import__('os').environ.get("GRASP_FULLBODY", "0") == "1"

# REF_TRACK MODE (GRASP_REF_TRACK=1, 2026-07-28): reference-state
# initialization + annealed tracking income over a bank of recorded
# full-task trajectories (see __init__ for the full story). Module-level
# flag mirrors FULLBODY because the observation space gains a 2-dim phase
# clock and cfg sizing happens at import time. Default OFF: with the flag
# unset, no observation, reward, or reset path changes by a single bit.
REF_TRACK = __import__('os').environ.get("GRASP_REF_TRACK", "0") == "1"

# legacy: waist pitch + right arm (7) + right hand (12)
ACTION_JOINTS_ARM_HAND = [
    "waist_pitch_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "R_index_proximal_joint", "R_index_intermediate_joint",
    "R_middle_proximal_joint", "R_middle_intermediate_joint",
    "R_ring_proximal_joint", "R_ring_intermediate_joint",
    "R_pinky_proximal_joint", "R_pinky_intermediate_joint",
    "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint", "R_thumb_distal_joint",
]
# unified: 29 body joints + right hand (12) = 41; left hand stays PD-parked
ACTION_JOINTS_FULL = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "R_index_proximal_joint", "R_index_intermediate_joint",
    "R_middle_proximal_joint", "R_middle_intermediate_joint",
    "R_ring_proximal_joint", "R_ring_intermediate_joint",
    "R_pinky_proximal_joint", "R_pinky_intermediate_joint",
    "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint", "R_thumb_distal_joint",
]
ACTION_JOINTS = ACTION_JOINTS_FULL if FULLBODY else ACTION_JOINTS_ARM_HAND
PALM_BODY = "right_wrist_yaw_link"
FINGER_GROUPS = {
    "index": ["R_index_proximal", "R_index_intermediate"],
    "middle": ["R_middle_proximal", "R_middle_intermediate"],
    "ring": ["R_ring_proximal", "R_ring_intermediate"],
    "pinky": ["R_pinky_proximal", "R_pinky_intermediate"],
    "thumb": ["R_thumb_proximal", "R_thumb_intermediate", "R_thumb_distal"],
}
FINGER_BODIES = [b for grp in FINGER_GROUPS.values() for b in grp]
# distal-most link per finger — the policy's "fingertips"
TIP_BODIES = ["R_index_intermediate", "R_middle_intermediate",
              "R_ring_intermediate", "R_pinky_intermediate", "R_thumb_distal"]

# Round 2: kneel stance (probe_kneel 2026-07-11 — pelvis 0.20 with folded
# legs puts the finger workspace floor at z~0.07, so protocol-spec 10-25 cm
# objects are reachable; round-1 squat bottomed out at z~0.25)
CROUCH_LEGS = {
    ".*_hip_pitch_joint": -2.2, ".*_knee_joint": 2.8, ".*_ankle_pitch_joint": -0.87,
    ".*_hip_roll_joint": 0.4, ".*_hip_yaw_joint": 0.0, ".*_ankle_roll_joint": 0.0,
}

# Backward-chaining curriculum over the mined pose bank (mine_demos.py):
# stage 0 = start holding the object mid-air (learn: don't drop)
# stage 1 = start caged around the object (learn: close + lift)
# stage 2 = start in kneel, object at a bank-verified reachable xy
# stage 3 = full task, object anywhere in the annulus
# Per-env stage advances when that env's recent success EMA clears the bar.
BANK_PATH = "/workspace/pregrasp_bank.pt"
N_STAGES = 4
STAGE_UP_EMA = 0.5
STAGE_DOWN_EMA = 0.02
EMA_ALPHA = 0.05


@configclass
class G1GraspEnvCfg(DirectRLEnvCfg):
    decimation = 4
    # 8 -> 12 s (round 6b): the delta-action policy is deliberate — 41% of
    # eval episodes ran out of clock before the palm arrived. Protocol
    # allows 30 s, so the 8 s limit was self-imposed harshness.
    episode_length_s = 12.0
    action_space = len(ACTION_JOINTS)
    # q, qd, palm, obj_rel, obj_vel, prev_a + TACTILE (round 3): 5 tip-to-object
    # vectors + 5 per-finger contact flags + grasped flag. The round-2 policy
    # was grasping blind — reward paid for contact but the policy could never
    # PERCEIVE it (dominant failures contact_no_grasp/lift_no_hold are exactly
    # blind-hand signatures; cf. 2504.05287 teacher obs = per-link contacts).
    # + ROUND 6: normalized joint targets — with a delta (integrating)
    # action space the target vector is internal state the policy steers, so
    # it must observe it. Layout: q(n) qd(n) palm(3) obj_rel(3) obj_vel(3)
    # prev_a(n) tip_rel(15) flags+grasped(6) tgt_norm(n) = 4n + 30.
    # FULLBODY adds a 9-dim vestibular head (lin vel, ang vel, gravity in
    # base frame) — a walking policy needs balance senses the seated grasp
    # policy never did.
    # + REF_TRACK: a 2-dim phase clock (sin/cos of the reference time
    # index) appended LAST — zeros for non-ref episodes and past the demo's
    # end (see _get_observations note).
    observation_space = (4 * len(ACTION_JOINTS) + 30 + (9 if FULLBODY else 0)
                         + (2 if REF_TRACK else 0))
    state_space = 0
    action_scale = 0.5

    # gpu_max_rigid_patch_count: PhysX contact-patch buffer. The engine
    # default (163,840) overflowed at ~iteration 2500 of Lane V (4096 envs,
    # self-collision on, policy learning to make MORE contact) — "Patch
    # buffer overflow" spams at 164-165k and PhysX SILENTLY DROPS the
    # excess contacts: corrupted grasp flags, phantom slips, the
    # silent-physics-change class this project treats as radioactive
    # (grasped_now slid 0.955 -> 0.633 in the same window, 2026-07-30).
    # 524,288 = 3.2x observed peak. Capacity only — bit-exact for any
    # scene that fits in the old buffer.
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200, render_interval=4,
        physx=sim_utils.PhysxCfg(gpu_max_rigid_patch_count=524288))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=2.5, replicate_physics=True
    )

    robot = G1_INSPIRE_FTP_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # protocol-spec object (matches the mined bank: h=0.15 r=0.03 m=0.25;
    # domain randomization widens per protocol once the skill exists)
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CylinderCfg(
            # easy end of the eval protocol's legal range (r<=45mm, m<=0.8kg):
            # a fat, heavy cylinder resists the tip-over that dominated every
            # thin-object run (30-45k tips/window). Curriculum toward the thin
            # spec later once the skill exists.
            radius=0.045, height=float(__import__('os').environ.get('GRASP_OBJ_H', '0.15')),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                # protocol legal max (μ 0.4-1.2): directly targets the
                # dominant "lift then slip" failure (151/400 fat-object eval)
                static_friction=1.2, dynamic_friction=1.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, -0.15,
                 float(__import__('os').environ.get('GRASP_OBJ_H', '0.15')) / 2)),
    )
    # NOTE: PhysX contact filtering requires the sensor prim to match exactly
    # ONE body per env — a single "R_.*" sensor silently returns an empty
    # force matrix ("Filter pattern did not match the correct number of
    # entries"). This zeroed every contact reward and the success check for
    # ALL of round 1. One sensor per finger body, created in _setup_scene.

    def __post_init__(self):
        import os as _os
        # protocol allows 30 s; the 12 s default is the seated-grasp clock.
        # Unified training sets GRASP_EP_S=30 (walk + descend + grasp + hold).
        self.episode_length_s = float(
            _os.environ.get("GRASP_EP_S", str(self.episode_length_s)))
        # GRASP_FREE=1 (phase 3): free root + honest gravity on links, on
        # the mimic-stripped USD — the composition-ready physics. Default
        # stays the fixed-root rig the r6 checkpoints were trained on.
        free = _os.environ.get("GRASP_FREE", "0") == "1"
        self.robot.spawn.usd_path = (
            "/workspace/assets/G1/g1_29dof_inspire_hand_free.usd" if free
            else LOCAL_USD)
        self.robot.spawn.articulation_props.fix_root_link = not free
        self.robot.spawn.rigid_props.disable_gravity = not free and \
            self.robot.spawn.rigid_props.disable_gravity
        if free:
            # probe_stance2: shipped leg PD cannot hold the deep crouch under
            # gravity — the robot sags and pitches from every spawn height
            # (settled pelvis 0.42-0.81, up 0.4-0.6). Stiffen the parked leg
            # actuators so the stance approximates "a balance controller
            # holds the kneel" (which is what composition will actually do
            # via the LocoKneel policy).
            k = float(_os.environ.get("GRASP_LEG_STIFF", "4.0"))
            for name, act in self.robot.actuators.items():
                jn = act.joint_names_expr
                pats = jn if isinstance(jn, list) else [jn]
                if any(("hip" in p or "knee" in p or "ankle" in p) for p in pats):
                    act.stiffness = {p: (act.stiffness if isinstance(act.stiffness, (int, float))
                                         else 150.0) * k for p in pats} \
                        if not isinstance(act.stiffness, dict) else \
                        {kk: v * k for kk, v in act.stiffness.items()}
                    act.damping = {p: (act.damping if isinstance(act.damping, (int, float))
                                       else 5.0) * k for p in pats} \
                        if not isinstance(act.damping, dict) else \
                        {kk: v * k for kk, v in act.damping.items()}
        # user-flagged 2026-07-11: hand was phasing through the legs — the
        # shipped G1 cfg disables self-collision, which is a cheat vector
        # (interpenetrating fingers could brace the object invisibly)
        self.robot.spawn.articulation_props.enabled_self_collisions = True
        # free mode: 0.24 penetrated the ground with the folded legs — the
        # depenetration kick launched robots to 0.8 m (probe_stance). Spawn
        # with 8 cm of drop clearance and let the stance settle onto feet.
        _spawn_z = float(_os.environ.get("GRASP_SPAWN_Z", "0.32"))
        self.robot.init_state.pos = (0.0, 0.0, _spawn_z if free else 0.20)


class G1GraspEnv(DirectRLEnv):
    cfg: G1GraspEnvCfg

    def __init__(self, cfg: G1GraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.joint_ids, _ = self.robot.find_joints(ACTION_JOINTS, preserve_order=True)
        # affine map of [-1,1] onto each joint's full limit range — a fixed
        # 0.5 rad action_scale left fingers unable to close (probe 2026-07-10)
        limits = self.robot.data.joint_pos_limits[0, self.joint_ids]
        self.act_mid = (limits[:, 0] + limits[:, 1]) / 2
        self.act_half = (limits[:, 1] - limits[:, 0]) / 2
        self.act_low, self.act_high = limits[:, 0], limits[:, 1]
        # ROUND 6 — delta action space. Four rounds of invariant ~17%
        # lift->hold conversion traced to a structural flaw: absolute
        # position targets over the full joint range at 50 Hz mean PPO's
        # exploration noise teleports finger targets by up to half their
        # travel every step — every grip that forms during training is
        # shaken apart by exploration itself, so the value function
        # correctly learns "holds always break" and yank-for-income is the
        # optimal policy. Delta actions make holding the FIXED POINT under
        # noise (zero-mean noise = tremble in place), and "do nothing" —
        # the correct hold policy — becomes trivially expressible.
        self.delta_max = 0.06  # rad per control step (3 rad/s at 50 Hz)
        # FULLBODY: per-group slew — gait needs faster legs (0.12 rad/step =
        # 6 rad/s) than the finger-tuned 0.06; arms/hands keep the r6 value
        self.delta_vec = torch.full((len(self.joint_ids),), self.delta_max,
                                    device=self.device)
        if FULLBODY:
            for _i, _nm in enumerate(ACTION_JOINTS):
                if any(k in _nm for k in ("hip", "knee", "ankle")):
                    self.delta_vec[_i] = 0.12
        # grip-conditioned gearing: precision mode while the grasp is
        # active (60% slower targets), power mode elsewhere — slew-limiting
        # only where it helps, no approach-passivity trap
        self.grasped_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.palm_id = self.robot.find_bodies(PALM_BODY)[0][0]
        self.finger_ids, _ = self.robot.find_bodies(FINGER_BODIES, preserve_order=True)
        self.tip_ids, _ = self.robot.find_bodies(TIP_BODIES, preserve_order=True)
        # bake the crouch into the default joint state (regex defaults in the
        # cfg collide with the asset's catch-all '.*' entry)
        # free-root: the crouch sags under gravity (pelvis 0.217m vs the
        # bolted-root pose), which pushes the 0.40m strict bar out of the arm
        # workspace (eval PEAK_Z p95=0.399). Blend the parked-leg targets
        # toward stand to buy the pelvis height back.
        import os as _os
        blend = float(_os.environ.get("GRASP_LEG_BLEND", "0.0"))
        stand_legs = {
            ".*_hip_pitch_joint": -0.20, ".*_knee_joint": 0.42,
            # hip_roll 0.10 spawned an ASYMMETRIC lateral shear (one hip
            # abducts, the other adducts) — the loco stand pose is 0.0
            ".*_ankle_pitch_joint": -0.23, ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0, ".*_ankle_roll_joint": 0.0,
        }
        for pattern, val in CROUCH_LEGS.items():
            ids, _ = self.robot.find_joints(pattern)
            self.robot.data.default_joint_pos[:, ids] = (
                (1.0 - blend) * val + blend * stand_legs[pattern])
        if blend > 0.0:
            # Standing/chain mode: spawn in the WALKER's stand pose, arms
            # included. Spawning with straight arms (elbow 0) while the walker
            # immediately commands 0.87 snapped them up over the first few
            # timesteps and pitched the robot over — it fell at ~0.9 s at
            # EVERY commanded speed (0.15-0.50), which is a spawn transient,
            # not a gait failure.
            for pattern, val in {".*_elbow_joint": 0.87,
                                 "left_shoulder_pitch_joint": 0.35,
                                 "right_shoulder_pitch_joint": 0.35}.items():
                ids, _ = self.robot.find_joints(pattern)
                self.robot.data.default_joint_pos[:, ids] = val
        self.default_targets = self.robot.data.default_joint_pos[:, self.joint_ids].clone()
        self.targets = self.default_targets.clone()
        self.prev_actions = torch.zeros(self.num_envs, len(self.joint_ids), device=self.device)
        # object half-height = its resting centre z. The G1's arms bottom out
        # near fingertip z 0.21 at the walker's 0.33 kneel (probe_kneel: 0.070
        # at pelvis 0.20, 0.25 at 0.36), so a 15 cm cylinder is unreachable
        # there. The eval protocol allows 10-25 cm; a taller object puts its
        # upper barrel inside the reachable envelope.
        self.obj_h2 = float(_os.environ.get("GRASP_OBJ_H", "0.15")) / 2
        # Reachable radius shrinks as the base rises. The walker's floor is a
        # 0.297 kneel, where zero-shot never_near was 52% — if only the near
        # half of the 0.20-0.35 corridor is reachable there, the mode layer
        # can just stop the walk closer, which costs nothing.
        self.obj_rmin = float(_os.environ.get("GRASP_OBJ_RMIN", "0.20"))
        self.obj_rmax = float(_os.environ.get("GRASP_OBJ_RMAX", "0.35"))
        # sampled corridor coords, kept so the object can be re-placed in the
        # robot's SETTLED base frame — see _replace_object
        self.obj_r = torch.zeros(self.num_envs, device=self.device)
        self.obj_th = torch.zeros(self.num_envs, device=self.device)
        self.hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.high_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # RUN 9S: GRASP_SUCCESS_TERM=<N> terminates the episode when
        # high_counter reaches N (150 = the strict judged event: grasped &
        # obj_z>0.40 sustained 3 s) and pays a one-time lump bonus. This
        # deletes the hold-occupancy income stream that captured runs 4-8
        # (camping in a hold paid ~32/step ≈ V 3,200; batch occupancy went
        # hold-dominated even from 100% entry starts). The r2c/r2d/r2e-era
        # failure of success-termination (hold-below-the-bar-forever) was
        # termination WITHOUT the compensating bonus: V(complete) collapsed
        # below V(camp-sub-bar). The bonus restores destination value
        # (~2,930 vs legacy 3,200 at B=2000) while making the camping-
        # duration gradient exactly zero. Default off: legacy training and
        # every eval keep bit-identical semantics.
        self.success_term = int(_os.environ.get("GRASP_SUCCESS_TERM", "0"))
        self.success_bonus = float(_os.environ.get("GRASP_SUCCESS_BONUS", "2000.0"))
        self.succ_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 9S-b: STREAM the destination value across the 150-step qualifying
        # window instead of a terminal lump. 9S post-mortem (code-verified):
        # rsl_rl normalizes advantages globally per rollout and clips value
        # updates to +-clip_param in ABSOLUTE units, so a 2000-point lump is
        # (a) unlearnable by the critic (0.2/iter toward a 2000 gap) and
        # (b) a permanent advantage outlier that divides every other signal
        # — standing balance included — to ~1e-3 sigma. 9S's policy fell at
        # 0.42 s from every eval spawn by model_1400. Stream 13.3/step while
        # high_counter > 0 (total 1995 over the window, same destination
        # value; max step reward ~46 vs legacy 32 — same order, no outlier)
        # + a ~1-sigma terminal marker via GRASP_SUCCESS_BONUS=300.
        self.success_stream = float(_os.environ.get("GRASP_SUCCESS_STREAM", "0.0"))
        # telemetry blindspots exposed by the 9S post-mortem
        self._qual_count = 0    # above-bar streak starts per window
        self._tip1s_count = 0   # tips younger than 1 s (spawn-collapse tell)
        # 9S-d POSTURE GATE (GRASP_POSTURE_GATE=1): the 9S-c economy was
        # satisfied by a sprawl — the policy lay down and used its own body
        # as a shelf to put the object at 0.40 m (clause-5 audit: 0/200
        # clean, above-knee links at 0.002 m, up 0.07-0.20). Under the gate,
        # the strict counter and ALL elevation/hold income require the
        # certified-feasible upright family (wbik floor pregrasp: pelvis
        # 0.387-0.396, up 0.97+ — thresholds sit far below certified, far
        # above sprawl); sustained above-knee floor-lying terminates the
        # episode (the protocol disqualifies; training now does too);
        # upright itself earns income so the ambient comparison favors
        # standing. Grip formation (r_reach/r_contact) stays ungated.
        self.posture_gate = _os.environ.get("GRASP_POSTURE_GATE", "0") == "1"
        self.gate_up = float(_os.environ.get("GRASP_GATE_UP", "0.5"))
        self.gate_pelv = float(_os.environ.get("GRASP_GATE_PELVIS", "0.35"))
        self.gate_minz = float(_os.environ.get("GRASP_GATE_MINZ", "0.05"))
        # ASYMMETRIC GATE (grasp-wall branch): the INCOME gate may sit lower
        # than the STRICT-counter gate — the dynamic dip of a real grasp
        # transiently crosses the certified-statics pelvis line (0.387
        # static; both A/B lanes stalled at touch-without-grasp with the
        # pelvis conjunct degraded to 0.82-0.92 while working the object).
        # Learn with slack (income at PELVIS_INC), qualify certified-tight
        # (strict counter stays at GATE_PELVIS). Defaults collapse to the
        # symmetric gate.
        self.gate_pelv_inc = float(_os.environ.get("GRASP_GATE_PELVIS_INC",
                                                   str(self.gate_pelv)))
        self.c5_term_n = int(_os.environ.get("GRASP_C5_TERM_STEPS", "25"))
        _names = list(self.robot.data.body_names)
        _c5_allowed = ("foot", "ankle", "knee", "shin", "calf",
                       "wrist", "hand", "palm", "thumb", "index", "middle",
                       "ring", "little", "pinky")
        self.c5_bodies = [i for i, nm in enumerate(_names)
                          if not any(a in nm for a in _c5_allowed)]
        # Lane X waist-fold terms (WAISTFOLD_AUTOPSY_2026-07-30): the crab
        # sprawl satisfied every root-link gate term. GRASP_GATE_TORSO>0
        # additionally requires the TORSO link's up-z above the bar;
        # GRASP_GATE_HANDPLANT=1 marks a hand-chain link that is both near
        # the floor (<0.05m) AND far from the object (>0.25m) as illegal —
        # the gripping hand at floor level next to the object stays legal
        # (the reason the c5 allowlist exempts hands). Defaults off =
        # bit-exact legacy gate.
        self.gate_torso = float(_os.environ.get("GRASP_GATE_TORSO", "0"))
        self.gate_handplant = _os.environ.get("GRASP_GATE_HANDPLANT",
                                              "0") == "1"
        self.rise_w = float(_os.environ.get("GRASP_RISE_W", "0"))
        _torso_ids = [i for i, nm in enumerate(_names) if "torso" in nm]
        self._torso_id = _torso_ids[0] if _torso_ids else 0
        _hand_keys = ("wrist", "hand", "palm", "thumb", "index", "middle",
                      "ring", "little", "pinky")
        self._handchain_ids = [i for i, nm in enumerate(_names)
                               if any(a in nm for a in _hand_keys)]
        if self.gate_torso > 0.0 or self.gate_handplant or self.rise_w > 0.0:
            print(f"[waistfold] torso>{self.gate_torso} "
                  f"({_names[self._torso_id]}) handplant="
                  f"{int(self.gate_handplant)} rise_w={self.rise_w}",
                  flush=True)
        if self.posture_gate:
            print(f"[posture-gate] up>{self.gate_up} pelvis>{self.gate_pelv} "
                  f"minz>{self.gate_minz} c5term@{self.c5_term_n}; "
                  f"links: " + ",".join(_names[i] for i in self.c5_bodies),
                  flush=True)
        self.c5_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._gate_stats = [0, 0, 0, 0, 0]  # samples, ok, up_ok, pelv_ok, minz_ok
        self.hc_grace = int(_os.environ.get("GRASP_HC_GRACE", "0"))
        self._hc_miss = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # GRASP_HC_GRACE_ANNEAL=<steps> (default 0 = off, bit-exact): the
        # effective grace K(t) anneals FROM GRASP_HC_GRACE_START (default
        # 2) TO GRASP_HC_GRACE over ANNEAL common steps —
        #   K(t) = round(lerp(START, GRACE, clamp(t/ANNEAL, 0, 1)))
        # so K=5->2 and K=2->0 curricula are expressible across resumes
        # via the two knobs. Integer, monotone (lerp is monotone in t and
        # round() preserves it; int(round()) uses banker's rounding at the
        # .5 boundaries — documented, still monotone). Interacts with
        # _hc_miss only through the K threshold below — no new buffers.
        self.hc_grace_anneal = float(
            _os.environ.get("GRASP_HC_GRACE_ANNEAL", "0"))
        self.hc_grace_start = int(
            _os.environ.get("GRASP_HC_GRACE_START", "2"))
        if self.hc_grace_anneal > 0:
            print(f"[hc-grace] anneal K {self.hc_grace_start} -> "
                  f"{self.hc_grace} over {self.hc_grace_anneal:.0f} common "
                  f"steps", flush=True)
        # backward-chaining curriculum state (bank absent => curriculum off,
        # e.g. while mine_demos.py is building it)
        import os
        if os.path.exists(BANK_PATH):
            bank = torch.load(BANK_PATH, weights_only=True)
            self.bank_cage_q = bank["joint_pos"].to(self.device)
            self.bank_cage_obj = bank["obj_pos"].to(self.device)
            self.bank_held_q = bank["held_joint_pos"].to(self.device)
            self.bank_held_obj = bank["held_obj_pos"].to(self.device)
            print(f"[curriculum] loaded pose bank: {len(self.bank_cage_q)} entries")
        else:
            self.bank_cage_q = None
            print("[curriculum] no pose bank — full-task starts only")
        self.stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ema = torch.zeros(self.num_envs, device=self.device)
        self.term_counts = [0, 0, 0, 0, 0, 0]  # held/flung/tipped/timeout/succ/c5term
        # eval override: GRASP_FORCE_STAGE=3 -> every episode is a full-task
        # start (bank starts otherwise dominate a fresh env's stage-0 EMA)
        import os as _os
        self.force_stage = int(_os.environ.get("GRASP_FORCE_STAGE", "-1"))
        # Round 5: annealed gravity assist (GRASP_ASSIST=1, training only).
        # Under full gravity a stable hold is a needle in policy space, so
        # every from-scratch policy converged to slap-and-yank before hold
        # gradients ever flowed. Early training lightens the OBJECT (upward
        # force = frac * m * g, frac 0.5 -> 0 over the first ~40% of the
        # run), so marginal grips succeed and holding becomes the native
        # style; gravity then tightens like a curriculum screw. Eval runs
        # WITHOUT assist (default off) — protocol-honest.
        self.free_root = _os.environ.get("GRASP_FREE", "0") == "1"
        # GRASP_FLUNG_REF=robot (chain mode): the training-era flung check
        # (object >1.2 m from ORIGIN) fires at step 1 on every chain episode
        # because the chain legitimately STARTS objects 1.2-1.8 m out — the
        # env then resets every step and the chain measures nothing (0/256
        # with 1-step episodes, 2026-07-25 night; same trap as the walk
        # probe's parked object in STATUS). Robot-referenced flung (>3.0 m
        # from the base) keeps the purpose — a thrown object recycles — and
        # is valid in every phase. Default stays origin/1.2 EXACTLY so no
        # certified eval number moves.
        self.flung_from_robot = _os.environ.get("GRASP_FLUNG_REF", "origin") == "robot"
        # M3 handoff: start episodes from LocoKneel ARRIVAL poses (probe_handoff
        # bank) instead of the hand-authored crouch — the composition chain
        # hands over in whatever kneel the walker actually settles in
        # (pelvis 0.343 vs 0.217, hip_pitch gap up to 1.3 rad).
        self.arrival_jp = None
        _bank = _os.environ.get("GRASP_START_BANK", "")
        if _bank:
            _b = torch.load(_bank, map_location=self.device, weights_only=True)
            _remap = [_b["joint_names"].index(nm)
                      for nm in self.robot.data.joint_names]
            self.arrival_jp = _b["joint_pos"][:, _remap].to(self.device)
            self.arrival_rs = _b["root_state"].to(self.device)
            # frozen-settled banks carry the PD targets that HOLD the pose
            # (measured pose is the equilibrium under those targets, not
            # under itself — holding the measured pose sags)
            self.arrival_tg = (_b["joint_targets"][:, _remap].to(self.device)
                               if "joint_targets" in _b else None)
            # stiffness lock (outside-review finding #2, 2026-07-30): the
            # stored joint_targets are equilibria under a SPECIFIC leg
            # stiffness, same class as the ref bank's baked targets — the
            # guard must cover this path too, not just GRASP_REF_BANK
            if self.arrival_tg is not None:
                from .bank_guard import check_leg_stiff
                check_leg_stiff(_b, _bank,
                                float(_os.environ.get("GRASP_LEG_STIFF",
                                                      "4.0")))
            # fraction of resets drawn from the bank (rest keep the crouch)
            self.arrival_frac = float(_os.environ.get("GRASP_ARRIVAL_FRAC", "1.0"))
        self.park_crouch = _os.environ.get("GRASP_PARK_CROUCH", "0") == "1"
        # SPLIT CONTROL (M3 v2): a frozen LocoKneel policy drives legs/waist/
        # left arm every step while the grasp policy drives right arm + hand.
        # No parked legs, no static-holdability requirement — the walker
        # balances its own kneel throughout the grasp, exactly as it will in
        # the composed chain. Actor MLP is rebuilt from the checkpoint's
        # state dict directly (dims inferred from weights, ELU stack, mean
        # head) so no rsl-rl runner/config coupling exists here.
        self.loco = None
        _lc = _os.environ.get("GRASP_LOCO_CKPT", "")
        if _lc:
            ck = torch.load(_lc, map_location=self.device, weights_only=False)
            sd = ck["model_state_dict"]
            self.loco_layers = []
            i = 0
            while f"actor.{i}.weight" in sd:
                self.loco_layers.append((sd[f"actor.{i}.weight"].to(self.device),
                                         sd[f"actor.{i}.bias"].to(self.device)))
                i += 2  # Linear + ELU pairs; final Linear has no activation
            self.loco = True
            self.loco_hcmd = float(_os.environ.get("GRASP_LOCO_HCMD", "0.32"))
            # The walker's obs (joint_pos - default) and its action write
            # (default + 0.5*a) are BOTH referenced to the pose it trained
            # against — loco_env_cfg init_state. This env overwrites
            # default_joint_pos with CROUCH_LEGS for its own resets/targets,
            # which fed the walker a reference ~2 rad off on every leg joint;
            # the two errors cancel at the crouch, so it looked like a stable
            # kneel while actually being a different controller that cannot
            # walk. Keep the walker's own reference, exactly as trained.
            self.loco_dflt = torch.zeros_like(self.robot.data.default_joint_pos)
            for pattern, val in {
                ".*_hip_pitch_joint": -0.20, ".*_knee_joint": 0.42,
                ".*_ankle_pitch_joint": -0.23, ".*_elbow_joint": 0.87,
                "left_shoulder_pitch_joint": 0.35,
                "right_shoulder_pitch_joint": 0.35,
            }.items():
                _ids, _ = self.robot.find_joints(pattern)
                self.loco_dflt[:, _ids] = val
            body_joints = [
                ".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint",
                ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint", ".*_wrist_yaw_joint",
            ]
            ids = []
            for p in body_joints:
                jids, _ = self.robot.find_joints(p)
                ids += jids
            ids = sorted(ids)
            self.loco_joint_ids = torch.tensor(ids, device=self.device)
            # loco output slots NOT applied: the grasp policy owns these
            grasp_owned = set(int(j) for j in self.joint_ids)
            self.loco_apply_mask = torch.tensor(
                [int(j) not in grasp_owned for j in ids], device=self.device)
            self.loco_prev_act = torch.zeros(
                self.num_envs, len(ids), device=self.device)
            # SETTLE + PARK (M3 v3). Earlier parking attempts latched at the
            # arrival pose (0.342) which is not PD-holdable, so they sagged.
            # The walker settles itself to ~0.239 — inside the holdable deep
            # crouch band — so latch AFTER the settle window instead. Gives
            # the grasp policy the static base it was trained on; the 3s hold
            # no longer has to survive a live balancer carrying the hand.
            self.loco_settle = int(_os.environ.get("GRASP_LOCO_SETTLE", "0"))
            self.loco_park = _os.environ.get("GRASP_LOCO_PARK", "0") == "1"
            self.park_tgt = torch.zeros(
                self.num_envs, len(ids), device=self.device)
            # per-env command/mode tensors — the composed-chain runner mutates
            # these in place to drive walk -> kneel -> settle -> grasp. The
            # settle/park env vars above just script the same tensors.
            self.loco_vel_cmd = torch.zeros(self.num_envs, 3, device=self.device)
            self.loco_hcmd_t = torch.full(
                (self.num_envs,), self.loco_hcmd, device=self.device)
            self.park_now = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device)
            self.suppress_grasp = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device)
            # the chain runner sets this True for WALK/KNEEL and False once
            # the robot is settling in place, ready to hand over
            self.arm_follows_walker = (
                _os.environ.get("GRASP_ARM_FOLLOWS_WALKER", "0") == "1")
            # ...per-ENV, phase-driven (2026-07-25 night): a scalar flag
            # cannot serve 64 envs in different phases — and with it False
            # during WALK the right arm is parked, the walker leans and
            # cannot move (the recorded 0.010 m/s stall; re-hit by the
            # re-aim run: 0/256 standoff arrivals, upright, object never
            # touched). The env var seeds the whole-episode default; the
            # runner drives this tensor per step.
            self.arm_follow_t = torch.full(
                (self.num_envs,), self.arm_follows_walker,
                dtype=torch.bool, device=self.device)
            # GRASP MODE = legs hold the folded crouch. The walker's deepest
            # stable kneel is 0.297 (hcmd sweep 0.24/0.28/0.32 -> .297/.310/
            # .332) but the arms only reach a floor object near 0.217, which
            # is the folded crouch — measured statically PD-holdable (up 0.98,
            # motionless). Ramp the leg targets there over park_ramp steps so
            # the mode switch is not a PD step input.
            self.park_to_crouch = (
                _os.environ.get("GRASP_PARK_CROUCH_TGT", "0") == "1")
            self.park_ramp = float(_os.environ.get("GRASP_PARK_RAMP", "50"))
            self.crouch_tgt = self.loco_dflt[:, self.loco_joint_ids].clone()
            for pattern, val in CROUCH_LEGS.items():
                _cids, _ = self.robot.find_joints(pattern)
                _sel = torch.isin(
                    self.loco_joint_ids,
                    torch.tensor(_cids, device=self.device))
                self.crouch_tgt[:, _sel] = val
            self.park_steps = torch.zeros(self.num_envs, device=self.device)
            # Joints the grasp policy owns that the walker ALSO drives:
            # waist_pitch + the 7 right-arm joints. Freezing them for the
            # whole episode meant walking with one arm dead and no waist
            # pitch — the walker trained with counter-rotating arm swing, so
            # it leaned and could not move (0.010 m/s vs 0.5 commanded).
            # Ownership is phase-dependent: the walker keeps them until the
            # grasp policy is released.
            _loco_list = [int(j) for j in self.loco_joint_ids]
            _og, _ol = [], []
            for _gi, _j in enumerate(self.joint_ids):
                if int(_j) in _loco_list:
                    _og.append(_gi)
                    _ol.append(_loco_list.index(int(_j)))
            self.ov_grasp_idx = torch.tensor(_og, dtype=torch.long,
                                             device=self.device)
            self.ov_loco_idx = torch.tensor(_ol, dtype=torch.long,
                                            device=self.device)
            self.loco_tgt = torch.zeros(
                self.num_envs, len(ids), device=self.device)
        # GRASP_INCOME_SCALE (lane C, pre-registered): hold-side income
        # (lift/hold/success/high/strict) scaled down to shrink the
        # held-vs-far return gap that produced income-capture in BOTH run 4
        # (conservative collapse) and run 5 (nursery-only learning).
        # Ordering preserved; formation terms untouched.
        self.income_scale = float(_os.environ.get("GRASP_INCOME_SCALE", "1.0"))
        # run 8: schedule-aware — full income during phase 1 (entry learning
        # needs the true destination value; lane C proved compression starves
        # nascent skills), treatment active only after unlock (where its job
        # is protecting an EXISTING skill from capture — run 7 showed entry
        # collapse 37->1 within 600 post-unlock iters at full income)
        self.sched_steps = float(_os.environ.get("GRASP_STAGE_SCHED", "0"))
        self.assist = _os.environ.get("GRASP_ASSIST", "0") == "1"
        self.assist_max = 0.5
        self.assist_end = 120_000  # policy steps to fully anneal by
        # UNIFIED MODE state — far-object spawns (the walk leg of the task)
        # and the potential-based approach shaping's memory
        self.far_frac = float(_os.environ.get("GRASP_FAR_FRAC", "0.0"))
        # far-spawn radius band; defaults reproduce the historical 0.8-1.8 m
        self.far_rmin = float(_os.environ.get("GRASP_FAR_RMIN", "0.8"))
        self.far_rspan = float(_os.environ.get("GRASP_FAR_RSPAN", "1.0"))
        if self.far_frac > 0.0:
            print(f"[far-spawn] frac={self.far_frac} radius "
                  f"{self.far_rmin:.2f}-{self.far_rmin + self.far_rspan:.2f} m",
                  flush=True)
        self.prev_base_d = torch.zeros(self.num_envs, device=self.device)
        # bank2 = free-root m2d SUCCESS states (harvest_success.py): cage +
        # held families with joint vel, PD targets, ROOT state and object
        # state — the r2c bank trick, but physics-earned under the exact
        # free-root physics this env trains in. Replaces the fixed-root-era
        # pregrasp bank for stages 0/1 when present.
        self.bank2 = None
        self.bank2g = {0: None, 1: None, 2: None}
        _b2 = _os.environ.get("GRASP_BANK2", "")
        if _b2 and os.path.exists(_b2):
            _raw2 = torch.load(_b2, map_location=self.device, weights_only=True)
            # stiffness lock (outside-review finding #2, 2026-07-30): bank2
            # families carry stored PD targets ("tgt") — equilibria under a
            # specific leg stiffness, the exact silent-sag class the lock
            # exists for. Guard here too, not just GRASP_REF_BANK.
            from .bank_guard import check_leg_stiff
            check_leg_stiff(_raw2, _b2,
                            float(_os.environ.get("GRASP_LEG_STIFF", "4.0")))
            _remap2 = [_raw2["joint_names"].index(nm)
                       for nm in self.robot.data.joint_names]
            self.bank2 = {}
            # run-5: the harvester families widened to the teacher's whole
            # LADDER (harvest_traj.py) — load whatever the file carries
            for _fam in ("cage", "held", "approach", "closing", "lift", "hold"):
                if _fam in _raw2:
                    _d = _raw2[_fam]
                    self.bank2[_fam] = {
                        "jp": _d["jp"][:, _remap2].to(self.device),
                        "jv": _d["jv"][:, _remap2].to(self.device),
                        "tgt": _d["tgt"][:, _remap2].to(self.device),
                        "root": _d["root"].to(self.device),
                        "obj": _d["obj"].to(self.device),
                    }
                    print(f"[bank2] {_fam}: {len(_d['jp'])} states")

            # stage groups (gate-4 conservative-collapse fix): the policy
            # repeatedly WAKES UP mid-crouch/mid-close/mid-lift, so the value
            # of the whole grab motion is experienced, not extrapolated
            def _cat(fams):
                fams = [f for f in fams if f in self.bank2]
                if not fams:
                    return None
                return {k: torch.cat([self.bank2[f][k] for f in fams])
                        for k in ("jp", "jv", "tgt", "root", "obj")}
            self.bank2g = {0: _cat(["hold", "held", "lift"]),
                           1: _cat(["closing", "cage"]),
                           2: _cat(["approach"])}
            for gi, G in self.bank2g.items():
                if G is not None:
                    print(f"[bank2] stage-{gi} group: {len(G['jp'])} states")
        # drop-ET (run 5): a grasp-family start that loses ALL finger contact
        # with the object back on the floor is a dead episode — recycle it
        # instead of grinding hundreds of off-task steps (DeepMimic/ResMimic
        # early-termination pattern)
        self.drop_et = _os.environ.get("GRASP_DROP_ET", "0") == "1"
        self.started_grasped = torch.zeros(self.num_envs, dtype=torch.bool,
                                           device=self.device)
        self.nograsp_count = torch.zeros(self.num_envs, dtype=torch.long,
                                         device=self.device)
        # REF_TRACK (GRASP_REF_TRACK=1): reference-state initialization +
        # annealed tracking income over a bank of recorded full-task
        # trajectories (GRASP_REF_BANK npz: joint_pos/joint_targets
        # [R,T,J_ref], root [R,T,13], obj [R,T,13], phase [R,T],
        # joint_names). The DeepMimic/RSI recipe, already proven here in
        # miniature by the bank2 stage groups: the policy wakes up
        # EVERYWHERE along a working motion, so the value of the whole
        # motion is experienced, not extrapolated — plus a dense "stay on
        # the demonstrated rail" income that anneals linearly to zero over
        # GRASP_REF_ANNEAL common steps, leaving the certified task economy
        # in sole charge. Default OFF: no certified number moves.
        self.ref_track = _os.environ.get("GRASP_REF_TRACK", "0") == "1"
        # buffers exist unconditionally (probes/telemetry may read them);
        # ref_t0 == -1 is the "not a ref episode" sentinel everywhere
        self.ref_id = torch.zeros(self.num_envs, dtype=torch.long,
                                  device=self.device)
        self.ref_t0 = torch.full((self.num_envs,), -1, dtype=torch.long,
                                 device=self.device)
        if self.ref_track:
            import numpy as _np
            _rpath = _os.environ.get("GRASP_REF_BANK",
                                     "/workspace/ref_traj_bank.npz")
            # allow_pickle: bank produced by this project's own harvester on
            # the pod — same trust model as the torch.load banks above
            # (joint_names may be an object array)
            _rz = _np.load(_rpath, allow_pickle=True)
            # BANK STIFFNESS LOCK (2026-07-29): gc banks bake joint_targets
            # gravity-compensated against a specific leg stiffness; nothing
            # here reads tau/kp_eff back (grep-verified), so a stiffness
            # mismatch replays a different physics SILENTLY. Loud
            # RuntimeError on mismatch; prominent warning if unstamped.
            from .bank_guard import check_leg_stiff
            check_leg_stiff(_rz, _rpath,
                            float(_os.environ.get("GRASP_LEG_STIFF", "4.0")))
            _rnames = [x.decode() if isinstance(x, bytes) else str(x)
                       for x in _rz["joint_names"].tolist()]
            _env_names = list(self.robot.data.joint_names)
            # bank column -> this robot's joint index. The bank carries BODY
            # joints only by design — fingers are grip state, not
            # demonstration (env defaults / CLOSE_FING below). A name the
            # robot lacks raises: a bank recorded for a different robot must
            # fail loud, never remap silently.
            self.ref_joint_ids = torch.tensor(
                [_env_names.index(nm) for nm in _rnames],
                dtype=torch.long, device=self.device)
            self.ref_jp = torch.as_tensor(
                _rz["joint_pos"], dtype=torch.float32, device=self.device)
            self.ref_tgt = torch.as_tensor(
                _rz["joint_targets"], dtype=torch.float32, device=self.device)
            self.ref_root = torch.as_tensor(
                _rz["root"], dtype=torch.float32, device=self.device)
            self.ref_obj = torch.as_tensor(
                _rz["obj"], dtype=torch.float32, device=self.device)
            self.ref_phase = torch.as_tensor(
                _rz["phase"], dtype=torch.float32, device=self.device)
            # root velocities ride inside root[..., 7:13] (finite-diffed by
            # the harvester); JOINT velocities only if the file carries them
            # — else zeros at restore (a still start ON the rail, not a
            # falling one)
            self.ref_jv = (torch.as_tensor(_rz["joint_vel"],
                                           dtype=torch.float32,
                                           device=self.device)
                           if "joint_vel" in _rz.files else None)
            self.ref_R = int(self.ref_jp.shape[0])
            self.ref_T = int(self.ref_jp.shape[1])
            self.ref_frac = float(_os.environ.get("GRASP_REF_FRAC", "0.7"))
            # GRASP_REF_T0ZERO=1: force every reference start to t0=0
            # (probe/certificate use — the playback certificate must test
            # the full path from stand, not mid-trajectory aloft restores
            # whose birth credit polluted the first playback's lift stat)
            self.ref_t0zero = _os.environ.get("GRASP_REF_T0ZERO", "0") == "1"
            self.ref_anneal = float(
                _os.environ.get("GRASP_REF_ANNEAL", "60000"))
            self.ref_w = float(_os.environ.get("GRASP_REF_W", "1.0"))
            self.ref_wj = float(_os.environ.get("GRASP_REF_WJ", "8.0"))
            self.ref_wr = float(_os.environ.get("GRASP_REF_WR", "4.0"))
            # GRASP_REF_BACKCURR=<steps> (2026-08-01): BACKWARD CURRICULUM.
            # Reference episodes start at the END of the demonstrated motion
            # (standing, object held — trivially finishable, immediately paid)
            # and the start point marches BACKWARD toward frame 0 over
            # <steps> env-steps, so the policy never has to discover the
            # squat: it only ever extends a motion it can already finish.
            # Why this and not another reward: every postural reward tried
            # (rise ramp, full tracking, base tracking, success gate) was a
            # minority signal arguing with an 85% majority economy that pays
            # for dive-and-grab. This adds no reward at all — it changes
            # WHERE episodes begin, so the reward already being chased is
            # only reachable through the motion we want. 0 = off, bit-exact.
            self.ref_backcurr = float(
                _os.environ.get("GRASP_REF_BACKCURR", "0"))
            self.ref_t0min_final = float(
                _os.environ.get("GRASP_REF_T0MIN_FINAL", "0"))
            # in-hand (phase>2) restores need a formed grip. This file has
            # no native grip-target constant (bank2/arrival starts carry
            # their own stored targets instead), so reuse the scripted-grasp
            # CLOSE_FING convention from mine_demos.py VERBATIM — order is
            # the last 12 ACTION_JOINTS (index/middle/ring/pinky prox+inter,
            # thumb yaw/pitch/inter/distal), identical in both action sets.
            self.ref_grip_ids = torch.tensor(self.joint_ids[-12:],
                                             dtype=torch.long,
                                             device=self.device)
            self.ref_grip_close = torch.tensor(
                [0.9, 0.7, 0.9, 0.7, 0.9, 0.7, 0.9, 0.7, 0.9, 0.4, 0.5, 0.7],
                device=self.device)
            # GRASP_REF_GRIP_POS / GRASP_REF_GRIP_TGT (aloft-restore repair,
            # 2026-07-29): 12 comma-separated values (ACTION_JOINTS[-12:]
            # order) overriding the finger POSE / PD TARGETS used for
            # in-hand (phase>2) reference restores. Measured on the old
            # bank: CLOSE_FING-gripped aloft restores read finger contact at
            # t=1 yet the object free-falls out in ~0.2 s (306/306 tip
            # terminations), while bank2 restores of MEASURED grip pose +
            # MEASURED squeeze targets are statically stable — the sag
            # lesson at the fingers: a grip is the equilibrium under the
            # targets that squeeze it, not under a synthesized pose.
            # Unset (default) -> ref_grip_close for both, bit-exact legacy.
            def _grip12(var, fallback):
                s = _os.environ.get(var, "")
                if not s:
                    return fallback
                v = [float(x) for x in s.split(",")]
                assert len(v) == 12, f"{var} needs 12 values, got {len(v)}"
                return torch.tensor(v, device=self.device)
            self.ref_grip_pose = _grip12("GRASP_REF_GRIP_POS",
                                         self.ref_grip_close)
            self.ref_grip_tgt = _grip12("GRASP_REF_GRIP_TGT",
                                        self.ref_grip_close)
            # telemetry (9S blindspot lesson: instrument BEFORE training):
            # per-window mean RMS joint tracking error over ref envs +
            # strict-success terminations of ref-started envs
            self._ref_err_sum = 0.0
            self._ref_err_cnt = 0
            self._ref_succ_count = 0
            print(f"[ref-track] {_rpath}: R={self.ref_R} T={self.ref_T} "
                  f"joints={len(_rnames)} frac={self.ref_frac} "
                  f"anneal={self.ref_anneal:.0f} w={self.ref_w}", flush=True)
        # MISS-BRANCH #2 (GRASP_PBRS_C, pre-registered): potential-based
        # reward shaping on object height. phi(s) = C * clamp((obj_z - Z0)
        # / (Z1 - Z0), 0, 1), paid as r = gamma*phi(s') - phi(s) (Ng/
        # Harada/Russell 1999 — optimal policy invariant up to the
        # (1-gamma) leak). UNGATED by design: no grasped/legal conjunct —
        # a gated potential is not a potential; the gate flip would mint
        # free energy at the flip. NO terminal clawback: residual phi on
        # any done is simply kept — a -phi terminal lump would be exactly
        # the 9S advantage-outlier poison in the other direction. Camping
        # pays ~0 by construction: a stationary object earns gamma*phi -
        # phi = -(1-gamma)*phi per step (slightly NEGATIVE, never income).
        # Default C=0: fully off — no buffers allocated, no kernels added,
        # reward stream bit-exact (probe_pbrs.py CTRL certifies).
        self.pbrs_c = float(_os.environ.get("GRASP_PBRS_C", "0"))
        if self.pbrs_c != 0.0:
            self.pbrs_z0 = float(_os.environ.get("GRASP_PBRS_Z0", "0.075"))
            self.pbrs_z1 = float(_os.environ.get("GRASP_PBRS_Z1", "0.50"))
            # gamma must be THE PPO discount the trainer actually uses or
            # the telescoping leaks at the wrong rate. Read from the agent
            # cfg (single source of truth: agents.G1GraspPPORunnerCfg.
            # algorithm.gamma); the 0.99 fallback is that cfg's value as
            # of 2026-07-29, for contexts where the import is unavailable.
            try:
                from .agents import G1GraspPPORunnerCfg as _RC
                self.pbrs_gamma = float(_RC().algorithm.gamma)
            except Exception:
                self.pbrs_gamma = 0.99
            self.prev_phi = torch.zeros(self.num_envs, device=self.device)
            # last-step term, exposed for probe_pbrs' certificates — the
            # probe must measure the ENV's own computation, not re-derive
            # the formula beside it
            self._pbrs_last = torch.zeros(self.num_envs, device=self.device)
            print(f"[pbrs] C={self.pbrs_c} Z0={self.pbrs_z0} "
                  f"Z1={self.pbrs_z1} gamma={self.pbrs_gamma}", flush=True)
        # FAMSHARE (GRASP_FAMSHARE=1): per-start-family reward-composition
        # telemetry. Buckets every PAID reward term by the episode's start
        # family (fresh spawn / arrival-bank / REF restore / far spawn) and
        # prints per-window shares next to [curriculum] — so "what fraction
        # does r_walk contribute in far-start episodes" is a measurement,
        # not a belief. Read-only instrument: the returned reward is never
        # touched; default off skips every kernel (bit-exact).
        # ECON V2 (2026-07-30 relaunch spec): consolidated economy +
        # load-bearing grip qualifier. See _get_rewards override block.
        self.econ_v2 = _os.environ.get("GRASP_ECON_V2", "0") == "1"
        self._secure_frac = 0.0
        if self.econ_v2:
            print("[econ-v2] consolidated economy: reach + secure-scaled "
                  "contact + secure elevation ramp + walk x2.5 + alive + "
                  "stream/bonus; grip secure bar 3.0 N", flush=True)
        self.famshare = _os.environ.get("GRASP_FAMSHARE", "0") == "1"
        if self.famshare:
            self._fam_terms = ("reach", "contact", "cage", "walk", "alive",
                               "track", "lift", "hold", "success", "high",
                               "strict", "stream", "bonus", "pbrs",
                               "objvel", "throw", "rate")
            self._fam_names = ("fresh", "arrival", "ref", "far")
            self.start_fam = torch.zeros(self.num_envs, dtype=torch.long,
                                         device=self.device)
            self.ep_ret = torch.zeros(self.num_envs, device=self.device)
            self._fam_sums = torch.zeros(len(self._fam_names),
                                         len(self._fam_terms),
                                         device=self.device)
            self._fam_done_ret = torch.zeros(len(self._fam_names),
                                             device=self.device)
            self._fam_done_n = torch.zeros(len(self._fam_names),
                                           dtype=torch.long,
                                           device=self.device)
            print(f"[famshare] on — families={self._fam_names} "
                  f"terms={len(self._fam_terms)}", flush=True)
        # PROTOCOL SPAWN AXES (2026-07-29 — EVAL_PROTOCOL.md verified
        # firsthand: object distance 1.0-4.0 m, bearing 0-360 deg, robot
        # initial yaw RANDOM, object standing or lying with random yaw; the
        # shipped spawn is protocol-soft on all four axes).
        # GRASP_SPAWN_YAW_MAX=<rad>: fresh-spawn robot yaw ~ U(-max, +max)
        # — the yaw-curriculum knob. GRASP_PROTOCOL_SPAWN=1 (eval intent):
        # fresh spawns at distance U(1,4) m, bearing U(0,2pi), robot yaw
        # U(-pi,pi). Both default off = the spawn path is bit-exact.
        self.spawn_yaw_max = float(_os.environ.get("GRASP_SPAWN_YAW_MAX", "0"))
        self.protocol_spawn = _os.environ.get("GRASP_PROTOCOL_SPAWN", "0") == "1"
        if self.protocol_spawn:
            # flung is origin-referenced by default (>1.2 m) — protocol
            # spawns START 1-4 m out, so every episode would terminate
            # "flung" at step 1 (the one-step-episode trap that zeroed the
            # chain measurements, 2026-07-25). Robot-referenced flung is
            # forced ON with this flag; it is the only semantics under
            # which these spawns are measurable at all.
            self.flung_from_robot = True
            print("[protocol-spawn] ACTIVE: dist U(1.0,4.0) m, bearing "
                  "U(0,2pi), robot yaw U(-pi,pi); flung ref forced to "
                  "robot. lying-object clause NOT implemented — "
                  "STANDING-ONLY (random yaw): evals under this flag must "
                  "not claim full protocol spawn compliance.", flush=True)
        # robot-referenced flung radius. 3.0 was the chain-mode constant;
        # protocol spawns go to 4.0 m, so under GRASP_PROTOCOL_SPAWN the
        # default rises to 4.5 or a third of protocol episodes read
        # flung=True at step 1, before the robot moves (outside-review
        # finding #1, 2026-07-30 — the same one-step-episode trap as the
        # 2026-07-25 chain measurements, one reference frame later).
        # Chain-mode (non-protocol) default stays 3.0: certified numbers
        # keep their exact termination semantics.
        self.flung_robot_dist = float(_os.environ.get(
            "GRASP_FLUNG_ROBOT_DIST",
            "4.5" if self.protocol_spawn else "3.0"))
        if self.spawn_yaw_max > 0.0:
            print(f"[spawn-yaw] fresh-spawn robot yaw ~ U(+-"
                  f"{self.spawn_yaw_max:.3f} rad); bank/arrival/REF "
                  f"restores keep their recorded roots", flush=True)
        # BASEFRAME OBS (GRASP_BASEFRAME_OBS=1): rotate exactly FOUR
        # world-frame obs blocks — palm(3), obj_rel(3), obj lin vel(3),
        # tip_rel(15) — into the base-YAW frame via R_z(-psi). Dims
        # identical, no origin shifts, nothing else touched. This is the
        # audited yaw pathology's fix: with spawn yaw randomized, world-
        # frame task vectors alias per-heading — the same physical scene
        # reads as different observations at every yaw, so the policy must
        # relearn the task once per heading. psi extraction is HYSTERETIC:
        # updated from the ground projection of the base FORWARD axis only
        # while that projection's norm exceeds 0.15 (deep pitch — the
        # ~85-deg grasp hinge — makes yaw gimbal-degenerate and a naive
        # atan2 whips the frame around), held at last-good otherwise;
        # re-initialized from the spawn quaternion at reset, never carried
        # across (the reset-boundary law). Default off = zero kernels,
        # obs bit-exact.
        self.baseframe_obs = _os.environ.get("GRASP_BASEFRAME_OBS", "0") == "1"
        if self.baseframe_obs:
            self.bf_psi = torch.zeros(self.num_envs, device=self.device)
            print("[baseframe-obs] ACTIVE: palm/obj_rel/obj_vel/tip_rel "
                  "-> base-yaw frame; hysteresis at fwd-proj norm 0.15",
                  flush=True)

    def _pbrs_phi(self, obj_z):
        """Miss-branch #2 potential: C * clamp((z - Z0)/(Z1 - Z0), 0, 1).

        Pure function of env-frame object z — callable on live state
        (_get_rewards) and on the restored/spawned z at the reset boundary
        (_reset_idx), which is what keeps the two sides of the lag buffer
        in the same units by construction.
        """
        return self.pbrs_c * ((obj_z - self.pbrs_z0)
                              / (self.pbrs_z1 - self.pbrs_z0)).clamp(0.0, 1.0)

    def _hc_grace_k(self):
        """Effective strict-counter grace K(t) — see the GRACE_ANNEAL note
        in __init__. ANNEAL=0 returns GRASP_HC_GRACE exactly (static path,
        bit-exact with the pre-anneal build); otherwise a monotone integer
        lerp START -> GRACE over ANNEAL common steps."""
        if self.hc_grace_anneal <= 0:
            return self.hc_grace
        f = min(1.0, max(0.0,
                         self.common_step_counter / self.hc_grace_anneal))
        return int(round(self.hc_grace_start
                         + (self.hc_grace - self.hc_grace_start) * f))

    def _loco_step(self):
        """Run the frozen LocoKneel policy one step and write its PD targets
        for the joints the grasp policy does not own."""
        r = self.robot.data
        dflt = self.loco_dflt
        obs = torch.cat([
            r.root_lin_vel_b, r.root_ang_vel_b, r.projected_gravity_b,
            self.loco_vel_cmd,
            r.joint_pos - dflt, r.joint_vel,
            self.loco_prev_act,
            self.loco_hcmd_t.unsqueeze(-1),
        ], dim=-1)
        x = obs
        for k, (w, b) in enumerate(self.loco_layers):
            x = torch.nn.functional.linear(x, w, b)
            if k < len(self.loco_layers) - 1:
                x = torch.nn.functional.elu(x)
        self.loco_prev_act = x
        tgt = dflt[:, self.loco_joint_ids] + 0.5 * x
        if self.loco_park:
            self.park_now = self.episode_length_buf >= self.loco_settle
        # while not latched park_tgt tracks the walker, so freezing it is just
        # "stop updating" — the last live targets keep holding the pose
        hold = self.park_now.unsqueeze(-1)
        self.park_tgt = torch.where(hold, self.park_tgt, tgt)
        tgt = self.park_tgt
        # write ALL 29 body joints; _apply_action runs after this and
        # overwrites the grasp-owned ones once the grasp policy is released
        self.loco_tgt = tgt
        self.robot.set_joint_position_target(
            tgt, joint_ids=self.loco_joint_ids.tolist())

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.obj = RigidObject(self.cfg.object_cfg)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.obj
        # one filtered contact sensor per finger body (see cfg note)
        self.finger_sensors = {}
        # GRASP_SENSOR_CR=1 (run-6 package, needs one 38%-regression eval
        # before trust): update sensors at the CONTROL rate (0.02 s) instead
        # of every physics substep — consumers only read the instantaneous
        # post-substep-4 force, so 3 of 4 updates per step are overwritten
        # unread. Default stays substep-rate: contact numbers are sacred.
        import os as _os2
        _sp = 0.02 if _os2.environ.get("GRASP_SENSOR_CR", "0") == "1" else 0.005
        for body in FINGER_BODIES:
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{body}",
                filter_prim_paths_expr=["/World/envs/env_.*/Object"],
                update_period=_sp,
            ))
            self.finger_sensors[body] = sensor
            self.scene.sensors[f"contact_{body}"] = sensor
        # The walker trained on terrain with friction 1.0/1.0 in "multiply"
        # mode plus a startup event pinning every robot body to 0.8/0.6 —
        # effective 0.8/0.6. A bare GroundPlaneCfg() is 0.5/0.5 "average",
        # so the feet slipped: a static kneel needs no traction and survived,
        # but every push-off slid (0.025 m/s against a 0.5 command, torso
        # pitching into the push). Reproduce the training contact exactly.
        # Ground material is left at the stock default ON PURPOSE. Switching
        # it to 1.0/1.0 "multiply" gave the feet their training traction but
        # also changed OBJECT-ground friction, and the grasp policy fell from
        # 35.5% to 20.5% strict. The feet get their 0.8/0.6 from the leg-body
        # material instead (see _set_body_friction), which no object ever
        # touches, so every contact the 35.5% was measured under is preserved.
        spawn = sim_utils.GroundPlaneCfg()
        spawn.func("/World/ground", spawn)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        light = sim_utils.DomeLightCfg(intensity=2500.0)
        light.func("/World/Light", light)

    def _set_body_friction(self):
        """Pin robot-body friction to the walker's training values (0.8/0.6).

        The loco env did this with a startup randomize_rigid_body_material
        event; a DirectRLEnv has no event manager, so write it once through
        the physx view after the sim is live.

        Applied to the LEG/FOOT bodies only. The training event covered every
        body, but the grasp policy earned its 35.5% under the hand material
        shipped in the USD — rewriting finger friction would silently change
        the physics the judged number was measured under, in either direction.
        """
        view = self.robot.root_physx_view
        mats = view.get_material_properties().clone()
        names = self.robot.body_names
        print(f"BODY_FRICTION_BEFORE static_p50={mats[..., 0].median():.3f} "
              f"dynamic_p50={mats[..., 1].median():.3f} "
              f"shape={tuple(mats.shape)} n_bodies={len(names)}", flush=True)
        # Material rows are per COLLISION SHAPE, not per body (62 vs 54 here),
        # so a body-index write hits the wrong shapes. Count shapes per link
        # the way Isaac Lab's randomize_rigid_body_material does; shapes are
        # contiguous per body.
        sim_view = self.robot._physics_sim_view
        counts = [sim_view.create_rigid_body_view(lp).max_shapes
                  for lp in view.link_paths[0]]
        if sum(counts) != mats.shape[1] or len(counts) != len(names):
            # Cannot target precisely -> change NOTHING. Rewriting every shape
            # raised finger friction from 0.5 to 0.8 and cost the grasp policy
            # 35.5% -> 23.0% strict. Silent global edits to contact physics are
            # exactly the failure this project cannot afford.
            print(f"BODY_FRICTION ABORT shape_map_mismatch sum={sum(counts)} "
                  f"shapes={mats.shape[1]} links={len(counts)} "
                  f"bodies={len(names)} — left untouched", flush=True)
            return
        leg_shapes, start = [], 0
        for i, n in enumerate(names):
            if any(k in n for k in ("hip", "knee", "ankle", "foot")):
                leg_shapes += list(range(start, start + counts[i]))
            start += counts[i]
        # Ground stays at its stock 0.5/0.5 "average", so to land the feet on
        # the training value the leg material must be pre-averaged:
        #   (0.5 + 1.1)/2 = 0.8 static,  (0.5 + 0.7)/2 = 0.6 dynamic
        mats[:, leg_shapes, 0] = 1.1
        mats[:, leg_shapes, 1] = 0.7
        view.set_material_properties(
            mats, torch.arange(self.num_envs, dtype=torch.int32))
        print(f"BODY_FRICTION legs=1.1/0.7 -> foot-ground 0.8/0.6 after "
              f"averaging with the 0.5/0.5 ground, on "
              f"{len(leg_shapes)}/{mats.shape[1]} leg shapes "
              f"(hands and object contacts untouched)", flush=True)

    def _replace_object(self, ids):
        """Put the object in the robot's SETTLED base frame.

        It is spawned at reset relative to the env origin, but the walker then
        settles for ~100 steps and the base translates and yaws while doing it
        — so by handoff the object sat outside the workspace in about half the
        episodes (never_near 104/201, and identical at 104/202 when the
        corridor was narrowed to 0.20-0.25, which is what ruled out reach).
        Re-placing here also matches the composed chain, where the standoff
        servo parks the robot relative to the object rather than the reverse.
        """
        base = self.robot.data.root_pos_w[ids]
        q = self.robot.data.root_quat_w[ids]
        yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        th = self.obj_th[ids] + yaw
        pos = torch.stack([
            base[:, 0] + self.obj_r[ids] * torch.cos(th),
            base[:, 1] + self.obj_r[ids] * torch.sin(th),
            self.scene.env_origins[ids, 2] + self.obj_h2], dim=-1)
        quat = torch.zeros(len(ids), 4, device=self.device)
        quat[:, 0] = 1.0
        self.obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), ids)
        self.obj.write_root_velocity_to_sim(
            torch.zeros(len(ids), 6, device=self.device), ids)

    def _pre_physics_step(self, actions: torch.Tensor):
        if self.loco is not None and self.loco_settle > 0:
            at_end = self.episode_length_buf == self.loco_settle
            if at_end.any():
                self._replace_object(at_end.nonzero(as_tuple=False).squeeze(-1))
        if not getattr(self, "_friction_done", False):
            self._friction_done = True
            # ONLY when a walker is driving the legs. Without one the legs are
            # PD-parked and the 35.5% baseline was measured at stock 0.5/0.5 —
            # touching foot friction there changes the crouch stance and is a
            # confound, not a fix.
            if self.loco is not None:
                self._set_body_friction()
        if self.loco is not None:
            with torch.no_grad():
                self._loco_step()
        self.actions = actions.clamp(-1.0, 1.0)
        if self.loco is not None:
            # arm holds its targets while the base is still moving — reaching
            # during the descent drags the hand off the object
            if self.loco_settle > 0:
                self.suppress_grasp = self.episode_length_buf < self.loco_settle
            self.actions = torch.where(
                self.suppress_grasp.unsqueeze(-1),
                torch.zeros_like(self.actions), self.actions)
            # ...and park them at the WALKER's reference posture, not the
            # grasp defaults. The walker trained on a body with both elbows
            # at 0.87; a straight right arm is a permanent 0.87 rad error in
            # its obs plus asymmetric arm mass. Tracking loco_dflt here also
            # means no target jump when the grasp policy is released.
            sup = self.suppress_grasp.unsqueeze(-1)
            # arm_follow_t True (WALK/KNEEL): the walker needs its right arm
            # to swing or it leans and cannot move. False (SETTLE): hold the
            # GRASP policy's own start pose — releasing it from the walker's
            # stand arm (elbow 0.87) is a pose it never trained from, and its
            # delta integrator starts there; never_near went 38/200 ->
            # 104/201 when follow was applied unconditionally.
            held_walk = self.loco_dflt[:, self.joint_ids].clone()
            if self.ov_grasp_idx.numel():
                held_walk[:, self.ov_grasp_idx] = self.loco_tgt[:, self.ov_loco_idx]
            held = torch.where(self.arm_follow_t.unsqueeze(-1),
                               held_walk, self.default_targets)
            self.targets = torch.where(sup, held, self.targets)
        gear = 1.0 - 0.6 * self.grasped_state.float()
        self.targets = (self.targets
                        + self.actions * self.delta_vec.unsqueeze(0)
                        * gear.unsqueeze(-1)
                        ).clamp(self.act_low, self.act_high)
        if self.assist:
            frac = self.assist_max * max(0.0, 1.0 - self.common_step_counter / self.assist_end)
            f = torch.zeros(self.num_envs, 1, 3, device=self.device)
            f[:, 0, 2] = frac * 0.5 * 9.81  # object mass 0.5 kg
            self.obj.set_external_force_and_torque(f, torch.zeros_like(f))

    def _apply_action(self):
        self.robot.set_joint_position_target(self.targets, joint_ids=self.joint_ids)

    def _contact_flags(self):
        # per-body filtered force vs the object: (envs, 1, 1, 3) each ->
        # (envs, 5) boolean, group order = FINGER_GROUPS (thumb last)
        # PER-STEP MEMO (throughput fork, run-6 package): this executes 3x
        # per control step (dones drop-ET, rewards, obs) against buffers that
        # only change inside sim.step() — the calls are bit-identical by
        # construction, including identically-stale post-reset reads. Cache
        # eliminates ~2/3 of the 36+ small kernel launches per step.
        if getattr(self, "_flags_cache_step", -1) == self.common_step_counter:
            return self._flags_cache
        flags = []
        for g, bodies in FINGER_GROUPS.items():
            hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for b in bodies:
                fm = self.finger_sensors[b].data.force_matrix_w
                hit |= fm.norm(dim=-1).flatten(1).amax(dim=1) > 0.1
            flags.append(hit)
        out = torch.stack(flags, dim=-1)
        self._flags_cache = out
        self._flags_cache_step = self.common_step_counter
        return out

    def _finger_contacts(self):
        flags = self._contact_flags()
        return flags.sum(dim=-1), flags[:, -1]

    def _grip_force(self):
        """Total finger-object contact force (N) — the load-bearing grip
        qualifier for ECON V2. The 0.1 N _contact_flags threshold counts
        grazes; a grip that carries a 0.5 kg object must react ~5 N plus
        squeeze. Cached per control step like _contact_flags."""
        if getattr(self, "_gf_cache_step", -1) == self.common_step_counter:
            return self._gf_cache
        tot = torch.zeros(self.num_envs, device=self.device)
        for b in FINGER_BODIES:
            fm = self.finger_sensors[b].data.force_matrix_w
            tot = tot + fm.norm(dim=-1).flatten(1).amax(dim=1)
        self._gf_cache = tot
        self._gf_cache_step = self.common_step_counter
        return tot

    def _get_observations(self):
        q = self.robot.data.joint_pos[:, self.joint_ids]
        qd = self.robot.data.joint_vel[:, self.joint_ids]
        palm = self.robot.data.body_pos_w[:, self.palm_id] - self.scene.env_origins
        obj = self.obj.data.root_pos_w - self.scene.env_origins
        obj_rel = (self.obj.data.root_pos_w - self.robot.data.body_pos_w[:, self.palm_id])
        # tactile block (appended last — see cfg.observation_space note)
        tips = self.robot.data.body_pos_w[:, self.tip_ids]  # (N, 5, 3)
        tip_rel = (self.obj.data.root_pos_w.unsqueeze(1) - tips).reshape(self.num_envs, 15)
        objv = self.obj.data.root_lin_vel_w
        # BASEFRAME OBS (GRASP_BASEFRAME_OBS=1, see __init__): hysteretic
        # psi update, then R_z(-psi) on exactly the four blocks. Flag off:
        # objv is the same tensor object and no kernel runs — bit-exact.
        if self.baseframe_obs:
            _qb = self.robot.data.root_quat_w
            # base forward axis in world = R(q) @ x_hat; its ground
            # projection carries the yaw. Norm > 0.15 (=sq 0.0225) or hold.
            _bfx = 1.0 - 2.0 * (_qb[:, 2] ** 2 + _qb[:, 3] ** 2)
            _bfy = 2.0 * (_qb[:, 1] * _qb[:, 2] + _qb[:, 0] * _qb[:, 3])
            _bf_ok = (_bfx * _bfx + _bfy * _bfy) > 0.0225
            self.bf_psi = torch.where(_bf_ok, torch.atan2(_bfy, _bfx),
                                      self.bf_psi)
            _bc, _bs = torch.cos(self.bf_psi), torch.sin(self.bf_psi)

            def _rz(v):  # R_z(-psi) @ v for (N, 3): world -> base-yaw
                return torch.stack([_bc * v[:, 0] + _bs * v[:, 1],
                                    -_bs * v[:, 0] + _bc * v[:, 1],
                                    v[:, 2]], dim=-1)
            palm = _rz(palm)
            obj_rel = _rz(obj_rel)
            objv = _rz(objv)
            _t3 = tip_rel.reshape(self.num_envs, 5, 3)
            _bcu, _bsu = _bc.unsqueeze(1), _bs.unsqueeze(1)
            tip_rel = torch.stack([
                _bcu * _t3[..., 0] + _bsu * _t3[..., 1],
                -_bsu * _t3[..., 0] + _bcu * _t3[..., 1],
                _t3[..., 2]], dim=-1).reshape(self.num_envs, 15)
        flags = self._contact_flags()
        grasped = (flags.sum(dim=-1) >= 3) & flags[:, -1]
        tgt_norm = (self.targets - self.act_mid) / self.act_half
        # FULLBODY vestibular head: a walking policy needs balance senses
        # (base-frame velocities + gravity) the seated grasp policy never did
        head = []
        if FULLBODY:
            rd = self.robot.data
            head = [rd.root_lin_vel_b, rd.root_ang_vel_b, rd.projected_gravity_b]
        # REF_TRACK phase clock (+2 dims, appended LAST per the cfg note):
        # under RSI the policy wakes up at arbitrary points along the demo —
        # it must be able to KNOW where, or identical-looking states demand
        # different actions (partial observability by construction). Zeroed
        # for non-ref envs and past-T via the mask, not a clamp: sin/cos
        # never hits (0,0) on the circle, so "no reference" cannot alias
        # any live phase; a clamped clock would alias "end of demo" with
        # "just past the end".
        tail = []
        if self.ref_track:
            _pt = self.ref_t0 + self.episode_length_buf
            _pv = ((self.ref_t0 >= 0) & (_pt < self.ref_T)).float().unsqueeze(-1)
            _pa = 2.0 * torch.pi * _pt.float() / float(self.ref_T)
            tail = [torch.stack([torch.sin(_pa), torch.cos(_pa)], dim=-1) * _pv]
        obs = torch.cat(
            head + [q, qd, palm, obj_rel, objv,
                    self.prev_actions, tip_rel, flags.float(),
                    grasped.float().unsqueeze(-1), tgt_norm] + tail,
            dim=-1,
        )
        self.prev_actions = self.actions.clone()
        return {"policy": obs}

    def _get_rewards(self):
        palm = self.robot.data.body_pos_w[:, self.palm_id]
        obj = self.obj.data.root_pos_w
        d = (palm - obj).norm(dim=-1)
        n_fingers, thumb = self._finger_contacts()

        obj_z = obj[:, 2] - self.scene.env_origins[:, 2]
        lifted = obj_z > 0.22
        grasped = (n_fingers >= 3) & thumb
        self.grasped_state = grasped  # drives next step's action gearing

        # reach pays on FINGERTIP proximity, not palm (round 3): palm-based
        # reach was satisfiable with an open or misplaced hand; per-tip
        # distance makes wrapping the barrel itself the shaded objective
        # (2504.05287 weights fingertip distance 4x other links). Subsumes
        # the old r_pocket centroid shaping. Gated fresh like r_pocket was —
        # once contact exists, contact/hold income takes over (anti-camping).
        tips = self.robot.data.body_pos_w[:, self.tip_ids]
        d_tips = (tips - obj.unsqueeze(1)).norm(dim=-1).mean(dim=-1)
        fresh = (n_fingers == 0) & (self.hold_counter == 0)
        decay = (1.0 - self.episode_length_buf.float() / 200.0).clamp(min=0.2)
        r_reach = (0.15 * torch.exp(-4.0 * d)
                   + 0.5 * torch.exp(-8.0 * d_tips) * fresh.float() * decay)
        r_contact = 0.4 * n_fingers.float() + 0.6 * thumb.float()
        # HOLD_DIAG post-mortem (r3, 2026-07-14): median grip-break speed
        # 2.8 m/s, median true-hold streak 0 — "lifts" were throws. Cause:
        # lift income started at FIRST touch, so touch-and-yank out-earned
        # settle-then-grip. Restructured: token income for touch-lift, the
        # real payout requires the full grasp; plus r_cage pays for the
        # skipped step (stable floor-level cage before lifting).
        any_contact = n_fingers >= 1
        # saturation at 0.325m taught a hover-below-the-bar exploit under free
        # root (strict 0/400 at 67% loose): full income below the 0.40m strict
        # bar, zero marginal payout for crossing it. Ramp now tops out at 0.45m
        # and holding above the bar pays its own income stream.
        h = (obj_z - self.obj_h2).clamp(0.0, 0.375) / 0.375
        r_lift = 6.0 * h * (0.3 * any_contact.float() + 1.7 * grasped.float())
        # binary gate at 0.42 was never visited (PEAK_Z frac>=0.42 stuck at
        # 0.005 after 1100 iters) — dead gradient. Dense ramp starts at 0.36,
        # inside the policy's actual peak distribution (p50 0.380), so the
        # pull toward the bar is collected immediately.
        r_high = 6.0 * ((obj_z - 0.36) / 0.09).clamp(0.0, 1.0) * grasped.float()
        # sustained time above the strict bar is the judged metric — pay the
        # streak itself, ramping over the 3 s (150-step) requirement
        # posture legality (9S-d gate): computed every step; `legal` is 1.0
        # everywhere when the gate is off so legacy semantics are bit-exact
        if self.posture_gate:
            _q_pg = self.robot.data.root_quat_w
            _up_pg = 1.0 - 2.0 * (_q_pg[:, 1] ** 2 + _q_pg[:, 2] ** 2)
            _pelv_pg = (self.robot.data.root_pos_w[:, 2]
                        - self.scene.env_origins[:, 2])
            _minz_pg = (self.robot.data.body_pos_w[:, self.c5_bodies, 2]
                        - self.scene.env_origins[:, 2:3]).min(dim=1).values
            _ok_up = _up_pg > self.gate_up
            _ok_pelv = _pelv_pg > self.gate_pelv
            _ok_minz = _minz_pg > self.gate_minz
            # waistfold terms (lane X): torso-link uprightness + no
            # load-bearing hand plant. Off (all-ones) by default.
            if self.gate_torso > 0.0:
                _tq_wf = self.robot.data.body_quat_w[:, self._torso_id]
                _ok_torso = (1.0 - 2.0 * (_tq_wf[:, 1] ** 2
                                          + _tq_wf[:, 2] ** 2)
                             ) > self.gate_torso
            else:
                _ok_torso = torch.ones_like(_ok_up)
            if self.gate_handplant:
                _hp = self.robot.data.body_pos_w[:, self._handchain_ids]
                _hp_low = (_hp[..., 2]
                           - self.scene.env_origins[:, 2:3]) < 0.05
                _hp_far = (_hp - self.obj.data.root_pos_w[:, None, :]
                           ).norm(dim=-1) > 0.25
                # scoped to lifted contexts (ref-mode cert 2026-07-30:
                # 28.4% of legit DESCENT frames have a hand near the
                # floor; a strut is only evidence while the object is up)
                _ok_plant = ~((_hp_low & _hp_far).any(dim=1)
                              & (obj_z > 0.22))
            else:
                _ok_plant = torch.ones_like(_ok_up)
            posture_ok = (_ok_up & _ok_pelv & _ok_minz
                          & _ok_torso & _ok_plant)
            # income legality uses the (possibly lower) PELVIS_INC bar
            legal = (_ok_up & (_pelv_pg > self.gate_pelv_inc)
                     & _ok_minz & _ok_torso & _ok_plant).float()
            self._gate_stats[0] += self.num_envs
            self._gate_stats[1] += int(posture_ok.sum())
            self._gate_stats[2] += int(_ok_up.sum())
            self._gate_stats[3] += int(_ok_pelv.sum())
            self._gate_stats[4] += int(_ok_minz.sum())
            self.c5_counter = torch.where(
                _minz_pg < 0.03, self.c5_counter + 1,
                torch.zeros_like(self.c5_counter))
        else:
            posture_ok = torch.ones(self.num_envs, dtype=torch.bool,
                                    device=self.device)
            legal = torch.ones(self.num_envs, device=self.device)
        # GRASP_HC_GRACE=K (training-side only; eval stays protocol-exact):
        # the strict counter tolerates up to K consecutive failing steps
        # before resetting — a 0.1 s bobble no longer guillotines a 3 s
        # streak. Measured basis: the learned grasp posture PASSES the gate
        # (up p5=0.60 > 0.5) yet succ=0 over 4,400 iters with lifts
        # flickering in eval — the streak dies to transient grip/height
        # dips during the raise, not to posture. K=0 (default) = legacy
        # instant reset, bit-exact.
        _hc_ok = grasped & (obj_z > 0.40) & posture_ok
        # GRACE_ANNEAL: K is time-scheduled when GRASP_HC_GRACE_ANNEAL>0
        # (see __init__/_hc_grace_k); with ANNEAL=0 this IS self.hc_grace
        # and both the gate and the threshold below are bit-exact legacy.
        # An annealed K that reaches 0 falls through to the legacy instant-
        # reset branch — identical semantics to static K=0 by construction.
        _hc_k = self._hc_grace_k()
        if _hc_k > 0:
            self._hc_miss = torch.where(
                _hc_ok, torch.zeros_like(self._hc_miss), self._hc_miss + 1)
            _hc_keep = self._hc_miss <= _hc_k
            self.high_counter = torch.where(
                _hc_ok, self.high_counter + 1,
                torch.where(_hc_keep, self.high_counter,
                            torch.zeros_like(self.high_counter)))
        else:
            self.high_counter = torch.where(
                _hc_ok, self.high_counter + 1,
                torch.zeros_like(self.high_counter))
        self._qual_count += int((self.high_counter == 1).sum())
        r_strict = 3.0 * (self.high_counter.float() / 150.0).clamp(max=1.5)
        obj_speed = self.obj.data.root_lin_vel_w.norm(dim=-1)
        r_cage = 1.2 * (grasped & ~lifted & (obj_speed < 0.2)).float()
        # per-step while in the success state; episodes no longer terminate
        # on success, so keep the per-step scale sane over long holds
        r_success = 5.0 * (lifted & grasped).float()
        # object momentum priced in 3D at all times (was: horizontal-only and
        # only below lift height — a vertical yank was completely free, and
        # yanking was the measured dominant behavior). 0.3 m/s deadband keeps
        # a gentle deliberate carry unpenalized.
        p_objvel = 1.5 * (obj_speed - 0.3).clamp(min=0.0)
        # throwing exploit (~40% of terminations): airborne object moving
        # fast with no grasp = it was launched, not carried
        p_throw = (0.5 * self.obj.data.root_lin_vel_w.norm(dim=-1)
                   * ((obj_z > 0.22) & ~grasped).float())
        p_rate = 0.02 * (self.actions - self.prev_actions).square().sum(dim=-1)

        self.hold_counter = torch.where(
            lifted & grasped, self.hold_counter + 1, torch.zeros_like(self.hold_counter)
        )
        # persistence pays: eval showed grips form and lift but collapse <1s —
        # make sustained holding itself the gradient, ramping over the 1s bar
        r_hold = 3.0 * (self.hold_counter.float() / 50.0).clamp(max=1.5)
        # UNIFIED MODE — exactly two added terms, frozen at launch:
        # r_walk: potential-based shaping on base->object distance. Telescopes
        #   over the episode, pays only PROGRESS — no camping optimum exists
        #   (the r_reach gradient is numerically dead beyond ~1 m: exp(-4d)).
        # r_alive: dense "stay up" income so early falls price themselves
        #   before the value function understands termination.
        r_walk = torch.zeros_like(r_hold)
        r_alive = torch.zeros_like(r_hold)
        if FULLBODY:
            based = (self.robot.data.root_pos_w[:, :2] - obj[:, :2]).norm(dim=-1)
            r_walk = 8.0 * (self.prev_base_d - based).clamp(-0.05, 0.05)
            self.prev_base_d = based
            q_rr = self.robot.data.root_quat_w
            up_rr = 1.0 - 2.0 * (q_rr[:, 1] ** 2 + q_rr[:, 2] ** 2)
            r_alive = 0.05 * (up_rr > 0.7).float()
        # REF_TRACK income (GRASP_REF_TRACK=1): stay on the demonstrated
        # rail. Joint term over the remapped BODY joints only (fingers are
        # grip state, not demonstration); root term pays position AND
        # attitude (quat-dot^2 is double-cover safe). Anneals linearly to
        # zero by GRASP_REF_ANNEAL common steps — scaffolding, not economy.
        # Max magnitude 8 + 4 = 12/step at w=1, deliberately the same order
        # as the task's dense terms, and NO lump anywhere in this term: the
        # 9S post-mortem (code-verified) showed a single 2000-point outlier
        # divides every other advantage to ~1e-3 sigma under rsl_rl's global
        # advantage normalization and is unlearnable through +-clip_param
        # value clipping. Streams only. Added UNSCALED next to the nursery
        # terms — existing income (_sc/legal gating included) untouched.
        r_track = torch.zeros_like(r_hold)
        if self.ref_track:
            _rm = self.ref_t0 >= 0
            if _rm.any():
                _rw = _rm.nonzero(as_tuple=False).squeeze(-1)
                _ri = self.ref_id[_rw]
                # same clock the phase obs uses: t = t0 + steps since reset,
                # clamped at the last frame (the demo ends; holding its
                # final pose keeps paying — no cliff at T)
                _ti = (self.ref_t0[_rw]
                       + self.episode_length_buf[_rw]).clamp(max=self.ref_T - 1)
                _qerr = (self.robot.data.joint_pos[_rw][:, self.ref_joint_ids]
                         - self.ref_jp[_ri, _ti]).square().mean(dim=-1)
                _rref = self.ref_root[_ri, _ti]
                _perr = (self.robot.data.root_pos_w[_rw]
                         - self.scene.env_origins[_rw]
                         - _rref[:, :3]).square().sum(dim=-1)
                _qdot = (self.robot.data.root_quat_w[_rw]
                         * _rref[:, 3:7]).sum(dim=-1)
                _wt = min(1.0, max(0.0, 1.0 - self.common_step_counter
                                   / max(self.ref_anneal, 1.0)))
                # GRASP_REF_WJ / GRASP_REF_WR (2026-07-31): split the joint
                # and root halves of the tracking term. Defaults 8.0/4.0 are
                # bit-exact with the historical form.
                # Rationale: the literature's working configuration for
                # crouch-to-stand-while-carrying is a BASE-POSITION tracking
                # term, not full joint mimicry. This project's own record
                # says why — the optimizer references are statically
                # certified and FAILED every dynamic playback, so their
                # joint angles are a good description of where the pelvis
                # belongs and a bad description of how to hold an object
                # with a compliant five-finger hand. WJ=0 keeps the rise
                # signal (pelvis 0.442 -> 0.600) while leaving the learned
                # grasp posture alone.
                r_track[_rw] = _wt * self.ref_w * (
                    self.ref_wj * torch.exp(-2.0 * _qerr)
                    + self.ref_wr * torch.exp(-10.0 * _perr)
                    * torch.exp(-5.0 * (1.0 - _qdot ** 2)))
                # telemetry: mean over ref envs of per-env RMS joint error
                self._ref_err_sum += float(_qerr.sqrt().mean())
                self._ref_err_cnt += 1
        if self.common_step_counter % 2400 == 0:
            s = self.get_curriculum_stats()
            tc = self.term_counts
            af = (self.assist_max * max(0.0, 1.0 - self.common_step_counter / self.assist_end)
                  if self.assist else 0.0)
            _campbb = int(((self.hold_counter >= 300)
                           & (self.high_counter == 0)).sum())
            _gs = self._gate_stats
            _gate_str = ""
            if self.posture_gate and _gs[0] > 0:
                _gate_str = (f" gate_ok={_gs[1]/_gs[0]:.2f}"
                             f" (up={_gs[2]/_gs[0]:.2f}"
                             f" pelv={_gs[3]/_gs[0]:.2f}"
                             f" minz={_gs[4]/_gs[0]:.2f})")
            # REF_TRACK telemetry: mean RMS joint tracking error over ref
            # envs this window + strict-success terminations of ref-started
            # envs (the RSI cohort's conversion rate)
            _ref_str = ""
            if self.ref_track:
                _te = (self._ref_err_sum / self._ref_err_cnt
                       if self._ref_err_cnt else float("nan"))
                _ref_str = (f" track_err={_te:.3f}"
                            f" ref_succ={self._ref_succ_count}")
            print(f"[curriculum] step {self.common_step_counter} " +
                  " ".join(f"{k}={v:.2f}" for k, v in s.items()) +
                  f" grasped_now={grasped.float().mean():.3f} assist={af:.2f}"
                  f" term(held/flung/tip/timeout/succ/c5)="
                  f"{tc[0]}/{tc[1]}/{tc[2]}/{tc[3]}/{tc[4]}/{tc[5]}"
                  f" qual={self._qual_count} tip1s={self._tip1s_count}"
                  f" campBB={_campbb}" + _gate_str + _ref_str)
            self.term_counts = [0, 0, 0, 0, 0, 0]
            self._qual_count = 0
            self._tip1s_count = 0
            self._gate_stats = [0, 0, 0, 0, 0]
            if self.ref_track:
                self._ref_err_sum = 0.0
                self._ref_err_cnt = 0
                self._ref_succ_count = 0
            # FAMSHARE window print: one line per family that saw traffic.
            # shares are signed fractions of the family's TOTAL paid reward
            # (penalties come out negative); ret is the mean undiscounted
            # return over episodes that ENDED this window.
            if self.famshare:
                for _fi, _fn in enumerate(self._fam_names):
                    _n_f = int(self._fam_done_n[_fi])
                    _sums = self._fam_sums[_fi]
                    if _n_f == 0 and float(_sums.abs().sum()) == 0.0:
                        continue
                    _tot = float(_sums.sum())
                    _den = abs(_tot) if abs(_tot) > 1e-9 else 1.0
                    _sh = " ".join(
                        f"{_k}={float(_v) / _den:.3f}"
                        for _k, _v in zip(self._fam_terms, _sums))
                    print(f"[famshare] fam={_fn} n={_n_f} "
                          f"ret={float(self._fam_done_ret[_fi]) / max(_n_f, 1):.1f} "
                          f"shares: {_sh}", flush=True)
                self._fam_sums.zero_()
                self._fam_done_ret.zero_()
                self._fam_done_n.zero_()
        _sc = 1.0 if (self.sched_steps > 0
                      and self.common_step_counter < self.sched_steps) \
            else self.income_scale
        # 9S-d: ALL elevation/hold income requires legal posture (r_cage
        # included — a sprawled cage is the camp we must not pay); the
        # nursery terms r_reach/r_contact stay ungated. Upright itself
        # earns 0.45/step so the ambient comparison favors standing even
        # before task income flows. `legal` is all-ones when the gate is
        # off — legacy return bit-exact.
        # ECON V2 (GRASP_ECON_V2=1, 2026-07-30 relaunch spec §A4/B6): the
        # consolidated <=6-term economy, implemented as OVERRIDES of the
        # legacy term values so the assembly line and famshare stack below
        # are shared verbatim (zeroed terms just read 0 in famshare).
        # Default off: every branch below is skipped, legacy bit-exact.
        # Design (research pass, 3-0 verified): single-critic summing at
        # ~30:1 is a documented interference mechanism; V2 collapses the
        # elevation family (lift/high/cage) into ONE secure-grip-gated ramp,
        # drops the redundant streak shadows (hold/strict/success — the
        # stream IS the success economy), rescales the walk term toward
        # parity, and pays contact income at half rate unless the grip is
        # LOAD-BEARING (B6: the 0.1 N flag counts grazes-that-tip as
        # "grasped" — model_14400 film). Peak ambient hold-side ~8.6/step
        # vs walk ~1.0 — ~8:1, down from ~30:1.
        if self.econ_v2:
            secure = grasped & (self._grip_force() > 3.0)
            self._secure_frac = float(secure.float().mean())
            r_contact = ((0.4 * n_fingers.float() + 0.6 * thumb.float())
                         * (0.5 + 0.5 * secure.float()))
            r_lift = 6.0 * h * secure.float()      # consolidated elevation
            _z = torch.zeros_like(r_lift)
            r_cage, r_high, r_hold, r_success, r_strict = _z, _z, _z, _z, _z
            r_walk = r_walk * 2.5                  # 8.0 -> 20.0 gain
            # rise ramp: pays carrying the object toward standing pelvis
            # height. Onset 0.36 sits just BELOW the posture gate's floor
            # (GATE_PELVIS 0.35 / _INC 0.37) so a gradient exists everywhere
            # the gate permits standing at all.
            #
            # Onset was 0.43 through lane X/Y/Yb and paid ZERO: strict-hold
            # pelvis_z ran 0.374-0.434 (p75 below onset at every checkpoint),
            # so the term fired on 4.2% of recorded frames with mean value
            # 0.0020 against a 6.0 elevation ramp — noise, not an incentive,
            # and the policy correctly sank toward the gate floor. Repaired
            # after tools/cert_rise_activation.py (RISE_CERT_PASS: old dead
            # on 95.8% of the OBSERVED distribution, new nonzero and
            # monotone across the legal band). Standing law from this bug:
            # every reward term ships with an activation certificate against
            # the observed state distribution, never the intended one.
            if self.rise_w > 0.0:
                _pz_rise = (self.robot.data.root_pos_w[:, 2]
                            - self.scene.env_origins[:, 2])
                r_hold = r_hold + (self.rise_w
                                   * ((_pz_rise - 0.36) / 0.26).clamp(0.0, 1.0)
                                   * secure.float())
        if self.posture_gate:
            r_cage = r_cage * legal
            r_alive = r_alive + 0.45 * legal
        # success economy (both unscaled by _sc — they ARE the destination
        # value the income schedule exists to protect): the qualifying
        # stream pays densely while the strict streak runs (9S-b: no
        # advantage outlier), the bonus on the termination step (9S-b uses
        # ~300, a marker; the 9S-era 2000 lump is the documented poison)
        rew = (r_reach + r_contact + r_cage + r_walk + r_alive + r_track
               + _sc * legal * (r_lift + r_hold + r_success + r_high + r_strict)
               + self.success_stream * (self.high_counter > 0).float()
               + self.success_bonus * self.succ_done.float()
               - p_objvel - p_throw - p_rate)
        # GRASP_POSTURE_ONLY=1 (2026-08-01): train a PURE POSTURE EXPERT.
        # Every task reward is removed — reach, contact, elevation, hold,
        # stream, bonus — leaving only reference tracking, alive income and
        # the action-rate penalty. The objective becomes exactly "reproduce
        # the demonstrated squat-and-stand and don't fall", with no object
        # income to compete with it.
        # Why: five attempts to add posture to a grasp policy failed, all
        # of them minority signals losing to an 85% economy that pays for
        # dive-and-grab. This removes the competition instead of trying to
        # outbid it, and produces the SECOND EXPERT that mcp_compose.py
        # needs — a mixture of one expert is that expert.
        # Default off = every term above is untouched = bit-exact.
        if __import__("os").environ.get("GRASP_POSTURE_ONLY", "0") == "1":
            rew = r_track + r_alive - p_rate
        # GRASP_WALK_ONLY=1 (2026-08-01): train a PURE APPROACH EXPERT on
        # the second GPU, in parallel with the posture expert. Only walk
        # shaping + alive income; no object reward at all, so the policy
        # cannot short-circuit by diving at the cylinder. Pairs with
        # FAR_FRAC=1.0 (every episode starts at walking distance) and, for a
        # from-scratch run, with observation normalization ACTUALLY enabled
        # — the defect diagnosed 2026-07-31 that a warm-started lane could
        # never test, because banked weights expect raw inputs.
        if __import__("os").environ.get("GRASP_WALK_ONLY", "0") == "1":
            rew = r_walk + r_alive - p_rate
        # MISS-BRANCH #2 PBRS (see __init__): lagged-buffer form — pay
        # gamma*phi(now) - prev_phi, then advance the buffer. Ordering
        # note: for envs terminating THIS step, this reward still reads the
        # pre-reset state (dones -> rewards -> resets in DirectRLEnv), and
        # _reset_idx then re-seeds prev_phi from the restored object, so no
        # value ever crosses the reset boundary. Guarded ADD, not a
        # zeros-add: the C=0 path must stay BIT-EXACT with zero extra
        # kernels (probe_pbrs.py CTRL certificate).
        if self.pbrs_c != 0.0:
            _phi_now = self._pbrs_phi(obj_z)
            self._pbrs_last = self.pbrs_gamma * _phi_now - self.prev_phi
            self.prev_phi = _phi_now
            rew = rew + self._pbrs_last
        # FAMSHARE accumulation (GRASP_FAMSHARE=1): PAID per-term values —
        # gated/scaled exactly as the return above pays them (r_cage and
        # r_alive already carry their `legal` gating by this point) —
        # bucketed by each env's start family. Pure reads; `rew` is
        # untouched, so flag-off AND flag-on leave training bit-identical.
        if self.famshare:
            _fam_pb = (self._pbrs_last if self.pbrs_c != 0.0
                       else torch.zeros_like(rew))
            _fam_t = torch.stack([
                r_reach, r_contact, r_cage, r_walk, r_alive, r_track,
                _sc * legal * r_lift, _sc * legal * r_hold,
                _sc * legal * r_success, _sc * legal * r_high,
                _sc * legal * r_strict,
                self.success_stream * (self.high_counter > 0).float(),
                self.success_bonus * self.succ_done.float(),
                _fam_pb, -p_objvel, -p_throw, -p_rate,
            ], dim=-1)
            self._fam_sums.index_add_(0, self.start_fam, _fam_t)
            self.ep_ret += rew
        return rew

    def _get_dones(self):
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        obj = self.obj.data.root_pos_w - self.scene.env_origins
        if self.flung_from_robot:  # chain mode — see __init__ note
            flung = ((self.obj.data.root_pos_w[:, :2]
                      - self.robot.data.root_pos_w[:, :2]).norm(dim=-1)
                     > self.flung_robot_dist)
        else:
            flung = obj[:, :2].norm(dim=-1) > 1.2
        # success is RECORDED at 1 s continuous hold but (legacy default) does
        # NOT terminate: r2c/r2d/r2e ended episodes on success WITHOUT a
        # compensating bonus — that cut off ~18/step hold income, so the
        # policy learned to hold just below the bar forever. RUN 9S
        # (GRASP_SUCCESS_TERM>0) terminates on the 3 s STRICT event instead,
        # WITH a lump bonus sized to keep destination value neutral — see
        # __init__ note. succ_done is computed here (dones run before rewards
        # in DirectRLEnv) and consumed by _get_rewards the same step; it reads
        # high_counter as of the previous reward pass (1-step lag, harmless),
        # and _reset_idx zeroes high_counter so a fresh episode cannot
        # inherit the event.
        if self.success_term > 0:
            self.succ_done = self.high_counter >= self.success_term
        else:
            self.succ_done = torch.zeros_like(self.succ_done)
        held = self.hold_counter == 50  # crossing event, telemetry only
        # tipped cylinders are below the reachable workspace: dead episode,
        # recycle immediately so experience concentrates on acquire-and-lift
        q = self.obj.data.root_quat_w
        up_z = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)  # world-z of object z-axis
        obj_z = obj[:, 2]
        tipped = (up_z < 0.7071) & (obj_z < self.obj_h2 + 0.045)  # >45 deg, near ground
        # free-root mode: the robot must keep its own balance — a collapsed
        # kneel (pelvis sunk or torso pitched into the floor) ends the episode
        if self.free_root:
            pelvis_z = (self.robot.data.root_pos_w[:, 2]
                        - self.scene.env_origins[:, 2])
            q_r = self.robot.data.root_quat_w
            up_r = 1.0 - 2.0 * (q_r[:, 1] ** 2 + q_r[:, 2] ** 2)
            # FULLBODY: 0.10 -> 0.07. m2d's real hold posture (braced
            # kneel-lean) keeps the pelvis at 0.11-0.14; the zero-action
            # probe showed restored holds are STATICALLY STABLE (grasped
            # 0.83+ for 5 s) yet ~90% got executed at birth by a momentary
            # dip under 0.10. A fall line must catch collapse, not the
            # policy's working altitude. Legacy evals keep 0.10 exactly.
            fallen = (pelvis_z < (0.07 if FULLBODY else 0.10)) | (up_r < 0.0)
            tipped = tipped | fallen
        # drop-ET (run 5): grasp-family starts that lose ALL finger contact
        # with the object back on the floor for 0.5 s are dead episodes
        if self.drop_et:
            nf_d, _ = self._finger_contacts()
            below_d = obj[:, 2] < 0.22
            self.nograsp_count = torch.where(
                self.started_grasped & (nf_d == 0) & below_d
                & (self.episode_length_buf > 10),
                self.nograsp_count + 1, torch.zeros_like(self.nograsp_count))
            tipped = tipped | (self.nograsp_count >= 25)
        # termination-reason telemetry (accumulated, printed with curriculum)
        self.term_counts[0] += int(held.sum())
        self.term_counts[1] += int(flung.sum())
        # 9S-d: sustained above-knee floor-lying disqualifies the episode
        # (0.03 bar with a 0.5 s grace window; income bar is 0.05 —
        # hysteresis keeps "not paying" and "disqualified" separate)
        c5term = self.c5_counter >= self.c5_term_n if self.posture_gate \
            else torch.zeros_like(self.succ_done)
        self.term_counts[2] += int((tipped & ~held & ~flung & ~self.succ_done
                                    & ~c5term).sum())
        self.term_counts[3] += int((truncated & ~flung & ~tipped
                                    & ~self.succ_done & ~c5term).sum())
        self.term_counts[4] += int(self.succ_done.sum())
        self.term_counts[5] += int((c5term & ~self.succ_done).sum())
        self._tip1s_count += int((tipped & ~held & ~flung & ~self.succ_done
                                  & (self.episode_length_buf < 50)).sum())
        # REF_TRACK telemetry: strict-success terminations of REF-started
        # envs — counted here (dones run before rewards, and _reset_idx
        # clears ref_t0 after) exactly like the _qual_count pattern
        if self.ref_track:
            self._ref_succ_count += int(
                (self.succ_done & (self.ref_t0 >= 0)).sum())
        return flung | tipped | self.succ_done | c5term, truncated

    def _reset_idx(self, env_ids):
        # Stage sampling: fixed stochastic mixture. The adaptive EMA
        # graduation (bar 0.5, then any bar above the policy's success rate)
        # froze at stage 0 for an entire run and the policy never trained
        # approach -> 0/200 full-task. A fixed mixture cannot stall.
        # mixture shifted toward approach/full-task now that hold and
        # close-and-lift are learned (25/15/25/35)
        roll = torch.rand(len(env_ids), device=self.device)
        # GRASP_STAGE_SCHED=<phase1_steps> (run 7, WoCoCo-lite staging —
        # the project's own origin recipe promoted to a schedule): during
        # phase 1 ONLY entry-family starts exist (stand-near/approach s2 +
        # corridor full-task s3, no far, no hold/closing starts) so contact
        # is the summit income with nothing competing; phase 2 restores the
        # full ladder. Four flat mixtures failed on entry; every published
        # from-scratch success on this task class staged its skills.
        import os as _os_s
        _sched = float(_os_s.environ.get("GRASP_STAGE_SCHED", "0"))
        self._in_phase1 = _sched > 0 and self.common_step_counter < _sched
        if self._in_phase1:
            self.stage[env_ids] = (roll > 0.70).long() + 2  # 70% s2 / 30% s3
        elif FULLBODY:
            if _os_s.environ.get("GRASP_P2_ENTRY", "0") == "1":
                # run 8: phase-2 keeps ENTRY exposure dominant (40% s2) so
                # the phase-1 skill keeps earning after the ladder unlocks —
                # run 7's phase 2 dropped stand-near to ~14% and lost it
                self.stage[env_ids] = (
                    (roll > 0.15).long() + (roll > 0.30).long()
                    + (roll > 0.70).long())
            else:
                # run-5 mixture: 20% hold/lift, 25% closing, 20% approach,
                # 35% full-task
                self.stage[env_ids] = (
                    (roll > 0.20).long() + (roll > 0.45).long()
                    + (roll > 0.65).long())
        else:
            self.stage[env_ids] = (
                (roll > 0.25).long() + (roll > 0.40).long() + (roll > 0.65).long())

        # FAMSHARE: episode length BEFORE super() zeroes it — separates
        # real episode ends from the startup reset (which would otherwise
        # log num_envs fake zero-return episodes into the first window)
        if self.famshare:
            self._fam_prelen = self.episode_length_buf[env_ids].clone()
        super()._reset_idx(env_ids)
        n = len(env_ids)
        if self.force_stage >= 0:
            self.stage[env_ids] = self.force_stage
        stage = self.stage[env_ids]
        dev = self.device

        # base: default pose (kneel legacy / stand under LEG_BLEND) + noise
        jp = self.robot.data.default_joint_pos[env_ids].clone()
        jp[:, self.joint_ids] += 0.05 * (torch.rand(n, len(self.joint_ids), device=dev) - 0.5)
        jv = torch.zeros_like(jp)  # bank2 held states carry real velocities
        # object: full-task region fitted to the mined bank's reachable xy
        # (r 0.20-0.37, bearing -1.1..-0.40 = front-right corridor; spawns
        # outside this are outside the seated-reach workspace and would only
        # teach hover-avoidance again)
        r = self.obj_rmin + (self.obj_rmax - self.obj_rmin) * torch.rand(n, device=dev)
        th = (-1.05 + 0.60 * torch.rand(n, device=dev))
        # PROTOCOL SPAWN (GRASP_PROTOCOL_SPAWN=1): the protocol's spawn law
        # replaces the trained corridor for FRESH spawns — distance
        # U(1,4) m, full-circle bearing (robot yaw handled below). Bank /
        # arrival / REF restores overwrite pos later and are unaffected.
        # The far-spawn block below is skipped under this flag (protocol
        # distances subsume it; letting far re-narrow to 0.8-1.8 m would
        # silently un-protocol a fraction of episodes).
        if self.protocol_spawn:
            r = 1.0 + 3.0 * torch.rand(n, device=dev)
            th = 6.283185307179586 * torch.rand(n, device=dev)
        # UNIFIED MODE far spawns: a fraction of full-task episodes put the
        # object at walking distance in the forward arc — the walk leg of the
        # task, trained in the same episodes as everything else
        far = torch.zeros(n, dtype=torch.bool, device=dev)
        if self.far_frac > 0.0 and not getattr(self, "_in_phase1", False) \
                and not self.protocol_spawn:
            far = (stage == 3) & (torch.rand(n, device=dev) < self.far_frac)
            if far.any():
                m = int(far.sum())
                # GRASP_FAR_RMIN/RSPAN (2026-07-31): far-spawn radius made
                # configurable for the survivability probe. Defaults 0.8/1.0
                # reproduce the historical 0.8-1.8 m band exactly, so unset
                # is bit-exact. Rationale: laneW2 died in a handful of
                # updates when far starts fed base->object distance at
                # 0.8-1.8 m into an UNNORMALIZED network that had only ever
                # seen ~0.2-0.5 m (2-4x raw-feature excursion) — grasp
                # capability went 81/100 -> 0/101, contact_no_grasp 100.
                # The probe asks whether a 1.6x excursion (0.6-0.8 m) is
                # survivable at all before any walk claim is entertained.
                r[far] = (self.far_rmin
                          + self.far_rspan * torch.rand(m, device=dev))
                th[far] = -0.6 + 1.2 * torch.rand(m, device=dev)
        self.obj_r[env_ids] = r
        self.obj_th[env_ids] = th
        pos = torch.stack([r * torch.cos(th), r * torch.sin(th),
                           torch.full((n,), self.obj_h2, device=dev)], dim=-1)

        # root FIRST — bank2 states must restore it (free root: a held-in-air
        # state without its root is a different, falling state)
        root = self.robot.data.default_root_state[env_ids].clone()

        # stage-specific overrides: bank2 (free-root m2d successes, complete
        # state) preferred; the fixed-root-era mined bank is the fallback
        have_old = self.bank_cage_q is not None
        have_b2 = self.bank2 is not None and len(self.bank2) > 0
        if not have_old and not have_b2:
            stage = torch.full_like(stage, N_STAGES - 1)
        elif not have_old and self.bank2g.get(2) is None:
            # s2 needs the old bank OR an approach group; fold into full-task
            stage = torch.where(stage == 2, torch.full_like(stage, 3), stage)
        s0, s1, s2 = stage == 0, stage == 1, stage == 2
        # drop-ET bookkeeping resets BEFORE the stage loop marks grasp starts
        self.started_grasped[env_ids] = False
        self.nograsp_count[env_ids] = 0
        b2mask = torch.zeros(n, dtype=torch.bool, device=dev)
        b2quat = torch.zeros(n, 4, device=dev)
        b2vel = torch.zeros(n, 6, device=dev)
        b2tgt = torch.zeros_like(jp)
        # fixed-root-era bank starts (fallback branches below) — tracked so
        # the spawn-yaw block can exempt them like every other restore
        oldbank = torch.zeros(n, dtype=torch.bool, device=dev)
        # stage-group restores (run 5): s0 = hold/lift, s1 = closing/cage,
        # s2 = approach — FULL state restore (root+vel+targets+object) from
        # the teacher ladder; grasp-family fallbacks stay within bank2 (the
        # fixed-root-era bank has no root state — a kneel spawned at the
        # standing default root z is a 0.4 m physics shock)
        for gi, mask in ((0, s0), (1, s1), (2, s2)):
            if not mask.any():
                continue
            G = self.bank2g.get(gi) if have_b2 else None
            if G is None and gi in (0, 1) and have_b2:
                G = self.bank2g.get(1 - gi) or self.bank2g.get(0)
            if G is not None:
                k = torch.randint(len(G["jp"]), (int(mask.sum()),), device=dev)
                jp[mask] = G["jp"][k]
                jv[mask] = G["jv"][k]
                root[mask] = G["root"][k]
                pos[mask] = G["obj"][k, :3]
                b2quat[mask] = G["obj"][k, 3:7]
                b2vel[mask] = G["obj"][k, 7:13]
                b2tgt[mask] = G["tgt"][k]
                b2mask |= mask
                if gi in (0, 1):  # woke up in (or entering) a grasp
                    self.started_grasped[env_ids[mask]] = True
            elif gi in (0, 1) and have_old:
                oldbank |= mask
                pk = torch.randint(len(self.bank_cage_q), (int(mask.sum()),), device=dev)
                if gi == 0:
                    jp[mask] = self.bank_held_q[pk]
                    pos[mask] = self.bank_held_obj[pk]
                else:
                    jp[mask] = self.bank_cage_q[pk]
                    pos[mask] = self.bank_cage_obj[pk]
            elif gi == 2 and have_old:
                oldbank |= mask
                # legacy: object at a bank-verified reachable xy, robot default
                pk2 = torch.randint(len(self.bank_cage_q), (int(mask.sum()),), device=dev)
                pos[mask] = self.bank_cage_obj[pk2]
                pos[mask, :2] += 0.03 * (torch.rand(int(mask.sum()), 2, device=dev) - 0.5)
                pos[mask, 2] = self.obj_h2
        # GRASP_S2_STAND_FRAC (run-6, gate-A curve fix): run 5's mixture
        # taught the grab but never the ENTRY — no episodes began standing
        # near the object, and the task eval starts exactly there (three
        # 0/100 rolling evals at 1400-1800 vs nursery grasped 0.9). This
        # fraction of s2 draws reverts to the plain standing-near start:
        # robot at its standing defaults, object in the corridor.
        import os as _os_r
        s2_stand = float(_os_r.environ.get("GRASP_S2_STAND_FRAC", "0.0"))
        if s2_stand > 0.0 and s2.any():
            back = s2 & (torch.rand(n, device=dev) < s2_stand)
            if back.any():
                m = int(back.sum())
                jp[back] = self.robot.data.default_joint_pos[env_ids][back] \
                    + 0.05 * (torch.rand(m, jp.shape[1], device=dev) - 0.5)
                jv[back] = 0.0
                root[back] = self.robot.data.default_root_state[env_ids][back]
                rb = self.obj_rmin + (self.obj_rmax - self.obj_rmin) \
                    * torch.rand(m, device=dev)
                tb = (-1.05 + 0.60 * torch.rand(m, device=dev))
                pos[back, 0] = rb * torch.cos(tb)
                pos[back, 1] = rb * torch.sin(tb)
                pos[back, 2] = self.obj_h2
                b2mask[back] = False
                self.started_grasped[env_ids[back]] = False
        # arrival-pose starts (M3): joint pose + root z/attitude from the
        # LocoKneel handoff bank; yaw removed so the object corridor stays in
        # the robot's frame. Full-task (stage 3) episodes only.
        if self.arrival_jp is not None:
            # far spawns walk from a STAND — arrival kneels are for in-corridor
            use = (stage == 3) & ~far & (torch.rand(n, device=dev) < self.arrival_frac)
            if use.any():
                bpick = torch.randint(len(self.arrival_jp), (int(use.sum()),), device=dev)
                bjp = self.arrival_jp[bpick]
                # frozen-settled banks are PD equilibria — spawn exactly on
                # them (noise would re-introduce the settle transient)
                noise = 0.0 if self.arrival_tg is not None else 0.03
                jp[use] = bjp + noise * (torch.rand_like(bjp) - 0.5)
                brs = self.arrival_rs[bpick]
                root[use, 2] = brs[:, 2]
                q = brs[:, 3:7]
                yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                                  1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
                c, s = torch.cos(-yaw / 2), torch.sin(-yaw / 2)
                # q_new = q_yawinv * q  (w,x,y,z convention)
                w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                root[use, 3] = c * w - s * z
                root[use, 4] = c * x - s * y
                root[use, 5] = c * y + s * x
                root[use, 6] = c * z + s * w
                root[use, 7:] = 0.0
        # REF_TRACK starts (GRASP_REF_TRACK=1): GRASP_REF_FRAC of full-task
        # (stage-3, in-corridor — the arrival starts' mask family) resets
        # wake up ON a reference trajectory at a uniform frame. FULL-state
        # restore per the reset-boundary law (five instruments died to
        # partial restores): root incl. its finite-diff velocities FIRST,
        # then joints, then PD targets (below, after the tgt assembly), then
        # the OBJECT's own pose/velocity — a held-aloft object without its
        # velocity is a different, falling object. Placed AFTER the arrival
        # block on purpose: a reset that is both arrival- and ref-selected
        # is a ref start (the ref frame carries its own root state).
        refm = torch.zeros(n, dtype=torch.bool, device=dev)
        if self.ref_track:
            self.ref_t0[env_ids] = -1  # sentinel: non-ref episode
            refm = (stage == 3) & ~far \
                & (torch.rand(n, device=dev) < self.ref_frac)
        if refm.any():
            rrows = refm.nonzero(as_tuple=False).squeeze(-1)
            rr = torch.randint(self.ref_R, (len(rrows),), device=dev)
            tt = torch.randint(self.ref_T, (len(rrows),), device=dev)
            if self.ref_t0zero:
                tt = torch.zeros_like(tt)
            # GRASP_REF_T0MIN: restrict start indices to [T0MIN, T) — e.g.
            # 100 = P3+ only, giving posture-legal ALOFT starts (object in
            # hand above the floor) that teach the raise-and-hold segment
            # by backward chaining. With GRASP_REF_W=0 these are pure RSI
            # starts: no tracking income, just the certified state family.
            _t0min = int(__import__("os").environ.get("GRASP_REF_T0MIN", "0"))
            # BACKWARD CURRICULUM (GRASP_REF_BACKCURR>0): override the fixed
            # floor with one that marches from near the LAST frame back to
            # ref_t0min_final as training proceeds. Early episodes start
            # standing-with-object (finishable now); later ones start
            # progressively earlier, ending at the full descent.
            if self.ref_backcurr > 0:
                _frac = min(1.0, self.common_step_counter / self.ref_backcurr)
                _hi = float(self.ref_T - 10)
                _t0min = int(_hi + (self.ref_t0min_final - _hi) * _frac)
                _t0min = max(0, min(_t0min, self.ref_T - 2))
                self._backcurr_t0min = _t0min
            if _t0min > 0:
                tt = _t0min + torch.randint(
                    max(1, self.ref_T - _t0min), (len(rrows),), device=dev)
            root[refm] = self.ref_root[rr, tt]
            # body joints from the bank; missing joints (fingers) keep the
            # env defaults already in jp — the bank is body-only by design
            jp[rrows.unsqueeze(-1), self.ref_joint_ids.unsqueeze(0)] = \
                self.ref_jp[rr, tt]
            jv[refm] = 0.0  # explicit: zeros unless the file carries joint_vel
            if self.ref_jv is not None:
                jv[rrows.unsqueeze(-1), self.ref_joint_ids.unsqueeze(0)] = \
                    self.ref_jv[rr, tt]
            pos[refm] = self.ref_obj[rr, tt, :3]
            # phase <= 2: object still at floor rest — pose only, zero
            # velocity. phase > 2: in-hand aloft — pose AND velocity (see
            # law above), and the hand must actually HOLD it: finger POSE
            # goes to the closed grip too, because an open hand around an
            # in-hand object interpenetrates and the depenetration kick
            # launches it (the probe_stance spawn lesson, at the hand).
            aloft = self.ref_phase[rr, tt] > 2.0
            if aloft.any():
                jp[rrows[aloft].unsqueeze(-1),
                   self.ref_grip_ids.unsqueeze(0)] = self.ref_grip_pose
            self.ref_id[env_ids[rrows]] = rr
            self.ref_t0[env_ids[rrows]] = tt
        # FAMSHARE (GRASP_FAMSHARE=1): close the ledger of the episodes
        # that just ENDED under their recorded family, then record the new
        # family. Taxonomy is fresh/arrival/ref/far ONLY (bank2 stage
        # starts count as fresh — the question this instrument exists to
        # answer is the far-start r_walk share); ref beats arrival exactly
        # like the restore override order above; far is exclusive of both
        # by construction of their masks.
        if self.famshare:
            _lived = self._fam_prelen > 0
            if _lived.any():
                _fam_old = self.start_fam[env_ids]
                self._fam_done_ret.index_add_(
                    0, _fam_old[_lived], self.ep_ret[env_ids][_lived])
                self._fam_done_n.index_add_(
                    0, _fam_old[_lived],
                    torch.ones(int(_lived.sum()), dtype=torch.long,
                               device=dev))
            self.ep_ret[env_ids] = 0.0
            _famv = torch.zeros(n, dtype=torch.long, device=dev)
            if self.arrival_jp is not None:
                _famv[use] = 1
            _famv[refm] = 2
            _famv[far] = 3
            self.start_fam[env_ids] = _famv
        # SPAWN YAW (GRASP_SPAWN_YAW_MAX, and U(-pi,pi) under
        # GRASP_PROTOCOL_SPAWN): rotate the FRESH-spawn root about world z.
        # Restores are exempt BY LAW, not convenience: bank2/REF frames are
        # full-state equilibria in their recorded frame — yawing the root
        # without co-rotating object pose AND every velocity vector is a
        # different, broken state (the reset-boundary bug family again) —
        # and arrival starts already re-yaw themselves to keep the object
        # corridor in the robot frame. Fresh spawns place the object in the
        # ENV frame, so yawing only the robot is exactly the protocol's
        # "robot initial yaw random relative to object bearing".
        _yawmax = 3.141592653589793 if self.protocol_spawn \
            else self.spawn_yaw_max
        if _yawmax > 0.0:
            fresh_root = ~b2mask & ~refm & ~oldbank
            if self.arrival_jp is not None:
                fresh_root &= ~use
            if fresh_root.any():
                _psi = (torch.rand(n, device=dev) * 2.0 - 1.0) * _yawmax
                _cy, _sy = torch.cos(_psi / 2), torch.sin(_psi / 2)
                _q0 = root[:, 3:7].clone()
                _qw, _qx = _q0[:, 0], _q0[:, 1]
                _qy, _qz = _q0[:, 2], _q0[:, 3]
                # q_new = q_yaw(psi) * q0 (w,x,y,z) — the arrival block's
                # composition with the sign flipped (adding yaw, not
                # removing it). World-z rotation leaves roll/pitch (the
                # up-axis z component) untouched by construction; the
                # spawn cert asserts it anyway.
                _qn = torch.stack([
                    _cy * _qw - _sy * _qz, _cy * _qx - _sy * _qy,
                    _cy * _qy + _sy * _qx, _cy * _qz + _sy * _qw], dim=-1)
                root[fresh_root, 3:7] = _qn[fresh_root]
        # BASEFRAME OBS: psi re-initialized from the SPAWN quaternion for
        # exactly these envs — never carried across a reset (a held
        # pre-reset psi would rotate the first post-reset obs into a stale
        # frame: the reset-boundary bug family, sixth time radioactive).
        # Same forward-axis extraction as the live path; runs AFTER the
        # spawn-yaw block above so a yawed fresh spawn seeds its own frame.
        if self.baseframe_obs:
            _qbr = root[:, 3:7]
            self.bf_psi[env_ids] = torch.atan2(
                2.0 * (_qbr[:, 1] * _qbr[:, 2] + _qbr[:, 0] * _qbr[:, 3]),
                1.0 - 2.0 * (_qbr[:, 2] ** 2 + _qbr[:, 3] ** 2))
        rxy_env = root[:, :2].clone()  # env-frame base xy, for r_walk's memory
        root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
        # set PD TARGETS for every joint to the reset pose. Only action
        # joints get targets each step — the parked legs/hands otherwise pull
        # toward the actuator built-in defaults (straight legs). Invisible
        # with a bolted pelvis and gravity-free links; under free-root
        # gravity the robot unfolded out of the crouch (probe_stance4).
        # GRASP_PARK_CROUCH=1 (M3 handoff mechanic): spawn in the arrival pose
        # but set the parked-joint PD targets to the static crouch — the
        # arrival kneel is only dynamically stable under the loco policy, so
        # the robot must sink into the crouch it can actually hold.
        tgt = jp
        if self.arrival_jp is not None and self.park_crouch:
            tgt = jp.clone()
            dflt = self.robot.data.default_joint_pos[env_ids]
            non_action = [i for i in range(tgt.shape[1])
                          if i not in set(self.joint_ids)]
            tgt[:, non_action] = dflt[:, non_action]
        # frozen-settled banks: the pose is the equilibrium UNDER its stored
        # targets — hold those, not the pose itself (holding the measured
        # pose removes the preload and the stance sags)
        if self.arrival_jp is not None and self.arrival_tg is not None and use.any():
            tgt = tgt.clone() if tgt is jp else tgt
            tgt[use] = self.arrival_tg[bpick]
        # bank2 states are equilibria under their STORED targets (the
        # frozen-settled-bank lesson: holding the measured pose sags)
        if b2mask.any():
            tgt = tgt.clone() if tgt is jp else tgt
            tgt[b2mask] = b2tgt[b2mask]
        # REF_TRACK: ref frames are equilibria under their RECORDED targets
        # (same frozen-settled-bank lesson — holding the measured pose
        # sags). Non-bank joints (fingers) get env defaults on floor-rest
        # phases and the CLOSE_FING grip when in-hand.
        if refm.any():
            tgt = tgt.clone() if tgt is jp else tgt
            _rdflt = self.robot.data.default_joint_pos[env_ids]
            tgt[refm] = _rdflt[refm]
            tgt[rrows.unsqueeze(-1), self.ref_joint_ids.unsqueeze(0)] = \
                self.ref_tgt[rr, tt]
            if aloft.any():
                tgt[rrows[aloft].unsqueeze(-1),
                    self.ref_grip_ids.unsqueeze(0)] = self.ref_grip_tgt
        self.robot.set_joint_position_target(tgt, env_ids=env_ids)

        yaw = torch.rand(n, device=dev) * 6.283
        quat = torch.zeros(n, 4, device=dev)
        quat[:, 0] = torch.cos(yaw / 2)
        quat[:, 3] = torch.sin(yaw / 2)
        # held starts keep the mined orientation upright (yaw irrelevant for
        # a cylinder); zero velocity everywhere — except bank2 states, which
        # restore the object's HARVESTED orientation and velocity
        if b2mask.any():
            quat[b2mask] = b2quat[b2mask]
        # REF_TRACK: the recorded object attitude, both phases (a floor
        # cylinder's roll/tilt matters to the grasp the demo performed)
        if refm.any():
            quat[refm] = self.ref_obj[rr, tt, 3:7]
        state = torch.cat([pos + self.scene.env_origins[env_ids], quat,
                           torch.zeros(n, 6, device=dev)], dim=-1)
        if b2mask.any():
            state[b2mask, 7:] = b2vel[b2mask]
        # REF_TRACK: in-hand (phase>2) restores carry the object's recorded
        # velocity; floor-rest restores keep the zeros set above
        if refm.any() and aloft.any():
            state[rrows[aloft], 7:] = self.ref_obj[rr[aloft], tt[aloft], 7:13]
        self.obj.write_root_pose_to_sim(state[:, :7], env_ids)
        self.obj.write_root_velocity_to_sim(state[:, 7:], env_ids)

        # MISS-BRANCH #2 PBRS reset boundary (the recurrent bug family —
        # SIX instruments in this project died to reads across dones; treat
        # as radioactive): re-seed prev_phi from the object state AS
        # RESTORED for exactly these envs, AFTER the object write above.
        # pos[:, 2] is the finalized env-frame z for EVERY start family —
        # fresh corridor spawns, s2-stand reverts, bank2 grasp-family
        # restores and REF_TRACK aloft frames all overwrite pos before this
        # point. Carrying pre-reset phi across the boundary would pay a
        # +-phi-scale lump on the first post-reset step (probe_pbrs
        # KA-reset certifies the absence of that spike).
        if self.pbrs_c != 0.0:
            self.prev_phi[env_ids] = self._pbrs_phi(pos[:, 2])

        self.hold_counter[env_ids] = 0
        self.high_counter[env_ids] = 0
        self.c5_counter[env_ids] = 0
        self._hc_miss[env_ids] = 0
        self.prev_actions[env_ids] = 0.0
        if self.loco is not None:
            self.loco_prev_act[env_ids] = 0.0
            self.loco_vel_cmd[env_ids] = 0.0
            self.loco_hcmd_t[env_ids] = self.loco_hcmd
            self.park_now[env_ids] = False
            self.suppress_grasp[env_ids] = False
        self.grasped_state[env_ids] = False
        # delta integrator starts from the actual post-reset pose
        # delta integrator starts from the PD targets actually applied — for
        # frozen-settled bank starts these differ from the measured pose
        self.targets[env_ids] = tgt[:, self.joint_ids]
        # r_walk potential memory: the true post-reset base->object distance
        self.prev_base_d[env_ids] = (pos[:, :2] - rxy_env).norm(dim=-1)

    def get_curriculum_stats(self):
        return {f"stage_{k}": float((self.stage == k).float().mean())
                for k in range(N_STAGES)}
