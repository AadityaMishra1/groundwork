"""G1 locomotion + commanded base height (crouch) — Phase 1 extension.

Extends the in-tree G1 flat velocity task with a resampled base-height
command, its observation, and an exponential tracking reward, so one policy
walks AND holds commanded pelvis heights down to a deep crouch (HOMIE-style).
"""

from __future__ import annotations

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import (
    CommandTerm,
    CommandTermCfg,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
)


class HeightCommand(CommandTerm):
    """Uniformly resampled base-height target, shape (num_envs, 1)."""

    cfg: "HeightCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.height_command

    def _update_metrics(self):
        self.metrics["height_error"] = torch.abs(
            self.height_command[:, 0] - self.robot.data.root_pos_w[:, 2]
        )

    def _resample_command(self, env_ids):
        r = torch.empty(len(env_ids), device=self.device)
        self.height_command[env_ids, 0] = r.uniform_(*self.cfg.height_range)

    def _update_command(self):
        pass


@configclass
class HeightCommandCfg(CommandTermCfg):
    class_type: type = HeightCommand
    asset_name: str = "robot"
    # G1 stands at ~0.74 m pelvis height; 0.35 is a deep crouch
    height_range: tuple = (0.35, 0.72)


def track_height_exp(env, std: float, command_name: str,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, 0]
    err = torch.square(asset.data.root_pos_w[:, 2] - cmd)
    return torch.exp(-err / std**2)


@configclass
class G1CrouchEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_height = HeightCommandCfg(resampling_time_range=(4.0, 8.0))
        self.observations.policy.height_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_height"}
        )
        self.rewards.track_height = RewTerm(
            func=track_height_exp, weight=2.0,
            params={"std": 0.12, "command_name": "base_height"},
        )
        # gentler velocity envelope while the crouch skill forms
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)


@configclass
class G1CrouchEnvCfg_PLAY(G1CrouchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
