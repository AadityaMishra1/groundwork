import gymnasium as gym

gym.register(
    id="G1-LocoKneel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "g1_loco.loco_env_cfg:G1LocoKneelEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.g1."
            "agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id="G1-LocoKneel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "g1_loco.loco_env_cfg:G1LocoKneelEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.g1."
            "agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg"
        ),
    },
)
