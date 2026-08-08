import gymnasium as gym

from . import agents

gym.register(
    id="G1-Grasp-v0",
    entry_point="g1_grasp.grasp_env:G1GraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "g1_grasp.grasp_env:G1GraspEnvCfg",
        "rsl_rl_cfg_entry_point": "g1_grasp.agents:G1GraspPPORunnerCfg",
    },
)
