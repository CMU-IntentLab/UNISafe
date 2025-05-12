from gymnasium.envs.registration import (
    load_plugin_envs,
    make,
    pprint_registry,
    register,
    registry,
    spec,
)


register(
    id="Takeoff_WM-v0",
    entry_point="reachability_dreamer.env.takeoff_wm:Takeoff_WM",
    max_episode_steps=30,
    reward_threshold=1e8,
)

register(
    id="Takeoff_WM_Failure-v0",
    entry_point="reachability_dreamer.env.takeoff_wm_failure:Takeoff_WM_Failure",
    max_episode_steps=30,
    reward_threshold=1e8,
)

register(
    id="Takeoff_WM_Failure_nf-v0",
    entry_point="reachability_dreamer.env.takeoff_wm_failure_nf:Takeoff_WM_Failure_Nf",
    max_episode_steps=30,
    reward_threshold=1e8,
)