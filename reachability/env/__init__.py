"""
Reachability Environment Registration

This module registers various reachability-based environments for safety-critical
reinforcement learning tasks. These environments are designed to work with
world models and uncertainty quantification for safe exploration.

All environments are configured with:
- Short episodes (30 steps) for focused safety evaluation
- High reward threshold to encourage safe behavior
"""

from gymnasium.envs.registration import (
    load_plugin_envs,
    make,
    pprint_registry,
    register,
    registry,
    spec,
)

# Base takeoff environment with world model and uncertainty quantification only
register(
    id="Takeoff_WM-v0",
    entry_point="reachability_dreamer.env.takeoff_wm_uncertainty_only:Takeoff_WM",
    max_episode_steps=30,
    reward_threshold=1e8,
)

# Extended takeoff environment with explicit failure prediction
register(
    id="Takeoff_WM_Failure-v0",
    entry_point="reachability_dreamer.env.takeoff_wm_failure:Takeoff_WM_Failure",
    max_episode_steps=30,
    reward_threshold=1e8,
)

