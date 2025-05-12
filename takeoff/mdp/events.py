from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv



def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # Set the default pose for robots in all envs
    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos[env_ids] = torch.tensor(default_pose, device=env.device)


def reset_root_states_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfgs: list[SceneEntityCfg],
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """

    # poses
    asset = env.scene[asset_cfgs[0].name]
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples_pos = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples_vel = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    for asset_cfg in asset_cfgs:
        asset: RigidObject | Articulation  = env.scene[asset_cfg.name]
        # get default root state
        root_states = asset.data.default_root_state[env_ids].clone()

    
        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples_pos[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples_pos[:, 3], rand_samples_pos[:, 4], rand_samples_pos[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    

        velocities = root_states[:, 7:13] + rand_samples_vel

        # set into the physics simulation
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)