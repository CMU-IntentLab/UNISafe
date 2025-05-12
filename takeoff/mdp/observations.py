# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject, RigidObjectCollection
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer

if TYPE_CHECKING:
	from omni.isaac.lab.envs import ManagerBasedRLEnv


def cube_positions_in_world_frame(
	env: ManagerBasedRLEnv,
	cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
	cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
	cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
	"""The position of the cubes in the world frame."""
	cube_1: RigidObject = env.scene[cube_1_cfg.name]
	cube_2: RigidObject = env.scene[cube_2_cfg.name]
	cube_3: RigidObject = env.scene[cube_3_cfg.name]

	return torch.cat((cube_1.data.root_link_pos_w, cube_2.data.root_link_pos_w, cube_3.data.root_link_pos_w), dim=1)

def instance_randomize_cube_positions_in_world_frame(
	env: ManagerBasedRLEnv,
	cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
	cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
	cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
	"""The position of the cubes in the world frame."""
	if not hasattr(env, "rigid_objects_in_focus"):
		return torch.full((env.num_envs, 9), fill_value=-1)

	cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
	cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
	cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]

	cube_1_pos_w = []
	cube_2_pos_w = []
	cube_3_pos_w = []
	for env_id in range(env.num_envs):
		cube_1_pos_w.append(cube_1.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
		cube_2_pos_w.append(cube_2.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][1], :3])
		cube_3_pos_w.append(cube_3.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][2], :3])
	cube_1_pos_w = torch.stack(cube_1_pos_w)
	cube_2_pos_w = torch.stack(cube_2_pos_w)
	cube_3_pos_w = torch.stack(cube_3_pos_w)

	return torch.cat((cube_1_pos_w, cube_2_pos_w, cube_3_pos_w), dim=1)


def cube_orientations_in_world_frame(
	env: ManagerBasedRLEnv,
	cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
	cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
	cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
):
	"""The orientation of the cubes in the world frame."""
	cube_1: RigidObject = env.scene[cube_1_cfg.name]
	cube_2: RigidObject = env.scene[cube_2_cfg.name]
	cube_3: RigidObject = env.scene[cube_3_cfg.name]

	return torch.cat((cube_1.data.root_link_quat_w, cube_2.data.root_link_quat_w, cube_3.data.root_link_quat_w), dim=1)


def instance_randomize_cube_orientations_in_world_frame(
	env: ManagerBasedRLEnv,
	cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
	cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
	cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
	"""The orientation of the cubes in the world frame."""
	if not hasattr(env, "rigid_objects_in_focus"):
		return torch.full((env.num_envs, 9), fill_value=-1)

	cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
	cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
	cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]

	cube_1_quat_w = []
	cube_2_quat_w = []
	cube_3_quat_w = []
	for env_id in range(env.num_envs):
		cube_1_quat_w.append(cube_1.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
		cube_2_quat_w.append(cube_2.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][1], :4])
		cube_3_quat_w.append(cube_3.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][2], :4])
	cube_1_quat_w = torch.stack(cube_1_quat_w)
	cube_2_quat_w = torch.stack(cube_2_quat_w)
	cube_3_quat_w = torch.stack(cube_3_quat_w)

	return torch.cat((cube_1_quat_w, cube_2_quat_w, cube_3_quat_w), dim=1)


def object_obs(
	env: ManagerBasedRLEnv,
	cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
	cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
	cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
	ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
	"""
	Object observations (in world frame):
		cube_1 pos,
		cube_1 quat,
		cube_2 pos,
		cube_2 quat,
		cube_3 pos,
		cube_3 quat,
		gripper to cube_1,
		gripper to cube_2,
		gripper to cube_3,
		cube_1 to cube_2,
		cube_2 to cube_3,
		cube_1 to cube_3,
	"""
	cube_1: RigidObject = env.scene[cube_1_cfg.name]
	cube_2: RigidObject = env.scene[cube_2_cfg.name]
	cube_3: RigidObject = env.scene[cube_3_cfg.name]
	ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

	cube_1_pos_w = cube_1.data.root_link_pos_w
	cube_1_quat_w = cube_1.data.root_link_quat_w

	cube_2_pos_w = cube_2.data.root_link_pos_w
	cube_2_quat_w = cube_2.data.root_link_quat_w

	cube_3_pos_w = cube_3.data.root_link_pos_w
	cube_3_quat_w = cube_3.data.root_link_quat_w

	ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
	gripper_to_cube_1 = cube_1_pos_w - ee_pos_w
	gripper_to_cube_2 = cube_2_pos_w - ee_pos_w
	gripper_to_cube_3 = cube_3_pos_w - ee_pos_w

	cube_1_to_2 = cube_1_pos_w - cube_2_pos_w
	cube_2_to_3 = cube_2_pos_w - cube_3_pos_w
	cube_1_to_3 = cube_1_pos_w - cube_3_pos_w

	return torch.cat(
		(
			cube_1_pos_w,
			cube_1_quat_w,
			cube_2_pos_w,
			cube_2_quat_w,
			cube_3_pos_w,
			cube_3_quat_w,
			gripper_to_cube_1,
			gripper_to_cube_2,
			gripper_to_cube_3,
			cube_1_to_2,
			cube_2_to_3,
			cube_1_to_3,
		),
		dim=1,
	)


def instance_randomize_object_obs(
	env: ManagerBasedRLEnv,
	cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
	cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
	cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
	ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
	"""
	Object observations (in world frame):
		cube_1 pos,
		cube_1 quat,
		cube_2 pos,
		cube_2 quat,
		cube_3 pos,
		cube_3 quat,
		gripper to cube_1,
		gripper to cube_2,
		gripper to cube_3,
		cube_1 to cube_2,
		cube_2 to cube_3,
		cube_1 to cube_3,
	"""
	if not hasattr(env, "rigid_objects_in_focus"):
		return torch.full((env.num_envs, 9), fill_value=-1)

	cube_1: RigidObjectCollection = env.scene[cube_1_cfg.name]
	cube_2: RigidObjectCollection = env.scene[cube_2_cfg.name]
	cube_3: RigidObjectCollection = env.scene[cube_3_cfg.name]
	ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

	cube_1_pos_w = []
	cube_2_pos_w = []
	cube_3_pos_w = []
	cube_1_quat_w = []
	cube_2_quat_w = []
	cube_3_quat_w = []
	for env_id in range(env.num_envs):
		cube_1_pos_w.append(cube_1.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
		cube_2_pos_w.append(cube_2.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][1], :3])
		cube_3_pos_w.append(cube_3.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][2], :3])
		cube_1_quat_w.append(cube_1.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
		cube_2_quat_w.append(cube_2.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][1], :4])
		cube_3_quat_w.append(cube_3.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][2], :4])
	cube_1_pos_w = torch.stack(cube_1_pos_w)
	cube_2_pos_w = torch.stack(cube_2_pos_w)
	cube_3_pos_w = torch.stack(cube_3_pos_w)
	cube_1_quat_w = torch.stack(cube_1_quat_w)
	cube_2_quat_w = torch.stack(cube_2_quat_w)
	cube_3_quat_w = torch.stack(cube_3_quat_w)

	ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
	gripper_to_cube_1 = cube_1_pos_w - ee_pos_w
	gripper_to_cube_2 = cube_2_pos_w - ee_pos_w
	gripper_to_cube_3 = cube_3_pos_w - ee_pos_w

	cube_1_to_2 = cube_1_pos_w - cube_2_pos_w
	cube_2_to_3 = cube_2_pos_w - cube_3_pos_w
	cube_1_to_3 = cube_1_pos_w - cube_3_pos_w

	return torch.cat(
		(
			cube_1_pos_w,
			cube_1_quat_w,
			cube_2_pos_w,
			cube_2_quat_w,
			cube_3_pos_w,
			cube_3_quat_w,
			gripper_to_cube_1,
			gripper_to_cube_2,
			gripper_to_cube_3,
			cube_1_to_2,
			cube_2_to_3,
			cube_1_to_3,
		),
		dim=1,
	)


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
	ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
	ee_frame_pos = ee_frame.data.target_pos_source[:, 0, :]

	return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
	ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
	ee_frame_quat = ee_frame.data.target_quat_source[:, 0, :]

	return ee_frame_quat


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
	robot: Articulation = env.scene[robot_cfg.name]
	finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
	finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)

	return torch.cat((finger_joint_1, finger_joint_2), dim=1)


def object_grasped(
	env: ManagerBasedRLEnv,
	robot_cfg: SceneEntityCfg,
	ee_frame_cfg: SceneEntityCfg,
	object_cfg: SceneEntityCfg,
	diff_threshold: float = 0.06,
	gripper_open_val: torch.tensor = torch.tensor([0.04]),
	gripper_threshold: float = 0.005,
) -> torch.Tensor:
	"""Check if an object is grasped by the specified robot."""

	robot: Articulation = env.scene[robot_cfg.name]
	ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
	object: RigidObject = env.scene[object_cfg.name]

	object_pos = object.data.root_link_pos_w
	end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
	pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

	grasped = torch.logical_and(
		pose_diff < diff_threshold,
		torch.abs(robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device)) > gripper_threshold,
	)
	grasped = torch.logical_and(
		grasped, torch.abs(robot.data.joint_pos[:, -2] - gripper_open_val.to(env.device)) > gripper_threshold
	)

	return grasped


def object_stacked(
	env: ManagerBasedRLEnv,
	robot_cfg: SceneEntityCfg,
	upper_object_cfg: SceneEntityCfg,
	lower_object_cfg: SceneEntityCfg,
	xy_threshold: float = 0.05,
	height_threshold: float = 0.005,
	height_diff: float = 0.0468,
	gripper_open_val: torch.tensor = torch.tensor([0.04]),
) -> torch.Tensor:
	"""Check if an object is stacked by the specified robot."""

	robot: Articulation = env.scene[robot_cfg.name]
	upper_object: RigidObject = env.scene[upper_object_cfg.name]
	lower_object: RigidObject = env.scene[lower_object_cfg.name]

	pos_diff = upper_object.data.root_link_pos_w - lower_object.data.root_link_pos_w
	height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
	xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

	stacked = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)

	stacked = torch.logical_and(
		torch.isclose(robot.data.joint_pos[:, -1], gripper_open_val.to(env.device), atol=1e-4, rtol=1e-4), stacked
	)
	stacked = torch.logical_and(
		torch.isclose(robot.data.joint_pos[:, -2], gripper_open_val.to(env.device), atol=1e-4, rtol=1e-4), stacked
	)

	return stacked


def ee_frame_out_of_boundary(
	env: ManagerBasedRLEnv,
	ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
	x_min: float = 0.3, x_max = 0.7, y_min = -0.3, y_max = 0.3
) -> torch.Tensor:

	ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
	ee_frame_pos = ee_frame.data.target_pos_source[:, 0, :]  # Extract position
	
	# Extract x, y positions
	x_pos = ee_frame_pos[:, 0]
	y_pos = ee_frame_pos[:, 1]
	
	# Check if x and y are within the boundary
	within_x = (x_pos >= x_min) & (x_pos <= x_max)
	within_y = (y_pos >= y_min) & (y_pos <= y_max)
	within_boundary = within_x & within_y
		
	return ~within_boundary


def check_stacked(
	env,
	upper_obj_name: str,
	lower_obj_name: str,
	xy_threshold: float,
	height_diff: float,
	height_threshold: float,
	min_height_diff:float
) -> torch.Tensor:
	"""
	Returns a tensor of booleans indicating whether `upper_obj_name` is stacked on `lower_obj_name`
	for each environment instance in the batch.
	"""
	upper_obj = env.scene[upper_obj_name]
	lower_obj = env.scene[lower_obj_name]

	# position differences: shape [N, 3] for N environments
	pos_diff = upper_obj.data.root_link_pos_w - lower_obj.data.root_link_pos_w

	# XY distance
	xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)  # shape [N]

	# Z distance
	height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)

	# Condition 1: XY must be close
	cond_xy = (xy_dist < xy_threshold)

	# Condition 2: Z must be near the known stack height
	# i.e., z_dist ~ height_diff within ± height_threshold
	cond_z = torch.abs(height_dist - height_diff) < height_threshold

	# upper one's height should be bigger than min_height diff -> success
	over_z = (height_dist > min_height_diff)

	stacked = cond_xy & cond_z & over_z
	return stacked

def is_failure(
	env,
	cube_1_name: str = "cube_1",
	cube_3_name: str = "cube_3",
	max_steps: int = 500,
	xy_threshold: float = 0.10,
	height_diff: float = 0.02,
	height_threshold: float = 0.010,
	min_height_diff = 0.014,
) -> torch.Tensor:
	"""
	Returns a boolean tensor indicating failure in each environment instance.

	- Fails if we exceed max_steps
	- Fails if cube_3 is obviously not stacked on cube_1 
	  (e.g., it has fallen off or is out of place).
	"""
	# We consider it a failure if cube_3 is definitely not on cube_1
	# (Looser XY threshold to detect a bigger displacement.)
	
	pos_diff = env.scene["cube_3"].data.root_link_pos_w - env.scene["cube_1"].data.root_link_pos_w
	xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)  # shape [N]
	# Condition 1: XY must be close
	cond_xy = (xy_dist > xy_threshold)

	c3_drop = (env.scene['cube_3'].data.root_link_pos_w[:, 2] < 0.13) | cond_xy
	c2_drop = (env.scene['cube_2'].data.root_link_pos_w[:, 2] < 0.13)

	fail_tensor = c2_drop | c3_drop 

	return fail_tensor


def check_stacked_3on1(
	env,
	cube_1_name: str = "cube_1",
	cube_2_name: str = "cube_2",
	cube_3_name: str = "cube_3",
	xy_threshold: float = 0.05,
	height_diff: float = 0.03,
	height_threshold: float = 0.01,
	min_height_diff = 0.014
) -> torch.Tensor:
	stacked_3on1 = check_stacked(
		env=env,
		upper_obj_name=cube_3_name,
		lower_obj_name=cube_1_name,
		xy_threshold=xy_threshold,
		height_diff=height_diff,
		height_threshold=height_threshold,
		min_height_diff = 0.014
	)

	return stacked_3on1

def check_not_stacked_2on1(
	env,
	cube_1_name: str = "cube_1",
	cube_2_name: str = "cube_2",
	cube_3_name: str = "cube_3",
	xy_threshold: float = 0.05,
	height_diff: float = 0.03,
	height_threshold: float = 0.01,
	min_height_diff = 0.014
) -> torch.Tensor:
	
	stacked_2on1 = check_stacked(
		env=env,
		upper_obj_name=cube_2_name,
		lower_obj_name=cube_1_name,
		xy_threshold=xy_threshold,
		height_diff=height_diff,
		height_threshold=height_threshold,
		min_height_diff = 0.014
	)

	return ~stacked_2on1

def check_not_stacked_3on2(
	env,
	cube_1_name: str = "cube_1",
	cube_2_name: str = "cube_2",
	cube_3_name: str = "cube_3",
	xy_threshold: float = 0.05,
	height_diff: float = 0.03,
	height_threshold: float = 0.01,
	min_height_diff = 0.014
) -> torch.Tensor:
	
	stacked_3on2 = check_stacked(
		env=env,
		upper_obj_name=cube_3_name,
		lower_obj_name=cube_2_name,
		xy_threshold=xy_threshold,
		height_diff=height_diff,
		height_threshold=height_threshold,
		min_height_diff = 0.014
	)

	return ~stacked_3on2

def is_success(
	env,
	cube_1_name: str = "cube_1",
	cube_2_name: str = "cube_2",
	cube_3_name: str = "cube_3",
	xy_threshold: float = 0.05,
	height_diff: float = 0.03,
	height_threshold: float = 0.01,
	min_height_diff = 0.014
) -> torch.Tensor:
	"""
	Returns a boolean tensor indicating success in each environment instance.
	- We want cube_3 on cube_1
	- We do NOT want cube_2 on cube_1
	- We do NOT want cube_2 on cube_3
	"""

	stacked_3on1 = check_stacked(
		env=env,
		upper_obj_name=cube_3_name,
		lower_obj_name=cube_1_name,
		xy_threshold=xy_threshold,
		height_diff=height_diff,
		height_threshold=height_threshold,
		min_height_diff = 0.014
	)

	stacked_2on1 = check_stacked(
		env=env,
		upper_obj_name=cube_2_name,
		lower_obj_name=cube_1_name,
		xy_threshold=xy_threshold,
		height_diff=height_diff,
		height_threshold=height_threshold,
		min_height_diff = 0.014
	)

	stacked_3on2 = check_stacked(
		env=env,
		upper_obj_name=cube_3_name,
		lower_obj_name=cube_2_name,
		xy_threshold=xy_threshold,
		height_diff=height_diff,
		height_threshold=height_threshold,
		min_height_diff = 0.014
	)

	# cube 2 distance from cube 3
	pos_diff = env.scene["cube_2"].data.root_link_pos_w - env.scene["cube_1"].data.root_link_pos_w
	xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)  # shape [N]
	# Condition 1: XY must be distant
	# c2_xy_dist = (xy_dist > 0.08) #0.08 -> 0.15 -> 0.12 -> 0.08
	c2_xy_dist = (xy_dist > 0.07) #0.08 -> 0.15 -> 0.12 -> 0.08
	c2_drop = env.scene['cube_2'].data.root_link_pos_w[:, 2] < 0.13

	success_tensor = stacked_3on1 & (~stacked_2on1) & (~stacked_3on2) & (~c2_drop) & c2_xy_dist

	return success_tensor

def rel_pos_12(
	env,
	cube_1_name: str = "cube_1",
	cube_2_name: str = "cube_2",
) -> torch.Tensor:

	# cube 2 distance from cube 3
	pos_diff = env.scene["cube_2"].data.root_link_pos_w - env.scene["cube_1"].data.root_link_pos_w
	xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)  # shape [N]

	return xy_dist