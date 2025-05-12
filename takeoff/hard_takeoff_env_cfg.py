# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import TiledCameraCfg, CameraCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
	"""Configuration for the lift scene with a robot and a object.
	This is the abstract base implementation, the exact scene is defined in the derived classes
	which need to set the target object, robot and end-effector frames
	"""

	# robots: will be populated by agent env cfg
	robot: ArticulationCfg = MISSING
	# end-effector sensor: will be populated by agent env cfg
	ee_frame: FrameTransformerCfg = MISSING

	# Cameras
	wrist_cam: CameraCfg | TiledCameraCfg = MISSING
	front_cam: CameraCfg | TiledCameraCfg = MISSING

	# Table
	table = AssetBaseCfg(
		prim_path="{ENV_REGEX_NS}/Table",
		init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
		spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
	)

	# plane
	plane = AssetBaseCfg(
		prim_path="/World/GroundPlane",
		init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
		spawn=GroundPlaneCfg(),
	)

	# lights
	light = AssetBaseCfg(
		prim_path="/World/light",
		spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
	)

	base =  RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Base",
		spawn=sim_utils.CuboidCfg(
			size=[0.7, 0.7, 0.1],
			rigid_props=sim_utils.RigidBodyPropertiesCfg(
			),
			mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), metallic=0.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.00, 0.05), rot=(1.0, 0.0, 0.0, 0.0)),
	)

	cube_1 =  RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Cube_1",
		spawn=sim_utils.CuboidCfg(
			size=[0.07, 0.07, 0.03], #size=[0.1, 0.1, 0.03],
			rigid_props=sim_utils.RigidBodyPropertiesCfg(
				solver_position_iteration_count=16,
				solver_velocity_iteration_count=1,
				max_angular_velocity=1000.0,
				max_linear_velocity=1000.0,
				max_depenetration_velocity=5.0,
				disable_gravity=False,
			),
			mass_props=sim_utils.MassPropertiesCfg(mass=2),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.1, 1.0), metallic=0.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(0.530, -0.00, 0.114), rot=(1.0, 0.0, 0.0, 0.0)),
	)

	cube_2 =  RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Cube_2",
		spawn=sim_utils.CuboidCfg(
			size=[0.09, 0.03, 0.02], # size=[0.11, 0.025, 0.02],
			rigid_props=sim_utils.RigidBodyPropertiesCfg(
				solver_position_iteration_count=16,
				solver_velocity_iteration_count=1,
				max_angular_velocity=1000.0,
				max_linear_velocity=1000.0,
				max_depenetration_velocity=5.0,
				disable_gravity=False,
			),
			mass_props=sim_utils.MassPropertiesCfg(mass=0.6),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.86, 1.0, 0.0), metallic=0.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(0.52, 0.01, 0.141), rot=(1.0, 0.0, 0.0, 0.0)),
	)

	cube_3 =  RigidObjectCfg(
		prim_path="{ENV_REGEX_NS}/Cube_3",
		spawn=sim_utils.CuboidCfg(
			size=[0.05, 0.06, 0.02], #size=[0.04, 0.06, 0.02],
			rigid_props=sim_utils.RigidBodyPropertiesCfg(
				solver_position_iteration_count=16,
				solver_velocity_iteration_count=1,
				max_angular_velocity=1000.0,
				max_linear_velocity=1000.0,
				max_depenetration_velocity=5.0,
				disable_gravity=False,
			),
			mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
			collision_props=sim_utils.CollisionPropertiesCfg(),
			visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.0), metallic=0.2),
		),
		init_state=RigidObjectCfg.InitialStateCfg(pos=(0.53851, 0.01, 0.16), rot=(0.970, 0.0, 0.0, -0.24)),
	)


##
# MDP settings
##
@configclass
class ActionsCfg:
	"""Action specifications for the MDP."""

	# will be set by agent env cfg
	arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
	gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
	"""Observation specifications for the MDP."""

	@configclass
	class PolicyCfg(ObsGroup):
		"""Observations for policy group with state values."""

		actions = ObsTerm(func=mdp.last_action)
		joint_pos = ObsTerm(func=mdp.joint_pos_rel)
		joint_pos_abs = ObsTerm(func=mdp.joint_pos)
		joint_vel = ObsTerm(func=mdp.joint_vel_rel)
		object = ObsTerm(func=mdp.object_obs)
		cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
		cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
		eef_pos = ObsTerm(func=mdp.ee_frame_pos)
		eef_quat = ObsTerm(func=mdp.ee_frame_quat)
		gripper_pos = ObsTerm(func=mdp.gripper_pos)

		def __post_init__(self):
			self.enable_corruption = False
			self.concatenate_terms = False

	@configclass
	class RGBCameraPolicyCfg(ObsGroup):
		"""Observations for policy group with RGB images."""

		front_cam = ObsTerm(
			func=mdp.image, params={"sensor_cfg": SceneEntityCfg("front_cam"), "data_type": "rgb", "normalize": False}
		)
		wrist_cam = ObsTerm(
			func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False}
		)

		# recording = ObsTerm(
		# 	func=mdp.image, params={"sensor_cfg": SceneEntityCfg("recording"), "data_type": "rgb", "normalize": False}
		# )

		def __post_init__(self):
			self.enable_corruption = False
			self.concatenate_terms = False

	@configclass
	class SubtaskCfg(ObsGroup):
		"""Observations for subtask group."""

		failure = ObsTerm(
			func=mdp.is_failure, params={"xy_threshold": 0.07}
		)

		success = ObsTerm(
			func=mdp.is_success, params={"xy_threshold": 0.03}
		)

		def __post_init__(self):
			self.enable_corruption = False
			self.concatenate_terms = False

	# observation groups
	policy: PolicyCfg = PolicyCfg()
	rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
	subtask_terms: SubtaskCfg = SubtaskCfg()

@configclass
class RewardsCfg:
	"""Reward terms for the MDP."""
	# action penalty
	action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

	joint_vel = RewTerm(
		func=mdp.joint_vel_l2,
		weight=-1e-4,
		params={"asset_cfg": SceneEntityCfg("robot")},
	)

	cube_2_dropping = RewTerm(
		func=mdp.root_height_below_minimum, weight=-1, params={"minimum_height": 0.13, "asset_cfg": SceneEntityCfg("cube_2")}
	)

	cube_3_dropping = RewTerm(
		func=mdp.root_height_below_minimum, weight=-5, params={"minimum_height": 0.13, "asset_cfg": SceneEntityCfg("cube_3")}
	)

	ee_position = RewTerm(
		func=mdp.ee_frame_out_of_boundary, weight=-5
	)

	failure = RewTerm(
			func=mdp.is_failure, weight = -10, params={"xy_threshold": 0.07}
	)

	success = RewTerm(
			func=mdp.is_success, weight = 10, params={"xy_threshold": 0.03}
	)

	check_3on1 = RewTerm(
			func=mdp.check_stacked_3on1, weight = 1
	)

	not_stacked_2on1 = RewTerm(
			func=mdp.check_not_stacked_2on1, weight = 1
	)

	not_stacked_3on2 = RewTerm(
			func=mdp.check_not_stacked_3on2, weight = 2
	)

	dist_12 = RewTerm(
		func=mdp.rel_pos_12, weight=10 #0.5
	)


@configclass
class TerminationsCfg:
	"""Termination terms for the MDP."""

	# time_out = DoneTerm(func=mdp.time_out, time_out=True)

	# ee_position = DoneTerm(func=mdp.ee_frame_out_of_boundary)

	# cube_2_dropping = DoneTerm(
	# 	func=mdp.root_height_below_minimum, params={"minimum_height": 0.13, "asset_cfg": SceneEntityCfg("cube_2")}
	# )

	# cube_3_dropping = DoneTerm(
	# 	func=mdp.root_height_below_minimum, params={"minimum_height": 0.13, "asset_cfg": SceneEntityCfg("cube_3")}
	# )

	# success = DoneTerm(func=mdp.is_success, params={"xy_threshold": 0.03})
	# failure = DoneTerm(func=mdp.is_failure, params={"xy_threshold": 0.07})


@configclass
class Hard_TakeoffEnvCfg(ManagerBasedRLEnvCfg):
	"""Configuration for the stacking environment."""

	# Scene settings
	scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
	# Basic settings
	observations: ObservationsCfg = ObservationsCfg()
	actions: ActionsCfg = ActionsCfg()
	# MDP settings
	terminations: TerminationsCfg = TerminationsCfg()

	# Unused managers
	commands = None
	rewards: RewardsCfg = RewardsCfg()
	events = None
	curriculum = None

	def __post_init__(self):
		"""Post initialization."""
		# general settings
		self.decimation = 5 # # env step every 5 sim steps: 100Hz / 5 = 20Hz
		self.episode_length_s = 15 #30.0 # 30 seconds.
		self.rerender_on_reset = True
		
		# simulation settings
		self.sim.dt = 0.01  # 100Hz
		self.sim.render_interval = self.decimation

		self.sim.physx.bounce_threshold_velocity = 0.2
		self.sim.physx.bounce_threshold_velocity = 0.01
		self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
		self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
		self.sim.physx.friction_correlation_distance = 0.00625

		self.seed = 0
