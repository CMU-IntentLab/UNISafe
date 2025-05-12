# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import CameraCfg, FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab_tasks.manager_based.manipulation.stack import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from omni.isaac.lab_tasks.manager_based.manipulation.stack.stack_env_cfg import StackEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG  # isort: skip

import sys
sys.path.append('source/latent_safety')

from takeoff import mdp
from takeoff.hard_takeoff_env_cfg import Hard_TakeoffEnvCfg
from omni.isaac.lab.sensors import TiledCameraCfg
from omni.isaac.lab.sim import PinholeCameraCfg

@configclass
class EventCfg:
    """Configuration for events."""

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="startup",
        params={
            "default_pose": [0.0857, -0.0894, -0.1006, -2.6492,  0.1896,  2.7419,  0.7095, 0.04, 0.04], #[0.0648, -0.0311, -0.0581, -2.6653,  0.2607,  2.8246,  0.6610, 0.0400, 0.0400],
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


    reset_object_position = EventTerm(
        func=mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.0, 0.0), "yaw": (-0.2, 0.2)},
            "velocity_range": {},
            "asset_cfgs": [SceneEntityCfg("cube_1")],
        },
    )

    reset_object_position = EventTerm(
        func=mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.0, 0.0), "yaw": (-0.5, 0.5)},
            "velocity_range": {},
            "asset_cfgs": [SceneEntityCfg("cube_2"),  SceneEntityCfg("cube_3")],
        },
    )



@configclass
class Hard_FrankaTakeoffEnvCfg(Hard_TakeoffEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

        self.scene.wrist_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.01,
            height=128,
            width=128,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20)
            ),
            offset=TiledCameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.0), rot=(0.17501, 0.68485, 0.6854, 0.17487), convention="opengl")
            # offset=TiledCameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.0), rot=(0.6848517, -0.1750107, -0.1748713, 0.6853973), convention="ros"),
        ) 
        
        self.scene.front_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/front_cam",
            update_period=0.01,
            height=128,
            width=128,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20)
            ),
            offset=TiledCameraCfg.OffsetCfg(pos=(0.72, 0.0, 0.32), rot=(0.65328,0.2706,0.2706,0.65328), convention="opengl"),
        )

        # self.scene.recording = TiledCameraCfg(
        #     prim_path="{ENV_REGEX_NS}/recording",
        #     update_period=0.01,
        #     height=1024,
        #     width=1024,
        #     data_types=["rgb"],
        #     spawn=PinholeCameraCfg(
        #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20)
        #     ),
        #     offset=TiledCameraCfg.OffsetCfg(pos=(0.72, 0.0, 0.32), rot=(0.65328,0.2706,0.2706,0.65328), convention="opengl"),
        # )
