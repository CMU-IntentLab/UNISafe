"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments, 
collecting and saving demonstrations only on success."""

import argparse
import os
import datetime
import numpy as np


from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument(
	"--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="spacemouse", help="Device for interacting with environment")
# parser.add_argument("--task", type=str, default="Isaac-Takeoff-Franka-IK-Rel-v0", help="Name of the task.")
parser.add_argument("--task", type=str, default="Isaac-Takeoff-Hard-Franka-IK-Rel-v0", help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
print(args_cli)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import carb

from omni.isaac.lab.devices import Se3Keyboard, Se3SpaceMouse

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

import sys
sys.path.append('latent_safety')

from takeoff import mdp
from takeoff.config import franka
from dreamer_wrapper import DreamerVecEnvWrapper
import dreamerv3_torch.envs.wrappers as wrappers

def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
	"""Pre-process actions for the environment."""
	if "Reach" in args_cli.task:
		# "Reach" environment has a different action space
		return delta_pose
	else:
		# resolve gripper command
		gripper_vel = torch.zeros(delta_pose.shape[0], 1, device=delta_pose.device)
		gripper_vel[:] = -1.0 if gripper_command else 1.0
		# compute actions
		return torch.concat([delta_pose, gripper_vel], dim=1)


def save_episode(transitions, cnt, save_dir):
	"""Save collected episode transitions to an .npz file."""
	# Convert lists -> np arrays
	episode = {}
	for key, val in transitions.items():
		print(key)
		episode[key] = np.array(val, dtype=np.float32)

	
	for i in range(len(episode["failure"])):
		reward = episode["reward"][i]
		if reward < -0.2:
			episode["failure"][max(0, i-2) : i+1] = 1.0

	# Create a unique file name with timestamp
	timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
	filename = f"expert_{cnt:03d}_{timestamp}.npz"
	os.makedirs(save_dir, exist_ok=True)

	# Save compressed numpy file
	np.savez_compressed(os.path.join(save_dir, filename), **episode)
	print(f"Saved demonstration to {filename}")

	return filename, episode

import cv2
import pathlib
def save_video(directory, filename, video, failure):
	directory = pathlib.Path(directory).expanduser()
	directory.mkdir(parents=True, exist_ok=True)

	length = len(video)
	filename = directory / f"{filename}.mp4"
		
	# Get video dimensions
	height, width, channels = video[0].shape
	fps = 20  # You can adjust FPS if needed

	# Initialize video writer
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
	video_writer = cv2.VideoWriter(str(filename), fourcc, fps, (width, height))

	# Write frames to the video file
	for i, frame in enumerate(video):
		failure_i = failure[i]
		frame = frame.astype(np.uint8)

		if failure_i > 0 :
			video_writer.write(frame)
		else:
			video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

	# Release the video writer
	video_writer.release()
	print("Saved Video!")
	return True


def main():
	# parse environment config
	env_cfg = parse_env_cfg(
		args_cli.task, device='cuda', num_envs=args_cli.num_envs, use_fabric=True
	)


	# create environment
	env = gym.make(args_cli.task, cfg=env_cfg)


	teleop_interface = Se3SpaceMouse(
		pos_sensitivity=0.5 * args_cli.sensitivity, 
		rot_sensitivity=0.5 * args_cli.sensitivity
	)

	# Optional: set up keyboard callback for environment reset
	keyboard = Se3Keyboard(
		pos_sensitivity=0.05 * args_cli.sensitivity, 
		rot_sensitivity=0.05 * args_cli.sensitivity
	)

	# wrap environment for Dreamer
	env = DreamerVecEnvWrapper(env=env, device="cuda")
	env = wrappers.NormalizeActions(env)
	
	# reset environment
	obs = env.reset()
	teleop_interface.reset()

	# Each episode's transitions
	transitions = {
		"front_cam": [],
		"wrist_cam": [],
		"eef_pos": [],
		"eef_quat": [],
		"success": [],
		"failure": [],
		"is_first": [],
		"is_last": [],
		"is_terminal": [],
		"action": [],
		"reward": [],
		"discount": [],
		"logprob": [],
	}

	# Store the initial observation o_0 with reward=0
	#    (If single environment, index 0)
	#    discount=1.0 means we haven't ended the episode yet

	for key, val in obs.items():
		transitions[key].append(val[0].cpu().numpy())

	transitions["action"].append(np.zeros((7), dtype=np.float32))
	transitions["reward"].append(np.array(0.0, dtype=np.float32))
	transitions["discount"].append(np.array(1.0, dtype=np.float32))
	transitions["logprob"].append(np.array(0.0, dtype=np.float32))


	cnt = 0

	def reset_episode_and_save():
		nonlocal cnt
		save_dir="source/latent_safety/takeoff/failure_dataset"
		# If we have at least one step recorded, decide whether to saveex
		cnt += 1
		filename, episode = save_episode(transitions, cnt=cnt, save_dir=save_dir)
		front_video = episode['front_cam']
		failure = episode['failure']
		save_video(save_dir, filename[:-4], front_video, failure)

		# Clear transitions
		transitions.clear()
		transitions.update({
			"front_cam": [],
			"wrist_cam": [],
			"eef_pos": [],
			"eef_quat": [],
			"success": [],
			"failure": [],
			"is_first": [],
			"is_last": [],
			"is_terminal": [],
			"action": [],
			"reward": [],
			"discount": [],
			"logprob": []
		})

		# Reset environment + teleop
		new_obs = env.reset()
		teleop_interface.reset()

		# Store new initial observation with reward=0
		for key, val in new_obs.items():
			transitions[key].append(val[0].cpu().numpy())

		transitions["action"].append(np.zeros((7), dtype=np.float32))
		transitions["reward"].append(np.array(0.0, dtype=np.float32))
		transitions["discount"].append(np.array(1.0, dtype=np.float32))
		transitions["logprob"].append(np.array(0.0, dtype=np.float32))

	def reset_episode():
		# Clear transitions
		transitions.clear()
		transitions.update({
			"front_cam": [],
			"wrist_cam": [],
			"eef_pos": [],
			"eef_quat": [],
			"success": [],
			"failure": [],
			"is_first": [],
			"is_last": [],
			"is_terminal": [],
			"action": [],
			"reward": [],
			"discount": [],
			"logprob": []
		})

		# Reset environment + teleop
		new_obs = env.reset()
		teleop_interface.reset()

		# Store new initial observation with reward=0
		for key, val in new_obs.items():
			transitions[key].append(val[0].cpu().numpy())

		transitions["action"].append(np.zeros((7), dtype=np.float32))
		transitions["reward"].append(np.array(0.0, dtype=np.float32))
		transitions["discount"].append(np.array(1.0, dtype=np.float32))
		transitions["logprob"].append(np.array(0.0, dtype=np.float32))


	# Make the keyboard’s L key also call our custom reset + save
	keyboard.add_callback("K", reset_episode_and_save)
	keyboard.add_callback("L", reset_episode)

	while simulation_app.is_running():
		with torch.inference_mode():
			# Teleop interface input
			delta_pose, gripper_cmd = teleop_interface.advance()
			delta_pose = delta_pose.astype("float32")
			delta_pose = torch.tensor(
				delta_pose,
				device=env.unwrapped.device
			).repeat(env.unwrapped.num_envs, 1)

			# Pre-process the action
			actions = pre_process_actions(delta_pose, gripper_cmd)
			# print(actions.min(), actions.max())
			actions = torch.clamp(actions, min=-1, max=1)

			# Step env
			next_obs, reward, done, info = env.step(actions)

			print(f"{reward.item():.4f}", next_obs['success'].item(), next_obs['failure'].item())


			# Flatten to CPU if single env
			act_cpu = (
				actions[0].cpu().numpy() 
				if isinstance(actions, torch.Tensor)
				else actions
			)
			reward_cpu = (
				reward[0].cpu().numpy() 
				if isinstance(reward, torch.Tensor)
				else reward
			)

			if next_obs['success'].item():
				next_obs['is_terminal'] = torch.tensor([1]).to(done.device)

			# Record transition (o_t, a_t, r_t)
			for key, val in next_obs.items():
				transitions[key].append(val[0].cpu().numpy())
			transitions["action"].append(np.atleast_1d(act_cpu))
			transitions["reward"].append(reward_cpu)
			# discount=1.0 for continuing episode
			transitions["discount"].append(np.array(1.0, dtype=np.float32))
			transitions["logprob"].append(np.array(0.0, dtype=np.float32))

			


if __name__ == "__main__":
	main()