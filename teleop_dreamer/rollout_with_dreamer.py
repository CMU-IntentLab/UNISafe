import os
import time
import torch
import gymnasium as gym
import numpy as np
import ruamel.yaml as yaml
from pathlib import Path
import argparse
import sys
sys.path.append('latent_safety')
import dreamerv3_torch.dreamer as dreamer
import dreamerv3_torch.tools as tools
import dreamerv3_torch.envs.wrappers as wrappers
from dreamer_wrapper import DreamerVecEnvWrapper
from reachability_dreamer.policy.SAC_Reachability import SAC 
from omni.isaac.lab.devices import Se3Keyboard, Se3SpaceMouse
from omni.isaac.lab_tasks.utils import parse_env_cfg

from takeoff import mdp
from takeoff.config import franka
import matplotlib.pyplot as plt
import cv2
import pathlib
from datetime import datetime


class TeleoperationWithWM:
	def __init__(self, config):
		self.config = config
		self.device = "cuda"
		self.env = None
		self.wm = None
		self.ensemble = None
		self.teleop_interface = None
		self.transitions = {}

		self.config.sensitivity = 1
		self.latent = None

		self.save_dir="source/latent_safety/teleop_dreamer/uncertainty_random"

	def setup_environment(self):
		"""
		Initialize the environment and attach the teleoperation interface.
		"""
		env_cfg = parse_env_cfg(
			self.config.task, device=self.device, num_envs=1, use_fabric=True
		)

		self.env = gym.make(self.config.task, cfg=env_cfg)
		self.env = DreamerVecEnvWrapper(self.env, device=self.device)
		self.env = wrappers.NormalizeActions(self.env)

		self.teleop_interface = Se3SpaceMouse(
			pos_sensitivity=0.5 * self.config.sensitivity,
			rot_sensitivity=0.5 * self.config.sensitivity
		)

		# Optional keyboard reset interface
		self.keyboard = Se3Keyboard(
			pos_sensitivity=0.05 * self.config.sensitivity,
			rot_sensitivity=0.05 * self.config.sensitivity
		)

		print("Environment and teleoperation interface initialized.")

	def load_world_model_and_ensemble(self):
		"""
		Load the world model (wm) and ensemble (if applicable).
		"""
		acts = self.env.single_action_space
		acts.low = np.ones_like(acts.low) * -1
		acts.high = np.ones_like(acts.high)
		self.config.num_actions = acts.shape[0]

		logger = dreamer.tools.DummyLogger(None, 1)

		agent = dreamer.Dreamer(
			self.env.single_observation_space,
			acts,
			self.config,
			logger=logger,
			dataset=None
		)

		agent.requires_grad_(requires_grad=False)

		checkpoint = torch.load(self.config.model_path, map_location=torch.device('cpu'))
		agent.load_state_dict(checkpoint["agent_state_dict"])

		self.wm = agent._wm.to(self.device)
		self.ensemble = agent._disag_ensemble.to(self.device)

		print("World model and ensemble loaded.")


	def pre_process_actions(self, delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
		"""
		Pre-process actions for the environment based on the task.
		"""
		if "Reach" in self.config.task:
			return delta_pose
		else:
			gripper_vel = torch.zeros(delta_pose.shape[0], 1, device=delta_pose.device)
			gripper_vel[:] = -1.0 if gripper_command else 1.0
			return torch.cat([delta_pose, gripper_vel], dim=1)

	def post_process_actions(self, action_cmd: torch.Tensor) -> torch.Tensor:

		action_cmd = torch.clamp(action_cmd, -1, 1)
		neg_mask = action_cmd[:, -1] < 0
		action_cmd[neg_mask, -1] = -1.0
		action_cmd[~neg_mask, -1] = 1.0
		
		return action_cmd

	def reset_episode(self):
		"""
		Reset the environment and prepare for a new episode.
		"""
		self.transitions.clear()
		self.transitions.update({
			"front_cam": [], "wrist_cam": [], "eef_pos": [], "eef_quat": [],
			"success": [], "failure": [], "is_first": [], "is_last": [], "is_terminal": [],
			"action": [], "reward": [], "discount": [], "logprob": [], "uncertainty": [], "reachability": []
		})

		obs = self.env.reset()
		for key, val in obs.items():
			self.transitions[key].append(val[0].cpu().numpy())

		self.transitions["action"].append(np.zeros((7), dtype=np.float32))
		self.transitions["reward"].append(np.array(0.0, dtype=np.float32))
		self.transitions["discount"].append(np.array(1.0, dtype=np.float32))
		self.transitions["logprob"].append(np.array(0.0, dtype=np.float32))
		self.transitions["uncertainty"].append(np.array(0.0, dtype=np.float32))

		init_data = {}
		for key, val in obs.items():
			init_data[key] = val.cpu().numpy()
		
		init_data["action"] = np.zeros((7), dtype=np.float32)
		init_data["reward"] = np.array(0.0, dtype=np.float32)
		init_data["discount"] = np.array(1.0, dtype=np.float32)
		init_data["logprob"] = np.array(0.0, dtype=np.float32)


		with torch.no_grad():
			data = self.wm.preprocess(init_data)
			embed = self.wm.encoder(data)

			# Latent and action is None for the initial
			latent = None
			action = None
			latent, _ = self.wm.dynamics.obs_step(latent, action, embed, obs["is_first"], sample=False)

			self.latent = latent

			return latent # shape: [1, D] for each key
	
	def forward_latent(self, latent_prev, act_prev, obs):
		with torch.no_grad():
			data = self.wm.preprocess(obs)
			embed = self.wm.encoder(data)

			# Latent and action is None for the initial
			latent_post, latent_prior = self.wm.dynamics.obs_step(latent_prev, act_prev, embed, obs["is_first"], sample=False)

			return latent_post, latent_prior

	def visualize_recon(self, pred, gt, fig, axes):

		# Convert tensors to NumPy arrays
		prior_image = pred.cpu().numpy()
		real_observation_image = gt.cpu().numpy()


		axes[0].imshow(prior_image)
		axes[1].imshow(real_observation_image)
		# Refresh the plot
		fig.canvas.draw()
		fig.canvas.flush_events()
		plt.pause(0.05)  # Pause briefly to render the updates

	def save_video(self):

		video = self.transitions['front_cam']  # List of frames (numpy arrays)
		uncertainties = self.transitions['uncertainty']  # List or array of uncertainties
		reachability = self.transitions['reachability']

		# Determine maximum uncertainty
		# max_uncertainty = np.max(uncertainties)
		max_uncertainty = np.quantile(uncertainties, 0.95)

		# Create directory for saving
		directory = pathlib.Path(self.save_dir).expanduser()
		directory.mkdir(parents=True, exist_ok=True)

		# Generate filename with max uncertainty
		current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		filename = directory / f"{max_uncertainty:.2f}_{current_time}.mp4"

		# Get video dimensions
		height, width, channels = video[0].shape
		fps = 10  # Frames per second for the video

		# Initialize video writer
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
		video_writer = cv2.VideoWriter(str(filename), fourcc, fps, (width, height))

		# Font settings for overlay
		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = 0.35
		font_color = (0, 255, 0)  # Green
		thickness = 1

		# Write frames with uncertainty overlay
		for frame, uncertainty in zip(video, uncertainties):
			# Convert RGB to BGR for OpenCV
			frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

			# Add uncertainty text to the frame
			text = f"Uncertainty: {uncertainty:.2f}"
			position = (10, height - 10)  # Bottom-left corner
			cv2.putText(frame_bgr, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

			# Write the frame to the video
			video_writer.write(frame_bgr)

		# Release the video writer
		video_writer.release()
		print(f"Saved video to {filename}")

		self.save_episode(self.transitions, self.save_dir)

		self.reset_episode()

		return True

	def save_episode(self, transitions, save_dir):
		"""Save collected episode transitions to an .npz file."""
		# Convert lists -> np arrays
		episode = {}
		for key, val in transitions.items():
			print(key)
			episode[key] = np.array(val, dtype=np.float32)

		# Create a unique file name with timestamp
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
		filename = f"rollout_{timestamp}.npz"
		os.makedirs(save_dir, exist_ok=True)

		# Save compressed numpy file
		np.savez_compressed(os.path.join(save_dir, filename), **episode)
		print(f"Saved demonstration to {filename}")

		return filename
	
	def run(self):
		"""
		Execute the teleoperation loop.
		"""
		self.setup_environment()
		self.load_world_model_and_ensemble()

		obs = self.env.reset()
		self.teleop_interface.reset()
		
		self.keyboard.add_callback("K", self.save_video)
		self.keyboard.add_callback("L", self.reset_episode)

		self.latent = self.reset_episode()

		while True:
			with torch.inference_mode():
			
				delta_pose, gripper_cmd = self.teleop_interface.advance()
				delta_pose = torch.tensor(delta_pose.astype("float32"), device=self.device).repeat(self.env.num_envs, 1)
				actions = self.pre_process_actions(delta_pose, gripper_cmd)
				actions = torch.clamp(actions, min=-1, max=1)
				# print(actions)

				# 1. Check uncertainty using (s_t = latent (posterior), a_t)
				feats = self.wm.dynamics.get_feat(self.latent)
				inputs = torch.concat([feats, actions], -1)
				uncertainty = self.ensemble.intrinsic_reward_penn(inputs).item()

				next_obs, reward, done, info = self.env.step(actions)

				# Update Latent
				# cnt += 1
				# if cnt % 20 == 0:
				# 	self.latent = None
				post, prior = self.forward_latent(latent_prev=self.latent, act_prev=actions, obs=next_obs)
				self.latent = post

				# recon_prior = self.wm.heads["decoder"](self.wm.dynamics.get_feat(prior).unsqueeze(0))["front_cam"].mode().squeeze()
				# real_observation = next_obs['front_cam'].squeeze()

				# self.visualize_recon(recon_prior, real_observation, fig, axes)

				print("{:.4f}".format(uncertainty))

				if done.item():
					self.latent = self.reset_episode()

def recursive_update(base, update):
	for key, value in update.items():
		if isinstance(value, dict) and key in base:
			recursive_update(base[key], value)
		else:
			base[key] = value

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--configs", nargs="+")
	parser.add_argument(
	"--enable_cameras", action="store_true", default=True
	)
	parser.add_argument(
	"--headless", action="store_true", default=True
	)
	parser.add_argument(
		"--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
	)
	parser.add_argument("--task", type=str, default="Isaac-Takeoff-Franka-IK-Rel-v0", help="Name of the task.")
	config, remaining = parser.parse_known_args()

	yaml = yaml.YAML(typ="safe", pure=True)
	configs = yaml.load(Path('source/latent_safety/reachability_dreamer/config.yaml').read_text())
	name_list = ["defaults"]

	defaults = {}
	for name in name_list:
		recursive_update(defaults, configs[name])
	
	# Merge CLI arguments into defaults
	for key, value in vars(config).items():
		if value is not None:
			defaults[key] = value

	parser = argparse.ArgumentParser()
	for key, value in sorted(defaults.items(), key=lambda x: x[0]):
		arg_type = tools.args_type(value)
		parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
		
	final_config = parser.parse_args(remaining)

	teleop = TeleoperationWithWM(final_config)
	teleop.run()
