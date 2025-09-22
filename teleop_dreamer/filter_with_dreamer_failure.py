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
from reachability.policy.SAC_Reachability_Env import SAC 
from omni.isaac.lab.devices import Se3Keyboard, Se3SpaceMouse
from omni.isaac.lab_tasks.utils import parse_env_cfg

from takeoff import mdp
from takeoff.config import franka
import matplotlib.pyplot as plt
import cv2
import pathlib
import imageio
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

		self.save_dir="source/latent_safety/teleop_dreamer/failure_filter/uncertaintyonly"

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

		# self.teleop_interface = Se3SpaceMouse(
		# 	pos_sensitivity=0.5 * self.config.sensitivity,
		# 	rot_sensitivity=0.5 * self.config.sensitivity
		# )

		self.teleop_interface = Se3Keyboard(
			pos_sensitivity=0.05 * self.config.sensitivity,
			rot_sensitivity=0.05 * self.config.sensitivity
		)

		# Optional keyboard reset interface
		self.keyboard = Se3Keyboard(
			pos_sensitivity=0.05 * self.config.sensitivity,
			rot_sensitivity=0.05 * self.config.sensitivity
		)

		print("Environment and teleoperation interface initialized.")

	def load_world_model_and_ensemble(self):
		"""
		Load the world model, ensemble, and reachability filter.
		
		This method loads all required models for safe teleoperation:
		1. World model for environment simulation
		2. Uncertainty ensemble for epistemic uncertainty estimation  
		3. Reachability filter for safety interventions
		"""
		print("Loading world model and uncertainty ensemble...")
		
		# Configure action space
		acts = self.env.single_action_space
		acts.low = np.ones_like(acts.low) * -1
		acts.high = np.ones_like(acts.high)
		self.config.num_actions = acts.shape[0]

		# Load world model
		if not os.path.exists(self.config.model_path):
			raise FileNotFoundError(
				f"World model checkpoint not found at: {self.config.model_path}.\n"
				f"Please ensure you have the pretrained world model file."
			)

		logger = dreamer.tools.DummyLogger(None, 1)

		agent = dreamer.Dreamer(
			self.env.single_observation_space,
			acts,
			self.config,
			logger=logger,
			dataset=None
		)

		agent.requires_grad_(requires_grad=False)

		print(f"Loading world model from: {self.config.model_path}")
		checkpoint = torch.load(self.config.model_path, map_location=torch.device('cpu'))
		agent.load_state_dict(checkpoint["agent_state_dict"])

		# Extract world model components
		self.wm = agent._wm.to(self.device)
		self.ensemble = agent._disag_ensemble.to(self.device)
		self.density = agent._density_estimator.to(self.device)

		print("World model and ensemble loaded successfully.")

		# Load reachability filter
		print("Setting up reachability safety filter...")
		
		# Calculate state and action dimensions
		if self.config.dyn_discrete:
			state_dim = (self.config.dyn_stoch * self.config.dyn_discrete) + self.config.dyn_deter
		else:
			state_dim = self.config.dyn_stoch + self.config.dyn_deter
		action_dim = self.config.num_actions
		actor_network_dims = self.config.control_net
		critic_network_dims = self.config.critic_net

		# Initialize safety filter
		self.agent = SAC(
			CONFIG=self.config,
			dim_state=state_dim,
			dim_action=action_dim,
			actor_dimList=actor_network_dims,
			critic_dimList=critic_network_dims
		)

		# Load filter checkpoint if available
		if hasattr(self.config, 'reachability_model_path') and self.config.reachability_model_path:
			if os.path.exists(self.config.reachability_model_path):
				filter_step = getattr(self.config, 'filter_step', 200000)
				print(f"Loading reachability filter from: {self.config.reachability_model_path}")
				print(f"Using filter checkpoint at step: {filter_step}")
				
				try:
					self.agent.restore(filter_step, self.config.reachability_model_path)
					print("Reachability filter loaded successfully!")
					print("🛡️  Safety filter is ACTIVE - will intervene when detecting unsafe actions")
				except Exception as e:
					print(f"Warning: Error loading reachability filter: {e}")
					print("Continuing without safety filter...")
			else:
				print(f"Warning: Reachability filter path not found: {self.config.reachability_model_path}")
				print("Continuing without safety filter...")
		else:
			print("No reachability filter specified - running base policy only")
			print("⚠️  WARNING: No safety filter active!")

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
		self.teleop_interface.reset()
		self.transitions.clear()
		self.transitions.update({
			"front_cam": [], "wrist_cam": [], "eef_pos": [], "eef_quat": [],
			"success": [], "failure": [], "is_first": [], "is_last": [], "is_terminal": [],
			"action": [], "reward": [], "discount": [], "logprob": [], "uncertainty": [], "reachability": [],
			"is_filtered":[], "failure": [], "recording":[]
		})

		obs = self.env.reset(seed=0)
		for key, val in obs.items():
			self.transitions[key].append(val[0].cpu().numpy())

		self.transitions["action"].append(np.zeros((7), dtype=np.float32))
		self.transitions["reward"].append(np.array(0.0, dtype=np.float32))
		self.transitions["discount"].append(np.array(1.0, dtype=np.float32))
		self.transitions["logprob"].append(np.array(0.0, dtype=np.float32))
		self.transitions["uncertainty"].append(np.array(0.0, dtype=np.float32))
		self.transitions["failure"].append(np.array(0.0, dtype=np.float32))
		self.transitions["is_filtered"].append(False)

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

			feat = self.wm.dynamics.get_feat(latent_post)
			pred = self.wm.heads["decoder"](feat.unsqueeze(0))
			loss_front = -pred['front_cam'].log_prob(data["front_cam"].unsqueeze(0)).item()
			loss_wrist = -pred['wrist_cam'].log_prob(data["wrist_cam"].unsqueeze(0)).item()

			return latent_post, latent_prior, loss_front + loss_wrist

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
		# Get your data from transitions
		video =  self.transitions['front_cam']       # List of frames (numpy arrays)
		uncertainties = self.transitions['uncertainty']  # List/array of uncertainties
		reachability = self.transitions['reachability']  # List/array of reachability values
		is_filtered = self.transitions['is_filtered']    # List/array of boolean flags
		failure = self.transitions['failure']            # List/array of boolean flags

		# Determine maximum uncertainty (95th percentile)
		max_uncertainty = np.quantile(uncertainties, 0.95)

		# Create directory for saving
		directory = pathlib.Path(self.save_dir).expanduser()
		directory.mkdir(parents=True, exist_ok=True)

		# Generate filename with max uncertainty
		current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		filename = directory / f"{max_uncertainty:.2f}_{current_time}.mp4"

		# Get video dimensions from the first frame
		height, width, channels = video[0].shape
		fps = 10  # Frames per second for the video

		# Font settings for overlay
		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = 0.3
		font_color = (0, 255, 0)  # Green in BGR
		thickness = 1

		# Initialize imageio video writer (using the ffmpeg plugin)
		writer = imageio.get_writer(str(filename), fps=fps, codec='libx264')

		# Iterate through frames along with uncertainties, reachability, and filter status
		for frame, uncertainty, reach, fail, filt in zip(video, uncertainties, reachability, failure, is_filtered):
			# Convert frame based on filter flag
			if filt:
				# If filtered, assume frame is already in BGR (or does not require conversion)
				frame_bgr = frame.copy()
			else:
				# Convert from RGB to BGR for consistency with OpenCV drawing functions
				frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

			# Prepare overlay texts
			uncertainty_text = f"Uncertainty: {uncertainty:.2f}"
			reachability_text = f"Reachability: {reach:.2f}"
			failure_text = f"Failure Pred: {fail:.2f}"

			# Define text positions
			uncertainty_position = (10, height - 10)  # Bottom-left corner
			failure_position = (10, height - 20)        # Above uncertainty text
			reachability_position = (10, height - 30)     # Above failure text

			# Add texts to the frame
			cv2.putText(frame_bgr, uncertainty_text, uncertainty_position, font, 
						font_scale, font_color, thickness, cv2.LINE_AA)
			cv2.putText(frame_bgr, failure_text, failure_position, font, 
						font_scale, font_color, thickness, cv2.LINE_AA)
			cv2.putText(frame_bgr, reachability_text, reachability_position, font, 
						font_scale, font_color, thickness, cv2.LINE_AA)

			# Convert the frame back to RGB for imageio (since OpenCV uses BGR)
			frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
			writer.append_data(frame_rgb)

		writer.close()
		print(f"Video saved to {filename}")

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

		obs = self.env.reset(seed=0)
		self.teleop_interface.reset()
		
		self.keyboard.add_callback("K", self.save_video)
		self.keyboard.add_callback("L", self.reset_episode)

		self.latent = self.reset_episode()
		value_thr = 0.0
		uq_thr = 4.0 

		fig, axes = plt.subplots(1, 2, figsize=(10, 5))
		axes[0].set_title("Predicted (Econ Prior)")
		axes[1].set_title("Ground Truth")
		for ax in axes:
			ax.axis("off")

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

				pred_next_latent = self.wm.dynamics.img_step(self.latent, actions)
				pred_next_feat = self.wm.dynamics.get_feat(pred_next_latent)
				next_best_act, log_prob = self.agent.select_action(pred_next_feat, eval_mode=True)

				reachability_value_Q1 = self.agent.critic1(pred_next_feat, next_best_act).item()
				reachability_value_Q2 = self.agent.critic2(pred_next_feat, next_best_act).item()
				reachability_value = min(reachability_value_Q1, reachability_value_Q2)

				# # 3. Filter action with optimal policy
				is_filtered = False
				if (uncertainty > uq_thr) or (reachability_value < value_thr) :
				# if (reachability_value < value_thr) :
					new_actions, log_prob = self.agent.select_action(feats, eval_mode=True)

					actions[:, :6] = self.post_process_actions(new_actions)[:, :6]
					is_filtered = True
					reachability_value_filtered = self.agent.critic1(feats, new_actions).item()
					# self.latent = None

				failure_dist = self.wm.heads["failure"](feats)
				failure = failure_dist.mean.item()

				inputs = torch.concat([feats, actions], -1)
				post_uncertainty = self.ensemble.intrinsic_reward_penn(inputs).item()
				

				next_obs, reward, done, info = self.env.step(actions)

				post, prior, recon_loss = self.forward_latent(latent_prev=self.latent, act_prev=actions, obs=next_obs)
				self.latent = post
				
				for key, val in next_obs.items():
					self.transitions[key].append(val[0].cpu().numpy())

				self.transitions["action"].append(actions[0].cpu().numpy())
				self.transitions["reward"].append(reward[0].cpu().numpy())
				self.transitions["discount"].append(np.array(1.0, dtype=np.float32))
				self.transitions["logprob"].append(np.array(0.0, dtype=np.float32))
				self.transitions["uncertainty"].append(np.array(uncertainty, dtype=np.float32))
				self.transitions["reachability"].append(np.array(reachability_value, dtype=np.float32))
				self.transitions["is_filtered"].append(is_filtered)
				self.transitions["failure"].append(np.array(failure, dtype=np.float32))

				if not is_filtered:
					print("failure {:.4f} uq {:.4f}, recon {:.4f} reachability {:.4f}".format(failure, uncertainty, recon_loss, reachability_value))
				else:
					print("failure {:.4f} uq {:.4f}-> {:.4f}, recon {:.4f}, reachability{:.4f}, {:.4f}".format(failure, uncertainty, post_uncertainty, recon_loss, reachability_value, reachability_value_filtered)) #, actions.tolist())
				if done.item():
					self.latent = self.reset_episode()
					self.env.reset(seed=0)


def recursive_update(base, update):
	for key, value in update.items():
		if isinstance(value, dict) and key in base:
			recursive_update(base[key], value)
		else:
			base[key] = value

if __name__ == "__main__":
	"""
	Interactive Teleoperation with Reachability Safety Filter
	
	This script allows manual teleoperation of a robotic system with an active
	safety filter that intervenes when detecting potentially unsafe actions.
	
	Usage:
		python filter_with_dreamer_failure.py \
			--model_path "path/to/dreamer.pt" \
			--reachability_model_path "path/to/filter" \
			--enable_cameras
	
	Controls:
		- WASD: Move in X-Y plane
		- QE: Move up/down (Z axis)  
		- RF: Rotate around Z axis
		- K: Save current episode
		- L: Reset without saving
	"""
	
	parser = argparse.ArgumentParser(description="Interactive teleoperation with safety filter")
	parser.add_argument("--configs", nargs="+", help="Configuration names to use")
	parser.add_argument("--enable_cameras", action="store_true", default=True, help="Enable camera rendering")
	parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
	parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations")
	parser.add_argument("--task", type=str, default="Isaac-Takeoff-Franka-IK-Rel-v0", help="Isaac environment task name")
	
	# Model path arguments for easy CLI usage
	parser.add_argument("--model_path", type=str, help="Path to pretrained world model (dreamer.pt)")
	parser.add_argument("--reachability_model_path", type=str, help="Path to pretrained reachability filter directory")
	parser.add_argument("--filter_step", type=int, default=200000, help="Filter checkpoint step to load")
	
	config, remaining = parser.parse_known_args()

	# Load default configuration
	yaml = yaml.YAML(typ="safe", pure=True)
	config_path = Path('source/latent_safety/reachability_dreamer/config_failure.yaml')
	
	if not config_path.exists():
		# Fallback to alternative path
		config_path = Path('latent_safety/reachability/config.yaml')
	
	if config_path.exists():
		configs = yaml.load(config_path.read_text())
		name_list = ["defaults"]
	else:
		print(f"Warning: Configuration file not found. Using minimal defaults.")
		configs = {"defaults": {}}
		name_list = ["defaults"]

	defaults = {}
	for name in name_list:
		if name in configs:
			recursive_update(defaults, configs[name])
	
	# Merge CLI arguments into defaults
	for key, value in vars(config).items():
		if value is not None:
			defaults[key] = value

	# Create final parser with all configuration options
	parser = argparse.ArgumentParser(description="Interactive teleoperation with safety filter")
	for key, value in sorted(defaults.items(), key=lambda x: x[0]):
		arg_type = tools.args_type(value)
		parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
		
	final_config = parser.parse_args(remaining)
	
	# Validate required paths
	if not hasattr(final_config, 'model_path') or not final_config.model_path:
		print("Error: --model_path is required. Please specify the path to the pretrained world model.")
		print("Example: --model_path 'latent_safety/dreamer.pt'")
		exit(1)
	
	if not os.path.exists(final_config.model_path):
		print(f"Error: World model file not found: {final_config.model_path}")
		print("Please ensure you have downloaded the pretrained models.")
		exit(1)
	
	print("🎮 Starting Interactive Teleoperation with Safety Filter")
	print("=" * 60)
	print(f"World Model: {final_config.model_path}")
	print(f"Safety Filter: {getattr(final_config, 'reachability_model_path', 'None (disabled)')}")
	print(f"Task: {final_config.task}")
	print("=" * 60)

	teleop = TeleoperationWithWM(final_config)
	teleop.run()
