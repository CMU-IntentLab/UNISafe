"""
Reachability Filter Evaluation Script

This script evaluates trained reachability filters in combination with base policies
to assess safety performance in challenging environments. The evaluation compares
the performance of:
1. Base policy alone (no safety filter)
2. Base policy with reachability safety filter

Key Features:
- Loads pre-trained world models for environment simulation
- Loads pre-trained base policies (e.g., Dreamer agents)
- Loads trained reachability filters (SAC-based safety filters)
- Comprehensive safety metrics evaluation
- Uncertainty and failure prediction analysis
- Deterministic evaluation for reproducible results
"""

import os
# Configure CUDA for deterministic operation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import argparse
import time
from datetime import datetime
from warnings import simplefilter
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ruamel.yaml as yaml
from termcolor import cprint
import wandb 
import json
import collections
import gymnasium as gym

# Add latent_safety to Python path for imports
sys.path.append('latent_safety')
import dreamerv3_torch.dreamer as dreamer
import dreamerv3_torch.models as models
import dreamerv3_torch.tools as tools
from reachability.policy.SAC_Reachability_Env import SAC 

import torch
import omni.isaac.core.utils.torch as torch_utils

# Configure matplotlib for headless operation
matplotlib.use('Agg')
simplefilter(action='ignore', category=FutureWarning)

# Global timestamp for logging
timestr = time.strftime("%Y-%m-%d-%H_%M")

# Enable deterministic operation for reproducible evaluation
tools.enable_deterministic_run()

class ReachabilityFilterEvaluator:
	"""
	Evaluates reachability-based safety filters in combination with base policies.
	
	This class loads pre-trained models and evaluates their safety performance:
	1. World model for environment simulation
	2. Base policy for task completion
	3. Reachability filter for safety constraints
	
	The evaluator runs multiple episodes and collects comprehensive metrics
	including success rates, failure rates, uncertainty statistics, and
	filter intervention frequency.
	
	Attributes:
		config: Evaluation configuration containing model paths and parameters
		device: GPU/CPU device for evaluation
		env: Environment for policy evaluation
		wm: Pre-trained world model for latent dynamics
		disag_ensemble: Ensemble model for uncertainty quantification
		policy: Base policy for task execution
		filter: Reachability filter for safety (optional)
	"""
	
	def __init__(self, config: Any):
		"""
		Initialize the reachability filter evaluator.
		
		Args:
			config: Configuration object containing evaluation parameters,
					model paths, and environment settings
		"""
		# Set up deterministic evaluation
		tools.set_seed_everywhere(config.seed)
		tools.enable_deterministic_run()
		torch_utils.set_seed(config.seed, torch_deterministic=True)

		self.config = config
		self.timestr = time.strftime("%Y-%m-%d-%H_%M")
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		print(f"Initializing evaluator with seed: {config.seed}")
		print(f"Device: {self.device}")
		
		# Initialize components
		self.setup_output_folders()
		self.initialize_environment()

	def setup_output_folders(self) -> None:
		"""
		Create necessary output directories for evaluation results.
		
		Creates experiment-specific folders for storing evaluation logs,
		episode data, and analysis results.
		"""
		self.outFolder = os.path.join(self.config.outFolder, self.config.remark)
		os.makedirs(self.outFolder, exist_ok=True)
		
		print(f"Output folder created: {self.outFolder}")

	def initialize_environment(self) -> None:
		"""
		Initialize the evaluation environment and load all required models.
		
		Sets up:
		1. Dreamer environment for evaluation
		2. Pre-trained world model and uncertainty ensemble
		3. Base policy for task execution
		4. Reachability filter for safety (if specified)
		"""
		print("Initializing evaluation environment...")
		
		# Create evaluation environment
		self.env = dreamer.make_env(self.config, num_envs=1)
		self.env.seed(self.config.seed)
		print(f"Environment created and seeded with: {self.config.seed}")
		
		# Load all required models
		self.setup_world_model()
		
		print("Environment initialization complete.")

	def setup_world_model(self) -> None:
		"""
		Load and configure all pre-trained models for evaluation.
		
		This method loads:
		1. World model with dynamics and uncertainty ensemble
		2. Base policy for task execution
		3. Reachability safety filter (if available)
		
		All models should be pre-trained before running evaluation.
		"""
		print("Setting up pre-trained models...")
		
		# Configure action space normalization
		action_space = self.env.single_action_space
		action_space.low = 0.5 * np.ones_like(action_space.low) * -1
		action_space.high = 0.5 * np.ones_like(action_space.high) 
		self.config.num_actions = (
			action_space.n if hasattr(action_space, "n") 
			else action_space.shape[0]
		)
		
		print(f"Action space configured: {self.config.num_actions} dimensions")

		# Create dummy logger for model loading
		logger = tools.DummyLogger(None, 1)

		# 1. Load World Model for Latent Dynamics and Uncertainty
		print("Loading world model...")
		if not os.path.exists(self.config.model_path):
			raise FileNotFoundError(
				f"World model checkpoint not found at: {self.config.model_path}.\n"
				f"Please ensure you have uploaded the pre-trained world model file."
			)
		
		world_model_agent = dreamer.Dreamer(
			self.env.single_observation_space,
			action_space,
			self.config,
			logger=logger,
			dataset=None,
		)
		world_model_agent.requires_grad_(requires_grad=False)

		checkpoint = torch.load(self.config.model_path, map_location=torch.device('cpu'))
		world_model_agent.load_state_dict(checkpoint["agent_state_dict"])

		self.wm = world_model_agent._wm.to(self.config.device)
		self.disag_ensemble = world_model_agent._disag_ensemble.to(self.config.device)
		print(f"World model loaded from: {self.config.model_path}")

		# 2. Load Base Policy for Task Execution
		print("Loading base policy...")
		if not os.path.exists(self.config.policy_model_path):
			raise FileNotFoundError(
				f"Base policy checkpoint not found at: {self.config.policy_model_path}.\n"
				f"Please ensure you have uploaded the pre-trained base policy file."
			)
		
		self.policy = dreamer.Dreamer(
			self.env.single_observation_space,
			action_space,
			self.config,
			logger=logger,
			dataset=None,
		)

		# Load base policy checkpoint (excluding uncertainty components)
		policy_checkpoint = torch.load(self.config.policy_model_path, map_location=torch.device('cpu'))
		
		# Filter out uncertainty-related components from base policy
		filtered_state_dict = {
			k: v for k, v in policy_checkpoint["agent_state_dict"].items()
			if (not k.startswith('_disag_ensemble')) 
			and (not k.startswith('_density_estimator'))
		}
		self.policy.load_state_dict(filtered_state_dict, strict=False)
		print(f"Base policy loaded from: {self.config.policy_model_path}")

		# 3. Load Reachability Safety Filter (Optional)
		print("Setting up reachability filter...")
		
		# Calculate state dimensions for filter
		if self.config.dyn_discrete:
			state_dim = (self.config.dyn_stoch * self.config.dyn_discrete) + self.config.dyn_deter
		else:
			state_dim = self.config.dyn_stoch + self.config.dyn_deter
		action_dim = self.config.num_actions - 1
		actor_network_dims = self.config.control_net
		critic_network_dims = self.config.critic_net

		if hasattr(self.config, 'reachability_model_path') and self.config.reachability_model_path:
			if not os.path.exists(self.config.reachability_model_path):
				print(f"Warning: Reachability filter path specified but file not found: {self.config.reachability_model_path}")
				self.filter = None
			else:
				print(f"Loading reachability filter from: {self.config.reachability_model_path}")
				self.filter = SAC(
					CONFIG=self.config,
					dim_state=state_dim,
					dim_action=action_dim,
					actor_dimList=actor_network_dims,
					critic_dimList=critic_network_dims
				)
				# Load filter at a specific training step (adjust as needed)
				filter_step = getattr(self.config, 'filter_step', 200000)
				self.filter.restore(filter_step, self.config.reachability_model_path)
				print(f"Reachability filter loaded successfully (step {filter_step})")
		else:
			print("No reachability filter specified - running base policy only")
			self.filter = None
		
		print("Model setup complete!")
		print(f"  - World model device: {next(self.wm.parameters()).device}")
		print(f"  - Policy device: {next(self.policy._modules.parameters()).device}")
		print(f"  - Filter available: {self.filter is not None}")
			
	def execute(self) -> None:
		"""
		Main execution pipeline for reachability filter evaluation.
		
		This method orchestrates the complete evaluation process:
		1. Model loading and setup (done in __init__)
		2. Policy evaluation with and without safety filter
		3. Comprehensive safety metrics analysis
		4. Results logging and visualization
		"""
		print("Starting reachability filter evaluation...")
		print("=" * 60)
		
		# Execute evaluation
		self.evaluate_trajectories()
		
		print("=" * 60)
		print("Reachability filter evaluation completed!")

	def evaluate_trajectories(self) -> None:
		"""
		Execute trajectory evaluation with comprehensive safety metrics collection.
		
		Runs evaluation episodes using the loaded models and collects detailed
		statistics about safety performance, including:
		- Success/failure/timeout rates
		- Uncertainty statistics (mean, max)
		- KL divergence measures
		- Filter intervention frequency
		- Episode length distributions
		"""
		print(f"Starting trajectory evaluation...")
		print(f"Number of episodes: {self.config.num_episodes}")
		print(f"Filter enabled: {getattr(self.config, 'is_filter', False)}")
		
		with torch.no_grad():
			# Run evaluation using the specialized filtering evaluation function
			evaluation_results = tools.evaluate_venv_filtering(
				agent_base=self.policy,
				vecenv=self.env,
				latent_dynamics=self.wm,
				latent_ensemble=self.disag_ensemble,
				latent_filter=self.filter,
				episodes=self.config.num_episodes,
				config=self.config,
				is_filter=getattr(self.config, 'is_filter', False)
			)
			
			# Unpack evaluation results
			(total_episodes, success_rate, failure_rate, 
			 timeout_rate, finished_episodes_log) = evaluation_results

		# Save detailed episode logs to file
		episodes_log_path = os.path.join(self.outFolder, "episodes_log.json")
		trajectory_logs = convert_numpy_values(finished_episodes_log)
		
		with open(episodes_log_path, 'w') as f:
			json.dump(trajectory_logs, f, indent=4)
		
		print(f"Detailed episode logs saved to: {episodes_log_path}")

		# Log basic evaluation metrics to wandb
		print("\n=== Basic Evaluation Results ===")
		print(f"Total episodes: {total_episodes}")
		print(f"Success rate: {success_rate:.3f}")
		print(f"Failure rate: {failure_rate:.3f}")
		print(f"Timeout rate: {timeout_rate:.3f}")
		
		wandb.log({
			"eval/total_episodes": total_episodes,
			"eval/success_rate": success_rate,
			"eval/failure_rate": failure_rate,
			"eval/incompletion_rate": timeout_rate,
		})

		# Extract detailed statistics from episode logs
		print("\n=== Extracting Detailed Statistics ===")
		
		lengths = np.array([r['length'] for r in finished_episodes_log], dtype=np.float32)
		max_uncertainties = np.array([r['max_uncertainty'] for r in finished_episodes_log], dtype=np.float32)
		mean_uncertainties = np.array([r['mean_uncertainty'] for r in finished_episodes_log], dtype=np.float32)
		max_kld = np.array([r['max_kld'] for r in finished_episodes_log], dtype=np.float32)
		mean_kld = np.array([r['mean_kld'] for r in finished_episodes_log], dtype=np.float32)
		num_filtered = np.array([r['num_filtered'] for r in finished_episodes_log], dtype=np.float32)

		# Compute summary statistics
		episode_statistics = {
			"episode_count": len(finished_episodes_log),
			"length_mean": np.mean(lengths),
			"length_std": np.std(lengths),
			"max_unc_mean": np.mean(max_uncertainties),
			"max_unc_std": np.std(max_uncertainties),
			"mean_unc_mean": np.mean(mean_uncertainties),
			"mean_unc_std": np.std(mean_uncertainties),
			"max_kld_mean": np.mean(max_kld),
			"max_kld_std": np.std(max_kld),
			"mean_kld_mean": np.mean(mean_kld),
			"mean_kld_std": np.std(mean_kld),
			"num_filtered_mean": np.mean(num_filtered),
			"num_filtered_std": np.std(num_filtered),
		}

		# Print detailed results summary
		print("\n=== Detailed Episode Statistics ===")
		print(f"Episodes analyzed: {episode_statistics['episode_count']}")
		print(f"Episode length: {episode_statistics['length_mean']:.2f} ± {episode_statistics['length_std']:.2f}")
		print(f"Max uncertainty: {episode_statistics['max_unc_mean']:.4f} ± {episode_statistics['max_unc_std']:.4f}")
		print(f"Mean uncertainty: {episode_statistics['mean_unc_mean']:.4f} ± {episode_statistics['mean_unc_std']:.4f}")
		print(f"Max KLD: {episode_statistics['max_kld_mean']:.4f} ± {episode_statistics['max_kld_std']:.4f}")
		print(f"Mean KLD: {episode_statistics['mean_kld_mean']:.4f} ± {episode_statistics['mean_kld_std']:.4f}")
		print(f"Filter interventions: {episode_statistics['num_filtered_mean']:.2f} ± {episode_statistics['num_filtered_std']:.2f}")

		# Log detailed statistics to wandb
		wandb_detailed_metrics = {
			f"eval/{key}": value for key, value in episode_statistics.items()
		}
		wandb.log(wandb_detailed_metrics)
		
		print(f"\nAll evaluation metrics logged to wandb successfully!")

def convert_numpy_values(obj):
    """
    Recursively convert numpy dtypes in a nested dict/list structure to native Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_values(elem) for elem in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

def recursive_update(base, update):
	for key, value in update.items():
		if isinstance(value, dict) and key in base:
			recursive_update(base[key], value)
		else:
			base[key] = value

def custom_exclude_fn(path, root=None):
	"""
	Excludes all non-.py files and any files within the 'logs/' subdirectory.
	"""
	normalized_path = path.replace(os.sep, '/')
	
	# Exclude any files within the 'logs/' directory
	if normalized_path.startswith("logs/"):
		return True
	
	if normalized_path.startswith("imgs/"):
		return True
	
	# Include all other .py files
	return False

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
	# parser.add_argument("--task", type=str, default="Isaac-Takeoff-Hard-Franka-IK-Rel-v0", help="Name of the task.")
	parser.add_argument("--task", type=str, default="Isaac-Takeoff-Franka-IK-Rel-v0", help="Name of the task.")
	config, remaining = parser.parse_known_args()

	yaml = yaml.YAML(typ="safe", pure=True)
	configs = yaml.load((Path(sys.argv[0]).parent / "config_filter_normal.yaml").read_text())
	name_list = ["defaults", *config.configs] if config.configs else ["defaults"]

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
	curr_time = datetime.now().strftime("%m%d/%H%M%S")
	expt_name = ( curr_time + "_" + final_config.remark )
	final_config.outFolder  = f"{final_config.logdir}/{expt_name}"

	# Initialize wandb
	wandb.init(project="FINAL_Filter_EVAL", name=expt_name)
	wandb.run.log_code("source/latent_safety/dreamerv3_torch", exclude_fn=custom_exclude_fn)
	wandb.run.log_code("source/latent_safety/reachability_dreamer", exclude_fn=custom_exclude_fn)
	wandb.config.update(final_config)

	# Create the reachability filter evaluator
	evaluator = ReachabilityFilterEvaluator(final_config)
	evaluator.execute()

	wandb.finish()
