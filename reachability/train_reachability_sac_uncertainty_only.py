"""
Reachability SAC Training with Uncertainty Quantification Only

This script trains a SAC (Soft Actor-Critic) agent for reachability analysis using
a world model environment that relies solely on epistemic uncertainty quantification
for safety. Unlike the failure prediction version, this focuses purely on avoiding
regions where the world model is uncertain about its predictions.

Key Features:
- World model-based training with pre-trained dynamics
- Epistemic uncertainty quantification for out-of-distribution detection
- SAC algorithm for continuous action spaces
- Pure uncertainty-based rewards (no failure prediction)
- Comprehensive logging and evaluation metrics
"""

import os
import sys
import argparse
import time
from datetime import datetime
from warnings import simplefilter
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import ruamel.yaml as yaml
from termcolor import cprint
import wandb 
import collections
import gymnasium as gym

# Add latent_safety to Python path for imports
sys.path.append('latent_safety')
import dreamerv3_torch.dreamer as dreamer
import dreamerv3_torch.tools as tools
import dreamerv3_torch.models as models
import dreamerv3_torch.uncertainty as uncertainty

from reachability.policy.SAC_Reachability_Env import SAC 

# Configure matplotlib for headless operation
matplotlib.use('Agg')
simplefilter(action='ignore', category=FutureWarning)

# Global timestamp for logging
timestr = time.strftime("%Y-%m-%d-%H_%M")

class ReachabilityTrainerUncertaintyOnly:
	"""
	Reachability-aware SAC trainer that uses world models with uncertainty quantification only.
	
	This class orchestrates the training of a SAC agent in a world model environment
	that provides safety-oriented rewards based purely on epistemic uncertainty.
	The agent learns to avoid regions where the model predictions are uncertain,
	without explicit failure prediction.
	
	Attributes:
		config: Training configuration containing hyperparameters and paths
		device: GPU/CPU device for training
		wm: Pre-trained world model for environment simulation
		disag_ensemble: Ensemble model for uncertainty quantification
		expert_dataset: Dataset for initialization and offline evaluation
		agent: SAC agent for policy learning
	"""
	
	def __init__(self, config: Any):
		"""
		Initialize the reachability trainer with uncertainty quantification only.
		
		Args:
			config: Configuration object containing training parameters, model paths,
					and environment settings
		"""
		self.config = config
		self.timestr = time.strftime("%Y-%m-%d-%H_%M")
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		# Initialize components in order
		self.setup_output_folders()
		self.initialize_environment()
		self.initialize_agent()

	def setup_output_folders(self) -> None:
		"""
		Create necessary output directories for logging and saving results.
		
		Creates experiment-specific folders based on configuration parameters
		for storing training logs, figures, and model checkpoints.
		"""
		# Create main output folder with experiment-specific name
		self.outFolder = os.path.join(self.config.outFolder, self.config.remark)
		self.figureFolder = os.path.join(self.outFolder, 'figure')
		
		# Ensure directories exist
		os.makedirs(self.figureFolder, exist_ok=True)
		
		print(f"Output folder created: {self.outFolder}")
		print(f"Figure folder created: {self.figureFolder}")

	def initialize_environment(self) -> None:
		"""
		Initialize the training environment and associated components.
		
		Sets up the dreamer environment, loads the pre-trained world model,
		and prepares the expert dataset for training.
		"""
		print("Initializing environment...")
		
		# Create the dreamer environment with single environment instance
		self.env = dreamer.make_env(self.config, num_envs=1)
		print(f"Environment created: {self.env}")
		
		# Load pre-trained world model and uncertainty ensemble
		self.setup_world_model()
		
		# Load expert dataset for training
		self.setup_datasets()
		
		print("Environment initialization complete.")

	def setup_world_model(self) -> None:
		"""
		Load and configure the pre-trained world model and uncertainty ensemble.
		
		This method loads a pre-trained Dreamer world model that includes:
		1. Dynamics model for state transitions
		2. Disagreement ensemble for epistemic uncertainty quantification
		
		Note: This version does NOT require failure prediction capabilities.
		The world model should be trained separately before running this script.
		"""
		print("Setting up world model...")
		
		# Configure action space normalization
		action_space = self.env.single_action_space
		action_space.low = np.ones_like(action_space.low) * -1
		action_space.high = np.ones_like(action_space.high) 
		self.config.num_actions = (
			action_space.n if hasattr(action_space, "n") 
			else action_space.shape[0]
		)
		
		print(f"Action space configured: {self.config.num_actions} dimensions")

		# Create dummy logger for model loading
		logger = tools.DummyLogger(None, 1)

		# Initialize Dreamer agent architecture
		dreamer_agent = dreamer.Dreamer(
			self.env.single_observation_space,
			action_space,
			self.config,
			logger=logger,
			dataset=None,
		)
		dreamer_agent.requires_grad_(requires_grad=False)

		# Load pre-trained world model checkpoint
		if not os.path.exists(self.config.model_path):
			raise FileNotFoundError(
				f"World model checkpoint not found at: {self.config.model_path}.\n"
				f"Please ensure you have uploaded the pre-trained world model file."
			)
		
		print(f"Loading world model from: {self.config.model_path}")
		checkpoint = torch.load(self.config.model_path, map_location=torch.device('cpu'))
		dreamer_agent.load_state_dict(checkpoint["agent_state_dict"])

		# Extract world model components and move to GPU
		self.wm = dreamer_agent._wm.to(self.config.device)
		self.disag_ensemble = dreamer_agent._disag_ensemble.to(self.config.device)
		
		print("World model loaded successfully:")
		print(f"  - World model device: {next(self.wm.parameters()).device}")
		print(f"  - Ensemble device: {next(self.disag_ensemble.parameters()).device}")
		print(f"  - Note: This trainer uses uncertainty quantification only (no failure prediction)")

	def setup_datasets(self) -> None:
		"""
		Load and prepare the expert dataset for training and environment initialization.
		
		This method loads demonstration data that will be used for:
		1. Initializing the world model environment states
		2. Offline evaluation and comparison
		3. Potential imitation learning components
		
		The dataset should contain high-quality demonstrations of safe behavior.
		"""
		print("Loading expert dataset...")
		
		if not hasattr(self.config, 'offline_traindir') or not self.config.offline_traindir:
			raise ValueError(
				"No offline training directories specified in config.offline_traindir. "
				"Please provide paths to demonstration datasets."
			)
		
		# Load episodes from all specified directories
		train_episodes = None
		total_episodes_loaded = 0
		
		for offline_dir in self.config.offline_traindir:
			directory = offline_dir.format(**vars(self.config))
			
			if not os.path.exists(directory):
				print(f"Warning: Dataset directory not found: {directory}")
				continue
				
			print(f"Loading episodes from: {directory}")
			train_episodes = tools.load_episodes(
				directory, 
				limit=self.config.dataset_size, 
				episodes=train_episodes
			)
			
			if train_episodes is not None:
				current_count = len(train_episodes)
				print(f"  Loaded {current_count - total_episodes_loaded} episodes")
				total_episodes_loaded = current_count

		if train_episodes is None or len(train_episodes) == 0:
			raise ValueError(
				"No episodes were loaded from the specified directories. "
				"Please check that the dataset paths are correct and contain valid episode data."
			)

		print(f"Total episodes loaded: {total_episodes_loaded}")
		
		# Note: Batch configuration can be customized here if needed
		# self.config.batch_length = 64
		# self.config.batch_size = 1
		
		# Create dataset iterator for training
		self.expert_dataset = dreamer.make_dataset(train_episodes, self.config)
		print("Expert dataset prepared successfully.")

	def initialize_agent(self) -> None:
		"""
		Initialize the SAC agent for reachability learning with uncertainty quantification only.
		
		Creates a SAC agent configured for the world model feature space and
		sets up the environment to use uncertainty-based rewards only. The agent
		will learn to navigate safely by avoiding uncertain regions without
		explicit failure prediction.
		"""
		print("Initializing SAC agent...")
		
		# Calculate state dimension from world model feature space
		if self.config.dyn_discrete:
			# Discrete stochastic dynamics: stochastic * discrete + deterministic
			state_dim = (
				self.config.dyn_stoch * self.config.dyn_discrete + 
				self.config.dyn_deter
			)
		else:
			# Continuous stochastic dynamics: stochastic + deterministic
			state_dim = self.config.dyn_stoch + self.config.dyn_deter

		# Validate continuous action space for SAC
		if hasattr(self.env.action_space, 'shape'):
			action_dim = self.config.num_actions - 1 # -1 for the gripper action
		else:
			raise ValueError(
				"SAC requires a continuous action space. "
				f"Environment action space: {type(self.env.action_space)}"
			)

		# Network architecture configuration
		actor_network_dims = self.config.control_net
		critic_network_dims = self.config.critic_net
		
		print(f"Agent configuration:")
		print(f"  - State dimension: {state_dim}")
		print(f"  - Action dimension: {action_dim}")
		print(f"  - Actor network: {actor_network_dims}")
		print(f"  - Critic network: {critic_network_dims}")

		# Create SAC agent
		self.agent = SAC(
			CONFIG=self.config,
			dim_state=state_dim,
			dim_action=action_dim,
			actor_dimList=actor_network_dims,
			critic_dimList=critic_network_dims
		)

		# Set up the agent with world model environment (uncertainty-only version)
		# Note: No env_name specified, so it defaults to uncertainty-only environment
		self.agent.setup_env(self.config, self.wm, self.disag_ensemble, self.expert_dataset)
		
		print("SAC agent initialized with uncertainty-only environment")
		print("Agent setup complete.")

	def train_agent(self) -> None:
		"""
		Execute the main training loop for the reachability SAC agent.
		
		Trains the agent to learn safe navigation policies by maximizing rewards
		based purely on epistemic uncertainty avoidance. The training uses the
		world model environment for efficient and safe policy learning.
		"""
		print("\n=== Training Reachability SAC Agent (Uncertainty Only) ===")
		print(f"Max updates: {self.config.maxUpdates}")
		print(f"Check period: {self.config.checkPeriod}")
		print(f"Output folder: {self.outFolder}")
		
		# Set agent to training mode
		self.agent.train()

		# Execute training loop
		training_records = self.agent.learn(
			dataset=self.expert_dataset,
			wm=self.wm,
			ensemble=self.disag_ensemble,
			MAX_UPDATES=self.config.maxUpdates,
			checkPeriod=self.config.checkPeriod,
			outFolder=self.outFolder,
		)

		# training_records contains (critic_loss, actor_loss, alpha_loss) for each update
		print("Training completed successfully!")
		print(f"Training records shape: {len(training_records) if training_records else 0}")

	def execute(self) -> None:
		"""
		Main execution pipeline for reachability training with uncertainty quantification only.
		
		This method orchestrates the complete training process:
		1. Environment and world model setup (done in __init__)
		2. Agent initialization (done in __init__)
		3. Training execution
		
		The trained agent will learn to navigate safely by avoiding regions
		where the world model has high epistemic uncertainty.
		"""
		print("Starting reachability training execution...")
		print("=" * 60)
		
		# Execute training
		self.train_agent()
		
		print("=" * 60)
		print("Reachability training execution completed!")


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
	parser.add_argument("--task", type=str, default="Isaac-Takeoff-Franka-IK-Rel-v0", help="Name of the task.")
	config, remaining = parser.parse_known_args()

	yaml = yaml.YAML(typ="safe", pure=True)
	configs = yaml.load((Path(sys.argv[0]).parent / "config.yaml").read_text())
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
	wandb.init(project="Final_Isaac_Reachability", name=expt_name)
	wandb.run.log_code("source/latent_safety/reachability_dreamer", exclude_fn=custom_exclude_fn)
	wandb.config.update(final_config)

	# Create the reachability trainer
	trainer = ReachabilityTrainerUncertaintyOnly(final_config)
	trainer.execute()

	wandb.finish()
