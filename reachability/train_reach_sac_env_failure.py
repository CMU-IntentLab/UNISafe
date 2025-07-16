import os
import sys
import argparse
import time
from datetime import datetime
from warnings import simplefilter
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import ruamel.yaml as yaml
from termcolor import cprint
import wandb 
import collections
import gymnasium as gym

sys.path.append('latent_safety')
import dreamerv3_torch.dreamer as dreamer
import dreamerv3_torch.tools as tools
import dreamerv3_torch.models as models
import dreamerv3_torch.uncertainty as uncertainty

from reachability_dreamer.policy.SAC_Reachability_Env import SAC 

matplotlib.use('Agg')
simplefilter(action='ignore', category=FutureWarning)

timestr = time.strftime("%Y-%m-%d-%H_%M")

class RARLAgentDDPG:
	def __init__(self, config):
		self.config = config
		self.timestr = time.strftime("%Y-%m-%d-%H_%M")
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.setup_output_folders()
		self.initialize_environment()
		self.initialize_agent()

	def setup_output_folders(self):

		self.outFolder = os.path.join(self.config.outFolder, self.config.remark)
		self.figureFolder = os.path.join(self.outFolder, 'figure')
		os.makedirs(self.figureFolder, exist_ok=True)

	def initialize_environment(self):

		self.env = dreamer.make_env(config, num_envs=1)
		self.setup_world_model()
		self.setup_datasets()

	def setup_world_model(self):
		"""
		Loads a learned world model (if used), along with an LX MLP or ensemble, 
		then attaches them to the environment's car.
		"""

		acts = self.env.single_action_space
		# Normalized Action Space.
		acts.low = np.ones_like(acts.low) * -1
		acts.high = np.ones_like(acts.high) 
		self.config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

		logger = tools.DummyLogger(None, 1)

		agent = dreamer.Dreamer(
			self.env.single_observation_space,
			acts,
			self.config,
			logger=logger,
			dataset=None,
		)
		agent.requires_grad_(requires_grad=False)

		checkpoint = torch.load(self.config.model_path, map_location=torch.device('cpu'))
		agent.load_state_dict(checkpoint["agent_state_dict"])

		self.wm = agent._wm.to(self.config.device)
		self.disag_ensemble = agent._disag_ensemble.to(self.config.device)

	def setup_datasets(self):
		"""
		Example method that may load or create expert dataset for model-based RL or IRL.
		"""

		train_eps = None
		for offline_dir in self.config.offline_traindir:
			directory = offline_dir.format(**vars(self.config))
			train_eps = tools.load_episodes(directory, limit=self.config.dataset_size, episodes=train_eps)

		train_dataset = dreamer.make_dataset(train_eps, self.config)

		self.expert_dataset = train_dataset

	def initialize_agent(self):
		"""
		Initialize the SAC agent, building networks and replay buffer.
		"""

		# If using world model latents: dyn_stoch + dyn_deter, etc.
		if self.config.dyn_discrete:
			stateDim = (self.config.dyn_stoch * self.config.dyn_discrete) + self.config.dyn_deter
		else:
			stateDim = self.config.dyn_stoch + self.config.dyn_deter

		# Action dimension (assume continuous for SAC)
		if hasattr(self.env.action_space, 'shape'):
			actionDim = self.config.num_actions
		else:
			# if the environment is discrete, you must adapt the code accordingly.
			raise ValueError("SAC typically assumes a continuous action space.")

		actor_dimList = self.config.control_net  # Or choose a separate hyperparam for actor
		critic_dimList = self.config.critic_net

		self.agent = SAC(
			CONFIG=self.config,
			dim_state=stateDim,
			dim_action=actionDim,
			actor_dimList=actor_dimList,
			critic_dimList=critic_dimList
		)

		self.agent.setup_env(self.config, self.wm, self.disag_ensemble, self.expert_dataset, env_name="Takeoff_WM_Failure-v0")

	def train_agent(self):
		print("\n== Training DDPG Agent ==")
		self.agent.train()

		trainRecords = self.agent.learn(
			dataset=self.expert_dataset,
			wm=self.wm,
			ensemble=self.disag_ensemble,
			MAX_UPDATES=self.config.maxUpdates,
			checkPeriod=self.config.checkPeriod,
			outFolder=self.outFolder,
		)

		# trainRecords is an array of (critic_loss, actor_loss, alpha_loss) logged each update

	def execute(self):
		"""
		The main entry point if you want to do everything:
		1. Warm up
		2. Train
		3. Evaluate
		"""

		self.train_agent()


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

	# Create the SAC agent wrapper
	sac_agent = RARLAgentDDPG(final_config)
	sac_agent.execute()

	wandb.finish()
