import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import sys
import argparse
import time
from datetime import datetime
from warnings import simplefilter
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ruamel.yaml as yaml
from termcolor import cprint
import wandb 
import json
import collections
import gymnasium as gym

sys.path.append('latent_safety')
import dreamerv3_torch.dreamer as dreamer
import dreamerv3_torch.models as models
import dreamerv3_torch.tools as tools
from reachability_dreamer.policy.SAC_Reachability_Env import SAC 

import torch
import omni.isaac.core.utils.torch as torch_utils

matplotlib.use('Agg')
simplefilter(action='ignore', category=FutureWarning)

timestr = time.strftime("%Y-%m-%d-%H_%M")


tools.enable_deterministic_run()

class EvalAgent:
	def __init__(self, config):

		tools.set_seed_everywhere(config.seed)
		tools.enable_deterministic_run()
		torch_utils.set_seed(config.seed, torch_deterministic=True)

		self.config = config
		self.timestr = time.strftime("%Y-%m-%d-%H_%M")
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.setup_output_folders()

		self.initialize_environment()

	def setup_output_folders(self):

		self.outFolder = os.path.join(self.config.outFolder, self.config.remark)
		os.makedirs(self.outFolder, exist_ok=True)

	def initialize_environment(self):

		self.env = dreamer.make_env(config, num_envs=1)
		self.env.seed(self.config.seed)
		self.setup_world_model()

	def setup_world_model(self):
		"""
		Loads a learned world model (if used), along with an LX MLP or ensemble, 
		then attaches them to the environment's car.
		"""

		acts = self.env.single_action_space
		# Normalized Action Space.
		acts.low = 0.5 * np.ones_like(acts.low) * -1
		acts.high = 0.5 * np.ones_like(acts.high) 
		self.config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

		logger = tools.DummyLogger(None, 1)

		# 1. Load Dreamer for Latent Dynamics
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
		# self.disag_ensemble = agent._density_estimator.to(self.config.device)

		# 2. Load Dreamer for Base Policy
		self.policy = dreamer.Dreamer(
			self.env.single_observation_space,
			acts,
			self.config,
			logger=logger,
			dataset=None,
		)

		# Offline Policy
		checkpoint = torch.load(self.config.policy_model_path, map_location=torch.device('cpu'))

		# Online Policy
		filtered_state_dict = {k: v for k, v in checkpoint["agent_state_dict"].items()
			if (not k.startswith('_disag_ensemble')) 
			and (not k.startswith('_density_estimator'))
		}
		self.policy.load_state_dict(filtered_state_dict, strict=False)

		# 3. Load Trained Safety Filter
		if self.config.dyn_discrete:
			stateDim = (self.config.dyn_stoch * self.config.dyn_discrete) + self.config.dyn_deter
		else:
			stateDim = self.config.dyn_stoch + self.config.dyn_deter
		actionDim = self.config.num_actions
		actor_dimList = self.config.control_net
		critic_dimList = self.config.critic_net

		if self.config.reachability_model_path:	
			self.filter = SAC(
				CONFIG=self.config,
				dim_state=stateDim,
				dim_action=actionDim,
				actor_dimList=actor_dimList,
				critic_dimList=critic_dimList
			)
			step = 200000
			self.filter.restore(step, self.config.reachability_model_path)
		else:
			self.filter = None
			
	def execute(self):
		"""
		The main entry point if you want to do everything:
		1. Warm up
		2. Train
		3. Evaluate
		"""

		self.evaluate_trajectories()

	def evaluate_trajectories(self):

		with torch.no_grad():
			total_episodes, success_rate, failure_rate, timeout_rate, finished_episodes_log = tools.evaluate_venv_filtering(
				agent_base=self.policy,
				vecenv=self.env,
				latent_dynamics=self.wm,
				latent_ensemble=self.disag_ensemble,
				latent_filter=self.filter,
				episodes=self.config.num_episodes,
				config=self.config,
				is_filter=self.config.is_filter
			)

		filepath = os.path.join(self.outFolder, "episodes_log.json")
		traj_logs = convert_numpy_values(finished_episodes_log)
		
		with open(filepath, 'w') as f:
			json.dump(traj_logs, f, indent=4)

		# Logging
		wandb.log({
			"eval/total_episodes": total_episodes,
			"eval/success_rate": success_rate,
			"eval/failure_rate": failure_rate,
			"eval/incompletion_rate": timeout_rate,
		})

		lengths = np.array([r['length'] for r in finished_episodes_log], dtype=np.float32)
		max_uncs = np.array([r['max_uncertainty'] for r in finished_episodes_log], dtype=np.float32)
		mean_uncs = np.array([r['mean_uncertainty'] for r in finished_episodes_log], dtype=np.float32)
		max_kld = np.array([r['max_kld'] for r in finished_episodes_log], dtype=np.float32)
		mean_kld = np.array([r['mean_kld'] for r in finished_episodes_log], dtype=np.float32)
		num_filtered = np.array([r['num_filtered'] for r in finished_episodes_log], dtype=np.float32)

		# Compute means
		length_mean = np.mean(lengths)
		max_unc_mean = np.mean(max_uncs)
		mean_unc_mean = np.mean(mean_uncs)
		max_kld_mean = np.mean(max_kld)
		mean_kld_mean = np.mean(mean_kld)
		num_filtered_mean = np.mean(num_filtered)

		# Compute standard deviations
		length_std = np.std(lengths)
		max_unc_std = np.std(max_uncs)
		mean_unc_std = np.std(mean_uncs)
		max_kld_std = np.std(max_kld)
		mean_kld_std = np.std(mean_kld)
		num_filtered_std = np.std(num_filtered)

		# Print or log the results
		print("=== Episode Results Summary ===")
		print(f"Count of Episodes: {len(finished_episodes_log)}")
		print(f"Length (mean ± std): {length_mean:.2f} ± {length_std:.2f}")
		print(f"Max Uncertainty (mean ± std): {max_unc_mean:.4f} ± {max_unc_std:.4f}")
		print(f"Mean Uncertainty (mean ± std): {mean_unc_mean:.4f} ± {mean_unc_std:.4f}")
		print(f"Max KLD (mean ± std): {max_kld_mean:.4f} ± {max_kld_std:.4f}")
		print(f"Mean KLD (mean ± std): {mean_kld_mean:.4f} ± {mean_kld_std:.4f}")
		print(f"Num Filtered (mean ± std): {num_filtered_mean:.2f} ± {num_filtered_std:.2f}")

		# If you want, you can return these stats as a dict for further use or logging:
		stats = {
			"episode_count": len(finished_episodes_log),
			"length_mean": length_mean,
			"length_std": length_std,
			"max_unc_mean": max_unc_mean,
			"max_unc_std": max_unc_std,
			"mean_unc_mean": mean_unc_mean,
			"mean_unc_std": mean_unc_std,
			"max_kld_mean": max_kld_mean,
			"max_kld_std": max_kld_std,
			"mean_kld_mean": mean_kld_mean,
			"mean_kld_std": mean_kld_std,
			"num_filtered_mean": num_filtered_mean,
			"num_filtered_std": num_filtered_std,
		}

		wandb.log({
			"eval/episode_count": stats["episode_count"],
			"eval/length_mean": stats["length_mean"],
			"eval/length_std": stats["length_std"],
			"eval/max_unc_mean": stats["max_unc_mean"],
			"eval/max_unc_std": stats["max_unc_std"],
			"eval/mean_unc_mean": stats["mean_unc_mean"],
			"eval/mean_unc_std": stats["mean_unc_std"],

			"eval/max_kld_mean": stats["max_kld_mean"],
			"eval/max_kld_std": stats["max_kld_std"],
			"eval/mean_kld_mean": stats["mean_kld_mean"],
			"eval/mean_kld_std": stats["mean_kld_std"],
			
			"eval/num_filtered_mean": stats["num_filtered_mean"],
			"eval/num_filtered_std": stats["num_filtered_std"],
		})

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

	# Create the SAC agent wrapper
	sac_agent = EvalAgent(final_config)
	sac_agent.execute()

	wandb.finish()
