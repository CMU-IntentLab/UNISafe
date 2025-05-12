import argparse
import pathlib
import functools
import numpy as np
import ruamel.yaml as yaml
import gymnasium as gym
import torch
import sys
from tqdm import trange

import dreamerv3_torch.dreamer as dreamer
import dreamerv3_torch.tools as tools

sys.path.append('source/latent_safety')
from takeoff import mdp
from takeoff.config import franka

from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


def count_steps(folder):
	return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))

def main(config):
	tools.set_seed_everywhere(config.seed)
	if config.deterministic_run:
		tools.enable_deterministic_run()
	
	logdir = pathlib.Path(config.logdir).expanduser()
	config.traindir = config.traindir or logdir / "train_eps"
	config.evaldir = config.evaldir or logdir / "eval_eps"
	config.steps //= config.action_repeat
	config.eval_every //= config.action_repeat
	config.log_every //= config.action_repeat
	config.time_limit //= config.action_repeat

	print("Logdir", logdir)
	logdir.mkdir(parents=True, exist_ok=True)
	config.traindir.mkdir(parents=True, exist_ok=True)
	config.evaldir.mkdir(parents=True, exist_ok=True)
	step = count_steps(config.traindir)
	# step in logger is environmental step
	logger = tools.Logger(logdir, config.action_repeat * step)

	# Save Config
	logger.config(vars(config))
	logger.write()

	print("Create envs.")
	if config.offline_traindir:
		# Load from multiple offline traindir:
		train_eps = None
		for offline_dir in config.offline_traindir:
			directory = offline_dir.format(**vars(config))
			train_eps = tools.load_episodes(directory, limit=config.dataset_size, episodes=train_eps)
	else:
		directory = config.traindir
		train_eps = tools.load_episodes(directory, limit=config.dataset_size)
		
	if config.offline_evaldir:
		directory = config.offline_evaldir.format(**vars(config))
	else:
		directory = config.evaldir
	eval_eps = tools.load_episodes(directory, limit=1)

	train_envs = dreamer.make_env(config, num_envs=config.envs)
	acts = train_envs.single_action_space

	# Normalized Action!
	acts.low = 0.5 * np.ones_like(acts.low) * -1
	acts.high = 0.5 * np.ones_like(acts.high) 
	print("Action Space", acts)
	config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

	state = None

	if not config.offline_traindir:
		prefill = max(0, config.prefill - count_steps(config.traindir))
		print(f"Prefill dataset ({prefill} steps).")
		if hasattr(acts, "discrete"):
			random_actor = tools.OneHotDist(
				torch.zeros(config.num_actions).repeat(config.envs, 1)
			)
		else:
			random_actor = torchd.independent.Independent(
				torchd.uniform.Uniform(
					torch.tensor(acts.low).repeat(config.envs, 1),
					torch.tensor(acts.high).repeat(config.envs, 1),
				),
				1,
			)

		def random_agent(o, d, s):
			action = random_actor.sample()
			logprob = random_actor.log_prob(action)
			return {"action": action, "logprob": logprob}, None

		state = tools.simulate_vecenv(
			random_agent,
			train_envs,
			train_eps,
			config.traindir,
			logger,
			limit=config.dataset_size,
			steps=prefill,
		)
		logger.step += prefill * config.action_repeat
		print(f"Logger: ({logger.step} steps).")

	print("Simulate agent.")
	train_dataset = dreamer.make_dataset(train_eps, config)
	eval_dataset = dreamer.make_dataset(eval_eps, config)

	agent = dreamer.Dreamer(
		train_envs.single_observation_space,
		acts,
		config,
		logger,
		train_dataset,
	).to(config.device)
	agent.requires_grad_(requires_grad=False)
	
	# Load a pretrained model
	if config.model_path:
		checkpoint = torch.load(config.model_path)
		agent.load_state_dict(checkpoint["agent_state_dict"])
		del checkpoint
		torch.cuda.empty_cache()  # Clear GPU memory cache
		agent._should_pretrain._once = False

	if config.model_only:
		for idx_step in trange(int(config.steps), desc="Training Dreamer with Offline Dataset", ncols=0, leave=False):

			agent.train_model_only(training=True)
			# agent.train_uncertainty_only(training=True)

			if ((idx_step + 1) % config.log_every) == 0:
				items_to_save = {
					"agent_state_dict": agent.state_dict(),
					"optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
				}
				torch.save(items_to_save, logdir / "latest.pt")
			
			if ((idx_step + 1) % config.save_every) == 0:
				items_to_save = {
					"agent_state_dict": agent.state_dict(),
					"optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
				}
				torch.save(items_to_save, logdir / "model_{:04d}".format(idx_step + 1))
			
	else:
		while agent._step < config.steps + config.eval_every:
			logger.write()
			if config.eval_episode_num > 0:
				print("Start evaluation.")

				with torch.no_grad():
					eval_policy = functools.partial(agent, training=False)
					tools.simulate_vecenv(
						eval_policy,
						train_envs,
						eval_eps,
						config.evaldir,
						logger,
						is_eval=True,
						episodes=config.eval_episode_num,
						save_success=True
					)
					if config.video_pred_log:
						if config.use_ensemble:
							video_pred = agent._wm.video_pred(next(eval_dataset), ensemble=agent._disag_ensemble, flow=agent._density_estimator)
							logger.video("eval_openl", video_pred)
						else:
							video_pred = agent._wm.video_pred(next(eval_dataset))
							logger.video("eval_openl", to_np(video_pred))
						
			print("Start training.")
			state = tools.simulate_vecenv(
				agent,
				train_envs,
				train_eps,
				config.traindir,
				logger,
				limit=config.dataset_size,
				steps=config.eval_every,
				state=state,
				save_success=True
			)
			items_to_save = {
				"agent_state_dict": agent.state_dict(),
				"optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
			}
			torch.save(items_to_save, logdir / "latest.pt")
			# TODO: add save_every.

	try:
		train_envs.close()
	except Exception:
		pass

from datetime import datetime
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
	# parser.add_argument("--task", type=str, default="Isaac-Takeoff-Hard-Franka-IK-Rel-v0", help="Name of the task.")

	args, remaining = parser.parse_known_args()
	configs = yaml.safe_load(
		(pathlib.Path(sys.argv[0]).parent / "dreamerv3_torch/configs.yaml").read_text()
	)
	
	def recursive_update(base, update):
		for key, value in update.items():
			if isinstance(value, dict) and key in base:
				recursive_update(base[key], value)
			else:
				base[key] = value

	name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
	defaults = {}
	for name in name_list:
		recursive_update(defaults, configs[name])

	# Overwrite defaults with command-line arguments
	for key, value in vars(args).items():
		defaults[key] = value

	parser = argparse.ArgumentParser()
	for key, value in sorted(defaults.items(), key=lambda x: x[0]):
		arg_type = tools.args_type(value)
		parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
	
	final_config = parser.parse_args(remaining)
	curr_time = datetime.now().strftime("%m%d/%H%M%S")
	expt_name = ( curr_time + "_" + final_config.remark )
	final_config.logdir = f"{final_config.logdir}/{expt_name}"

	main(final_config)

	dreamer.simulation_app.close()