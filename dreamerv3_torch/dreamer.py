import argparse
import os
from omni.isaac.lab.app import AppLauncher

# Check if simulation app is already running (e.g., from Jupyter notebook)
def _is_app_already_running():
    try:
        import omni.kit.app
        return omni.kit.app.get_app() is not None
    except:
        return False

_APP_ALREADY_RUNNING = _is_app_already_running() or os.environ.get("ISAAC_JUPYTER_KERNEL", "0") == "1"

# Only launch app if no app is already running
if not _APP_ALREADY_RUNNING:
    # add argparse arguments
    parser = argparse.ArgumentParser(description="Isaac Lab environments.")
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    # parse the arguments
    args_cli, remaining = parser.parse_known_args()

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
else:
    # App is already running (e.g., from Jupyter notebook)
    simulation_app = None
    print("[INFO] Simulation app already running - skipping duplicate launch")

import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import gymnasium as gym
import torch

import carb

from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

import sys
sys.path.append('latent_safety')
from takeoff import mdp
from takeoff.config import franka

from dreamer_wrapper import DreamerVecEnvWrapper
import dreamerv3_torch.exploration as expl
import dreamerv3_torch.models as models
import dreamerv3_torch.tools as tools
import dreamerv3_torch.uncertainty as uncertainty
import envs.wrappers as wrappers

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
	def __init__(self, obs_space, act_space, config, logger, dataset):
		super(Dreamer, self).__init__()
		self._config = config
		self._logger = logger
		self._should_log = tools.Every(config.log_every)
		batch_steps = config.batch_size * config.batch_length
		self._should_train = tools.Every(batch_steps / config.train_ratio)
		self._should_pretrain = tools.Once()
		self._should_reset = tools.Every(config.reset_every)
		self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
		self._metrics = {}
		# this is update step
		self._step = logger.step // config.action_repeat
		self._update_count = 0
		self._dataset = dataset
		self._wm = models.WorldModel(obs_space, act_space, self._step, config)
		self._task_behavior = models.ImagBehavior(config, self._wm)
		if (
			config.compile and os.name != "nt"
		):  # compilation is not supported on windows
			self._wm = torch.compile(self._wm)
			self._task_behavior = torch.compile(self._task_behavior)
		reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
		self._expl_behavior = dict(
			greedy=lambda: self._task_behavior,
			random=lambda: expl.Random(config, act_space),
			plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
		)[config.expl_behavior]().to(self._config.device)

		if config.use_ensemble:
			self._disag_ensemble = uncertainty.OneStepPredictor(config, self._wm)

		else:
			self._disag_ensemble = None


	def __call__(self, obs, reset, state=None, training=True):
		step = self._step
		if training:
			steps = (
				self._config.pretrain
				if self._should_pretrain()
				else self._should_train(step)
			)
			for _ in range(steps):
				self._train(next(self._dataset))
				self._update_count += 1
				self._metrics["update_count"] = self._update_count
			if self._should_log(step):
				for name, values in self._metrics.items():
					self._logger.scalar(name, float(np.mean(values)))
					self._metrics[name] = []
				if self._config.video_pred_log:
					if self._config.use_ensemble:
						video_pred = self._wm.video_pred(next(self._dataset), ensemble=self._disag_ensemble)
						self._logger.video("train_openl", video_pred)
					else:
						openl = self._wm.video_pred(next(self._dataset))
						self._logger.video("train_openl", to_np(openl))
				self._logger.write(fps=True)

		policy_output, state = self._policy(obs, state, training)

		if training:
			self._step += len(reset)
			self._logger.step = self._config.action_repeat * self._step
		return policy_output, state
	
	def train_model_only(self, training=True):
		step = self._step
		if training:
			self._train(next(self._dataset))
			self._update_count += 1
			self._metrics["update_count"] = self._update_count

			if (step+1) % 1000 == 0:
				if self._config.video_pred_log:
					if self._config.use_ensemble:
						video_pred = self._wm.video_pred(next(self._dataset), ensemble=self._disag_ensemble)
						self._logger.video("train_openl", video_pred)
					else:
						openl = self._wm.video_pred(next(self._dataset))
						self._logger.video("train_openl", to_np(openl))

			for name, values in self._metrics.items():
				self._logger.scalar(name, float(np.mean(values)))
				self._metrics[name] = []

			self._logger.write(fps=True, print_cli=False)

		if training:
			self._step += 1
			self._logger.step = self._step

	
	def train_uncertainty_only(self, training=True):
		step = self._step
		if training:
			met = self._wm.train_uncertainty_only(data=next(self._dataset), ensemble=self._disag_ensemble)
			self._update_count += 1
			self._metrics["update_count"] = self._update_count

			if (step+1) % 1000 == 0:
				if self._config.video_pred_log:
					if self._config.use_ensemble:
						video_pred = self._wm.video_pred(next(self._dataset), ensemble=self._disag_ensemble)
						self._logger.video("train_openl", video_pred)
					else:
						openl = self._wm.video_pred(next(self._dataset))
						self._logger.video("train_openl", to_np(openl))

			for name, value in met.items():
				if not name in self._metrics.keys():
					self._metrics[name] = [value]
				else:
					self._metrics[name].append(value)

			for name, values in self._metrics.items():
				self._logger.scalar(name, float(np.mean(values)))
				self._metrics[name] = []

			self._logger.write(fps=True, print_cli=False)

		if training:
			self._step += 1
			self._logger.step = self._step

	def _policy(self, obs, state, training):
		if state is None:
			latent = action = None
		else:
			latent, action = state
		obs = self._wm.preprocess(obs)
		embed = self._wm.encoder(obs)
		latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
		if self._config.eval_state_mean:
			latent["stoch"] = latent["mean"]
		feat = self._wm.dynamics.get_feat(latent)
		if not training:
			actor = self._task_behavior.actor(feat)
			action = actor.mode()
		elif self._should_expl(self._step):
			actor = self._expl_behavior.actor(feat)
			action = actor.sample()
		else:
			actor = self._task_behavior.actor(feat)
			action = actor.sample()
		logprob = actor.log_prob(action)
		latent = {k: v.detach() for k, v in latent.items()}
		action = action.detach()
		if self._config.actor["dist"] == "onehot_gumble":
			action = torch.one_hot(
				torch.argmax(action, dim=-1), self._config.num_actions
			)
		policy_output = {"action": action, "logprob": logprob}
		state = (latent, action)
		return policy_output, state

	def _train(self, data):
		metrics = {}
		post, context, mets = self._wm._train(data, ensemble=self._disag_ensemble)
		metrics.update(mets)
		start = post
		reward = lambda f, s, a: self._wm.heads["reward"](
			self._wm.dynamics.get_feat(s)
		).mode()

		metrics.update(self._task_behavior._train(start, reward)[-1])
		
		if self._config.expl_behavior != "greedy":
			mets = self._expl_behavior.train(start, context, data)[-1]
			metrics.update({"expl_" + key: value for key, value in mets.items()})
		for name, value in metrics.items():
			if not name in self._metrics.keys():
				self._metrics[name] = [value]
			else:
				self._metrics[name].append(value)


def count_steps(folder):
	return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
	generator = tools.sample_episodes(episodes, config.batch_length)
	dataset = tools.from_generator(generator, config.batch_size)
	return dataset

def make_env(config, num_envs):

	env_cfg = parse_env_cfg(
		config.task, device='cuda', num_envs=num_envs, use_fabric=True
	)
	env_cfg.seed = 0

	# create environment
	env = gym.make(config.task, cfg=env_cfg)
	env = DreamerVecEnvWrapper(env, device=env_cfg.sim.device)
	env = wrappers.NormalizeActions(env)
	env = wrappers.SelectAction(env, key="action")
	env = wrappers.UUID(env)

	return env