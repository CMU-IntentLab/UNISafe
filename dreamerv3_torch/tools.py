import datetime
import collections
import io
import os
import json
import pathlib
import re
import time
import random

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as torchd
from torch.utils.tensorboard import SummaryWriter


to_np = lambda x: x.detach().cpu().numpy()


def symlog(x):
	return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
	return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class RequiresGrad:
	def __init__(self, model):
		self._model = model

	def __enter__(self):
		self._model.requires_grad_(requires_grad=True)

	def __exit__(self, *args):
		self._model.requires_grad_(requires_grad=False)


class TimeRecording:
	def __init__(self, comment):
		self._comment = comment

	def __enter__(self):
		self._st = torch.cuda.Event(enable_timing=True)
		self._nd = torch.cuda.Event(enable_timing=True)
		self._st.record()

	def __exit__(self, *args):
		self._nd.record()
		torch.cuda.synchronize()
		print(self._comment, self._st.elapsed_time(self._nd) / 1000)

class DummyLogger:
	def __init__(self, logdir, step):
		self._logdir = logdir
		self._last_step = None
		self._last_time = None
		self._scalars = {}
		self._images = {}
		self._videos = {}
		self.step = step

import wandb
class Logger:
	def __init__(self, logdir, step):
		self._logdir = logdir
		self._last_step = None
		self._last_time = None
		self._scalars = {}
		self._images = {}
		self._videos = {}
		self.step = step

		name = str(logdir).split('/')[-2] + '_' + str(logdir).split('/')[-1]
		# Initialize WandB
		wandb.init(project="isaac_takeoff", config={"logdir": str(logdir)}, name=name)

	def config(self, config_dict):
		# Convert PosixPath objects to strings
		config_dict = {
			k: str(v) if isinstance(v, pathlib.PosixPath) else v
			for k, v in config_dict.items()
		}
		# Log the config
		wandb.config.update(config_dict)

	def scalar(self, name, value):
		self._scalars[name] = float(value)

	def image(self, name, value):
		self._images[name] = np.array(value)

	def video(self, name, value):
		self._videos[name] = np.array(value)

	def write(self, fps=False, step=False, fps_namespace="", print_cli=True):
		if not step:
			step = self.step
		scalars = list(self._scalars.items())
		if fps:
			fps_str = fps_namespace + "fps"
			scalars.append((fps_str, self._compute_fps(step)))
		if print_cli:
			print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
		# Log metrics to WandB
		metrics = {"step": step, **dict(scalars)}
		wandb.log(metrics, step=step)
		
		for name, value in self._images.items():
			# Log images to WandB
			if np.shape(value)[0] == 3:
				value = np.transpose(value, (1, 2, 0))
			wandb.log({name: [wandb.Image(value, caption=name)]}, step=step)

		for name, value in self._videos.items():
			name = name if isinstance(name, str) else name.decode("utf-8")
			if np.issubdtype(value.dtype, np.floating):
				value = np.clip(255 * value, 0, 255).astype(np.uint8)
			B, T, H, W, C = value.shape
			value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
			# Log videos to WandB
			wandb.log({name: wandb.Video(value, fps=20, format="mp4")}, step=step)

		self._scalars = {}
		self._images = {}
		self._videos = {}

	def _compute_fps(self, step):
		if self._last_step is None:
			self._last_time = time.time()
			self._last_step = step
			return 0
		steps = step - self._last_step
		duration = time.time() - self._last_time
		self._last_time += duration
		self._last_step = step
		return steps / duration

	def offline_scalar(self, name, value, step):
		# Log scalar metrics to WandB
		wandb.log({f"scalars/{name}": value}, step=step)

	def offline_video(self, name, value, step):
		if np.issubdtype(value.dtype, np.floating):
			value = np.clip(255 * value, 0, 255).astype(np.uint8)
		B, T, H, W, C = value.shape
		value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
		# Log videos to WandB
		wandb.log({name: wandb.Video(value, fps=20, format="mp4")}, step=step)



def simulate(
	agent,
	envs,
	cache,
	directory,
	logger,
	is_eval=False,
	limit=None,
	steps=0,
	episodes=0,
	state=None,
):
	num_env = envs.num_envs
	# initialize or unpack simulation state
	if state is None:
		step, episode = 0, 0
		done = np.ones(envs.num_envs, bool)
		length = np.zeros(envs.num_envs, np.int32)
		obs = [None] * envs.num_envs
		agent_state = None
		reward = [0] * envs.num_envs
	else:
		step, episode, done, length, obs, agent_state, reward = state
	while (steps and step < steps) or (episodes and episode < episodes):
		# reset envs if necessary
		if done.any():
			indices = [index for index, d in enumerate(done) if d]
			results = [envs[i].reset() for i in indices]
			results = [r() for r in results]
			for index, result in zip(indices, results):
				t = result.copy()
				t = {k: convert(v) for k, v in t.items()}
				# action will be added to transition in add_to_cache
				t["reward"] = 0.0
				t["discount"] = 1.0
				# initial state should be added to cache
				add_to_cache(cache, envs[index].id, t)
				# replace obs with done by initial state
				obs[index] = result
		# step agents
		obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}
		action, agent_state = agent(obs, done, agent_state)
		if isinstance(action, dict):
			action = [
				{k: np.array(action[k][i].detach().cpu()) for k in action}
				for i in range(len(envs))
			]
		else:
			action = np.array(action)
		assert len(action) == len(envs)
		# step envs
		results = [e.step(a) for e, a in zip(envs, action)]
		results = [r() for r in results]
		obs, reward, done = zip(*[p[:3] for p in results])
		obs = list(obs)
		reward = list(reward)
		done = np.stack(done)
		episode += int(done.sum())
		length += 1
		step += len(envs)
		length *= 1 - done
		# add to cache
		for a, result, env in zip(action, results, envs):
			o, r, d, info = result
			o = {k: convert(v) for k, v in o.items()}
			transition = o.copy()
			if isinstance(a, dict):
				transition.update(a)
			else:
				transition["action"] = a
			transition["reward"] = r
			transition["discount"] = info.get("discount", np.array(1 - float(d)))
			add_to_cache(cache, env.id, transition)

		if done.any():
			indices = [index for index, d in enumerate(done) if d]
			# logging for done episode
			for i in indices:
				save_episodes(directory, {envs[i].id: cache[envs[i].id]})
				length = len(cache[envs[i].id]["reward"]) - 1
				score = float(np.array(cache[envs[i].id]["reward"]).sum())
				video = cache[envs[i].id]["image"]
				# record logs given from environments
				for key in list(cache[envs[i].id].keys()):
					if "log_" in key:
						logger.scalar(
							key, float(np.array(cache[envs[i].id][key]).sum())
						)
						# log items won't be used later
						cache[envs[i].id].pop(key)
				if not is_eval:
					step_in_dataset = erase_over_episodes(cache, limit)
					logger.scalar(f"dataset_size", step_in_dataset)
					logger.scalar(f"train_return", score)
					logger.scalar(f"train_length", length)
					logger.scalar(f"train_episodes", len(cache))
					logger.write(step=logger.step)
				else:
					if not "eval_lengths" in locals():
						eval_lengths = []
						eval_scores = []
						eval_done = False
					# start counting scores for evaluation
					eval_scores.append(score)
					eval_lengths.append(length)

					score = sum(eval_scores) / len(eval_scores)
					length = sum(eval_lengths) / len(eval_lengths)
					logger.video(f"eval_policy", np.array(video)[None])

					if len(eval_scores) >= episodes and not eval_done:
						logger.scalar(f"eval_return", score)
						logger.scalar(f"eval_length", length)
						logger.scalar(f"eval_episodes", len(eval_scores))
						logger.write(step=logger.step)
						eval_done = True
	if is_eval:
		# keep only last item for saving memory. this cache is used for video_pred later
		while len(cache) > 1:
			# FIFO
			cache.popitem(last=False)
	return (step - steps, episode - episodes, done, length, obs, agent_state, reward)

import datetime
import uuid
def simulate_vecenv(
	agent,
	vecenv,        # A single vectorized environment with N sub-envs
	cache,
	directory,
	logger,
	is_eval=False,
	limit=None,
	steps=0,
	episodes=0,
	state=None,
	save_success=False
):
	# Number of sub-envs in the vectorized env
	num_env = vecenv.num_envs
	id_bank = [str(uuid.uuid4()) for _ in range(num_env)]

	# Unpack or initialize simulation state
	if state is None:
		step, episode = 0, 0
		done = np.ones(num_env, bool)
		length = np.zeros(num_env, np.int32)
		obs = None
		agent_state = None
		reward = np.zeros(num_env, dtype=np.float32)
	else:
		step, episode, done, length, obs, agent_state, reward = state

	
	# If environment automatically resets sub-envs on done, we can just do a single reset initially:
	if obs is None:
		# shape might be (num_env, obs_dim, ...)
		obs = vecenv.reset()

		# If we want to add these initial states to the cache:
		for i in range(num_env):
			t = {key: value[i].detach().cpu() for key, value in obs.items()}
			t["reward"] = 0.0
			t["discount"] = 1.0
			t["failure"] = 0.0
			add_to_cache(cache, id_bank[i], t)  # Using i or "env_id=i" as the key

	# Main loop
	while ((steps and step < steps) or (episodes and episode < episodes)):

		step_info = f"{step + 1}/{steps}" if steps is not None else f"{step + 1} (no upper limit)"
		episode_info = f"{episode + 1}/{episodes}" if episodes is not None else f"{episode + 1} (no upper limit)"
		print(f"Current Step: {step_info} (Length: {length}) | Current Episode: {episode_info}", end="\r", flush=True)

		# Agent step
		# obs is shape (num_env, ...) 
		# done is shape (num_env,)
		# Convert obs into a batch dict if needed by agent
		obs_dict = obs
		
		action, agent_state = agent(obs_dict, done, agent_state)

		# Step the vector env once with the entire batch
		next_obs, next_reward, next_done, info = vecenv.step(action)
		# shapes:
		#   next_obs:   (num_env, ...)
		#   next_reward:(num_env,)
		#   next_done:  (num_env,)

		next_done = next_done.cpu().numpy()

		# Update counters
		episode += int(next_done.sum())   # how many sub-envs finished in this step
		length += 1                       # increment length for all ongoing episodes
		step += num_env                   # each sub-env took one step
		length *= (1 - next_done)         # reset to 0 where done

		# Add transitions to cache
		for i in range(num_env):

			transition = {key: value[i].detach().cpu().numpy() for key, value in next_obs.items()}

			# Handle missing last observation of a vectorized environment.
			# Remedy: Simply overwrite an observation using the recent one.
			if next_done[i]:
				overwrite_keys = ["front_cam", "wrist_cam", "eef_pos", "eef_quat"]
				for key in overwrite_keys:
					transition[key] = obs[key][i].detach().cpu().numpy()
				
			# Add action
			if isinstance(action, dict):
				for k, v in action.items():
					transition[k] = v[i].cpu().numpy()
			else:
				transition["action"] = np.array(action)

			transition["reward"] = next_reward[i].item()
			transition["discount"] = np.array(1.0, dtype=np.float32)
			
			# TODO: Label failure for previous transitions.
			if transition["reward"] < -0.2:
				transition["failure"] = 1.0 
				cache[id_bank[i]]["failure"][-3:] = [convert(1.0) for _ in range(len(cache[id_bank[i]]["failure"][-3:]))]

			else:
				transition["failure"] = 0.0

			add_to_cache(cache, id_bank[i], transition)

		# Logging for done sub-envs
		# So we gather the completed episodes from the cache for each done[i].
		done_indices = np.where(next_done)[0]
		if len(done_indices) > 0:
			for i in done_indices:
				# Save the completed episode
				index = id_bank[i]
				length = len(cache[index]["reward"]) - 1
				score = float(np.array(cache[index]["reward"]).sum())
				video_front = cache[index]["front_cam"]
				video_wrist = cache[index]["wrist_cam"]
				failure_data = cache[index]["failure"]
				video = np.concatenate((video_wrist, video_front), axis=1)

				# save recorded video
				# Save a successful trajectory
				if save_success and next_reward[i].item() > 0.6:
					filename = "success_" + index
					save_episodes(directory, {filename: cache[index]})
					save_video(directory, filename, video, failure_data)
				
				elif save_success and next_reward[i].item() < -0.2 and info['log']['Episode_Termination/failure']:
					filename = "failure_" + index
					save_episodes(directory, {filename: cache[index]})
					save_video(directory, filename, video, failure_data)

				# Log environment-specific items
				for key in list(cache[index].keys()):
					if "log_" in key:
						logger.scalar(key, float(np.array(cache[index][key]).sum()))
						cache[index].pop(key)

				if not is_eval:
					step_in_dataset = erase_over_episodes(cache, limit)
					logger.scalar(f"dataset_size", step_in_dataset)
					logger.scalar(f"train_return", score)
					logger.scalar(f"train_length", length)
					logger.scalar(f"train_episodes", len(cache))
					logger.write(step=logger.step)
				
				else:
					if not "eval_lengths" in locals():
						eval_lengths = []
						eval_scores = []
						eval_done = False
					# start counting scores for evaluation
					eval_scores.append(score)
					eval_lengths.append(length)

					score = sum(eval_scores) / len(eval_scores)
					length = sum(eval_lengths) / len(eval_lengths)
					logger.video(f"eval_policy", np.array(video)[None])

					if len(eval_scores) >= episodes and not eval_done:
						logger.scalar(f"eval_return", score)
						logger.scalar(f"eval_length", length)
						logger.scalar(f"eval_episodes", len(eval_scores))
						logger.write(step=logger.step)
						eval_done = True

				# Change id to a new one.
				id_bank[i] = str(uuid.uuid4())
				
		# Move on
		obs = next_obs
		reward = next_reward
		done = next_done

	# If evaluation, keep minimal cache
	if is_eval:
		while len(cache) > 1:
			cache.popitem(last=False)

	return (step - steps, episode - episodes, done, length, obs, agent_state, reward)

def get_uncertainty(wm, ensemble, latent, actions):

	with torch.no_grad():
		feats = wm.dynamics.get_feat(latent)
		
		inputs = torch.concat([feats, actions['action']], -1)
		uncertainty = ensemble.intrinsic_reward_penn(inputs)

		# FIXME: nf
		# uncertainty = 1 - ensemble.calculate_likelihood(feats)

		return uncertainty
	
def get_kld(wm, post, prior):

	with torch.no_grad():
		kl_free = 1.0
		dyn_scale = 0.5
		rep_scale = 0.1
		_, _, value, _ = wm.dynamics.kl_loss(post, prior, kl_free, dyn_scale, rep_scale
				)

		return value
	
def get_recon_loss(wm, post, obs):

	with torch.no_grad():

		data = wm.preprocess(obs)
		feat = wm.dynamics.get_feat(post)
		pred = wm.heads["decoder"](feat.unsqueeze(0))
		loss_front = -pred['front_cam'].log_prob(data["front_cam"].unsqueeze(0)).item()
		loss_wrist = -pred['wrist_cam'].log_prob(data["wrist_cam"].unsqueeze(0)).item()
	
		return loss_front + loss_wrist

def forward_latent(wm, latent_prev, act_prev, obs):
	with torch.no_grad():
		data = wm.preprocess(obs)
		embed = wm.encoder(data)

		# Latent and action is None for the initial
		latent_post, latent_prior = wm.dynamics.obs_step(latent_prev, act_prev['action'], embed, obs["is_first"], sample=False)

		return latent_post, latent_prior

def post_process_actions(action_cmd: torch.Tensor) -> torch.Tensor:

	action_cmd = torch.clamp(action_cmd, -0.5, 0.5)
	neg_mask = action_cmd[:, -1] < 0
	action_cmd[neg_mask, -1] = -1.0
	action_cmd[~neg_mask, -1] = 1.0
	
	return action_cmd

def filter_action(wm, ensemble, latent, actions, latent_filter, uq_thr=4.2, reachability_thr=0.1, denom=1):
	with torch.no_grad():
		action = actions['action']
		uncertainty = get_uncertainty(wm, ensemble, latent, actions)
		feats = wm.dynamics.get_feat(latent)

		pred_next_latent = wm.dynamics.img_step(latent, action)
		pred_next_feat = wm.dynamics.get_feat(pred_next_latent)
		next_best_act, log_prob = latent_filter.select_action(pred_next_feat, eval_mode=True)

		reachability_value_Q1 = latent_filter.critic1(pred_next_feat, next_best_act).item()
		reachability_value_Q2 = latent_filter.critic2(pred_next_feat, next_best_act).item()
		reachability_value = min(reachability_value_Q1, reachability_value_Q2)

		# # 3. Filter action with optimal policy
		is_filtered = False
		if (uncertainty > uq_thr) or (reachability_value < reachability_thr) :
			new_actions, log_prob = latent_filter.select_action(feats, eval_mode=True)
			action[:, :6] = post_process_actions(new_actions)[:, :6] / denom
			is_filtered = True
		
	return action, is_filtered

import omni.isaac.core.utils.torch as torch_utils

def reset_episode(vecenv, latent_dynamics, seed=None):
	vecenv.seed(seed)
	obs = vecenv.reset()

	# torch_utils.set_seed(seed, torch_deterministic=True)
	if seed is None:
		torch_utils.set_seed(-1, torch_deterministic=True)
	else:
		torch_utils.set_seed(seed, torch_deterministic=True)

	data = latent_dynamics.preprocess(obs)
	embed = latent_dynamics.encoder(data)
	latent = None
	action = None
	latent, _ = latent_dynamics.dynamics.obs_step(latent, action, embed, obs["is_first"], sample=False)

	return obs, latent, None

def reset_episode_df(vecenv, latent_dynamics, seed=None):
	vecenv.seed(seed)
	obs = vecenv.reset()

	if seed is None:
		torch_utils.set_seed(-1, torch_deterministic=True)
	else:
		torch_utils.set_seed(seed, torch_deterministic=True)

	obs_original = vecenv.original_obs
	data = latent_dynamics.preprocess(obs_original)
	embed = latent_dynamics.encoder(data)
	latent = None
	action = None
	latent, _ = latent_dynamics.dynamics.obs_step(latent, action, embed, obs_original["is_first"], sample=False)

	return obs, latent, obs_original

# Convert NumPy data types to Python native types
def convert_numpy(obj):
    if isinstance(obj, np.ndarray):  # Convert NumPy arrays to lists
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64)):  # Convert NumPy integers to Python int
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):  # Convert NumPy floats to Python float
        return float(obj)
    else:
        return obj
	
def evaluate_venv_filtering(
	agent_base,
	latent_dynamics,
	latent_ensemble,
	latent_filter,
	config,
	vecenv,        # A single vectorized environment with N sub-envs
	episodes=10,  # total number of finished trajectories to test
	state=None,
	is_filter=False
):

	total_episode_cnt = 0
	global_results = []
	result_file_path = os.path.join(config.outFolder, "results.json")
	# === Unpack or initialize simulation state ===
	num_env = vecenv.num_envs
	if state is None:
		step = 0
		episode = 0
		done = np.ones(num_env, bool)
		length = np.zeros(num_env, np.int32)
		obs = None
		agent_state = None
	else:
		step, episode, done, length, obs, agent_state, reward = state

	# Reset environment if needed and initialize the latent state.
	obs, latent, agent_state = reset_episode(vecenv=vecenv, latent_dynamics=latent_dynamics, seed=episode)

	# --- Containers for per-episode uncertainty logging ---
	# For each sub-environment, we will accumulate the uncertainty values over the trajectory.
	traj_uncertainties = [[] for _ in range(num_env)]
	traj_filtered = [[] for _ in range(num_env)]
	traj_kld = [[] for _ in range(num_env)]
	traj_recon_loss = [[] for _ in range(num_env)]

	# To record finished episode details:
	finished_episodes_log = []
	success_count = 0
	failure_count = 0
	timeout_count = 0

	# === Main evaluation loop ===
	# (Stop when we have finished the desired number of episodes or reached the max step count.)

	while (episode < episodes):

		# === (1) Nominal Policy ===
		obs_dict = obs
		# Get action from the nominal policy.
		action, agent_state = agent_base._policy(obs_dict, agent_state, training=False)
		action['action'][:, -1] *= 2 # gripper action

		# === (2) Filter Action ===
		is_filtered = False
		if is_filter:
			new_action, is_filtered = \
				filter_action(latent_dynamics, latent_ensemble, latent, action, latent_filter, \
				  uq_thr=config.ood_threshold, reachability_thr=config.reachability_threshold, denom=config.action_denom)
			action['action'] = new_action

		filtered_action = action
		next_obs, next_reward, next_done, info = vecenv.step(filtered_action)
		next_done = next_done.cpu().numpy()

		# === (3) Calculate and record uncertainty values ===
		final_uncertainties = get_uncertainty(latent_dynamics, latent_ensemble, latent, filtered_action)
		for i in range(num_env):
			traj_uncertainties[i].append(final_uncertainties[i].item())
			traj_filtered[i].append(is_filtered)

		# === (4) Update Latent ===
		latent, latent_prior = forward_latent(latent_dynamics, latent, filtered_action, next_obs)

		for i in range(num_env):
			kld_uncertaity = get_kld(latent_dynamics, latent, latent_prior)
			traj_kld[i].append(kld_uncertaity[i].item())
			recon_loss = get_recon_loss(latent_dynamics, latent, next_obs)
			traj_recon_loss[i].append(recon_loss)

		# === Update counters ===
		length += 1
		step += num_env
		# Capture the current length values (these represent the step count for this episode)
		current_lengths = length.copy()

		# Identify which environments have just finished an episode.
		done_indices = np.where(next_done)[0]
		if len(done_indices) > 0:
			# For each finished sub-env, determine the termination outcome and log uncertainty stats.
			for i in done_indices:

				# Success
				if info['log']['Episode_Termination/success']:
					success_count += 1
					outcome = 'success'
				
				# Failure
				elif info['log']['Episode_Termination/failure']:
					failure_count += 1
					outcome = 'failure'

				# Timeout.
				elif info['log']['Episode_Termination/time_out']:
					timeout_count += 1
					outcome = 'incompletion'

				else:
					failure_count += 1
					outcome = 'failure' #'incompletion'
					
				# --- Compute uncertainty statistics for this episode ---
				traj_unc = traj_uncertainties[i]
				num_filtered = np.sum(traj_filtered[i])
				max_unc = np.max(traj_unc) if len(traj_unc) > 0 else 0.0
				mean_unc = np.mean(traj_unc) if len(traj_unc) > 0 else 0.0

				traj_kld_value = traj_kld[i]
				max_kld = np.max(traj_kld_value) if len(traj_kld_value) > 0 else 0.0
				mean_kld = np.mean(traj_kld_value) if len(traj_kld_value) > 0 else 0.0

				traj_recon_loss_value = traj_recon_loss[i]
				max_recon_loss = np.max(traj_recon_loss_value) if len(traj_recon_loss_value) > 0 else 0.0
				mean_recon_loss = np.mean(traj_recon_loss_value) if len(traj_recon_loss_value) > 0 else 0.0

				result = {
					'episode_index': episode,
					'length': current_lengths[i],
					'max_uncertainty': max_unc,
					'mean_uncertainty': mean_unc,
					'max_kld': max_kld,
					'mean_kld': mean_kld,
					'max_recon': max_recon_loss,
					'mean_recon': mean_recon_loss,
					'num_filtered': num_filtered,
					'outcome': outcome
				}
				finished_episodes_log.append(result)
				# Reset uncertainty storage for that environment (so the next episode starts fresh)
				traj_uncertainties[i] = []
				traj_filtered[i] = []
				traj_kld[i] = []
				traj_recon_loss[i] = [] 
				print(result)

				# Save result
				total_episode_cnt += 1 
				converted_result = {k: convert_numpy(v) for k, v in result.items()}
				global_results.append(converted_result)

			# Increase our episode count by the number of environments that finished this step.
			episode += len(done_indices)

			next_obs, latent, agent_state = reset_episode(vecenv=vecenv, latent_dynamics=latent_dynamics, seed=episode)

			if episode % 10 == 0 :
				lengths = np.array([r['length'] for r in finished_episodes_log], dtype=np.float32)
				max_uncs = np.array([r['max_uncertainty'] for r in finished_episodes_log], dtype=np.float32)
				mean_uncs = np.array([r['mean_uncertainty'] for r in finished_episodes_log], dtype=np.float32)
				max_klds = np.array([r['max_kld'] for r in finished_episodes_log], dtype=np.float32)
				mean_klds = np.array([r['mean_kld'] for r in finished_episodes_log], dtype=np.float32)
				max_recon = np.array([r['max_recon'] for r in finished_episodes_log], dtype=np.float32)
				mean_recon = np.array([r['mean_recon'] for r in finished_episodes_log], dtype=np.float32)
				num_filtered = np.array([r['num_filtered'] for r in finished_episodes_log], dtype=np.float32)

				# Compute means
				length_mean = np.mean(lengths)
				max_unc_mean = np.mean(max_uncs)
				mean_unc_mean = np.mean(mean_uncs)
				max_klds_mean = np.mean(max_klds)
				mean_klds_mean = np.mean(mean_klds)
				max_recon_mean = np.mean(max_recon)
				mean_recon_mean = np.mean(mean_recon)
				num_filtered_mean = np.mean(num_filtered)

				# Compute standard deviations
				length_std = np.std(lengths)
				max_unc_std = np.std(max_uncs)
				mean_unc_std = np.std(mean_uncs)
				max_recon_std = np.std(max_recon)
				mean_recon_std = np.std(mean_recon)
				num_filtered_std = np.std(num_filtered)

				# Print or log the results
				success_rate = success_count / len(finished_episodes_log)
				failure_rate = failure_count / len(finished_episodes_log)
				timeout_rate = timeout_count / len(finished_episodes_log)
				print("=== Episode Results Summary ===")
				print(f"Count of Episodes: {len(finished_episodes_log)}")
				print(f"Success Rate: {success_rate:.2f}")
				print(f"Failure Rate: {failure_rate:.2f}")
				print(f"Timeout Rate: {timeout_rate:.2f}")
				print(f"Num Filtered (mean ± std): {num_filtered_mean:.2f} ± {num_filtered_std:.2f}")
				print(f"Length (mean ± std): {length_mean:.2f} ± {length_std:.2f}")
				print(f"Max Uncertainty (mean ± std): {max_unc_mean:.4f} ± {max_unc_std:.4f}")
				print(f"Mean Uncertainty (mean ± std): {mean_unc_mean:.4f} ± {mean_unc_std:.4f}")
				print(f"Max KLD (mean): {max_klds_mean:.4f}")
				print(f"Mean KLD (mean): {mean_klds_mean:.4f}")
				print(f"Max recon (mean ± std): {max_recon_mean:.4f} ± {max_recon_std:.4f}")
				print(f"Mean recon (mean ± std): {mean_recon_mean:.4f} ± {mean_recon_std:.4f}")
				print(f"Num Filtered (mean ± std): {num_filtered_mean:.2f} ± {num_filtered_std:.2f}")

		# Reset the length counters for those environments that just finished.
		length *= (1 - next_done)

		# Prepare for next step.
		obs = next_obs

		episode_info = f"{episode + 1}/{episodes}" if episodes is not None else f"{episode + 1} (no upper limit)"
		print(f"Filtered: {is_filtered} | (Lengths: {length}) | Current Episode: {episode_info}", end="\r", flush=True)


	# === Log overall results after evaluation ===
	total_episodes = episodes
	success_rate = success_count / total_episodes if total_episodes > 0 else 0.0
	failure_rate = failure_count / total_episodes if total_episodes > 0 else 0.0
	timeout_rate = timeout_count / total_episodes if total_episodes > 0 else 0.0

	print("\n===== Evaluation Results =====")
	print(f"Total Episodes: {total_episodes}")
	print(f"Success Rate: {success_rate:.2f}")
	print(f"Failure Rate: {failure_rate:.2f}")
	print(f"Timeout Rate: {timeout_rate:.2f}")

	wandb.log({
			"eval/total_episodes": total_episodes,
			"eval/success_rate": success_rate,
			"eval/failure_rate": failure_rate,
			"eval/incompletion_rate": timeout_rate,
		})
	
	with open(result_file_path, "w") as f:
		json.dump(global_results, f, indent=4)

	return total_episodes, success_rate, failure_rate, timeout_rate, finished_episodes_log

def evaluate_venv_filtering_df(
	agent_df,
	latent_dynamics,
	latent_ensemble,
	latent_filter,
	config,
	vecenv,        # A single vectorized environment with N sub-envs
	dict_apply,
	episodes=10,  # total number of finished trajectories to test
	state=None,
	is_filter=False,
):

	total_episode_cnt = 0
	global_results = []

	# === Unpack or initialize simulation state ===
	num_env = vecenv.num_envs
	if state is None:
		step = 0
		episode = 0
		length = 0
		obs = None

	# Reset environment if needed and initialize the latent state.
	obs, latent, obs_original = reset_episode_df(vecenv=vecenv, latent_dynamics=latent_dynamics, seed=episode)
	agent_df.reset()

	# --- Containers for per-episode uncertainty logging ---
	# For each sub-environment, we will accumulate the uncertainty values over the trajectory.
	traj_uncertainties = []
	traj_filtered = []
	traj_kld = []
	traj_recon_loss = []

	# To record finished episode details:
	finished_episodes_log = []
	success_count = 0
	failure_count = 0
	timeout_count = 0

	# === Main evaluation loop ===
	# (Stop when we have finished the desired number of episodes or reached the max step count.)
	while (episode < episodes):

		# === (1) Nominal Policy ===
		np_obs_dict = dict(obs)
		obs_dict = dict_apply(np_obs_dict, 
			lambda x: torch.from_numpy(x).to(
				device="cuda"))
		
		for k, v in obs_dict.items():
			obs_dict[k] = v.transpose(1,0)

		# run policy
		with torch.no_grad():
			action_dict = agent_df.predict_action(obs_dict)

		np_action_dict = dict_apply(action_dict,
					lambda x: x.detach().to('cpu').numpy())

		action = torch.tensor(np_action_dict['action'][0,:1], device="cuda")
		action[:, -1] *= 2

		# === (2) Filter Action ===
		is_filtered = False
		action_dict = {'action': action}
		if is_filter:
			new_action, is_filtered = \
				filter_action(latent_dynamics, latent_ensemble, latent, action_dict, latent_filter, \
				  uq_thr=config.ood_threshold, reachability_thr=config.reachability_threshold, denom=config.action_denom)
			action_dict['action'] = new_action

		filtered_action = action_dict['action'].cpu().numpy()
		next_obs, reward, next_done, info = vecenv.step(filtered_action)

		# === (3) Calculate and record uncertainty values ===
		action_dict = {'action': torch.tensor(filtered_action, device="cuda")}
		final_uncertainties = get_uncertainty(latent_dynamics, latent_ensemble, latent, action_dict)

		traj_uncertainties.append(final_uncertainties.item())
		traj_filtered.append(is_filtered)

		# === (4) Update Latent ===
		next_obs_original = vecenv.original_obs
		latent, latent_prior = forward_latent(latent_dynamics, latent, action_dict, next_obs_original)

		kld_uncertaity = get_kld(latent_dynamics, latent, latent_prior)
		traj_kld.append(kld_uncertaity.item())
		recon_loss = get_recon_loss(latent_dynamics, latent, next_obs_original)
		traj_recon_loss.append(recon_loss)

		# === Update counters ===
		length += 1
		step += num_env
		# Capture the current length values (these represent the step count for this episode)
		current_lengths = length

		if next_done:
			# Success
			if info['log'][-1]['Episode_Termination/success']:
				success_count += 1
				outcome = 'success'
			
			# Failure
			elif info['log'][-1]['Episode_Termination/failure']:
				failure_count += 1
				outcome = 'failure'

			# Timeout.
			elif info['log'][-1]['Episode_Termination/time_out']:
				timeout_count += 1
				outcome = 'incompletion'

			else:
				failure_count += 1
				outcome = 'failure' #'incompletion'
					
			num_filtered = np.sum(traj_filtered)
			max_unc = np.max(traj_uncertainties) if len(traj_uncertainties) > 0 else 0.0
			mean_unc = np.mean(traj_uncertainties) if len(traj_uncertainties) > 0 else 0.0
			traj_kld_value = traj_kld
			max_kld = np.max(traj_kld_value) if len(traj_kld_value) > 0 else 0.0
			mean_kld = np.mean(traj_kld_value) if len(traj_kld_value) > 0 else 0.0
			traj_recon_loss_value = traj_recon_loss
			max_recon_loss = np.max(traj_recon_loss_value) if len(traj_recon_loss_value) > 0 else 0.0
			mean_recon_loss = np.mean(traj_recon_loss_value) if len(traj_recon_loss_value) > 0 else 0.0

			# result = {
			# 		'episode_index': episode,
			# 		'length': current_lengths,
			# 		'max_uncertainty': max_unc,
			# 		'mean_uncertainty': mean_unc,
			# 		'num_filtered': num_filtered,
			# 		'outcome': outcome
			# 	}

			result = {
					'episode_index': episode,
					'length': current_lengths,
					'max_uncertainty': max_unc,
					'mean_uncertainty': mean_unc,
					'max_kld': max_kld,
					'mean_kld': mean_kld,
					'max_recon': max_recon_loss,
					'mean_recon': mean_recon_loss,
					'num_filtered': num_filtered,
					'outcome': outcome
				}
			finished_episodes_log.append(result)
			# Reset uncertainty storage for that environment (so the next episode starts fresh)
			traj_uncertainties = []
			traj_filtered = []
			traj_kld = []
			traj_recon_loss = [] 
			length = 0
			print(result)

			# Save result
			total_episode_cnt += 1 
			converted_result = {k: convert_numpy(v) for k, v in result.items()}
			global_results.append(converted_result)
				
			# Increase our episode count by the number of environments that finished this step.
			episode += 1

			next_obs, latent, _ = reset_episode_df(vecenv=vecenv, latent_dynamics=latent_dynamics, seed=episode)


			if episode % 10 == 0 :
				lengths = np.array([r['length'] for r in finished_episodes_log], dtype=np.float32)
				max_uncs = np.array([r['max_uncertainty'] for r in finished_episodes_log], dtype=np.float32)
				mean_uncs = np.array([r['mean_uncertainty'] for r in finished_episodes_log], dtype=np.float32)
				max_klds = np.array([r['max_kld'] for r in finished_episodes_log], dtype=np.float32)
				mean_klds = np.array([r['mean_kld'] for r in finished_episodes_log], dtype=np.float32)
				max_recon = np.array([r['max_recon'] for r in finished_episodes_log], dtype=np.float32)
				mean_recon = np.array([r['mean_recon'] for r in finished_episodes_log], dtype=np.float32)
				num_filtered = np.array([r['num_filtered'] for r in finished_episodes_log], dtype=np.float32)

				# Compute means
				length_mean = np.mean(lengths)
				max_unc_mean = np.mean(max_uncs)
				mean_unc_mean = np.mean(mean_uncs)
				max_klds_mean = np.mean(max_klds)
				mean_klds_mean = np.mean(mean_klds)
				max_recon_mean = np.mean(max_recon)
				mean_recon_mean = np.mean(mean_recon)
				num_filtered_mean = np.mean(num_filtered)

				# Compute standard deviations
				length_std = np.std(lengths)
				max_unc_std = np.std(max_uncs)
				mean_unc_std = np.std(mean_uncs)
				max_recon_std = np.std(max_recon)
				mean_recon_std = np.std(mean_recon)
				num_filtered_std = np.std(num_filtered)

				# Print or log the results
				success_rate = success_count / len(finished_episodes_log)
				failure_rate = failure_count / len(finished_episodes_log)
				timeout_rate = timeout_count / len(finished_episodes_log)
				print("=== Episode Results Summary ===")
				print(f"Count of Episodes: {len(finished_episodes_log)}")
				print(f"Success Rate: {success_rate:.2f}")
				print(f"Failure Rate: {failure_rate:.2f}")
				print(f"Timeout Rate: {timeout_rate:.2f}")
				print(f"Num Filtered (mean ± std): {num_filtered_mean:.2f} ± {num_filtered_std:.2f}")
				print(f"Length (mean ± std): {length_mean:.2f} ± {length_std:.2f}")
				print(f"Max Uncertainty (mean ± std): {max_unc_mean:.4f} ± {max_unc_std:.4f}")
				print(f"Mean Uncertainty (mean ± std): {mean_unc_mean:.4f} ± {mean_unc_std:.4f}")
				print(f"Max KLD (mean): {max_klds_mean:.4f}")
				print(f"Mean KLD (mean): {mean_klds_mean:.4f}")
				print(f"Max recon (mean ± std): {max_recon_mean:.4f} ± {max_recon_std:.4f}")
				print(f"Mean recon (mean ± std): {mean_recon_mean:.4f} ± {mean_recon_std:.4f}")
				print(f"Num Filtered (mean ± std): {num_filtered_mean:.2f} ± {num_filtered_std:.2f}")

		# Prepare for next step.
		obs = next_obs

		episode_info = f"{episode + 1}/{episodes}" if episodes is not None else f"{episode + 1} (no upper limit)"
		print(f"Filtered: {is_filtered} | Uncertainty {final_uncertainties.item():.4f} (Lengths: {length}) | Current Episode: {episode_info}", end="\r", flush=True)


	# === Log overall results after evaluation ===
	total_episodes = episodes
	success_rate = success_count / total_episodes if total_episodes > 0 else 0.0
	failure_rate = failure_count / total_episodes if total_episodes > 0 else 0.0
	timeout_rate = timeout_count / total_episodes if total_episodes > 0 else 0.0

	print("\n===== Evaluation Results =====")
	print(f"Total Episodes: {total_episodes}")
	print(f"Success Rate: {success_rate:.2f}")
	print(f"Failure Rate: {failure_rate:.2f}")
	print(f"Timeout Rate: {timeout_rate:.2f}")

	wandb.log({
			"eval/total_episodes": total_episodes,
			"eval/success_rate": success_rate,
			"eval/failure_rate": failure_rate,
			"eval/incompletion_rate": timeout_rate,
		})
	
	result_file_path = os.path.join(config.outFolder, "results.json")
	with open(result_file_path, "w") as f:
		json.dump(global_results, f, indent=4)

	return total_episodes, success_rate, failure_rate, timeout_rate, finished_episodes_log



def add_to_cache(cache, id, transition):

	if id not in cache:
		cache[id] = dict()
		for key, val in transition.items():
			cache[id][key] = [convert(val)]
	else:
		for key, val in transition.items():
			if key not in cache[id]:
				# fill missing data(action, etc.) at second time
				cache[id][key] = [convert(0 * val)]
				cache[id][key].append(convert(val))
			else:
				cache[id][key].append(convert(val))


def erase_over_episodes(cache, dataset_size):
	step_in_dataset = 0

	for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
		if (
			not dataset_size
			or step_in_dataset + (len(ep["reward"]) - 1) <= dataset_size
		):
			step_in_dataset += len(ep["reward"]) - 1
			# print(key, (len(ep["reward"]) - 1), step_in_dataset)
		else:
			#FIXME
			if key.startswith("expert_"):
				continue
			del cache[key]

	return step_in_dataset


def convert(value, precision=32):
	value = np.array(value)
	if np.issubdtype(value.dtype, np.floating):
		dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
	elif np.issubdtype(value.dtype, np.signedinteger):
		dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
	elif np.issubdtype(value.dtype, np.uint8):
		dtype = np.uint8
	elif np.issubdtype(value.dtype, bool):
		dtype = bool
	else:
		raise NotImplementedError(value.dtype)
	return value.astype(dtype)


def save_episodes(directory, episodes):
	directory = pathlib.Path(directory).expanduser()
	directory.mkdir(parents=True, exist_ok=True)
	for filename, episode in episodes.items():
		length = len(episode["reward"])
		filename = directory / f"{filename}-{length}.npz"
		with io.BytesIO() as f1:
			np.savez_compressed(f1, **episode)
			f1.seek(0)
			with filename.open("wb") as f2:
				f2.write(f1.read())
	return True

import cv2
import imageio
def save_video(directory, filename, video, failure):
	directory = pathlib.Path(directory).expanduser()
	directory.mkdir(parents=True, exist_ok=True)

	length = len(video)
	filename = directory / f"{filename}-{length}.mp4"
		
	# Get video dimensions
	height, width, channels = video[0].shape
	fps = 20  # You can adjust FPS if needed

	# Initialize video writer
	# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
	# video_writer = cv2.VideoWriter(str(filename), fourcc, fps, (width, height))

	writer = imageio.get_writer(str(filename), fps=fps, codec='libx264')

	# Write frames to the video file
	# for i, frame in video:
	# 	video_writer.write(frame)

	for i, frame in enumerate(video):
		failure_i = failure[i]
		frame = frame.astype(np.uint8)

		if failure_i > 0 :
			# video_writer.write(frame)
			writer.append_data(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
		else:
			# video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
			writer.append_data(frame)

	# Release the video writer
	# video_writer.release()
	writer.close()
	print("Saved Video!")
	return True


def from_generator(generator, batch_size):
	while True:
		batch = []
		for _ in range(batch_size):
			batch.append(next(generator))
		data = {}
		for key in batch[0].keys():
			data[key] = []
			for i in range(batch_size):
				data[key].append(batch[i][key])
			data[key] = np.stack(data[key], 0)
		yield data


def sample_episodes(episodes, length, seed=0):
	np_random = np.random.RandomState(seed)
	while True:
		size = 0
		ret = None
		p = np.array(
			[len(next(iter(episode.values()))) for episode in episodes.values()]
		)
		p = p / np.sum(p)
		while size < length:
			episode = np_random.choice(list(episodes.values()), p=p)
			total = len(next(iter(episode.values())))
			# make sure at least one transition included
			if total < 2:
				continue
			if not ret:
				index = int(np_random.randint(0, total - 1))
				ret = {
					k: v[index : min(index + length, total)].copy()
					for k, v in episode.items()
					if "log_" not in k
				}
				if "is_first" in ret:
					ret["is_first"][0] = True
			else:
				# 'is_first' comes after 'is_last'
				index = 0
				possible = length - size
				ret = {
					k: np.append(
						ret[k], v[index : min(index + possible, total)].copy(), axis=0
					)
					for k, v in episode.items()
					if "log_" not in k
				}
				if "is_first" in ret:
					ret["is_first"][size] = True
			size = len(next(iter(ret.values())))
		yield ret


def load_episodes(directory, limit=None, reverse=True, episodes=None):
	directory = pathlib.Path(directory).expanduser()

	if episodes is None :
		episodes = collections.OrderedDict()
	else:
		episodes = episodes

	total = 0
	if reverse:
		for filename in reversed(sorted(directory.glob("*.npz"))):
			try:
				with filename.open("rb") as f:
					episode = np.load(f)
					episode = {k: episode[k] for k in episode.keys()}
			except Exception as e:
				print(f"Could not load episode: {e}")
				continue
			
			# extract only filename without extension
			episodes["expert_" + str(os.path.splitext(os.path.basename(filename))[0])] = episode
			total += len(episode["reward"]) - 1

			print("loaded episode from {}, total len {} ({} episodes)".format(filename, total, len(episodes)))
			if limit and total >= limit:
				break
	else:
		for filename in sorted(directory.glob("*.npz")):
			try:
				with filename.open("rb") as f:
					episode = np.load(f)
					episode = {k: episode[k] for k in episode.keys()}
			except Exception as e:
				print(f"Could not load episode: {e}")
				continue
			episodes[str(filename)] = episode
			total += len(episode["reward"]) - 1
			if limit and total >= limit:
				break
	return episodes


class SampleDist:
	def __init__(self, dist, samples=100):
		self._dist = dist
		self._samples = samples

	@property
	def name(self):
		return "SampleDist"

	def __getattr__(self, name):
		return getattr(self._dist, name)

	def mean(self):
		samples = self._dist.sample(self._samples)
		return torch.mean(samples, 0)

	def mode(self):
		sample = self._dist.sample(self._samples)
		logprob = self._dist.log_prob(sample)
		return sample[torch.argmax(logprob)][0]

	def entropy(self):
		sample = self._dist.sample(self._samples)
		logprob = self.log_prob(sample)
		return -torch.mean(logprob, 0)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
	def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
		if logits is not None and unimix_ratio > 0.0:
			probs = F.softmax(logits, dim=-1)
			probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
			logits = torch.log(probs)
			super().__init__(logits=logits, probs=None)
		else:
			super().__init__(logits=logits, probs=probs)

	def mode(self):
		_mode = F.one_hot(
			torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
		)
		return _mode.detach() + super().logits - super().logits.detach()

	def sample(self, sample_shape=(), seed=None):
		if seed is not None:
			raise ValueError("need to check")
		sample = super().sample(sample_shape).detach()
		probs = super().probs
		while len(probs.shape) < len(sample.shape):
			probs = probs[None]
		sample += probs - probs.detach()
		return sample


class DiscDist:
	def __init__(
		self,
		logits,
		low=-20.0,
		high=20.0,
		transfwd=symlog,
		transbwd=symexp,
		device="cuda",
	):
		self.logits = logits
		self.probs = torch.softmax(logits, -1)
		self.buckets = torch.linspace(low, high, steps=255, device=device)
		self.width = (self.buckets[-1] - self.buckets[0]) / 255
		self.transfwd = transfwd
		self.transbwd = transbwd

	def mean(self):
		_mean = self.probs * self.buckets
		return self.transbwd(torch.sum(_mean, dim=-1, keepdim=True))

	def mode(self):
		_mode = self.probs * self.buckets
		return self.transbwd(torch.sum(_mode, dim=-1, keepdim=True))

	# Inside OneHotCategorical, log_prob is calculated using only max element in targets
	def log_prob(self, x):
		x = self.transfwd(x)
		# x(time, batch, 1)
		below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
		above = len(self.buckets) - torch.sum(
			(self.buckets > x[..., None]).to(torch.int32), dim=-1
		)
		# this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
		below = torch.clip(below, 0, len(self.buckets) - 1)
		above = torch.clip(above, 0, len(self.buckets) - 1)
		equal = below == above

		dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
		dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
		total = dist_to_below + dist_to_above
		weight_below = dist_to_above / total
		weight_above = dist_to_below / total
		target = (
			F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
			+ F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
		)
		log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
		target = target.squeeze(-2)

		return (target * log_pred).sum(-1)

	def log_prob_target(self, target):
		log_pred = super().logits - torch.logsumexp(super().logits, -1, keepdim=True)
		return (target * log_pred).sum(-1)


class MSEDist:
	def __init__(self, mode, agg="sum"):
		self._mode = mode
		self._agg = agg

	def mode(self):
		return self._mode

	def mean(self):
		return self._mode

	def log_prob(self, value):
		assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
		distance = (self._mode - value) ** 2
		if self._agg == "mean":
			loss = distance.mean(list(range(len(distance.shape)))[2:])
		elif self._agg == "sum":
			loss = distance.sum(list(range(len(distance.shape)))[2:])
		else:
			raise NotImplementedError(self._agg)
		return -loss


class SymlogDist:
	def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
		self._mode = mode
		self._dist = dist
		self._agg = agg
		self._tol = tol

	def mode(self):
		return symexp(self._mode)

	def mean(self):
		return symexp(self._mode)

	def log_prob(self, value):
		assert self._mode.shape == value.shape
		if self._dist == "mse":
			distance = (self._mode - symlog(value)) ** 2.0
			distance = torch.where(distance < self._tol, 0, distance)
		elif self._dist == "abs":
			distance = torch.abs(self._mode - symlog(value))
			distance = torch.where(distance < self._tol, 0, distance)
		else:
			raise NotImplementedError(self._dist)
		if self._agg == "mean":
			loss = distance.mean(list(range(len(distance.shape)))[2:])
		elif self._agg == "sum":
			loss = distance.sum(list(range(len(distance.shape)))[2:])
		else:
			raise NotImplementedError(self._agg)
		return -loss


class ContDist:
	def __init__(self, dist=None, absmax=None):
		super().__init__()
		self._dist = dist
		self.mean = dist.mean
		self.absmax = absmax

	def __getattr__(self, name):
		return getattr(self._dist, name)

	def entropy(self):
		return self._dist.entropy()

	def mode(self):
		out = self._dist.mean
		if self.absmax is not None:
			out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
		return out

	def sample(self, sample_shape=()):
		out = self._dist.rsample(sample_shape)
		if self.absmax is not None:
			out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
		return out

	def log_prob(self, x):
		return self._dist.log_prob(x)


class Bernoulli:
	def __init__(self, dist=None):
		super().__init__()
		self._dist = dist
		self.mean = dist.mean

	def __getattr__(self, name):
		return getattr(self._dist, name)

	def entropy(self):
		return self._dist.entropy()

	def mode(self):
		_mode = torch.round(self._dist.mean)
		return _mode.detach() + self._dist.mean - self._dist.mean.detach()

	def sample(self, sample_shape=()):
		return self._dist.rsample(sample_shape)

	def log_prob(self, x):
		_logits = self._dist.base_dist.logits
		log_probs0 = -F.softplus(_logits)
		log_probs1 = -F.softplus(-_logits)

		return torch.sum(log_probs0 * (1 - x) + log_probs1 * x, -1)


class UnnormalizedHuber(torchd.normal.Normal):
	def __init__(self, loc, scale, threshold=1, **kwargs):
		super().__init__(loc, scale, **kwargs)
		self._threshold = threshold

	def log_prob(self, event):
		return -(
			torch.sqrt((event - self.mean) ** 2 + self._threshold**2) - self._threshold
		)

	def mode(self):
		return self.mean


class SafeTruncatedNormal(torchd.normal.Normal):
	def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
		super().__init__(loc, scale)
		self._low = low
		self._high = high
		self._clip = clip
		self._mult = mult

	def sample(self, sample_shape):
		event = super().sample(sample_shape)
		if self._clip:
			clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
			event = event - event.detach() + clipped.detach()
		if self._mult:
			event *= self._mult
		return event


class TanhBijector(torchd.Transform):
	def __init__(self, validate_args=False, name="tanh"):
		super().__init__()

	def _forward(self, x):
		return torch.tanh(x)

	def _inverse(self, y):
		y = torch.where(
			(torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
		)
		y = torch.atanh(y)
		return y

	def _forward_log_det_jacobian(self, x):
		log2 = torch.math.log(2.0)
		return 2.0 * (log2 - x - torch.softplus(-2.0 * x))


def static_scan_for_lambda_return(fn, inputs, start):
	last = start
	indices = range(inputs[0].shape[0])
	indices = reversed(indices)
	flag = True
	for index in indices:
		# (inputs, pcont) -> (inputs[index], pcont[index])
		inp = lambda x: (_input[x] for _input in inputs)
		last = fn(last, *inp(index))
		if flag:
			outputs = last
			flag = False
		else:
			outputs = torch.cat([outputs, last], dim=-1)
	outputs = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
	outputs = torch.flip(outputs, [1])
	outputs = torch.unbind(outputs, dim=0)
	return outputs


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
	# Setting lambda=1 gives a discounted Monte Carlo return.
	# Setting lambda=0 gives a fixed 1-step return.
	# assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
	assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
	if isinstance(pcont, (int, float)):
		pcont = pcont * torch.ones_like(reward)
	dims = list(range(len(reward.shape)))
	dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
	if axis != 0:
		reward = reward.permute(dims)
		value = value.permute(dims)
		pcont = pcont.permute(dims)
	if bootstrap is None:
		bootstrap = torch.zeros_like(value[-1])
	next_values = torch.cat([value[1:], bootstrap[None]], 0)
	inputs = reward + pcont * next_values * (1 - lambda_)
	# returns = static_scan(
	#    lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
	#    (inputs, pcont), bootstrap, reverse=True)
	# reimplement to optimize performance
	returns = static_scan_for_lambda_return(
		lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap
	)
	if axis != 0:
		returns = returns.permute(dims)
	return returns


class Optimizer:
	def __init__(
		self,
		name,
		parameters,
		lr,
		eps=1e-4,
		clip=None,
		wd=None,
		wd_pattern=r".*",
		opt="adam",
		use_amp=False,
	):
		assert 0 <= wd < 1
		assert not clip or 1 <= clip
		self._name = name
		self._parameters = parameters
		self._clip = clip
		self._wd = wd
		self._wd_pattern = wd_pattern
		self._opt = {
			"adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
			"nadam": lambda: NotImplemented(f"{opt} is not implemented"),
			"adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
			"sgd": lambda: torch.optim.SGD(parameters, lr=lr),
			"momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
		}[opt]()
		self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

	def __call__(self, loss, params, retain_graph=True):
		assert len(loss.shape) == 0, loss.shape
		metrics = {}
		metrics[f"{self._name}_loss"] = loss.detach().cpu().numpy()
		self._opt.zero_grad()
		self._scaler.scale(loss).backward(retain_graph=retain_graph)
		self._scaler.unscale_(self._opt)
		# loss.backward(retain_graph=retain_graph)
		norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
		if self._wd:
			self._apply_weight_decay(params)
		self._scaler.step(self._opt)
		self._scaler.update()
		# self._opt.step()
		self._opt.zero_grad()
		metrics[f"{self._name}_grad_norm"] = to_np(norm)
		return metrics

	def _apply_weight_decay(self, varibs):
		nontrivial = self._wd_pattern != r".*"
		if nontrivial:
			raise NotImplementedError
		for var in varibs:
			var.data = (1 - self._wd) * var.data


def args_type(default):
	def parse_string(x):
		if default is None:
			return x
		if isinstance(default, bool):
			return bool(["False", "True"].index(x))
		if isinstance(default, int):
			return float(x) if ("e" in x or "." in x) else int(x)
		if isinstance(default, (list, tuple)):
			return tuple(args_type(default[0])(y) for y in x.split(","))
		return type(default)(x)

	def parse_object(x):
		if isinstance(default, (list, tuple)):
			return tuple(x)
		return x

	return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def static_scan(fn, inputs, start):
	last = start
	indices = range(inputs[0].shape[0])
	flag = True
	for index in indices:
		inp = lambda x: (_input[x] for _input in inputs)
		last = fn(last, *inp(index))
		if flag:
			if type(last) == type({}):
				outputs = {
					key: value.clone().unsqueeze(0) for key, value in last.items()
				}
			else:
				outputs = []
				for _last in last:
					if type(_last) == type({}):
						outputs.append(
							{
								key: value.clone().unsqueeze(0)
								for key, value in _last.items()
							}
						)
					else:
						outputs.append(_last.clone().unsqueeze(0))
			flag = False
		else:
			if type(last) == type({}):
				for key in last.keys():
					outputs[key] = torch.cat(
						[outputs[key], last[key].unsqueeze(0)], dim=0
					)
			else:
				for j in range(len(outputs)):
					if type(last[j]) == type({}):
						for key in last[j].keys():
							outputs[j][key] = torch.cat(
								[outputs[j][key], last[j][key].unsqueeze(0)], dim=0
							)
					else:
						outputs[j] = torch.cat(
							[outputs[j], last[j].unsqueeze(0)], dim=0
						)
	if type(last) == type({}):
		outputs = [outputs]
	return outputs


class Every:
	def __init__(self, every):
		self._every = every
		self._last = None

	def __call__(self, step):
		if not self._every:
			return 0
		if self._last is None:
			self._last = step
			return 1
		count = int((step - self._last) / self._every)
		self._last += self._every * count
		return count


class Once:
	def __init__(self):
		self._once = True

	def __call__(self):
		if self._once:
			self._once = False
			return True
		return False


class Until:
	def __init__(self, until):
		self._until = until

	def __call__(self, step):
		if not self._until:
			return True
		return step < self._until


def weight_init(m):
	if isinstance(m, nn.Linear):
		in_num = m.in_features
		out_num = m.out_features
		denoms = (in_num + out_num) / 2.0
		scale = 1.0 / denoms
		std = np.sqrt(scale) / 0.87962566103423978
		nn.init.trunc_normal_(
			m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
		)
		if hasattr(m.bias, "data"):
			m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		space = m.kernel_size[0] * m.kernel_size[1]
		in_num = space * m.in_channels
		out_num = space * m.out_channels
		denoms = (in_num + out_num) / 2.0
		scale = 1.0 / denoms
		std = np.sqrt(scale) / 0.87962566103423978
		nn.init.trunc_normal_(
			m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
		)
		if hasattr(m.bias, "data"):
			m.bias.data.fill_(0.0)
	elif isinstance(m, nn.LayerNorm):
		m.weight.data.fill_(1.0)
		if hasattr(m.bias, "data"):
			m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
	def f(m):
		if isinstance(m, nn.Linear):
			in_num = m.in_features
			out_num = m.out_features
			denoms = (in_num + out_num) / 2.0
			scale = given_scale / denoms
			limit = np.sqrt(3 * scale)
			nn.init.uniform_(m.weight.data, a=-limit, b=limit)
			if hasattr(m.bias, "data"):
				m.bias.data.fill_(0.0)
		elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
			space = m.kernel_size[0] * m.kernel_size[1]
			in_num = space * m.in_channels
			out_num = space * m.out_channels
			denoms = (in_num + out_num) / 2.0
			scale = given_scale / denoms
			limit = np.sqrt(3 * scale)
			nn.init.uniform_(m.weight.data, a=-limit, b=limit)
			if hasattr(m.bias, "data"):
				m.bias.data.fill_(0.0)
		elif isinstance(m, nn.LayerNorm):
			m.weight.data.fill_(1.0)
			if hasattr(m.bias, "data"):
				m.bias.data.fill_(0.0)

	return f


def tensorstats(tensor, prefix=None):
	metrics = {
		"mean": to_np(torch.mean(tensor)),
		"std": to_np(torch.std(tensor)),
		"min": to_np(torch.min(tensor)),
		"max": to_np(torch.max(tensor)),
	}
	if prefix:
		metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
	return metrics


def set_seed_everywhere(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


def enable_deterministic_run():
	os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.use_deterministic_algorithms(True)


def recursively_collect_optim_state_dict(
	obj, path="", optimizers_state_dicts=None, visited=None
):
	if optimizers_state_dicts is None:
		optimizers_state_dicts = {}
	if visited is None:
		visited = set()
	# avoid cyclic reference
	if id(obj) in visited:
		return optimizers_state_dicts
	else:
		visited.add(id(obj))
	attrs = obj.__dict__
	if isinstance(obj, torch.nn.Module):
		attrs.update(
			{k: attr for k, attr in obj.named_modules() if "." not in k and obj != attr}
		)
	for name, attr in attrs.items():
		new_path = path + "." + name if path else name
		if isinstance(attr, torch.optim.Optimizer):
			optimizers_state_dicts[new_path] = attr.state_dict()
		elif hasattr(attr, "__dict__"):
			optimizers_state_dicts.update(
				recursively_collect_optim_state_dict(
					attr, new_path, optimizers_state_dicts, visited
				)
			)
	return optimizers_state_dicts


def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
	for path, state_dict in optimizers_state_dicts.items():
		keys = path.split(".")
		obj_now = obj
		for key in keys:
			obj_now = getattr(obj_now, key)
		obj_now.load_state_dict(state_dict)
