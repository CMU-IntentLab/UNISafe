from __future__ import annotations

import gym.spaces  # needed for rl-games incompatibility: https://github.com/Denys88/rl_games/issues/261
import torch
import numpy as np
from rl_games.common.vecenv import IVecEnv
import matplotlib.pyplot as plt
from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv

"""
Vectorized environment wrapper.
"""

class DreamerVecEnvWrapper(IVecEnv):

	def __init__(self, env: ManagerBasedRLEnv, device: str):
		# check that input is valid
		if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
			raise ValueError(f"The environment must be inherited from ManagerBasedRLEnv. Environment type: {type(env)}")
		# initialize the wrapper
		self._env = env
		self._obs_is_dict = hasattr(self._env.observation_space, "spaces")
		self.size = (128,128)
		self.ac_lim = 0.15
		self.device = device

	@property
	def get_takeoff_obs_space(self):
		return self._env.observation_space.spaces.copy()

	@property
	def single_observation_space(self):
		if self._obs_is_dict:
			spaces = self._env.observation_space.spaces.copy()
			pol_pos_space = gym.spaces.Box(spaces['policy']['eef_pos'].low[0], spaces['policy']['eef_pos'].high[0], spaces['policy']['eef_pos'].shape[1:], dtype=spaces['policy']['eef_pos'].dtype)
			pol_quat_space = gym.spaces.Box(spaces['policy']['eef_quat'].low[0], spaces['policy']['eef_quat'].high[0], spaces['policy']['eef_quat'].shape[1:], dtype=spaces['policy']['eef_quat'].dtype)
			new_spaces = {"eef_pos": pol_pos_space, 'eef_quat': pol_quat_space}
			for k in spaces["rgb_camera"].keys():
				img_shape = [*spaces['rgb_camera'][k].shape[1:]]
				img_shape[-1] = 3
				img_space = gym.spaces.Box(np.zeros_like(spaces['rgb_camera'][k].low)[0, :, :, :3], 255*np.ones_like(spaces['rgb_camera'][k].high)[0, :, :, :3], img_shape, dtype='uint8')
				
				new_spaces[k] = img_space

		else:
			new_spaces = {self._obs_key: self._env.observation_space}
			

		return gym.spaces.Dict(
			{
				**new_spaces,
				"is_first": gym.spaces.Box(0, 1, (), dtype=bool),
				"is_last": gym.spaces.Box(0, 1, (), dtype=bool),
				"is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
			}
		)

	@property
	def action_space(self):
		space = self._env.action_space
		space.low = -self.ac_lim*np.ones_like(space.low)
		space.high = self.ac_lim*np.ones_like(space.high)
		return space
	
	@property
	def single_action_space(self):
		env_low = -self.ac_lim*np.ones_like(self.action_space.low[0,:])
		env_high = self.ac_lim*np.ones_like(self.action_space.high[0,:])
		space =  gym.spaces.Box(env_low, env_high, dtype=np.float32)
		return space
	

	@property
	def unwrapped(self) -> ManagerBasedRLEnv:
		"""Returns the base environment of the wrapper.

		This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
		"""
		return self._env.unwrapped

	@property
	def num_envs(self) -> int:
		"""Returns the number of sub-environment instances."""
		return self.unwrapped.num_envs

	def step(self, action):
		action = torch.where(action>self.ac_lim, self.ac_lim, action)
		action = torch.where(action<-self.ac_lim, -self.ac_lim, action)
		obs, reward, terminated, truncated, info = self._env.step(action)
		for k in obs["rgb_camera"].keys():
			obs[k] = obs["rgb_camera"][k][..., :3]
		obs.pop("rgb_camera")

		obs['eef_pos'] = obs['policy']['eef_pos']
		obs['eef_quat'] = obs['policy']['eef_quat']
		obs['failure'] = obs['subtask_terms']['failure']
		obs['success'] = obs['subtask_terms']['success']

		obs.pop("policy")
		obs.pop("subtask_terms")

		done = terminated | truncated

		obs["is_first"] = torch.tensor([0] * self.num_envs).to(done.device) # is_first should be 1 after a is_last signal.
		obs["is_last"] = done.int()
		obs["is_terminal"] = terminated.int()
		return obs, reward, done, info

	def reset(self, seed=None):
		obs, info = self._env.reset(seed=seed)

		for k in obs["rgb_camera"].keys():
			obs[k] = obs["rgb_camera"][k][..., :3]
		obs.pop("rgb_camera")

		obs['eef_pos'] = obs['policy']['eef_pos']
		obs['eef_quat'] = obs['policy']['eef_quat']
		obs['failure'] = obs['subtask_terms']['failure']
		obs['success'] = obs['subtask_terms']['success']

		obs.pop("policy")
		obs.pop("subtask_terms")

		obs["is_first"] = torch.tensor([1] * self.num_envs).to(obs['front_cam'].device)
		obs["is_last"] = torch.tensor([0] * self.num_envs).to(obs['front_cam'].device)
		obs["is_terminal"] = torch.tensor([0] * self.num_envs).to(obs['front_cam'].device)
		 
		return obs


import gym
import copy
class df_takeoff_wrapper(gym.Wrapper):
	def __init__(self, env):
		super().__init__(env)
		self.device = env.device

		self.original_obs = None

	@property
	def observation_space(self):
		spaces = self.get_takeoff_obs_space
		pol_pos_space = gym.spaces.Box(spaces['policy']['eef_pos'].low[0], spaces['policy']['eef_pos'].high[0], spaces['policy']['eef_pos'].shape[1:], dtype=spaces['policy']['eef_pos'].dtype)
		pol_quat_space = gym.spaces.Box(spaces['policy']['eef_quat'].low[0], spaces['policy']['eef_quat'].high[0], spaces['policy']['eef_quat'].shape[1:], dtype=spaces['policy']['eef_quat'].dtype)
		new_spaces = {"agent_pos": pol_pos_space, 'agent_quat': pol_quat_space}
		for k in spaces["rgb_camera"].keys():
			img_shape = [*spaces['rgb_camera'][k].shape[1:]]
			img_shape[-1] = 3
			img_space = gym.spaces.Box(
							  np.moveaxis(np.zeros_like(spaces['rgb_camera'][k].low)[0, :, :, :3], -1, 0), \
							  np.moveaxis(1*np.ones_like(spaces['rgb_camera'][k].high)[0, :, :, :3], -1, 0), 
							  [3, 128, 128], dtype='uint8'
							  )
			new_spaces[k] = img_space

		obs_space_dict = gym.spaces.Dict(
							{
								**new_spaces,
							}
						)

		return obs_space_dict

	def reset(self, seed=None):
		obs = self.env.reset(seed=seed)

		self.original_obs = copy.deepcopy(obs)

		obs['front_cam'] = obs['front_cam'].permute(0, 3, 1, 2) / 255.0
		obs['wrist_cam'] = obs['wrist_cam'].permute(0, 3, 1, 2) / 255.0
		obs['agent_pos'] = obs['eef_pos']
		obs['agent_quat'] = obs['eef_quat']

		obs.pop('eef_pos')
		obs.pop('eef_quat')

		for k, v in obs.items():
			obs[k] = v.cpu().numpy()

		return obs

	def step(self, action):
		obs, reward, done, info = self.env.step(action)

		self.original_obs = copy.deepcopy(obs)

		obs_new = {}
		obs_new['front_cam'] = obs['front_cam'].permute(0, 3, 1, 2) / 255.0
		obs_new['wrist_cam'] = obs['wrist_cam'].permute(0, 3, 1, 2) / 255.0
		obs_new['agent_pos'] = obs['eef_pos']
		obs_new['agent_quat'] = obs['eef_quat']

		for k, v in obs_new.items():
			obs_new[k] = v.cpu().numpy() # Need to squeeze

		reward = reward.cpu().numpy()
		done = done.item() # .cpu().numpy()

		return obs_new, reward, done, info
