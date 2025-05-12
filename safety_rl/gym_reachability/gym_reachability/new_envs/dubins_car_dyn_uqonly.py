"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu        ( kaichieh@princeton.edu )

This module implements the parent class for the Dubins car environments, e.g.,
one car environment and pursuit-evasion game between two Dubins cars.
"""
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
from .env_utils import calculate_margin_circle
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

import io
from PIL import Image
import random
import pickle

class DubinsCarDyn(object):
	"""
	This base class implements a Dubins car dynamical system as well as the
	environment with concentric circles. The inner circle is the target set
	boundary, while the outer circle is the boundary of the constraint set.
	"""

	def __init__(self, config, doneType='toEnd'):
		"""Initializes the environment with the episode termination criterion.

		Args:
				doneType (str, optional): conditions to raise `done` flag in
						training. Defaults to 'toEnd'.
		"""
		# State bounds.
		self.bounds = np.array([[config.x_min, config.x_max], [config.y_min, config.y_max], [0, 2 * np.pi]])
		self.low = self.bounds[:, 0]
		self.high = self.bounds[:, 1]

		self.learned_margin = False
		self.learned_dyn = False
		self.image = False
		self.debug = False
		self.use_wm = False
		self.gt_lx = False
		self.use_ensemble_disagreement = False


		# Dubins car parameters.
		self.alive = True
		self.time_step = config.dt
		self.speed = config.speed  # v

		# Control parameters.
		self.max_turning_rate = config.u_max # w
		self.R_turn = self.speed / self.max_turning_rate 
		self.discrete_controls = np.array([
				-self.max_turning_rate, 0., self.max_turning_rate
		])

		# Constraint set parameters.
		self.constraint_center = None
		self.constraint_radius = None

		# Target set parameters.
		self.target_center = None
		self.target_radius = None

		# Internal state.
		self.state = np.zeros(3)
		self.doneType = doneType

		# Set random seed.
		self.seed_val = 0
		np.random.seed(self.seed_val)

		# Cost Params
		self.targetScaling = 1.
		self.safetyScaling = 1.

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.config = config
		self.image_size = config.size[0] #128

	def set_wm(self, wm, config):
		self.encoder = wm.encoder.to(self.device)
		self.MLP_dyn = wm.dynamics.to(self.device)
		self.wm = wm.to(self.device)
		self.use_wm = True
		if config.wm:
			if config.dyn_discrete:
				self.feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
			else:
				self.feat_size = config.dyn_stoch + config.dyn_deter
		
	def set_ensemble(self, ensemble, config):
		self.disagreement_ensemble = ensemble.to(self.device)
		self.use_ensemble_disagreement = config.use_ensemble
		print("use ensemble disagreement: {}".format(self.use_ensemble_disagreement))


	def reset(
			self, start=None, theta=None, sample_inside_obs=False,
			sample_inside_tar=True
	):

		latent, gt_state = self.sample_random_state(
					sample_inside_obs=sample_inside_obs,
					sample_inside_tar=sample_inside_tar, theta=theta, init_state=start,
					rand_state=True
			)
		self.state = gt_state
		self.latent = latent
		return latent.copy(), np.copy(gt_state)


	def get_latent(self, xs, ys, thetas, imgs):
		states = np.expand_dims(np.expand_dims(thetas,1),1)
		imgs = np.expand_dims(imgs, 1)
		dummy_acs = np.zeros((np.shape(xs)[0], 1, 3))
		rand_idx = 1 #go straight #np.random.randint(0, 3, np.shape(xs)[0])
		dummy_acs[np.arange(np.shape(xs)[0]), :, rand_idx] = 1
		firsts = np.ones((np.shape(xs)[0], 1))
		lasts = np.zeros((np.shape(xs)[0], 1))
		
		cos = np.cos(states)
		sin = np.sin(states)

		states = np.concatenate([cos, sin], axis=-1)

		chunks = 21
		if np.shape(imgs)[0] > chunks:
			bs = int(np.shape(imgs)[0]/chunks)
		else:
			bs = int(np.shape(imgs)[0]/chunks)
		
		for i in range(chunks):
			if i == chunks-1:
				data = {'obs_state': states[i*bs:], 'image': imgs[i*bs:], 'action': dummy_acs[i*bs:], 'is_first': firsts[i*bs:], 'is_terminal': lasts[i*bs:]}
			else:
				data = {'obs_state': states[i*bs:(i+1)*bs], 'image': imgs[i*bs:(i+1)*bs], 'action': dummy_acs[i*bs:(i+1)*bs], 'is_first': firsts[i*bs:(i+1)*bs], 'is_terminal': lasts[i*bs:(i+1)*bs]}

			data = self.wm.preprocess(data)
			embeds = self.encoder(data)
			if i == 0:
				embed = embeds
			else:
				embed = torch.cat([embed, embeds], dim=0)

		data = {'obs_state': states, 'image': imgs, 'action': dummy_acs, 'is_first': firsts, 'is_terminal': lasts}
		data = self.wm.preprocess(data)
		post, prior = self.wm.dynamics.observe(
				embed, data["action"], data["is_first"]
				)
		
		if self.gt_lx:
			raise Exception('Not implemented')
		else:
			g_x = self.safety_margin(post)

		feat = self.wm.dynamics.get_feat(post).detach().cpu().numpy().squeeze()
		return g_x, feat, post
	

	def sample_random_state(
			self, sample_inside_obs=False, sample_inside_tar=True, theta=None, init_state=None, rand_state=False
	):
		"""Picks the state uniformly at random.

		Args:
				sample_inside_obs (bool, optional): consider sampling the state inside
						the obstacles if True. Defaults to False.
				sample_inside_tar (bool, optional): consider sampling the state inside
						the target if True. Defaults to True.
				theta (float, optional): if provided, set the initial heading angle
						(yaw). Defaults to None.

		Returns:
				np.ndarray: the sampled initial state.
		"""

		if rand_state:
			rnd_state = np.random.uniform(low=self.low[:2], high=self.high[:2])
			theta_rnd = 2.0 * np.random.uniform() * np.pi
			state0 = np.array([rnd_state[0], rnd_state[1], theta_rnd])
			img0 = self.capture_image(state0)

		else:
			random_idx = random.randint(0, len(self.observed_states) - 1)
			state0 = np.array(self.observed_states[random_idx])
			img0 = self.capture_image(state0)

		rand_u0 = np.zeros(3)
		data = {'obs_state': [[[np.cos(state0[-1]), np.sin(state0[-1])]]], 'image': [[img0]], 'action': [[rand_u0]], 'is_first': np.array([[[True]]]), 'is_terminal': np.array([[[False]]])}
		data = self.wm.preprocess(data)
		embed = self.encoder(data)

		post, prior = self.wm.dynamics.observe(
				embed, data["action"], data["is_first"]
				)

		return post, state0
				

	def capture_image(self, state=None):
		"""Captures an image of the current state of the environment."""


		# Region to stop (green area)
		stop_region_x_1 = (-1.0, 1.0)
		stop_region_y_1 = (-1.0, -0.6)

		# Region to stop (green area)
		stop_region_x_2 = (-1.0, 1.0)
		stop_region_y_2 = (0.6, 1.0)
		
		fig,ax = plt.subplots()
		plt.xlim([-1.0, 1.0])
		plt.ylim([-1.0, 1.0])
		plt.axis('off')
		fig.set_size_inches( 1, 1 )

		rect1 = patches.Rectangle(
			(stop_region_x_1[0], stop_region_y_1[0]),
			stop_region_x_1[1] - stop_region_x_1[0],
			stop_region_y_1[1] - stop_region_y_1[0],
			linewidth=3, facecolor='#BB96EB', edgecolor="none"
		)

		ax.add_patch(rect1)

		rect2 = patches.Rectangle(
			(stop_region_x_2[0], stop_region_y_2[0]),
			stop_region_x_2[1] - stop_region_x_2[0],
			stop_region_y_2[1] - stop_region_y_2[0],
			linewidth=3, facecolor='#BB96EB', edgecolor="none"
		)
		ax.add_patch(rect2)

		# Add the circle patch to the axis
		dt = self.time_step
		v = self.speed
		dpi=self.image_size

		plt.quiver(
			state[0], state[1],
			dt * v * np.cos(state[2]),
			dt * v * np.sin(state[2]),
			angles='xy', scale_units='xy', minlength=0,
			width=0.1, scale=0.18, color="black", zorder=3
		)

		plt.scatter(
				state[0], state[1],
				s=20, color="black", zorder=3
		)

		plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

		buf = io.BytesIO()
		plt.savefig(buf, format='png', dpi=dpi)
		buf.seek(0)

		# Load the buffer content as an RGB image
		img = Image.open(buf).convert('RGB')
		img_array = np.array(img)
		plt.close()
		return img_array
	
	def forward_latent(self, latent_prev, act_prev, image, state):

		action = np.zeros(3)
		action[act_prev] = 1
		obs = {'obs_state': [[[np.cos(state[-1]), np.sin(state[-1])]]], 'image': [[image]], 'action': [[action]], 'is_first': np.array([[[False]]]), 'is_terminal': np.array([[[False]]])}

		with torch.no_grad():
			data = self.wm.preprocess(obs)
			embed = self.wm.encoder(data)

			# Latent and action is None for the initial
			latent_post, latent_prior = self.wm.dynamics.obs_step(latent_prev, data['action'], embed, data["is_first"], sample=False)

			return latent_post, latent_prior
	
	# == Dynamics ==
	def step(self, action):
		"""Evolves the environment one step forward given an action.

		Args:
				action (int): the index of the action in the action set.

		Returns:
				np.ndarray: next state.
				bool: True if the episode is terminated.
		"""
		if type(action) == dict:
			action = np.argmax(action[ 'action' ])

		img_ac = torch.zeros(3).to(self.device)
		img_ac[action] = 1
		img_ac = img_ac.unsqueeze(0).unsqueeze(0)
		
		init = {k: v[:, -1] for k, v in self.latent.items()}
		self.latent = self.wm.dynamics.imagine_with_action(img_ac, init)
		

		# step gt state
		u = self.discrete_controls[action]
		state = self.integrate_forward(self.state, u)
		self.state = state

		return self.latent.copy(), np.copy(self.state) # need latent 

	def integrate_forward(self, state, u):
		"""Integrates the dynamics forward by one step.

		Args:
				state (np.ndarray): (x, y, yaw).
				u (float): the contol input, angular speed.

		Returns:
				np.ndarray: next state.
		"""
		x, y, theta = state
		x = x + self.time_step * self.speed * np.cos(theta)
		y = y + self.time_step * self.speed * np.sin(theta)
		theta = np.mod(theta + self.time_step * u, 2 * np.pi)
		assert theta >= 0 and theta < 2 * np.pi
		state_next = np.array([x, y, theta])
			
		return state_next

	# == Setting Hyper-Parameter Functions ==
	def set_bounds(self, bounds):
		"""Sets the boundary of the environment.

		Args:
				bounds (np.ndarray): of the shape (n_dim, 2). Each row is [LB, UB].
		"""
		self.bounds = bounds

		# Get lower and upper bounds
		self.low = np.array(self.bounds)[:, 0]
		self.high = np.array(self.bounds)[:, 1]

	def set_speed(self, speed=.5):
		"""Sets speed of the car. The speed influences the angular speed and the
				discrete control set.

		Args:
				speed (float, optional): speed of the car. Defaults to .5.
		"""
		self.speed = speed
		self.max_turning_rate = self.speed / self.R_turn  # w
		self.discrete_controls = np.array([
				-self.max_turning_rate, 0., self.max_turning_rate
		])

	def set_time_step(self, time_step=.05):
		"""Sets the time step for dynamics integration.

		Args:
				time_step (float, optional): time step used in the integrate_forward.
						Defaults to .05.
		"""
		self.time_step = time_step

	def set_radius(self, target_radius=.3, constraint_radius=1., R_turn=.6):
		"""Sets target_radius, constraint_radius and turning radius.

		Args:
				target_radius (float, optional): the radius of the target set.
						Defaults to .3.
				constraint_radius (float, optional): the radius of the constraint set.
						Defaults to 1.0.
				R_turn (float, optional): the radius of the car's circular motion.
						Defaults to .6.
		"""
		self.target_radius = target_radius
		self.constraint_radius = constraint_radius
		self.set_radius_rotation(R_turn=R_turn)

	def set_radius_rotation(self, R_turn=.6, verbose=False):
		"""Sets radius of the car's circular motion. The turning radius influences
				the angular speed and the discrete control set.

		Args:
				R_turn (float, optional): the radius of the car's circular motion.
						Defaults to .6.
				verbose (bool, optional): print messages if True. Defaults to False.
		"""
		self.R_turn = R_turn
		self.max_turning_rate = self.speed / self.R_turn  # w
		self.discrete_controls = np.array([
				-self.max_turning_rate, 0., self.max_turning_rate
		])
		if verbose:
			print(self.discrete_controls)

	def set_constraint(self, center, radius):
		"""Sets the constraint set (complement of failure set).

		Args:
				center (np.ndarray, optional): center of the constraint set.
				radius (float, optional): radius of the constraint set.
		"""
		self.constraint_center = center
		self.constraint_radius = radius

	def set_target(self, center, radius):
		"""Sets the target set.

		Args:
				center (np.ndarray, optional): center of the target set.
				radius (float, optional): radius of the target set.
		"""
		self.target_center = center
		self.target_radius = radius

	# == Getting Functions ==
	def check_within_bounds(self, state):
		"""Checks if the agent is still in the environment.

		Args:
				state (np.ndarray): the state of the agent.

		Returns:
				bool: False if the agent is not in the environment.
		"""
		new_bound = self.bounds.copy()
		margin = 0.0
		new_bound[0, 0] += margin
		new_bound[1, 0] += margin
		new_bound[0, 1] -= margin
		new_bound[1, 1] -= margin

		for dim, bound in enumerate(new_bound):
		# for dim, bound in enumerate(self.bounds):
			flagLow = state[dim] < (bound[0])
			flagHigh = state[dim] > (bound[1])
			if flagLow or flagHigh:
				return False
		return True
	
	def check_success(self, state):
		"""Checks if the agent is still in the environment.

		Args:
				state (np.ndarray): the state of the agent.

		Returns:
				bool: False if the agent is not in the environment.
		"""
		
		x = state[0]
		y = state[1]
		if ((x < -1.0) or (x > 1.0)) and (-0.6 < y < 0.6):
			return True
		else :
			return False
		
	def check_fail(self, state):
		"""Checks if the agent is still in the environment.

		Args:
				state (np.ndarray): the state of the agent.

		Returns:
				bool: False if the agent is not in the environment.
		"""
		
		y = state[1]
		if ((y < -1.0) or (y > 1.0)):
			return True
		else :
			return False

	# == Compute Margin ==
	def safety_margin(self, s, action=None):
		"compute learned safety margin"
		g_xList = []
		if self.gt_lx:
			raise Exception('Not implemented')
		
		g_x = np.array(1.)
		feat = self.wm.dynamics.get_feat(s).detach()
		with torch.no_grad():  # Disable gradient calculation
			if action is not None:
				# 1. Make action
				action_onehot = torch.zeros(3).to(self.device)
				action_onehot[action] = 1
				action_onehot = action_onehot.unsqueeze(0).unsqueeze(0)

				# # 2. measure disagreement
				disagreement = self.disagreement_ensemble._intrinsic_reward_penn(feat, action_onehot)
				disagreement = disagreement.squeeze(-1)
				if disagreement > self.config.ood_thr:
					g_x = np.array(-1.)

			g_xList.append(g_x)
		
		safety_margin = np.array(g_xList).squeeze()

		return self.safetyScaling * safety_margin
	
	def gt_safety_margin(self, states):
		# Region to stop (green area)
		stop_region_x_1 = (-1.0, 1.0)
		stop_region_y_1 = (-1.0, -0.6)

		# Region to stop (green area)
		stop_region_x_2 = (-1.0, 1.0)
		stop_region_y_2 = (0.6, 1.0)

		inside_region1 = (stop_region_x_1[0] <= states[0] <= stop_region_x_1[1] and 
						  stop_region_y_1[0] <= states[1] <= stop_region_y_1[1])
		
		inside_region2 = (stop_region_x_2[0] <= states[0] <= stop_region_x_2[1] and 
						  stop_region_y_2[0] <= states[1] <= stop_region_y_2[1])
		
		g_x = -1 if (inside_region1 or inside_region2) else 1
		
		safety_margin = np.array(g_x).squeeze()

		return self.safetyScaling * safety_margin
