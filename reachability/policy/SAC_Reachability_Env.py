from cv2 import threshold
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import abc

from collections import namedtuple
import os
import pickle
import numpy as np
from torch.distributions import Independent, Normal
import wandb
import random
import gymnasium as gym

# Reuse the same Transition namedtuple and ReplayMemory from your code
from .ReplayMemory import ReplayMemory
from .utils import soft_update, save_model, StepLRMargin, GaussianNoise


from .model import ActorProb, Critic, Net
from reachability_dreamer.policy.takeoff.visualize import visualize_uncertainty_with_value
import reachability_dreamer.env
from copy import deepcopy


Transition = namedtuple("Transition", ["s", "a", "r", "s_", "done", "info"])


class SAC(nn.Module):

	def __init__(self, CONFIG, dim_state, dim_action, actor_dimList, critic_dimList):
		super(SAC, self).__init__()
		self.CONFIG = CONFIG
		self.saved = False
		self.memory = ReplayMemory(CONFIG.memoryCapacity)

		self.training = True

		# Device (cpu or cuda)
		self.device = CONFIG.device

		# Discount factor
		self.gamma = CONFIG.gamma_reachability
		self.gamma_scheduler = StepLRMargin(initValue=0.85, period=4e3, goalValue=0.9999, decay=0.95)

		# Soft/hard update
		self.tau = CONFIG.tau
		self.soft_update = True 

		# Actor and Critic learning rates
		self.lr_actor = CONFIG.actor_lr
		self.lr_critic = CONFIG.critic_lr

		# Dimensions
		self.dim_state = dim_state
		self.dim_action = dim_action

		self.actor_dimList = actor_dimList
		self.critic_dimList = critic_dimList

		self.max_action = 0.5
		self.target_entropy = -self.dim_action  # e.g. -dim_action
		self.learn_alpha = True
		self.log_alpha = torch.zeros(1, requires_grad=self.learn_alpha, device=self.device)
		self.alpha = self.log_alpha.detach().exp() if self.learn_alpha else 0.2  # initial alpha

		# Keep track of gradient updates
		self.cntUpdate = 0
		self.max_grad_norm = 1  # gradient clipping if desired
		self.__eps = np.finfo(np.float32).eps.item()
		self.batch_size = CONFIG.reachability_batch_size
		self.epsilon = 0.9

		# Build networks
		self._build_networks()

		# Build optimizers
		self._build_optimizers()


	def setup_env(self, config, wm, ensemble, dataset, env_name="Takeoff_WM-v0"):
		self.env = gym.make(env_name)
		self.env.set_wm(config, wm, ensemble, dataset)

	def _actor_network(self, dim_state, dim_action):
		"""
		actor (policy) pi(a|s). The network should output distribution
		parameters (e.g., for a Gaussian).
		"""

		actor1_net = Net(dim_state, hidden_sizes=self.actor_dimList, norm_layer=nn.LayerNorm, device=self.device)

		actor = ActorProb(
			preprocess_net=actor1_net,
			action_shape=(dim_action,),            # continuous action dimension
			hidden_sizes=(),                       # extra MLP layers after preprocess_net if you wish
			max_action=self.max_action,                        # scale final action by tanh if not unbounded
			device=self.device,            			# if True, sigma also depends on obs
			preprocess_net_output_dim=None,        # let ActorProb infer from preprocess_net
			unbounded=True,                       # if True, skip tanh on mu
			conditioned_sigma=True,               # if True, sigma also depends on obs
		)
		return actor

	def _critic_network(self, critic_net, dim_state, dim_action):
		"""
		Abstract method that should return an nn.Module implementing
		Q(s,a).
		"""

		if critic_net is None:
			critic_net = Net(
				dim_state,
				dim_action,
				hidden_sizes=self.critic_dimList,
				norm_layer=nn.LayerNorm,
				concat=True,
				device=self.device
			)

		critic = Critic(
			preprocess_net=critic_net,
			hidden_sizes=(),                  # more layers after preprocess_net if desired
			device=self.device,
			preprocess_net_output_dim=None,   # let Critic infer from preprocess_net
			flatten_input=False,              # Critic will flatten in forward; safe to leave False
		)

		return critic
		
	def train(self, mode: bool = True):
		self.training = mode
		self.actor.train(mode)
		self.critic1.train(mode)
		self.critic2.train(mode)
		return self

	def _build_networks(self):
		"""Builds the actor, critics, and their target networks."""
		# Single Actor
		self.actor = self._actor_network(self.dim_state, self.dim_action).to(self.device)

		# Two critics
		critic_net = Net(
				self.dim_state,
				self.dim_action,
				hidden_sizes=self.critic_dimList,
				concat=True,
				device=self.device
			)
		
		critic_net_target = Net(
				self.dim_state,
				self.dim_action,
				hidden_sizes=self.critic_dimList,
				concat=True,
				device=self.device
			)
		
		self.critic1 = self._critic_network(critic_net, self.dim_state, self.dim_action).to(self.device)
		self.critic2 = self._critic_network(critic_net, self.dim_state, self.dim_action).to(self.device)

		# Targets for the critics
		self.critic1_target = self._critic_network(critic_net_target, self.dim_state, self.dim_action).to(self.device)
		self.critic2_target = self._critic_network(critic_net_target, self.dim_state, self.dim_action).to(self.device)

		# Initialize targets with same weights
		self.critic1_target.load_state_dict(self.critic1.state_dict())
		self.critic2_target.load_state_dict(self.critic2.state_dict())

		self.critic1_target.eval()
		self.critic2_target.eval()

	def _build_optimizers(self):
		"""Builds the optimizers for actor, critics, and alpha (if learnable)."""
		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
		self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr_critic)
		self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr_critic)

		if self.learn_alpha:
			self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_actor)
		else:
			self.alpha_optimizer = None

	def update_target_networks(self):
		"""
		Updates the target networks using soft update or, if desired, a
		periodic hard update.
		"""
		if self.soft_update:
			soft_update(self.critic1_target, self.critic1, self.tau)
			soft_update(self.critic2_target, self.critic2, self.tau)
		else:
			pass

	
	def select_action(self, state, eval_mode=False):
		
		logits, _ = self.actor(state)
		logits = logits[0], logits[1]
		dist = Independent(Normal(*logits), 1)

		if not eval_mode:
			action = dist.rsample()
		else:
			action = logits[0]
		
		log_prob = dist.log_prob(action).unsqueeze(-1)
		# apply correction for Tanh squashing when computing logprob from Gaussian
		# You can check out the original SAC paper (arXiv 1801.01290): Eq 21.
		# in appendix C to get some understanding of this equation.
		squashed_action = torch.tanh(action)
		output_action = self.max_action * squashed_action
		log_prob = log_prob - torch.log(self.max_action * (1 - squashed_action.pow(2)) +
										self.__eps).sum(-1, keepdim=True)
		
		return output_action, log_prob
	
	def sample_actions(self, state, n_action=1):
		
		logits, _ = self.actor(state)
		logits = logits[0], logits[1]
		dist = Independent(Normal(*logits), 1)
		actions = self.max_action * torch.tanh(dist.rsample(torch.Size([n_action])))

		action_mean = self.max_action * torch.tanh(logits[0])

		return actions, action_mean
	

	def update(self, states, actions, rewards, dones, next_states):

		non_final_mask = ~dones  # shape (N,)
		non_final_next_feats = next_states[non_final_mask]  # shape (N_nonfinal, feat_dim)
		
		with torch.no_grad():
			# Next actions from the current actor (target policy).
			next_actions, next_actions_log_prob = self.select_action(non_final_next_feats, eval_mode=False)    

			# Evaluate Q-target networks
			Q1_next = self.critic1_target(non_final_next_feats, next_actions)
			Q2_next = self.critic2_target(non_final_next_feats, next_actions)
			Q_next = torch.min(Q1_next, Q2_next)
			state_value_nxt = (Q_next - self.alpha * next_actions_log_prob).squeeze(-1)

			y_target = rewards.clone().squeeze(-1)  # shape (N,)

			non_terminal_value = torch.min(
				rewards[non_final_mask].squeeze(-1),  # (N_nonfinal,)
				state_value_nxt
			)

			y_target[non_final_mask] = (
				non_terminal_value * self.gamma 
				+ rewards[non_final_mask].squeeze(-1) * (1 - self.gamma)
			)

			y_target = y_target.unsqueeze(-1)

		# Critic Update
		Q1_pred = self.critic1(states, actions)
		Q2_pred = self.critic2(states, actions)
		critic1_loss = nn.MSELoss()(Q1_pred, y_target)
		critic2_loss = nn.MSELoss()(Q2_pred, y_target)
		critic_loss = critic1_loss + critic2_loss

		self.critic1_optimizer.zero_grad()
		self.critic2_optimizer.zero_grad()
		critic_loss.backward()
		self.critic1_optimizer.step()
		self.critic2_optimizer.step()

		# Actor Update
		new_actions, log_probs = self.select_action(states, eval_mode=False)
		Q1_new = self.critic1(states, new_actions)
		Q2_new = self.critic2(states, new_actions)
		Q_new = torch.min(Q1_new, Q2_new)

		if self.CONFIG.action_reg:
			action_reg_loss = 1e-2 * (new_actions[:,:6] ** 2).sum(dim=1).mean()
			actor_loss = (self.alpha * log_probs - Q_new).mean() + action_reg_loss
		else:
			actor_loss = (self.alpha * log_probs - Q_new).mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update Alpha
		if self.learn_alpha:
			alpha_loss = -self.log_alpha * (log_probs.detach() + self.target_entropy).mean()
			self.alpha_optimizer.zero_grad()
			alpha_loss.backward()
			self.alpha_optimizer.step()
			self.alpha = self.log_alpha.exp().item()

		# Soft-update target critics
		self.update_target_networks()
		self.cntUpdate += 1

		self.gamma_scheduler.step()
		self.gamma = self.gamma_scheduler.get_value()

		return (critic_loss.item(), actor_loss.item(), alpha_loss.item())

	def save(self, step, logs_path):
		"""Saves model weights and configuration."""
		save_model(self.actor, step, logs_path, "actor")
		save_model(self.critic1, step, logs_path, "critic1")
		save_model(self.critic2, step, logs_path, "critic2")

		if not self.saved:
			config_path = os.path.join(logs_path, "CONFIG.pkl")
			pickle.dump(self.CONFIG, open(config_path, "wb"))
			self.saved = True

	def restore(self, step, logs_path, verbose=True):
		"""Restores model weights from the given model path."""
		actor_path = os.path.join(logs_path, "model", f"actor-{step}.pth")
		critic1_path = os.path.join(logs_path, "model", f"critic1-{step}.pth")
		critic2_path = os.path.join(logs_path, "model", f"critic2-{step}.pth")

		self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
		self.critic1.load_state_dict(torch.load(critic1_path, map_location=self.device))
		self.critic2.load_state_dict(torch.load(critic2_path, map_location=self.device))
		self.critic1_target.load_state_dict(self.critic1.state_dict())
		self.critic2_target.load_state_dict(self.critic2.state_dict())

		if verbose:
			print(f"  => Restored actor from {actor_path}")
			print(f"  => Restored critic1 from {critic1_path}")
			print(f"  => Restored critic2 from {critic2_path}")

	def evaluate(self, wm, ensemble, critic, data, cnt):
		
		visualize_uncertainty_with_value(wm, ensemble, critic, data, cnt)

	def store_transition(self, *args):
		"""Stores the transition into the replay buffer.
		"""
		self.memory.update(Transition(*args))

	def unpack_batch(self, batch):
		"""Decomposes the batch into different variables.

		Args:
			batch (object): Transition of batch-arrays.

		Returns:
			A tuple of torch.Tensor objects, extracted from the elements in the
				batch and processed for update().
		"""

		state = torch.FloatTensor(batch.s).to(self.device)
		action = torch.FloatTensor(batch.a).to(self.device)
		reward = torch.FloatTensor(batch.r).to(self.device)
		done = torch.BoolTensor(batch.done).to(self.device)
		next_state = torch.FloatTensor(batch.s_).to(self.device)

		return (
			state, action, reward, done, next_state
		)

	def update_batch(self):
		# == EXPERIENCE REPLAY ==
		transitions = self.memory.sample(self.batch_size)
		# Transpose the batch (see https://stackoverflow.com/a/19343/3343043
		# for detailed explanation). This converts batch-array of Transitions
		# to Transition of batch-arrays.
		batch = Transition(*zip(*transitions))
		(states, actions, rewards, dones, next_states) = self.unpack_batch(batch)

		return self.update(states, actions, rewards, dones, next_states)
	

	def learn(
		self, dataset, wm, ensemble, MAX_UPDATES=200000, warmup_iter=10000,
		checkPeriod=5000, outFolder="sac_logs", step_repeat=10
	):
		os.makedirs(outFolder, exist_ok=True)
		modelFolder = os.path.join(outFolder, "model")
		os.makedirs(modelFolder, exist_ok=True)

		trainingRecords = []
		threshold = self.CONFIG.ood_threshold
		self.cnt = 0

		ep = 0
		env = self.env
		MAX_EP_STEPS = 30

		while self.cntUpdate <= MAX_UPDATES:

			s = env.reset()
			ep += self.CONFIG.batch_size * self.CONFIG.batch_length
			# Rollout
			for step_num in range(MAX_EP_STEPS):
				# Select action
				with torch.no_grad():
					a, _ = self.select_action(s, eval_mode=False) # Explore for a step.
					# Interact with env
					s_, r, done, _, info = env.step(a)

					# Truncate!
					if step_num == (MAX_EP_STEPS - 1):
						done = True

					for i in range(self.CONFIG.batch_size * self.CONFIG.batch_length):
						self.store_transition(s[i], a[i].squeeze().cpu().numpy(), r[i], s_[i].squeeze(), done, info)
					
					s = s_

				if len(self.memory) > self.CONFIG.batch_size:
					critic_loss, actor_loss, alpha_loss = self.update_batch()
				else: 
					continue

				# Check periodically
				if self.cntUpdate != 0 and self.cntUpdate % checkPeriod == 0:
					# Save model
					self.save(self.cntUpdate, modelFolder)
				
				print(f"[Training {self.cntUpdate:4d}] [Ep {ep}, Step {step_num}] Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}", end='\r', flush=True)

				wandb.log({
					"critic_loss": critic_loss,
					"actor_loss": actor_loss,
					"alpha_loss": alpha_loss,
					"alpha": self.alpha,
					"gamma": self.gamma,
					"step": self.cntUpdate  # Replace `training_step` with the current step variable
				})

				if done:
					break

		trainingRecords = np.array(trainingRecords)
		return trainingRecords
		
