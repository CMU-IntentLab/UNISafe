from os import path
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import io
from PIL import Image
import matplotlib.patches as patches
import torch
import math, random



class Takeoff_WM_Failure_Nf(gym.Env):
    def __init__(self):
        self.render_mode = None
        self.time_step = 0.05
        self.high = np.array([
            1., 1., np.pi,
        ])
        self.low = np.array([
            -1., -1., -np.pi
        ])
        self.device = 'cuda'

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,1,1536,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(7,), dtype=np.float32) # joint action space
        self.image_size=128
        self.failure_threshold = 0.7
        

    def set_wm(self, config, wm, ensemble, dataset):
        self.config = config
        self.wm = wm
        self.dataset = dataset
        self.nf = ensemble
        self.threshold = self.config.ood_threshold
        
        if self.config.dyn_discrete:
            self.feat_size = self.config.dyn_stoch * self.config.dyn_discrete + self.config.dyn_deter
        else:
            self.feat_size = self.config.dyn_stoch + self.config.dyn_deter
    

    def step(self, action):

        with torch.no_grad():
            action = action.detach()

            failure = self.failure_classifier(self.feat)[:, 0] # in [0, 1]
            s_failure = 0.5 - failure.clone()
            s_failure = 1.5 * torch.tanh(s_failure)

            # ALL
            if self.config.use_uq:
                epistemic_uncertainty = self.calculate_uncertainty(self.feat) # 0: in-dist, 1: ood
                # epistemic_uncertainty = 0.5 - uncertainty # (T,), tensor

                uncertainty = epistemic_uncertainty.clone()
                uncertainty[epistemic_uncertainty < 0.5] = 1.0
                uncertainty[epistemic_uncertainty >= 0.5] = -1.0
                # print(uncertainty)
                reward = self.calculate_reward(uncertainty, s_failure)
            
            else:
                reward = s_failure
            
            # Update
            action = action.reshape((self.config.batch_size, self.config.batch_length, -1))
            self.latent = self.wm.dynamics.img_step(self.latent, action)
            self.feat = self.wm.dynamics.get_feat(self.latent)
            self.feat = self.feat.reshape((self.config.batch_size * self.config.batch_length, -1))

            done = False
            truncated = False
            info = {}

        return self.feat.cpu().numpy(), reward, done, truncated, info
    
    def reset(self,):
        super().reset()

        init_traj = next(self.dataset)

        with torch.no_grad():
            data = self.wm.preprocess(init_traj)
            embed = self.wm.encoder(data)
            self.latent, _ = self.wm.dynamics.observe(
                embed, data["action"], data["is_first"]
            )

            self.feat = self.wm.dynamics.get_feat(self.latent)
            self.feat = self.feat.reshape((self.config.batch_size * self.config.batch_length, -1))

        return self.feat.cpu().numpy()
      
    def calculate_uncertainty(self, feat):
        with torch.no_grad():  # Disable gradient calculation
            epistemic_uncertainty = 1 - self.nf.calculate_likelihood(feat)
            return epistemic_uncertainty
        
    
    def failure_classifier(self, feats):
        with torch.no_grad():  # Disable gradient calculation
            failure_dist = self.wm.heads["failure"](feats)
            failure = failure_dist.mean # \in [0, 1] (Bernouli Dist)

            return failure

    def calculate_reward(self, uncertainty_reward, failure_reward):
        final_rewards = torch.minimum(uncertainty_reward, failure_reward)
    
        return final_rewards

# class Takeoff_WM_Failure(gym.Env):
#     def __init__(self):
#         self.render_mode = None
#         self.time_step = 0.05
#         self.high = np.array([
#             1., 1., np.pi,
#         ])
#         self.low = np.array([
#             -1., -1., -np.pi
#         ])
#         self.device = 'cuda'

#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,1,1536,), dtype=np.float32)
#         self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(7,), dtype=np.float32) # joint action space
#         self.image_size=128
#         self.failure_threshold = 0.7
        

#     def set_wm(self, config, wm, ensemble, dataset):
#         self.config = config
#         self.wm = wm
#         self.dataset = dataset
#         self.ensemble = ensemble
#         self.threshold = self.config.ood_threshold
        
#         if self.config.dyn_discrete:
#             self.feat_size = self.config.dyn_stoch * self.config.dyn_discrete + self.config.dyn_deter
#         else:
#             self.feat_size = self.config.dyn_stoch + self.config.dyn_deter
    

#     def step(self, action):

#         with torch.no_grad():
#             action = action.detach().unsqueeze(0)

#             inputs = torch.concat([self.feat, action], -1)
#             epistemic_uncertainty = self.threshold - self.ensemble.intrinsic_reward_penn(inputs) # (T,)
#             failure = self.failure_classifier(self.feat) # in [0, 1]
#             reward = self.calculate_reward(epistemic_uncertainty, failure).item()
#             done = (epistemic_uncertainty.item() < 0)

#             # Update
#             self.latent = self.wm.dynamics.img_step(self.latent, action)
#             self.feat = self.wm.dynamics.get_feat(self.latent)

#             truncated = False

#             info = {}

#         return self.feat.cpu().numpy(), reward, done, truncated, info
    
#     def reset(self,):
#         super().reset()

#         init_traj = next(self.dataset)

#         with torch.no_grad():
#             data = self.wm.preprocess(init_traj)
#             embed = self.wm.encoder(data)
#             self.latent, _ = self.wm.dynamics.observe(
#                 embed, data["action"], data["is_first"]
#             )

#             idx = random.randint(0, self.config.batch_length -1)
#             for k, v in self.latent.items(): 
#                 self.latent[k] = v[:, [idx]]

#             self.feat = self.wm.dynamics.get_feat(self.latent)

#         return self.feat.cpu().numpy()
    

#     def step_offline(self):
#         init_traj = next(self.dataset)

#         with torch.no_grad():
#             data = self.wm.preprocess(init_traj)
#             embed = self.wm.encoder(data)
#             post, _ = self.wm.dynamics.observe(
#                 embed, data["action"], data["is_first"]
#             )

#             feats_orig = self.wm.dynamics.get_feat(post)
#             feats = feats_orig[:, :-1]
#             actions = data["action"][:, 1:]
#             next_feats = feats_orig[:, 1:, :]
#             inputs = torch.concat([feats, actions], -1)
#             epistemic_uncertainty = self.threshold - self.ensemble.intrinsic_reward_penn(inputs) # (T,)
#             failure = data["failure"][:, :-1]

#             final_rewards = self.calculate_reward(epistemic_uncertainty, failure)

#             dones = data["is_first"][0].roll(-1)[:-1]

#         return feats.squeeze().cpu().numpy(), actions.squeeze().cpu().numpy(), \
#             final_rewards.squeeze().cpu().numpy(), next_feats.squeeze().cpu().numpy(), dones.cpu().numpy()
      
#     def calculate_uncertainty(self, feat, action):
#         with torch.no_grad():  # Disable gradient calculation
#             inputs = torch.concat([feat, action], -1)
#             epistemic_uncertainty = self.ensemble.intrinsic_reward_penn(inputs)[:, :, 0]
#             return epistemic_uncertainty.item()
        
    
#     def failure_classifier(self, feats):
#         with torch.no_grad():  # Disable gradient calculation
#             failure_dist = self.wm.heads["failure"](feats)
#             failure = failure_dist.mean # \in [0, 1] (Bernouli Dist)

#             return failure

#     def calculate_reward(self, uncertainty, failure):
#         final_rewards = torch.where(
#                 failure > self.failure_threshold,  # Case 1: If failure is above the threshold
#                 torch.minimum(-failure, uncertainty),  # Assign negative failure as penalty
#                 torch.where(
#                     uncertainty < 0,  # Case 2: If epistemic uncertainty is negative
#                     uncertainty,  # Assign epistemic uncertainty
#                     torch.minimum(1 - failure, uncertainty)  # Otherwise, assign a positive value inversely proportional to failure
#                 )
#             ) 
    
#         return final_rewards