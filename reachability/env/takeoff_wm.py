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

class Takeoff_WM(gym.Env):
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
        

    def set_wm(self, config, wm, ensemble, dataset):
        self.config = config
        self.wm = wm
        self.dataset = dataset
        self.ensemble = ensemble
        self.threshold = self.config.ood_threshold
        
        if self.config.dyn_discrete:
            self.feat_size = self.config.dyn_stoch * self.config.dyn_discrete + self.config.dyn_deter
        else:
            self.feat_size = self.config.dyn_stoch + self.config.dyn_deter
    

    def step(self, action):

        with torch.no_grad():
            # action = action.detach().unsqueeze(0)

            action = action.detach()
            uncertainty = self.calculate_uncertainty(self.feat, action)
            reward = self.threshold - uncertainty

            s_reward = reward.copy()
            s_reward[reward < -0.4] = -1.0
            s_reward[reward > 0.4] = 1.0
            reward = s_reward

            # update feat
            action = action.reshape((self.config.batch_size, self.config.batch_length, -1))
            self.latent = self.wm.dynamics.img_step(self.latent, action)
            self.feat = self.wm.dynamics.get_feat(self.latent)
            self.feat = self.feat.reshape((self.config.batch_size * self.config.batch_length, -1))

            # done = (reward < 0)
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

            # idx = random.randint(0, self.config.batch_length -1)
            # for k, v in self.latent.items(): 
            #     self.latent[k] = v[:, [idx]]

            self.feat = self.wm.dynamics.get_feat(self.latent)
            self.feat = self.feat.reshape((self.config.batch_size * self.config.batch_length, -1))

        return self.feat.cpu().numpy()

    def step_offline(self):
        init_traj = next(self.dataset)

        with torch.no_grad():
            data = self.wm.preprocess(init_traj)
            embed = self.wm.encoder(data)
            post, _ = self.wm.dynamics.observe(
                embed, data["action"], data["is_first"]
            )

            feats_orig = self.wm.dynamics.get_feat(post)
            feats = feats_orig[:, :-1]
            actions = data["action"][:, 1:]
            next_feats = feats_orig[:, 1:, :]
            inputs = torch.concat([feats, actions], -1)
            epistemic_uncertainty = self.threshold - self.ensemble.intrinsic_reward_penn(inputs) # (T,)
            # dones = data["is_first"][0].roll(-1)[:-1]
            dones = data["is_first"].roll(-1, dims=1)[:, :-1]

        return feats.cpu().numpy(), actions.cpu().numpy(), \
            epistemic_uncertainty.cpu().numpy(), next_feats.cpu().numpy(), dones.cpu().numpy()
      
    def calculate_uncertainty(self, feat, action):
        with torch.no_grad():  # Disable gradient calculation
            inputs = torch.concat([feat, action], -1)
            epistemic_uncertainty = self.ensemble.intrinsic_reward_penn(inputs)
            return epistemic_uncertainty[:, 0].cpu().numpy()
 