import copy
import torch
import torch.optim
from torch import nn
from termcolor import cprint
from functools import partial
import numpy as np
import torch.nn.functional as F

import dreamer.networks as networks
import dreamer.tools as tools
from tqdm import trange
from torch.nn.utils import spectral_norm

def to_np(x):
    return x.detach().cpu().numpy()

class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}

        self._init_model(shapes, config)
        self._init_heads(shapes, config)
        self._init_optims(config)
        self.obs_step = config.obs_step

        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )
        self.always_frozen_layers = []
        self.freeze_encoder = config.freeze_encoder
        if self.freeze_encoder:
            cprint(
                "Freezing embeddings from encoder during training",
                color="red",
                attrs=["bold"],
            )
            self.always_frozen_layers = [
                name
                for name, _ in self.named_parameters()
                if "encoder." in name or "_obs" in name
            ]

    def _init_model(self, shapes, config):
        self.encoder = networks.MultiEncoder(
            shapes, augment_images=config.augment_images, **config.encoder
        )
        self.embed_size = self.encoder.outdim

        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )

    def _init_heads(self, shapes, config):
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
    
    def _init_lx_mlp(self, config, obs_shape):
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter

        lx_mlp = nn.Sequential(
            spectral_norm(nn.Linear(feat_size, 16)),
            nn.ReLU(),
            spectral_norm(nn.Linear(16, obs_shape)),
        )
        
        lx_mlp.to(config.device)
        standard_kwargs = {
            "lr": config.obs_lr,
            "eps": config.opt_eps,
            "clip": config.grad_clip,
            "wd": config.weight_decay,
            "opt": config.opt,
            "use_amp": self._use_amp,
        }
        lx_recon_opt = tools.Optimizer(
            "lx_mlp", lx_mlp.parameters(), **standard_kwargs
        )
        return lx_mlp, lx_recon_opt

    def _init_optims(self, config):
        standard_kwargs = {
            "lr": config.model_lr,
            "eps": config.opt_eps,
            "clip": config.grad_clip,
            "wd": config.weight_decay,
            "opt": config.opt,
            "use_amp": self._use_amp,
            "lr_decay": config.decay_model_lr,
        }
        self.decay_model_lr = config.decay_model_lr
        self._model_opt = tools.Optimizer("model", self.parameters(), **standard_kwargs)
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )


    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = obs.copy()
        # For all keys in obs which contain "image", normalize the values by 255.0
        for key, value in obs.items():
            if "image" in key:
                obs[key] = torch.Tensor(np.array(value)) / 255.0

        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(np.array(v)).to(self._config.device) for k, v in obs.items()}
        return obs

    def video_pred(self, data, ensemble=None):
        data = self.preprocess(data)
        embed = self.encoder(data)

        obs_steps = self.obs_step #obs_step: how many steps using actual observation? = 1.

        # Visualize 6 images
        states, _ = self.dynamics.observe(
            embed[:6, :obs_steps], data["action"][:6, :obs_steps], data["is_first"][:6, :obs_steps]
        )
        # Reconstruction based on the posterior (with image embeddings)
        recon = self.heads["decoder"](self.dynamics.get_feat(states))[
            "image"
        ].mode()[:6]

        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, obs_steps:], init)    

        # Openl only with priors.
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))[
            "image"
        ].mode()
 
        truth = data["image"][:6]

        row, col = torch.where(data['is_first'][:6, obs_steps:] == 1.)
        for i in range(row.size(0)):
            data['is_first'][row[i], obs_steps+col[i]:] = 1.
            openl[row[i], col[i]:] = openl[row[i], col[i]-1]
            truth[row[i], obs_steps+col[i]:] = truth[row[i], obs_steps+col[i]-1]

        # observed image is given until obs_steps
        model = torch.cat([recon[:, :obs_steps], openl], 1)
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)
    
    def video_pred_ensemble(self, data, ensemble=False, density=False, rssm_ensemble=False):
        data = self.preprocess(data)
        embed = self.encoder(data)

        obs_steps = self.obs_step #obs_step: how many steps using actual observation? = 1.

        # Visualize 6 images
        states, _ = self.dynamics.observe(
            embed[:6, :obs_steps], data["action"][:6, :obs_steps], data["is_first"][:6, :obs_steps]
        )
        # Reconstruction based on the posterior (with image embeddings)
        recon = self.heads["decoder"](self.dynamics.get_feat(states))[
            "image"
        ].mode()[:6]

        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, obs_steps:], init)    

        # Openl only with priors.
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))[
            "image"
        ].mode()
 
        truth = data["image"][:6]

        row, col = torch.where(data['is_first'][:6, obs_steps:] == 1.)
        for i in range(row.size(0)):
            data['is_first'][row[i], obs_steps+col[i]:] = 1.
            openl[row[i], col[i]:] = openl[row[i], col[i]-1]
            truth[row[i], obs_steps+col[i]:] = truth[row[i], obs_steps+col[i]-1]

        # observed image is given until obs_steps
        model = torch.cat([recon[:, :obs_steps], openl], 1)
        error = (model - truth + 1.0) / 2.0

        disagreement = {}
        if ensemble is not None:
            stoch = torch.cat([states['stoch'], prior['stoch']], 1)
            deter = torch.cat([states['deter'], prior['deter']], 1)

            feat = torch.cat([stoch, deter], -1)
            action = data["action"][:6]

            #penn
            disagreement_ensemble = ensemble._intrinsic_reward_penn(feat, action)
            disagreement["ensemble"] = disagreement_ensemble
        
        return torch.cat([truth, model, error], 2), disagreement
    
    def measure_disagreement(self, data, ensemble=None):
        data = self.preprocess(data)
        embed = self.encoder(data)

        # Only Imagination !!
        states_init, _ = self.dynamics.observe(
            embed[:, :1], data["action"][:, :1], data["is_first"][:, :1]
        )

        init_imagine = {k: v[:, -1] for k, v in states_init.items()}
        prior_imagine = self.dynamics.imagine_with_action(data["action"][:, 1:], init_imagine) 

        stoch_imagine = torch.cat([states_init['stoch'], prior_imagine['stoch']], 1)
        deter_imagine = torch.cat([states_init['deter'], prior_imagine['deter']], 1)

        feat_imagine = torch.cat([stoch_imagine, deter_imagine], -1)
        action = data["action"]
        disagreement_imagine = ensemble._intrinsic_reward_penn(feat_imagine, action)

        ###################################
        # Posterior (w. Neural Kalman Filter)
        states_post, _ = self.dynamics.observe(
            embed, data["action"], data["is_first"]
        )

        feat_post = torch.cat([states_post['stoch'], states_post['deter']], -1)
        action = data["action"]
        disagreement_post = ensemble._intrinsic_reward_penn(feat_post, action)

        
        return data['privileged_state'], disagreement_imagine, disagreement_post


