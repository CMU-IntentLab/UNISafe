import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import dreamer.models as models
import dreamer.networks as networks
import dreamer.tools as tools


class Random(nn.Module):
    def __init__(self, config, act_space):
        super(Random, self).__init__()
        self._config = config
        self._act_space = act_space

    def actor(self, feat):
        if self._config.actor["dist"] == "onehot":
            return tools.OneHotDist(
                torch.zeros(self._config.num_actions)
                .repeat(self._config.envs, 1)
                .to(self._config.device)
            )
        else:
            return torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(self._act_space.low)
                    .repeat(self._config.envs, 1)
                    .to(self._config.device),
                    torch.Tensor(self._act_space.high)
                    .repeat(self._config.envs, 1)
                    .to(self._config.device),
                ),
                1,
            )

    def train(self, start, context, data):
        return None, {}


from dreamer.penn import EnsembleStochasticLinear, EnsembleStochasticLinearUnitVariance

class OneStepPredictor(nn.Module):
    def __init__(self, config, world_model):
        super(OneStepPredictor, self).__init__()
        self._config = config
        self._use_amp = True if config.precision == 16 else False
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
            stoch = config.dyn_stoch * config.dyn_discrete
            dist = "onehot"
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
            stoch = config.dyn_stoch
            dist = "symlog_mse" #"normal_std_fixed"
        size = {
            "embed": world_model.embed_size,
            "stoch": stoch,
            "deter": config.dyn_deter,
            "feat": config.dyn_stoch + config.dyn_deter,
        }[self._config.disag_target]
        kw = dict(
            inp_dim=feat_size
            + (
                config.num_actions if config.disag_action_cond else 0
            ),  # pytorch version
            dist = dist, # Normal.
            shape=size,
            layers=config.disag_layers,
            units=config.disag_units,
            act=config.act,
        )
        # self._networks = nn.ModuleList(
        #     [networks.MLP(**kw) for _ in range(config.disag_models)]
        # )

        input_dim = feat_size + (config.num_actions if config.disag_action_cond else 0)

        self._networks = EnsembleStochasticLinear(in_features=input_dim, 
                                                 out_features=size,
                                                 hidden_features=input_dim, #hidden_features=config.disag_units, #
                                                 ensemble_size=config.disag_models,
                                                 explore_var='jrd', 
                                                 residual=True)
        
        # self._networks = nn.DataParallel(self._networks)
        torch.backends.cudnn.benchmark = True
        self.criterion = self.gaussian_nll_loss 
        
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._expl_opt = tools.Optimizer(
            "explorer",
            self._networks.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            **kw,
        )
        self.config = config

    def gaussian_nll_loss(self, mu, target, var):
        # Custom Gaussian Negative Log Likelihood Loss
        loss = 0.5 * (torch.log(var) + (target - mu) ** 2 / var)
        return torch.mean(loss)
    
    def _intrinsic_reward(self, feat, action):
        inputs = feat
        if self._config.disag_action_cond:
            inputs = torch.concat([inputs, action], -1)
        preds = torch.cat(
            [head(inputs, torch.float32).mode()[None] for head in self._networks], 0
        )
        disag = torch.mean(torch.std(preds, 0), -1)[..., None]
        if self._config.disag_log:
            disag = torch.log(disag)
        reward = self._config.expl_intr_scale * disag

        return reward
    
    def _intrinsic_reward_penn(self, feat, action):
        inputs = feat
        if self._config.disag_action_cond:
            inputs = torch.concat([inputs, action], -1)

        self._networks.eval()

        if len(inputs.shape) == 3:
            N, T, D = inputs.shape
            inputs = inputs.reshape(N * T, D)

            with torch.no_grad():
                ensemble_outputs = self._networks(inputs)
                div = ensemble_outputs[-1]
            
            div = div.view(N, T, -1)

        else:
            with torch.no_grad():
                ensemble_outputs = self._networks(inputs)
                div = ensemble_outputs[-1]

        if self._config.disag_log:
            div = torch.log(div)

        return div
    

    def _intrinsic_reward_categorical(self, feat, action):
        """
        Compute intrinsic reward based on disagreement (epistemic uncertainty)
        for categorical logits output by an ensemble of networks.

        Args:
            feat (torch.Tensor): Features of the state.
            action (torch.Tensor): Actions taken.
        
        Returns:
            torch.Tensor: Intrinsic reward based on disagreement.
        """
        # Combine features and actions if action-conditional
        inputs = feat
        if self._config.disag_action_cond:
            inputs = torch.cat([inputs, action], dim=-1)

        # Collect predictions from each network in the ensemble and process logits
        logits = torch.stack(
                    [head(inputs, torch.float32).logits for head in self._networks],
                    dim=0 
                )

        N, B, S, KC = logits.shape
        logits = logits.view(N, B, S, self.config.dyn_discrete, -1) 

        # Convert logits to log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)

        # Compute probabilities from log probabilities
        probs = torch.exp(log_probs)  # Shape: (N, B, S, K, C)

        mean_probs = probs.mean(dim=0)  # Shape: (B, S, K, C)

        # Compute entropy of the aggregated distribution
        entropy_aggregated = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)  # Shape: (B, S, K)

        # Compute mean entropy of individual distributions
        individual_entropies = -torch.sum(probs * log_probs, dim=-1)  # Shape: (N, B, S, K)
        mean_individual_entropy = individual_entropies.mean(dim=0)  # Shape: (B, S, K)

        # Compute disagreement (mutual information)
        disag = entropy_aggregated - mean_individual_entropy  # Shape: (B, S, K)

        # Scale disagreement to produce intrinsic reward
        # reward = self._config.expl_intr_scale * disag[..., None]  # Add dimension for consistency
        reward = disag.mean()
        return reward


    def _train_ensemble(self, inputs, targets):
        with torch.cuda.amp.autocast(self._use_amp):
            if self._config.disag_offset:
                targets = targets[:, self._config.disag_offset :]
                inputs = inputs[:, : -self._config.disag_offset]

            targets = targets.detach()
            inputs = inputs.detach()
            
            if self.config.dyn_discrete:
                kld = torchd.kl.kl_divergence
                logits = torch.stack(
                    [head(inputs, torch.float32).logits for head in self._networks],
                    dim=0 
                )
                B, S, K, C = targets.shape
                logits = logits.reshape(-1, B, S, K, C)

                targets = targets.unsqueeze(0).expand(logits.shape)

                dist_l = torchd.independent.Independent(
                    tools.OneHotDist(logits), 1 # Event: Last dimension.
                )
                dist_t = torchd.independent.Independent(
                    tools.OneHotDist(targets), 1
                )
                
                likes = kld(dist_l, dist_t)

                # Compute the loss
                loss = -likes.mean()  # Mean over (N)
            
            else:
                preds = [head(inputs) for head in self._networks]
                likes = torch.cat(
                    [torch.mean(pred.log_prob(targets))[None] for pred in preds], 0
                )

                loss = -torch.mean(likes)

        metrics = self._expl_opt(loss, self._networks.parameters())
        metrics['target_norm'] = targets.norm().cpu().numpy()

        return metrics
    
    def _train_ensemble_penn(self, inputs, targets, is_first=None):
        self._networks.train()
        with torch.cuda.amp.autocast(self._use_amp):
            if self._config.disag_offset:
                targets = targets[:, self._config.disag_offset :]
                inputs = inputs[:, : -self._config.disag_offset]

            # targets = targets.detach()
            # inputs = inputs.detach()

            valid_idx = torch.roll(is_first, shifts=-1, dims=1)[:, :-1] == 0.
            valid_inputs = inputs[valid_idx]
            valid_targets = targets[valid_idx]

            targets = valid_targets.detach()
            inputs = valid_inputs.detach()
            
            train_loss = torch.FloatTensor([0]).cuda()
            # N, T, D = inputs.shape
            # inputs = inputs.reshape(N * T, D)

            for i in range(self.config.disag_models):                
                (mu, log_std) = self._networks.single_forward(
                    inputs, index=i)
                
                # mu = mu.view(N, T, -1)
                # log_std = log_std.reshape(N, T, -1)

                # yhat_mu = mu
                # var = torch.square(torch.exp(log_std))
                yhat_mu = mu.unsqueeze(0)
                var = torch.square(torch.exp(log_std.unsqueeze(0)))
                loss = self.gaussian_nll_loss(yhat_mu, targets, var)
                loss = loss.mean()
                self._expl_opt(loss, self._networks.parameters())
                
                train_loss += loss

        metrics = {"explorer_loss": train_loss.item() / self.config.disag_models}

        return metrics
    


class OneStepPredictorUnitVariance(nn.Module):
    def __init__(self, config, world_model):
        super(OneStepPredictorUnitVariance, self).__init__()
        self._config = config
        self._use_amp = True if config.precision == 16 else False
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
            stoch = config.dyn_stoch * config.dyn_discrete
            dist = "onehot"
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
            stoch = config.dyn_stoch
            dist = "symlog_mse" #"normal_std_fixed"
        size = {
            "embed": world_model.embed_size,
            "stoch": stoch,
            "deter": config.dyn_deter,
            "feat": config.dyn_stoch + config.dyn_deter,
        }[self._config.disag_target]
        kw = dict(
            inp_dim=feat_size
            + (
                config.num_actions if config.disag_action_cond else 0
            ),  # pytorch version
            dist = dist, # Normal.
            shape=size,
            layers=config.disag_layers,
            units=config.disag_units,
            act=config.act,
        )

        input_dim = feat_size + (config.num_actions if config.disag_action_cond else 0)

        self._networks = EnsembleStochasticLinearUnitVariance(in_features=input_dim, 
                                                 out_features=size,
                                                 hidden_features=input_dim, #hidden_features=config.disag_units, #
                                                 ensemble_size=config.disag_models,
                                                 explore_var='jrd', 
                                                 residual=True)
        
        torch.backends.cudnn.benchmark = True
        
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._expl_opt = tools.Optimizer(
            "explorer",
            self._networks.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            **kw,
        )
        self.config = config
    
    def _intrinsic_reward_penn(self, feat, action):
        inputs = feat
        if self._config.disag_action_cond:
            inputs = torch.concat([inputs, action], -1)

        self._networks.eval()

        if len(inputs.shape) == 3:
            N, T, D = inputs.shape
            inputs = inputs.reshape(N * T, D)

            with torch.no_grad():
                ensemble_outputs = self._networks(inputs)
                div = ensemble_outputs[-1]
            
            div = div.view(N, T, -1)
        
        else:
            with torch.no_grad():
                ensemble_outputs = self._networks(inputs)
                div = ensemble_outputs[-1]

        # if self._config.disag_log:
        #     div = torch.log(div)

        return div
    
    def _train_ensemble_penn(self, inputs, targets, is_first=None):
        self._networks.train()
        with torch.cuda.amp.autocast(self._use_amp):
            if self._config.disag_offset:
                targets = targets[:, self._config.disag_offset :]
                inputs = inputs[:, : -self._config.disag_offset]


            valid_idx = torch.roll(is_first, shifts=-1, dims=1)[:, :-1] == 0.
            valid_inputs = inputs[valid_idx]
            valid_targets = targets[valid_idx]

            targets = valid_targets.detach()
            inputs = valid_inputs.detach()
            
            train_loss = torch.FloatTensor([0]).cuda()
            # N, T, D = inputs.shape
            # inputs = inputs.reshape(N * T, D)

            for i in range(self.config.disag_models):                
                mu = self._networks.single_forward(
                    inputs, index=i)
                
                yhat_mu = mu.unsqueeze(0)
                loss = (yhat_mu - targets).pow(2)
                loss = loss.mean()
                self._expl_opt(loss, self._networks.parameters())
                train_loss += loss

        metrics = {"explorer_loss": train_loss.item() / self.config.disag_models}

        return metrics
    


import normflows as nf
class DensityEstimator(nn.Module):
    def __init__(self, config, world_model):
        super(DensityEstimator, self).__init__()
        self._config = config
        self._use_amp = True if config.precision == 16 else False
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
            stoch = config.dyn_stoch * config.dyn_discrete
            dist = "onehot"
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
            stoch = config.dyn_stoch
            dist = "symlog_mse" #"normal_std_fixed"
        size = {
            "embed": world_model.embed_size,
            "stoch": stoch,
            "deter": config.dyn_deter,
            "feat": config.dyn_stoch + config.dyn_deter,
        }[self._config.disag_target]
        kw = dict(
            inp_dim=feat_size
            + (
                config.num_actions if config.disag_action_cond else 0
            ),  # pytorch version
            dist = dist, # Normal.
            shape=size,
            layers=config.disag_layers,
            units=config.disag_units,
            act=config.act,
        )

        input_dim = feat_size + config.num_actions
        
        self.config = config
        # Set base distribuiton
        self.q0 = nf.distributions.DiagGaussian(input_dim, trainable=True)
        flows = [
            nf.flows.CoupledRationalQuadraticSpline(num_input_channels=input_dim, num_blocks=2, num_hidden_channels=128),
            nf.flows.LULinearPermute(input_dim),
            nf.flows.CoupledRationalQuadraticSpline(num_input_channels=input_dim, num_blocks=2, num_hidden_channels=128),
            nf.flows.LULinearPermute(input_dim),
            nf.flows.CoupledRationalQuadraticSpline(num_input_channels=input_dim, num_blocks=2, num_hidden_channels=128),
            nf.flows.LULinearPermute(input_dim)
            ]

        # Construct flow model
        self._networks = nf.NormalizingFlow(q0=self.q0, flows=flows)        
        self._networks = self._networks.cuda()

        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._expl_opt = tools.Optimizer(
            "explorer",
            self._networks.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            **kw,
        )


    def train_density_estimator(self, x):
        x = x.detach()
        N, T, D = x.shape
        x = x.view(-1, D)

        torch.use_deterministic_algorithms(False)
        loss = self._networks.forward_kld(x)
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            self._expl_opt(loss, self._networks.parameters())
        torch.use_deterministic_algorithms(True)
        metrics = {"density_loss": loss.item()}

        return metrics
    
    def calculate_likelihood(self, x):

        self._networks.eval()
        N, T, D = x.shape
        x = x.view(-1, D)
        # torch.use_deterministic_algorithms(False)
        log_prob = self._networks.log_prob(x)
        # torch.use_deterministic_algorithms(True)
        prob = torch.exp(log_prob).view(N, T)
        prob[torch.isnan(prob)] = 0
        prob = torch.clamp(prob, min=0, max=1)
        self._networks.train()

        return prob
