import torch
from torch import nn
from torch import distributions as torchd

import sys
sys.path.append('latent_safety')
import dreamerv3_torch.tools as tools
from dreamerv3_torch.ensemble.penn import EnsembleStochasticLinear, EnsembleStochasticLinearUnitVariance
from dreamerv3_torch.ensemble.made import MaskedAutoregressiveFlow

def clamp_preserve_gradients(x: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """
    Clamps the values of the tensor into ``[lower, upper]`` but keeps the gradients.

    Args:
        x: The tensor whose values to constrain.
        lower: The lower limit for the values.
        upper: The upper limit for the values.

    Returns:
        The clamped tensor.
    """
    return x + (x.clamp(min=lower, max=upper) - x).detach()

class OneStepPredictor(nn.Module):
    def __init__(self, config, world_model):
        super(OneStepPredictor, self).__init__()
        self._config = config
        self._use_amp = True if config.precision == 16 else False
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
            stoch = config.dyn_stoch * config.dyn_discrete
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
            stoch = config.dyn_stoch
        size = {
            "embed": world_model.embed_size,
            "stoch": stoch,
            "deter": config.dyn_deter,
            "feat": config.dyn_stoch + config.dyn_deter,
        }[self._config.disag_target]

        input_dim = feat_size + (config.num_actions if config.disag_action_cond else 0)

        self._networks = EnsembleStochasticLinear(in_features=input_dim, 
                                                 out_features=size,
                                                 hidden_features=input_dim,
                                                 ensemble_size=config.disag_models,
                                                 explore_var='jrd', 
                                                 residual=True)
        
        
        self.criterion = self.gaussian_nll_loss 
        
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._expl_opt = tools.Optimizer(
            "ensemble",
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
    
    def intrinsic_reward_penn(self, inputs):

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
    
    def train_ensemble_penn_fixed(self, feats, actions, targets, is_first):
        self._networks.train()
        with torch.cuda.amp.autocast(self._use_amp):

            feats = feats[:, :-1] # N, T-1
            actions = actions[:, 1:] # N, T-1
            inputs = torch.concat([feats, actions], -1)
            targets = targets[:, 1:] # N, T-1

            valid_idx = torch.roll(is_first, shifts=-1, dims=1)[:, :-1] == 0.

            valid_inputs = inputs[valid_idx]
            valid_targets = targets[valid_idx]

            valid_inputs = valid_inputs.detach()
            valid_targets = valid_targets.detach()
            
            train_loss = torch.FloatTensor([0]).cuda()
            
            for i in range(self.config.disag_models):                
                (mu, log_std) = self._networks.single_forward(
                    valid_inputs, index=i)

                yhat_mu = mu.unsqueeze(0)
                var = torch.square(torch.exp(log_std.unsqueeze(0)))
                loss = self.gaussian_nll_loss(yhat_mu, valid_targets, var)
                loss = loss.mean()
                self._expl_opt(loss, self._networks.parameters())
                
                train_loss += loss

        metrics = {"ensemble_loss": train_loss.item() / self.config.disag_models}

        with torch.no_grad():
            div = self.intrinsic_reward_penn(valid_inputs).mean()
        metrics["log_disagreement"] = div.cpu().numpy()

        return metrics
    
class DensityEstimator_MAF(nn.Module):
    def __init__(self, config):
        super(DensityEstimator_MAF, self).__init__()
        self.config = config
        input_dim = config.dyn_deter
        self._use_amp = True if config.precision == 16 else False
        self._networks = MaskedAutoregressiveFlow(dim=input_dim, num_layers=4, hidden_layer_size=input_dim)
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._expl_opt = tools.Optimizer(
            "nf_density",
            self._networks.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            **kw,
        )
        self.norm_max = 30

    def train_density_estimator(self, x):
        x = x.detach()
        N, T, D = x.shape
        x = x.reshape(-1, D)

        log_prob = self._networks.forward(x)
        # log_prob = clamp_preserve_gradients(log_prob, lower=-30, upper=30)
        loss = -torch.mean(log_prob)

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            self._expl_opt(loss, self._networks.parameters())

        with torch.no_grad():
            prob = torch.exp(log_prob).view(N, T)
            prob[torch.isinf(prob)] = self.norm_max
            prob[torch.isnan(prob)] = 0
            density = prob

        metrics = {"density_loss": loss.item(), "density": density.mean().item()}

        return metrics
    
    def calculate_likelihood(self, x):

        self._networks.eval()
        N, T, D = x.shape
        x = x.reshape(-1, D)
        log_prob = self._networks.forward(x)
        prob = torch.exp(log_prob).view(N, T)
        prob[torch.isnan(prob)] = 0
        prob = torch.clamp(prob, min=0, max=self.norm_max)
        self._networks.train()

        return prob


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
    
    def intrinsic_reward_penn(self, inputs):

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
    
    def train_ensemble_penn_fixed(self, feats, actions, targets, is_first):
        self._networks.train()
        with torch.cuda.amp.autocast(self._use_amp):
            feats = feats[:, :-1] # N, T-1
            actions = actions[:, 1:] # N, T-1
            inputs = torch.concat([feats, actions], -1)
            targets = targets[:, 1:] # N, T-1

            valid_idx = torch.roll(is_first, shifts=-1, dims=1)[:, :-1] == 0.

            valid_inputs = inputs[valid_idx]
            valid_targets = targets[valid_idx]

            valid_inputs = valid_inputs.detach()
            valid_targets = valid_targets.detach()
            
            train_loss = torch.FloatTensor([0]).cuda()

            for i in range(self.config.disag_models):                
                mu = self._networks.single_forward(
                    valid_inputs, index=i)
                
                yhat_mu = mu.unsqueeze(0)
                loss = (yhat_mu - valid_targets).pow(2)
                loss = loss.mean()
                self._expl_opt(loss, self._networks.parameters())
                train_loss += loss
            
        metrics = {"explorer_loss": train_loss.item() / self.config.disag_models}

        with torch.no_grad():
            div = self.intrinsic_reward_penn(valid_inputs).mean()
        metrics["log_disagreement"] = div.cpu().numpy()

        return metrics
    



import normflows as nf
class DensityEstimator(nn.Module): 

    def __init__(self, config):
        super(DensityEstimator, self).__init__()
        self._config = config
        self._use_amp = True if config.precision == 16 else False

        feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        input_dim = feat_size     
        self.config = config

        # Set base distribuiton
        self.q0 = nf.distributions.DiagGaussian(input_dim, trainable=True)
        # self.q0 = nf.distributions.DiagGaussian(input_dim, trainable=False)
        flows = [
            nf.flows.CoupledRationalQuadraticSpline(num_input_channels=input_dim, num_blocks=2, num_hidden_channels=input_dim*2),
            nf.flows.LULinearPermute(input_dim),
            nf.flows.CoupledRationalQuadraticSpline(num_input_channels=input_dim, num_blocks=2, num_hidden_channels=input_dim*2),
            nf.flows.LULinearPermute(input_dim),
            nf.flows.CoupledRationalQuadraticSpline(num_input_channels=input_dim, num_blocks=2, num_hidden_channels=input_dim*2),
            nf.flows.LULinearPermute(input_dim),
            nf.flows.CoupledRationalQuadraticSpline(num_input_channels=input_dim, num_blocks=2, num_hidden_channels=input_dim*2),
            nf.flows.LULinearPermute(input_dim)
            ]

        self._networks = nf.NormalizingFlow(q0=self.q0, flows=flows)
        self._networks = self._networks.cuda()

        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._expl_opt = tools.Optimizer(
            "nf_density",
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
        # import pdb; pdb.set_trace()
        loss = self._networks.forward_kld(x)
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            self._expl_opt(loss, self._networks.parameters())
        torch.use_deterministic_algorithms(True)
        metrics = {"density_loss": loss.item()}

        return metrics
    
    def calculate_likelihood(self, x):

        self._networks.eval()

        if len(x.shape) == 3:
            N, T, D = x.shape
            x = x.view(-1, D)
            torch.use_deterministic_algorithms(False)
            log_prob = self._networks.log_prob(x)
            torch.use_deterministic_algorithms(True)
            prob = torch.exp(log_prob).view(N, T)
            prob = torch.clamp(prob, min=0, max=1)
            self._networks.train()
        
        else:
            torch.use_deterministic_algorithms(False)
            log_prob = self._networks.log_prob(x)
            torch.use_deterministic_algorithms(True)
            prob = torch.exp(log_prob)
            prob = torch.clamp(prob, min=0, max=1)
            self._networks.train()


        return prob