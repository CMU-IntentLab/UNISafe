from typing import List, Tuple, TypeVar
import torch
import torch.nn.functional as F
from torch import nn
from abc import ABC, abstractmethod
from typing import Optional
import math

class Transform(nn.Module, ABC):
    """
    Base class for all normalizing flow transforms
    """

    @abstractmethod
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transforms the given input.

        Args:
            z: A tensor of shape ``[*, dim]`` where ``dim`` is the dimensionality of this module.

        Returns:
            The transformed inputs, a tensor of shape ``[*, dim]`` and the log-determinants of the
            Jacobian evaluated at the inputs, a tensor of shape ``[*]``.
        """


class BatchNormTransform(Transform):
    r"""
    Batch Normalization layer for stabilizing deep normalizing flows. It was first introduced in
    `Density Estimation Using Real NVP <https://arxiv.org/pdf/1605.08803.pdf>`_ (Dinh et al.,
    2017).
    """

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, dim: int, momentum: float = 0.5, eps: float = 1e-5):
        """
        Args:
            dim: The dimension of the inputs.
            momentum: Value used for calculating running average statistics.
            eps: A small value added in the denominator for numerical stability.
        """
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.empty(dim))
        self.beta = nn.Parameter(torch.empty(dim))

        self.register_buffer("running_mean", torch.empty(dim))
        self.register_buffer("running_var", torch.empty(dim))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets this module's parameters.
        """
        nn.init.zeros_(self.log_gamma)  # equal to `init.ones_(self.gamma)`
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.running_mean)
        nn.init.ones_(self.running_var)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            reduce = list(range(z.dim() - 1))
            mean = z.detach().mean(reduce)
            var = z.detach().var(reduce, unbiased=True)

            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(mean * (1 - self.momentum))
                self.running_var.mul_(self.momentum).add_(var * (1 - self.momentum))
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize input
        x = (z - mean) / (var + self.eps).sqrt()
        out = x * self.log_gamma.exp() + self.beta

        # Compute log-determinant
        log_det = self.log_gamma - 0.5 * (var + self.eps).log()
        # Do repeat instead of expand to allow fixing the log_det below
        log_det = log_det.sum(-1).repeat(z.size()[:-1])

        # Fix numerical issues during evaluation
        if not self.training:
            # Find all output rows where at least one value is not finite
            rows = (~torch.isfinite(out)).sum(-1) > 0
            # Fill these rows with 0 and set the log-determinant to -inf to indicate that they have
            # a density of exactly 0
            out[rows] = 0
            log_det[rows] = float("-inf")

        return out, log_det


T = TypeVar("T", bound=Transform, covariant=True)


class NormalizingFlow(nn.Module):
    """
    pass
    """

    def __init__(self, transforms: List[T]):
        """
        Args:
            transforms: The transforms to use in the normalizing flow.
        """
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the log-probability of observing the given input, transformed by the flow's
        transforms under the standard Normal distribution.

        Args:
            z: A tensor of shape ``[*, dim]`` with the inputs.

        Returns:
            A tensor of shape ``[*]`` including the log-probabilities.
        """
        batch_size = z.size()[:-1]
        dim = z.size(-1)

        log_det_sum = z.new_zeros(batch_size)
        for transform in self.transforms:
            z, log_det = transform.forward(z)
            log_det_sum += log_det

        # Compute log-probability
        const = dim * math.log(2 * math.pi)
        norm = torch.einsum("...ij,...ij->...i", z, z)
        normal_log_prob = -0.5 * (const + norm)
        return normal_log_prob + log_det_sum
    
class MaskedAutoregressiveFlow(NormalizingFlow):
    """
    Normalizing flow that consists of masked autoregressive transforms with optional batch
    normalizing layers in between.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        num_hidden_layers: int = 1,
        hidden_layer_size: Optional[int] = None,
        use_batch_norm: bool = True,
    ):
        """
        Args:
            dim: The input dimension of the normalizing flow.
            num_layers: The number of sequential masked autoregressive transforms.
            num_hidden_layers: The number of hidden layers for each autoregressive transform.
            hidden_layer_size_multiplier: The dimension of each hidden layer. Defaults to
                ``3 * dim + 1``.
            use_batch_norm: Whether to insert batch normalizing transforms between transforms.
        """
        transforms = []
        for i in range(num_layers):
            if i > 0 and use_batch_norm:
                transforms.append(BatchNormTransform(dim))
            transform = MaskedAutoregressiveTransform(
                dim,
                [hidden_layer_size or (dim * 3 + 1)] * num_hidden_layers,
            )
            transforms.append(transform)
        super().__init__(transforms)

class MaskedAutoregressiveTransform(Transform):
    r"""
    Masked Autogressive Transform as introduced in `Masked Autoregressive Flow for Density
    Estimation <https://arxiv.org/abs/1705.07057>`_ (Papamakarios et al., 2018).
    """

    def __init__(
        self,
        dim: int,
        hidden_dims: List[int],
    ):
        """
        Args:
            dim: The dimension of the inputs.
            hidden_dims: The hidden dimensions of the MADE model.
        """
        super().__init__()
        self.net = MADE(dim, hidden_dims)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, logscale = self.net(z).chunk(2, dim=-1)
        logscale = logscale.tanh()
        out = (z - mean) * torch.exp(-logscale)
        log_det = -logscale.sum(-1)
        return out, log_det
    
class MADE(nn.Sequential):
    """
    Masked autoencoder for distribution estimation (MADE) as introduced in
    `MADE: Masked Autoencoder for Distribution Estimation <https://arxiv.org/abs/1502.03509>`_
    (Germain et al., 2015). In consists of a series of masked linear layers and a given
    non-linearity between them.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        """
        Initializes a new MADE model as a sequence of masked linear layers.

        Args:
            input_dim: The number of input dimensions.
            hidden_dims: The dimensions of the hidden layers.
        """
        assert len(hidden_dims) > 0, "MADE model must have at least one hidden layer."

        dims = [input_dim] + hidden_dims + [input_dim * 2]
        hidden_masks = _create_masks(input_dim, hidden_dims)

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
            if i > 0:
                layers.append(nn.LeakyReLU())
            layers.append(_MaskedLinear(in_dim, out_dim, mask=hidden_masks[i]))
        super().__init__(*layers)

class _MaskedLinear(nn.Linear):
    mask: torch.Tensor

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
        super().__init__(in_features, out_features)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-renamed
        return F.linear(x, self.weight * self.mask, self.bias)

    def __repr__(self):
        return f"MaskedLinear(in_features={self.in_features}, out_features={self.out_features})"


def _create_masks(input_dim: int, hidden_dims: List[int]) -> List[torch.Tensor]:
    permutation = torch.randperm(input_dim)

    input_degrees = permutation + 1
    hidden_degrees = [_sample_degrees(1, input_dim - 1, d) for d in hidden_dims]
    output_degrees = permutation.repeat(2)

    all_degrees = [input_degrees] + hidden_degrees + [output_degrees]
    hidden_masks = [
        _create_single_mask(in_deg, out_deg)
        for in_deg, out_deg in zip(all_degrees, all_degrees[1:])
    ]

    return hidden_masks


def _create_single_mask(in_degrees: torch.Tensor, out_degrees: torch.Tensor) -> torch.Tensor:
    return (out_degrees.unsqueeze(-1) >= in_degrees).float()


def _sample_degrees(minimum: int, maximum: int, num: int) -> torch.Tensor:
    return torch.linspace(minimum, maximum, steps=num).round()



# flow = MaskedAutoregressiveFlow(self.latent_dim, num_layers=self.flow_num_layers)
#  z = self.encoder.forward(x)
# if z.dim() > 2:
#     z = z.permute(0, 2, 3, 1)
# log_prob = self.flow.forward(z)


class RadialFlow(NormalizingFlow):
    """
    Normalizing flow that consists purely of a series of radial transforms.
    """

    def __init__(self, dim: int, num_layers: int = 8):
        """
        Args:
            dim: The input dimension of the normalizing flow.
            num_layers: The number of sequential radial transforms.
        """
        transforms = [RadialTransform(dim) for _ in range(num_layers)]
        super().__init__(transforms)


class RadialTransform(Transform):
    r"""
    A radial transformation may be used to apply radial contractions and expansions around a
    reference point. It was introduced in "Variational Inference with Normalizing Flows" (Rezende
    and Mohamed, 2015).
    """

    def __init__(self, dim: int):
        r"""
        Args:
            dim: The dimension of the transform.
        """
        super().__init__()

        self.reference = nn.Parameter(torch.empty(dim))
        self.alpha_prime = nn.Parameter(torch.empty(1))
        self.beta_prime = nn.Parameter(torch.empty(1))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets this module's parameters. All parameters are sampled uniformly depending on this
        module's dimension.
        """
        std = 1 / math.sqrt(self.reference.size(0))
        nn.init.uniform_(self.reference, -std, std)
        nn.init.uniform_(self.alpha_prime, -std, std)
        nn.init.uniform_(self.beta_prime, -std, std)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dim = self.reference.size(0)
        alpha = F.softplus(self.alpha_prime)  # [1]
        beta = -alpha + F.softplus(self.beta_prime)  # [1]

        # Compute output
        diff = z - self.reference  # [*, D]
        r = diff.norm(dim=-1, keepdim=True)  # [*, 1]
        h = (alpha + r).reciprocal()  # [*]
        beta_h = beta * h  # [*]
        y = z + beta_h * diff  # [*, D]

        # Compute log-determinant of Jacobian
        h_d = -(h ** 2)  # [*]
        log_det_lhs = (dim - 1) * beta_h.log1p()  # [*]
        log_det_rhs = (beta_h + beta * h_d * r).log1p()  # [*, 1]
        log_det = (log_det_lhs + log_det_rhs).squeeze(-1)  # [*]

        return y, log_det