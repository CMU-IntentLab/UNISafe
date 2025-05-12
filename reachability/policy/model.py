"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

This module implements a Sin activation function and neural network model as
torch.nn.Module. Also, it implements a scheduler for hyper-parameters.
"""

import abc
import torch
import torch.nn as nn
import numpy as np

import warnings

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    no_type_check,
)

ModuleType = Type[nn.Module]
ArgsType = Union[Tuple[Any, ...], Dict[Any, Any], Sequence[Tuple[Any, ...]],
                 Sequence[Dict[Any, Any]]]

SIGMA_MIN = -20
SIGMA_MAX = 2


def miniblock(
    input_size: int,
    output_size: int = 0,
    norm_layer: Optional[ModuleType] = None,
    norm_args: Optional[Union[Tuple[Any, ...], Dict[Any, Any]]] = None,
    activation: Optional[ModuleType] = None,
    act_args: Optional[Union[Tuple[Any, ...], Dict[Any, Any]]] = None,
    linear_layer: Type[nn.Linear] = nn.Linear,
) -> List[nn.Module]:
    """Construct a miniblock with given input/output-size, norm layer and \
    activation."""
    layers: List[nn.Module] = [linear_layer(input_size, output_size)]
    if norm_layer is not None:
        if isinstance(norm_args, tuple):
            layers += [norm_layer(output_size, *norm_args)]
        elif isinstance(norm_args, dict):
            layers += [norm_layer(output_size, **norm_args)]
        else:
            layers += [norm_layer(output_size)]
    if activation is not None:
        if isinstance(act_args, tuple):
            layers += [activation(*act_args)]
        elif isinstance(act_args, dict):
            layers += [activation(**act_args)]
        else:
            layers += [activation()]
    return layers


class MLP(nn.Module):
    """Simple MLP backbone.

    Create a MLP of size input_dim * hidden_sizes[0] * hidden_sizes[1] * ...
    * hidden_sizes[-1] * output_dim

    :param int input_dim: dimension of the input vector.
    :param int output_dim: dimension of the output vector. If set to 0, there
        is no final linear layer.
    :param hidden_sizes: shape of MLP passed in as a list, not including
        input_dim and output_dim.
    :param norm_layer: use which normalization before activation, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
        You can also pass a list of normalization modules with the same length
        of hidden_sizes, to use different normalization module in different
        layers. Default to no normalization.
    :param activation: which activation to use after each layer, can be both
        the same activation for all layers if passed in nn.Module, or different
        activation for different Modules if passed in a list. Default to
        nn.ReLU.
    :param device: which device to create this model on. Default to None.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    :param bool flatten_input: whether to flatten input data. Default to True.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        norm_args: Optional[ArgsType] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        act_args: Optional[ArgsType] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        flatten_input: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        if norm_layer:
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(hidden_sizes)
                norm_layer_list = norm_layer
                if isinstance(norm_args, list):
                    assert len(norm_args) == len(hidden_sizes)
                    norm_args_list = norm_args
                else:
                    norm_args_list = [norm_args for _ in range(len(hidden_sizes))]
            else:
                norm_layer_list = [norm_layer for _ in range(len(hidden_sizes))]
                norm_args_list = [norm_args for _ in range(len(hidden_sizes))]
        else:
            norm_layer_list = [None] * len(hidden_sizes)
            norm_args_list = [None] * len(hidden_sizes)
        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(hidden_sizes)
                activation_list = activation
                if isinstance(act_args, list):
                    assert len(act_args) == len(hidden_sizes)
                    act_args_list = act_args
                else:
                    act_args_list = [act_args for _ in range(len(hidden_sizes))]
            else:
                activation_list = [activation for _ in range(len(hidden_sizes))]
                act_args_list = [act_args for _ in range(len(hidden_sizes))]
        else:
            activation_list = [None] * len(hidden_sizes)
            act_args_list = [None] * len(hidden_sizes)
        hidden_sizes = [input_dim] + list(hidden_sizes)
        model = []
        for in_dim, out_dim, norm, norm_args, activ, act_args in zip(
            hidden_sizes[:-1], hidden_sizes[1:], norm_layer_list, norm_args_list,
            activation_list, act_args_list
        ):
            model += miniblock(
                in_dim, out_dim, norm, norm_args, activ, act_args, linear_layer
            )
        if output_dim > 0:
            model += [linear_layer(hidden_sizes[-1], output_dim)]
        self.output_dim = output_dim or hidden_sizes[-1]
        self.model = nn.Sequential(*model)
        self.flatten_input = flatten_input

    @no_type_check
    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if self.device is not None:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if self.flatten_input:
            obs = obs.flatten(1)
        return self.model(obs)

class Actor(nn.Module):
    """Simple actor network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> action_shape.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param float max_action: the scale for the final action logits. Default to
        1.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        max_action: float = 1.0,
        device: Union[str, int, torch.device] = "cpu",
        preprocess_net_output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = int(np.prod(action_shape))
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(
            input_dim,  # type: ignore
            self.output_dim,
            hidden_sizes,
            device=self.device
        )
        self.max_action = max_action

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: obs -> logits -> action."""
        logits, hidden = self.preprocess(obs, state)
        logits = self.max_action * torch.tanh(self.last(logits))
        return logits, hidden


class Critic(nn.Module):
    """Simple critic network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    :param bool flatten_input: whether to flatten input data for the last layer.
        Default to True.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_sizes: Sequence[int] = (),
        device: Union[str, int, torch.device] = "cpu",
        preprocess_net_output_dim: Optional[int] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        flatten_input: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = 1
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(
            input_dim,  # type: ignore
            1,
            hidden_sizes,
            device=self.device,
            linear_layer=linear_layer,
            flatten_input=flatten_input,
        )

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        ).flatten(1)
        if act is not None:
            act = torch.as_tensor(
                act,
                device=self.device,
                dtype=torch.float32,
            ).flatten(1)
            obs = torch.cat([obs, act], dim=1)
        logits, hidden = self.preprocess(obs)
        logits = self.last(logits)
        return logits

class ActorProb(nn.Module):
    """Simple actor network (output with a Gauss distribution).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param float max_action: the scale for the final action logits. Default to
        1.
    :param bool unbounded: whether to apply tanh activation on final logits.
        Default to False.
    :param bool conditioned_sigma: True when sigma is calculated from the
        input, False when sigma is an independent parameter. Default to False.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        max_action: float = 1.0,
        device: Union[str, int, torch.device] = "cpu",
        unbounded: bool = False,
        conditioned_sigma: bool = False,
        preprocess_net_output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if unbounded and not np.isclose(max_action, 1.0):
            warnings.warn(
                "Note that max_action input will be discarded when unbounded is True."
            )
            max_action = 1.0
        self.preprocess = preprocess_net
        self.device = device
        self.output_dim = int(np.prod(action_shape))
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.mu = MLP(
            input_dim,  # type: ignore
            self.output_dim,
            hidden_sizes,
            device=self.device
        )
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = MLP(
                input_dim,  # type: ignore
                self.output_dim,
                hidden_sizes,
                device=self.device
            )
        else:
            self.sigma_param = nn.Parameter(torch.zeros(self.output_dim, 1))
        self.max_action = max_action
        self._unbounded = unbounded

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: obs -> logits -> (mu, sigma)."""
        logits, hidden = self.preprocess(obs, state)
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self.max_action * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), state

class Net(nn.Module):
    """Wrapper of MLP to support more specific DRL usage.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    :param state_shape: int or a sequence of int of the shape of state.
    :param action_shape: int or a sequence of int of the shape of action.
    :param hidden_sizes: shape of MLP passed in as a list.
    :param norm_layer: use which normalization before activation, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
        You can also pass a list of normalization modules with the same length
        of hidden_sizes, to use different normalization module in different
        layers. Default to no normalization.
    :param activation: which activation to use after each layer, can be both
        the same activation for all layers if passed in nn.Module, or different
        activation for different Modules if passed in a list. Default to
        nn.ReLU.
    :param device: specify the device when the network actually runs. Default
        to "cpu".
    :param bool softmax: whether to apply a softmax layer over the last layer's
        output.
    :param bool concat: whether the input shape is concatenated by state_shape
        and action_shape. If it is True, ``action_shape`` is not the output
        shape, but affects the input shape only.
    :param int num_atoms: in order to expand to the net of distributional RL.
        Default to 1 (not use).
    :param bool dueling_param: whether to use dueling network to calculate Q
        values (for Dueling DQN). If you want to use dueling option, you should
        pass a tuple of two dict (first for Q and second for V) stating
        self-defined arguments as stated in
        class:`~tianshou.utils.net.common.MLP`. Default to None.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.MLP` for more
        detailed explanation on the usage of activation, norm_layer, etc.

        You can also refer to :class:`~tianshou.utils.net.continuous.Actor`,
        :class:`~tianshou.utils.net.continuous.Critic`, etc, to see how it's
        suggested be used.
    """

    def __init__(
        self,
        state_shape: Union[int, Sequence[int]],
        action_shape: Union[int, Sequence[int]] = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        norm_args: Optional[ArgsType] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        act_args: Optional[ArgsType] = None,
        device: Union[str, int, torch.device] = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
    ) -> None:
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        input_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim += action_dim
        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0
        self.model = MLP(
            input_dim, output_dim, hidden_sizes, norm_layer, norm_args, activation,
            act_args, device, linear_layer
        )
        self.output_dim = self.model.output_dim
        if self.use_dueling:  # dueling DQN
            q_kwargs, v_kwargs = dueling_param  # type: ignore
            q_output_dim, v_output_dim = 0, 0
            if not concat:
                q_output_dim, v_output_dim = action_dim, num_atoms
            q_kwargs: Dict[str, Any] = {
                **q_kwargs, "input_dim": self.output_dim,
                "output_dim": q_output_dim,
                "device": self.device
            }
            v_kwargs: Dict[str, Any] = {
                **v_kwargs, "input_dim": self.output_dim,
                "output_dim": v_output_dim,
                "device": self.device
            }
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits."""
        logits = self.model(obs)
        bsz = logits.shape[0]
        if self.use_dueling:  # Dueling DQN
            q, v = self.Q(logits), self.V(logits)
            if self.num_atoms > 1:
                q = q.view(bsz, -1, self.num_atoms)
                v = v.view(bsz, -1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        elif self.num_atoms > 1:
            logits = logits.view(bsz, -1, self.num_atoms)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state


class Model(nn.Module):
  """
  Constructs a fully-connected neural network with flexible depth, width and
  activation function choices.
  """

  def __init__(
      self, dimList, actType="Tanh", output_activation=nn.Identity,
      verbose=False
  ):
    """
    Initalizes the neural network with dimension of each layer and the
    following activation layer.

    Args:
        dimList (List): the dimension of each layer.
        actType (str, optional): the type of activation function. Defaults to
            'Tanh'. Currently supports 'Sin', 'Tanh' and 'ReLU'.
        verbose (bool, optional):print messages if True. Defaults to False.
    """
    super(Model, self).__init__()

    # Construct module list: if use `Python List`, the modules are not
    # added to computation graph. Instead, we should use `nn.ModuleList()`.
    self.moduleList = nn.ModuleList()
    numLayer = len(dimList) - 1
    for idx in range(numLayer):
      i_dim = dimList[idx]
      o_dim = dimList[idx + 1]

      self.moduleList.append(nn.Linear(in_features=i_dim, out_features=o_dim))
      if idx == numLayer - 1:  # final linear layer, no act.
        self.moduleList.append(output_activation())
      else:
        if actType == "Sin":
          self.moduleList.append(Sin())
        elif actType == "Tanh":
          self.moduleList.append(nn.Tanh())
        elif actType == "ReLU":
          self.moduleList.append(nn.ReLU())
        else:
          raise ValueError(
              "Activation type ({:s}) is not included!".format(actType)
          )
    if verbose:
      print(self.moduleList)

  def forward(self, x):
    for m in self.moduleList:
      x = m(x)
    return x


# == Scheduler ==
class _scheduler(abc.ABC):
  """
  The parent class for schedulers. It implements some basic functions that will
  be used in all scheduler.
  """

  def __init__(self, last_epoch=-1, verbose=False):
    """Initializes the scheduler with the index of last epoch.
    """
    self.cnt = last_epoch
    self.verbose = verbose
    self.variable = None
    self.step()

  def step(self):
    """Updates the index of the last epoch and the variable.
    """
    self.cnt += 1
    value = self.get_value()
    self.variable = value

  @abc.abstractmethod
  def get_value(self):
    raise NotImplementedError

  def get_variable(self):
    """Returns the variable.
    """
    return self.variable


class StepLR(_scheduler):
  """This scheduler will decay to end value periodically.
  """

  def __init__(
      self, initValue, period, decay=0.1, endValue=0., last_epoch=-1,
      verbose=False
  ):
    """Initializes an object of the scheduler with the specified attributes.

    Args:
        initValue (float): initial value of the variable.
        period (int): the period to update the variable.
        decay (float, optional): the amount by which the variable decays.
            Defaults to 0.1.
        endValue (float, optional): the target value to decay to.
            Defaults to 0.
        last_epoch (int, optional): the index of the last epoch.
            Defaults to -1.
        verbose (bool, optional): print messages if True. Defaults to False.
    """
    self.initValue = initValue
    self.period = period
    self.decay = decay
    self.endValue = endValue
    super(StepLR, self).__init__(last_epoch, verbose)

  def get_value(self):
    """Returns the value of the variable.
    """
    if self.cnt == -1:
      return self.initValue

    numDecay = int(self.cnt / self.period)
    tmpValue = self.initValue * (self.decay**numDecay)
    if self.endValue is not None and tmpValue <= self.endValue:
      return self.endValue
    return tmpValue


class StepLRMargin(_scheduler):

  def __init__(
      self, initValue, period, goalValue, decay=0.1, endValue=1, last_epoch=-1,
      verbose=False
  ):
    """Initializes an object of the scheduler with the specified attributes.

    Args:
        initValue (float): initial value of the variable.
        period (int): the period to update the variable.
        goalValue (float):the target value to anneal to.
        decay (float, optional): the amount by which the margin between the
            variable and the goal value decays. Defaults to 0.1.
        endValue (float, optional): the maximum value of the variable.
            Defaults to 1.
        last_epoch (int, optional): the index of the last epoch.
            Defaults to -1.
        verbose (bool, optional): print messages if True. Defaults to False.
    """
    self.initValue = initValue
    self.period = period
    self.decay = decay
    self.endValue = endValue
    self.goalValue = goalValue
    super(StepLRMargin, self).__init__(last_epoch, verbose)

  def get_value(self):
    """Returns the value of the variable.
    """
    if self.cnt == -1:
      return self.initValue

    numDecay = int(self.cnt / self.period)
    tmpValue = self.goalValue - (self.goalValue
                                 - self.initValue) * (self.decay**numDecay)
    if self.endValue is not None and tmpValue >= self.endValue:
      return self.endValue
    return tmpValue


class StepResetLR(_scheduler):

  def __init__(
      self, initValue, period, resetPeriod, decay=0.1, endValue=0,
      last_epoch=-1, verbose=False
  ):
    """Initializes an object of the scheduler with the specified attributes.

    Args:
        initValue (float): initial value of the variable.
        period (int): the period to update the variable.
        resetPeriod (int): the period to reset the variable to its initial
            value.
        decay (float, optional): the amount by which the variable decays.
            Defaults to 0.1.
        endValue (float, optional): the target value to decay to.
            Defaults to 0.
        last_epoch (int, optional): the index of the last epoch.
            Defaults to -1.
        verbose (bool, optional): print messages if True. Defaults to False.
    """
    self.initValue = initValue
    self.period = period
    self.decay = decay
    self.endValue = endValue
    self.resetPeriod = resetPeriod
    super(StepResetLR, self).__init__(last_epoch, verbose)

  def get_value(self):
    """Returns the value of the variable.
    """
    if self.cnt == -1:
      return self.initValue

    numDecay = int(self.cnt / self.period)
    tmpValue = self.initValue * (self.decay**numDecay)
    if self.endValue is not None and tmpValue <= self.endValue:
      return self.endValue
    return tmpValue

  def step(self):
    """
    Updates the index of the last epoch and the variable. It overrides the same
    function in the parent class.
    """
    self.cnt += 1
    value = self.get_value()
    self.variable = value
    if (self.cnt + 1) % self.resetPeriod == 0:
      self.cnt = -1
