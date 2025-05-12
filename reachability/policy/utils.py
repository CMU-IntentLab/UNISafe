"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

This module provides some utils functions for reinforcement learning
algortihms and some general save and load functions.
"""

import os
import glob
import pickle
import torch


def soft_update(target, source, tau):
  """Uses soft_update method to update the target network.

  Args:
      target (toch.nn.Module): target network in double deep Q-network.
      source (toch.nn.Module): Q-network in double deep Q-network.
      tau (float): the ratio of the weights in the target.
  """
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(target_param.data * (1.0-tau) + param.data * tau)


def save_model(model, step, logs_path, types):
  """Saves the weights of the model.

  Args:
      model (toch.nn.Module): the model to be saved.
      step (int): the number of updates so far.
      logs_path (str): the path to save the weights.
      types (str): the decorater of the file name.
      MAX_MODEL (int): the maximum number of models to be saved.
  """
  os.makedirs(logs_path, exist_ok=True)
  logs_path = os.path.join(logs_path, "{}-{}.pth".format(types, step))
  torch.save(model.state_dict(), logs_path)
  print("  => Save {} after [{}] updates".format(logs_path, step))


def save_obj(obj, filename):
  """Saves the object into a pickle file.

  Args:
      obj (object): the object to be saved.
      filename (str): the path to save the object.
  """
  with open(filename + ".pkl", "wb") as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
  """Loads the object and return the object.

  Args:
      filename (str): the path to save the object.
  """
  with open(filename + ".pkl", "rb") as f:
    return pickle.load(f)

# == Scheduler ==
import abc
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
      self, initValue, period, goalValue, decay=0.9, endValue=1, last_epoch=-1,
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
  
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

import numpy as np


class BaseNoise(ABC, object):
    """The action noise base class."""

    def __init__(self) -> None:
        super().__init__()

    def reset(self) -> None:
        """Reset to the initial state."""
        pass

    @abstractmethod
    def __call__(self, size: Sequence[int]) -> np.ndarray:
        """Generate new noise."""
        raise NotImplementedError


class GaussianNoise(BaseNoise):

    def __init__(self, mu: float = 0.0, sigma: float = 1.0, device: str = "cuda"):
        super().__init__()
        self._mu = mu
        assert sigma >= 0, "Noise std should not be negative."
        self._sigma = sigma
        self._device = device

    def __call__(self, size: Sequence[int]) -> torch.Tensor:
        # Generates random noise from a normal distribution 
        # with mean = self._mu and std = self._sigma.
        return self._sigma * torch.randn(size, device=self._device) + self._mu


class OUNoise(BaseNoise):
    """Class for Ornstein-Uhlenbeck process, as used for exploration in DDPG.

    Usage:
    ::

        # init
        self.noise = OUNoise()
        # generate noise
        noise = self.noise(logits.shape, eps)

    For required parameters, you can refer to the stackoverflow page. However,
    our experiment result shows that (similar to OpenAI SpinningUp) using
    vanilla Gaussian process has little difference from using the
    Ornstein-Uhlenbeck process.
    """

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 0.3,
        theta: float = 0.15,
        dt: float = 1e-2,
        x0: Optional[Union[float, np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self._mu = mu
        self._alpha = theta * dt
        self._beta = sigma * np.sqrt(dt)
        self._x0 = x0
        self.reset()

    def reset(self) -> None:
        """Reset to the initial state."""
        self._x = self._x0

    def __call__(self, size: Sequence[int], mu: Optional[float] = None) -> np.ndarray:
        """Generate new noise.

        Return an numpy array which size is equal to ``size``.
        """
        if self._x is None or isinstance(
            self._x, np.ndarray
        ) and self._x.shape != size:
            self._x = 0.0
        if mu is None:
            mu = self._mu
        r = self._beta * np.random.normal(size=size)
        self._x = self._x + self._alpha * (mu - self._x) + r
        return self._x  # type: ignore