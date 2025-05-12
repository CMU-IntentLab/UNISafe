"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

This module implements reach-avoid reinforcement learning with double deep
Q-network. It also supports the standard sum of discounted rewards (Lagrange
cost) reinforcement learning.

Here we aim to minimize the reach-avoid cost, given by the Bellman backup:
    - a' = argmin_a' Q_network(s', a')
    - V(s') = Q_target(s', a')
    - V(s) = gamma ( max{ g(s), min{ l(s), V(s') } }
             + (1-gamma) max{ g(s), l(s) }
    - loss = E[ ( V(f(s,a)) - Q_network(s,a) )^2 ]
"""

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, smooth_l1_loss

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import wandb

from .model import Model
from .DDQN import DDQN, Transition


class DDQNSingle(DDQN):
  """
  Implements the double deep Q-network algorithm. Supports minimizing the
  reach-avoid cost or the standard sum of discounted costs.

  Args:
      DDQN (object): an object implementing the basic utils functions.
  """

  def __init__(
      self, CONFIG, numAction, actionList, dimList, mode="RA",
      terminalType="g", verbose=True
  ):
    """
    Initializes with a configuration object, environment information, neural
    network architecture, reinforcement learning algorithm type and type of
    the terminal value for reach-avoid reinforcement learning.

    Args:
        CONFIG (object): configuration.
        numAction (int): the number of actions.
        actionList (list): action set.
        dimList (np.ndarray): dimensions of each layer in the neural network.
        mode (str, optional): the reinforcement learning mode.
            Defaults to 'RA'.
        terminalType (str, optional): type of the terminal value.
            Defaults to 'g'.
        verbose (bool, optional): print the messages if True. Defaults to True.
    """
    super(DDQNSingle, self).__init__(CONFIG)

    self.mode = mode  # 'normal' or 'RA'
    self.terminalType = terminalType

    # == ENV PARAM ==
    self.numAction = numAction
    self.actionList = actionList

    # == Build neural network for (D)DQN ==
    self.dimList = dimList
    self.actType = CONFIG.ACTIVATION
    self.build_network(dimList, self.actType, verbose)
    print(
        "DDQN: mode-{}; terminalType-{}".format(self.mode, self.terminalType)
    )

  def build_network(self, dimList, actType="Tanh", verbose=True):
    """Builds a neural network for the Q-network.

    Args:
        dimList (np.ndarray): dimensions of each layer in the neural network.
        actType (str, optional): activation function. Defaults to 'Tanh'.
        verbose (bool, optional): print the messages if True. Defaults to True.
    """
    self.Q_network = Model(dimList, actType, verbose=verbose)
    self.target_network = Model(dimList, actType)

    if self.device == torch.device("cuda"):
      self.Q_network.cuda()
      self.target_network.cuda()

    self.build_optimizer()

  def update(self, addBias=False, env=None):
    """Updates the Q-network using a batch of sampled replay transitions.

    Args:
        addBias (bool, optional): use biased version of value function if
            True. Defaults to False.

    Returns:
        float: critic loss.
    """
    if len(self.memory) < self.BATCH_SIZE * 20:
      return

    # == EXPERIENCE REPLAY ==
    transitions = self.memory.sample(self.BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    (non_final_mask, non_final_state_nxt, state, action, reward, g_x,
     ) = self.unpack_batch(batch)

    non_final_mask = non_final_mask.squeeze()
    non_final_state_nxt = non_final_state_nxt.squeeze()
    state = state.squeeze()
    action = action.squeeze().unsqueeze(1)
    g_x = g_x.squeeze()

    self.Q_network.train()
    state = state.squeeze()
    state_action_values = (
        self.Q_network(state).gather(dim=1, index=action).view(-1)
    )
    # == get a' by Q_network: a' = argmin_a' Q_network(s', a') ==
    with torch.no_grad():
      self.Q_network.eval()
      action_nxt = (
          self.Q_network(non_final_state_nxt).max(1, keepdim=True)[1]
      )

    # == get expected value ==
    state_value_nxt = torch.zeros(self.BATCH_SIZE).to(self.device)

    with torch.no_grad():  # V(s') = Q_target(s', a'), a' is from Q_network
      if self.double_network:
        self.target_network.eval()
        Q_expect = self.target_network(non_final_state_nxt)
      else:
        self.Q_network.eval()
        Q_expect = self.Q_network(non_final_state_nxt)

    state_value_nxt[non_final_mask] = \
        Q_expect.gather(dim=1, index=action_nxt).view(-1)

    # == Discounted Reach-Avoid Bellman Equation (DRABE) ==
    if self.mode == "RA":
      y = torch.zeros(self.BATCH_SIZE).float().to(self.device)
      final_mask = torch.logical_not(non_final_mask)

      non_terminal = torch.min(g_x[non_final_mask], state_value_nxt[non_final_mask])
      terminal = g_x

        # normal state
      y[non_final_mask] = non_terminal * self.GAMMA + terminal[
          non_final_mask] * (1 - self.GAMMA)

      # terminal state
      if self.terminalType == "g":
        y[final_mask] = g_x[final_mask]
      elif self.terminalType == "max":
        y[final_mask] = terminal[final_mask]
      else:
        raise ValueError("invalid terminalType")
      
    else:  # V(s) = c(s, a) + gamma * V(s')
      y = state_value_nxt * self.GAMMA + reward

    # == regression: Q(s, a) <- V(s) ==
    loss = smooth_l1_loss(
        input=state_action_values,
        target=y.detach(),
    )

    # == backpropagation ==
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.Q_network.parameters(), self.max_grad_norm)
    self.optimizer.step()

    self.update_target_network()

    return loss.item()
  
  def initBuffer(self, env):
    """Adds some transitions to the replay memory (buffer) randomly.

    Args:
        env (gym.Env): the environment we interact with.
    """
    cnt = 0
    s = env.reset()
    if type(s) == dict:
      s = s["state"]
    while len(self.memory) < self.memory.capacity:
      cnt += 1
      print("\rWarmup Buffer [{:d}]".format(cnt), end="")
      a, a_idx = self.select_action(s, explore=True)
      s_, r, done, info = env.step(a_idx)
      if type(s_) == dict:
        s_ = s_["state"]
      s_ = None if done else s_
      self.store_transition(s, a_idx, r, s_, info)
      if done:
        s = env.reset()
        if type(s) == dict:
          s = s["state"]
      else:
        s = s_
    print(" --- Warmup Buffer Ends")

  def initQ(
      self, env, warmupIter, outFolder, num_warmup_samples=200, vmin=-1,
      vmax=1, plotFigure=True, storeFigure=True
  ):
    """
    Initalizes the Q-network given that the environment can provide warmup
    examples with heuristic values.

    Args:
        env (gym.Env): the environment we interact with.
        warmupIter (int, optional): the number of iterations in the
            Q-network warmup.
        outFolder (str, optional): the path of the parent folder of model/ and
            figure/.
        num_warmup_samples (int, optional): the number of warmup samples.
            Defaults to 200.
        vmin (float, optional): the minmum value in the colorbar.
            Defaults to -1.
        vmax (float, optional): the maximum value in the colorbar.
            Defaults to 1.
        plotFigure (bool, optional): plot figures if True.
            Defaults to True.
        storeFigure (bool, optional): store figures if True.
            Defaults to False.

    Returns:
        np.ndarray: loss of fitting Q-values to heuristic values.
    """

    lossList = np.empty(warmupIter, dtype=float)
    for ep_tmp in range(warmupIter):
      states, heuristic_v = env.get_warmup_examples_offline(
          num_warmup_samples=num_warmup_samples
      )

      self.Q_network.train()
      heuristic_v = torch.from_numpy(heuristic_v).float().to(self.device)
      states = torch.from_numpy(states).float().to(self.device)
      v = self.Q_network(states)
      loss = mse_loss(input=v, target=heuristic_v, reduction="sum")
      self.optimizer.zero_grad()
      loss.backward()

      nn.utils.clip_grad_norm_(self.Q_network.parameters(), self.max_grad_norm)
      self.optimizer.step()
      lossList[ep_tmp] = loss.detach().cpu().numpy()
      print(
          "\rWarmup Q [{:d} / {:d}]. MSE = {:f}".format(ep_tmp + 1, warmupIter, loss),
          end="",
      )

    print(" --- Warmup Q Ends")

    if plotFigure or storeFigure:
      env.visualize(self.Q_network, vmin=vmin, vmax=vmax, cmap="seismic")
      if storeFigure:
        figureFolder = os.path.join(outFolder, "figure")
        os.makedirs(figureFolder, exist_ok=True)
        wandb.log({"InitQ": wandb.Image(plt.gcf())})
      if plotFigure:
        plt.pause(0.001)
      plt.clf()
      plt.close('all')
    self.target_network.load_state_dict(
        self.Q_network.state_dict()
    )  # hard replace
    self.build_optimizer()

    return lossList
  
  def visualize(self, env):
    v_nn_bool, tn,tp,fn,fp = env.visualize(
          self.Q_network, vmin=0, boolPlot=True, addBias=False
      )
    wandb.log({"env_vis_bool": wandb.Image(plt.gcf())}, step=self.cntUpdate)

    v_nn, _, _, _, _= env.visualize(
          self.Q_network, vmin=-2, vmax=2, cmap="seismic",
          addBias=False
      )
    wandb.log({"env_vis": wandb.Image(plt.gcf())}, step=self.cntUpdate)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tnr = specificity  # same as specificity
    balanced_accuracy = (recall + specificity) / 2

    result = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "recall": recall,         # tp / (tp + fn)
        "precision": precision,   # tp / (tp + fp)
        "f1_score": f1,
        "accuracy": accuracy,     # (tp + tn) / (tp + tn + fp + fn)
        "tnr": specificity,  # tn / (tn + fp)
        "npv": npv,               # tn / (tn + fn)
        "fpr": fpr,               # fp / (fp + tn)
        "balanced_accuracy": balanced_accuracy
    }

    return v_nn_bool, v_nn, result

  def learn(
      self, env, MAX_UPDATES=2000000, MAX_EP_STEPS=100, warmupBuffer=True,
      warmupQ=False, warmupIter=10000, addBias=False, doneTerminate=True,
      runningCostThr=None, curUpdates=None, checkPeriod=50000, plotFigure=True,
      storeFigure=False, showBool=False, vmin=-1, vmax=1, numRndTraj=200,
      storeModel=True, storeBest=False, outFolder="RA", verbose=True
  ):
    # == Warmup Buffer ==
    startInitBuffer = time.time()
    if warmupBuffer:
      self.initBuffer(env)
    endInitBuffer = time.time()

    # == Warmup Q ==
    startInitQ = time.time()
    if warmupQ:
      self.initQ(
          env, warmupIter=warmupIter, outFolder=outFolder,
          plotFigure=plotFigure, storeFigure=storeFigure, vmin=vmin, vmax=vmax
      )
    endInitQ = time.time()

    # == Main Training ==
    startLearning = time.time()
    trainingRecords = []
    runningCost = 0.0
    trainProgress = []
    ep = 0

    if curUpdates is not None:
      self.cntUpdate = curUpdates
      print("starting from {:d} updates".format(self.cntUpdate))

    if storeModel:
      modelFolder = os.path.join(outFolder, "model")
      os.makedirs(modelFolder, exist_ok=True)
    if storeFigure:
      figureFolder = os.path.join(outFolder, "figure")
      os.makedirs(figureFolder, exist_ok=True)

    while self.cntUpdate <= MAX_UPDATES:
      s = env.reset()
      if type(s) == dict:
        s = s["state"]
      epCost = 0.0
      ep += 1
      # Rollout
      for step_num in range(MAX_EP_STEPS):
        # Select action
        a, a_idx = self.select_action(s, explore=True)

        # Interact with env
        s_, r, done, info = env.step(a_idx)
        if type(s_) == dict:
          s_ = s_["state"]
        s_ = None if done else s_
        epCost += r

        # Store the transition in memory
        self.store_transition(s, a_idx, r, s_, info)
        s = s_
        
        # Check after fixed number of gradient updates
        if self.cntUpdate != 0 and self.cntUpdate % checkPeriod == 0:

          if storeModel:
              self.save(self.cntUpdate, modelFolder)

          if plotFigure or storeFigure:

            # resultMtx = self.evaluate_agent(env) # closed-loop evalutaion.
            # success_rate = sum(resultMtx == 1) / sum(resultMtx != 0)
            # wandb.log({"success_rate:": success_rate}, step=self.cntUpdate)
            # print("success rate: {} ({} / {})".format(success_rate, sum(resultMtx == 1), sum(resultMtx != 0)))

            v_nn, tn,tp,fn,fp = env.visualize(
                  self.Q_network, vmin=0, boolPlot=True, addBias=addBias
              )

            # v_nn, tn,tp,fn,fp = env.visualize_all_eval(
            #       self.Q_network, vmin=0, boolPlot=True, addBias=addBias
            #   )
            wandb.log({"env_vis_bool": wandb.Image(plt.gcf())}, step=self.cntUpdate)

            env.visualize(
                  self.Q_network, vmin=vmin, vmax=vmax, cmap="seismic",
                  addBias=addBias
              )
            wandb.log({"env_vis": wandb.Image(plt.gcf())}, step=self.cntUpdate)

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tnr = specificity  # same as specificity
            balanced_accuracy = (recall + specificity) / 2

            result = {
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "recall": recall,         # tp / (tp + fn)
                "precision": precision,   # tp / (tp + fp)
                "f1_score": f1,
                "accuracy": accuracy,     # (tp + tn) / (tp + tn + fp + fn)
                "tnr": specificity,  # tn / (tn + fp)
                "npv": npv,               # tn / (tn + fn)
                "fpr": fpr,               # fp / (fp + tn)
                "balanced_accuracy": balanced_accuracy
            }

            wandb.log(result, step=self.cntUpdate)
            print(result)
                    
            if plotFigure:
              plt.pause(0.001)

            plt.clf()
            plt.close('all')
          
        # Perform one step of the optimization (on the target network)
        lossC = self.update(addBias=addBias, env=env)
        trainingRecords.append(lossC)
        self.cntUpdate += 1
        self.updateHyperParam()

        # Terminate early
        if done and doneTerminate:
          break

      # Rollout report
      runningCost = runningCost*0.9 + epCost*0.1
      if verbose:
        print(
            "\r[{:d}-{:d}]: ".format(ep, self.cntUpdate)
            + "This episode gets running/episode cost = "
            + "({:3.2f}/{:.2f}) after {:d} steps.".
            format(runningCost, epCost, step_num + 1),
            end="",
        )

      # Check stopping criteria
      if runningCostThr is not None:
        if runningCost <= runningCostThr:
          print(
              "\n At Updates[{:3.0f}] Solved!".format(self.cntUpdate)
              + " Running cost is now {:3.2f}!".format(runningCost)
          )
          env.close()
          break

    endLearning = time.time()
    timeInitBuffer = endInitBuffer - startInitBuffer
    timeInitQ = endInitQ - startInitQ
    timeLearning = endLearning - startLearning
    self.save(self.cntUpdate, modelFolder)
    print(
        "\nInitBuffer: {:.1f}, InitQ: {:.1f}, Learning: {:.1f}".format(
            timeInitBuffer, timeInitQ, timeLearning
        )
    )
    trainingRecords = np.array(trainingRecords)
    trainProgress = np.array(trainProgress)
    return trainingRecords, trainProgress

  def select_action(self, state, explore=False):
    """Selects the action given the state and conditioned on `explore` flag.

    Args:
        state (np.ndarray): the state of the environment.
        explore (bool, optional): randomize the deterministic action by
            epsilon-greedy algorithm if True. Defaults to False.

    Returns:
        np.ndarray: action
        int: action index
    """
    self.Q_network.eval()
    state = torch.from_numpy(state).float().squeeze().unsqueeze(0).to(self.device)
    action_index = np.random.randint(0, self.numAction)
    if (np.random.rand() < self.EPSILON) and explore:
      action_index = np.random.randint(0, self.numAction)
    else:
      action_index = self.Q_network(state).max(dim=1)[1].item()
    return self.actionList[action_index], action_index


  def evaluate_agent(self, env):

    results = env.simulate_trajectories_post(self.Q_network)
    
    return results