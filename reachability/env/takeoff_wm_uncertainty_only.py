"""
Takeoff World Model Environment

A safety-critical reinforcement learning environment that uses world models
and uncertainty quantification for safe exploration in takeoff scenarios.
The environment provides rewards based on epistemic uncertainty to encourage
exploration in safe regions and avoid out-of-distribution states.
"""

from os import path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import io
from PIL import Image
import matplotlib.patches as patches
import torch
import math
import random


class Takeoff_WM(gym.Env):
    """
    World Model-based Takeoff Environment for Safe Reinforcement Learning.
    
    This environment uses a pre-trained world model to simulate takeoff dynamics
    and provides uncertainty-based rewards to encourage safe exploration.
    The agent receives higher rewards for actions that maintain low epistemic
    uncertainty, indicating confidence in the world model's predictions.
    
    Attributes:
        observation_space: Feature space from the world model (1536-dimensional)
        action_space: Joint action space (7-dimensional, bounded)
        time_step: Simulation time step
        threshold: Out-of-distribution threshold for uncertainty quantification
    """
    def __init__(self):
        """
        Initialize the Takeoff World Model environment.
        
        Sets up observation and action spaces, device configuration, and
        environment parameters for safe exploration using world models.
        """
        self.render_mode = None
        self.time_step = 0.05  # Simulation time step in seconds
        
        # State bounds for position and orientation (currently unused but kept for compatibility)
        self.state_bounds_high = np.array([1., 1., np.pi])  # [x, y, theta]
        self.state_bounds_low = np.array([-1., -1., -np.pi])
        
        # GPU device for model inference
        self.device = 'cuda'

        # Observation space: World model features (latent state representation)
        # Shape: (1, 1, 1536) - batch_size, sequence_length, feature_dim
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(1, 1, 1536), 
            dtype=np.float32
        )
        
        # Action space: 7-dimensional joint actions with symmetric bounds
        self.action_space = spaces.Box(
            low=-0.5, 
            high=0.5, 
            shape=(7,), 
            dtype=np.float32
        )
        
        self.image_size = 128  # Image size for rendering (if needed)
        
        # World model components (initialized later via set_wm)
        self.config = None
        self.wm = None
        self.dataset = None
        self.ensemble = None
        self.threshold = None
        self.feat_size = None
        self.feat = None  # Current feature representation
        self.latent = None  # Current latent state
        

    def set_wm(self, config: Any, wm: Any, ensemble: Any, dataset: Any) -> None:
        """
        Set up the world model components for the environment.
        
        Args:
            config: Configuration object containing model hyperparameters
            wm: Pre-trained world model for state transitions and predictions
            ensemble: Ensemble model for uncertainty quantification
            dataset: Training dataset for initialization and offline evaluation
        """
        self.config = config
        self.wm = wm
        self.dataset = dataset
        self.ensemble = ensemble
        self.threshold = self.config.ood_threshold
        
        # Calculate feature size based on dynamics configuration
        if self.config.dyn_discrete:
            # Discrete stochastic dynamics: combine discrete and deterministic components
            self.feat_size = (
                self.config.dyn_stoch * self.config.dyn_discrete + 
                self.config.dyn_deter
            )
        else:
            # Continuous stochastic dynamics: add stochastic and deterministic components
            self.feat_size = self.config.dyn_stoch + self.config.dyn_deter
    

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step using the world model.
        
        Args:
            action: Action to take in the environment (shape: [7])
            
        Returns:
            observation: Updated world model features (shape: [batch_size * seq_len, feat_size])
            reward: Uncertainty-based reward encouraging safe exploration
            done: Episode termination flag (always False for continuous operation)
            truncated: Episode truncation flag (always False)
            info: Additional information dictionary (empty)
        """
        with torch.no_grad():
            # Ensure action is detached from computation graph
            action_tensor = action.detach()
            
            # Calculate epistemic uncertainty for the current state-action pair
            epistemic_uncertainty = self.calculate_uncertainty(self.feat, action_tensor)
            
            # Convert uncertainty to reward: higher uncertainty = lower reward
            # This encourages the agent to stay in regions where the model is confident
            raw_reward = self.threshold - epistemic_uncertainty

            # Apply step function transformation to create discrete reward levels
            # This creates three reward levels: -1.0, original, 1.0
            processed_reward = raw_reward.copy()
            processed_reward[raw_reward < -0.4] = -1.0  # High uncertainty penalty
            processed_reward[raw_reward > 0.4] = 1.0    # Low uncertainty bonus
            
            # Update world model state by taking the action
            # Reshape action for batch processing
            batched_action = action_tensor.reshape(
                (self.config.batch_size, self.config.batch_length, -1)
            )
            
            # Predict next latent state using world model dynamics
            self.latent = self.wm.dynamics.img_step(self.latent, batched_action)
            
            # Extract features from updated latent state
            self.feat = self.wm.dynamics.get_feat(self.latent)
            self.feat = self.feat.reshape(
                (self.config.batch_size * self.config.batch_length, -1)
            )

            # Environment continues indefinitely (no terminal states)
            done = False
            truncated = False
            info = {}

        return self.feat.cpu().numpy(), processed_reward, done, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Samples a random trajectory from the dataset and uses it to initialize
        the world model's latent state and feature representation.
        
        Args:
            seed: Random seed for reproducibility (unused)
            options: Additional reset options (unused)
            
        Returns:
            observation: Initial world model features
            info: Reset information (empty)
        """
        super().reset(seed=seed)

        # Sample a random initial trajectory from the dataset
        initial_trajectory = next(self.dataset)

        with torch.no_grad():
            # Preprocess the trajectory data for the world model
            preprocessed_data = self.wm.preprocess(initial_trajectory)
            
            # Encode observations into latent representations
            encoded_observations = self.wm.encoder(preprocessed_data)
            
            # Use the world model dynamics to observe and create initial latent state
            # This processes the sequence of observations and actions to create
            # a coherent latent state representation
            self.latent, _ = self.wm.dynamics.observe(
                encoded_observations, 
                preprocessed_data["action"], 
                preprocessed_data["is_first"]
            )

            # Alternative: Sample a random timestep from the trajectory (currently disabled)
            # This could be used to start from a random point in the trajectory
            # random_idx = random.randint(0, self.config.batch_length - 1)
            # for state_key, state_value in self.latent.items(): 
            #     self.latent[state_key] = state_value[:, [random_idx]]

            # Extract feature representation from the latent state
            self.feat = self.wm.dynamics.get_feat(self.latent)
            self.feat = self.feat.reshape(
                (self.config.batch_size * self.config.batch_length, -1)
            )

        return self.feat.cpu().numpy(), {}

      
    def calculate_uncertainty(self, features: torch.Tensor, action: torch.Tensor) -> np.ndarray:
        """
        Calculate epistemic uncertainty for a given state-action pair.
        
        Uses an ensemble model to estimate epistemic (model) uncertainty,
        which indicates how confident the model is about its predictions
        in the given state-action region.
        
        Args:
            features: Current world model features (latent state representation)
            action: Action to evaluate
            
        Returns:
            epistemic_uncertainty: Uncertainty estimates for each sample
        """
        with torch.no_grad():
            # Concatenate features and actions to create input for uncertainty estimation
            state_action_input = torch.concat([features, action], dim=-1)
            
            # Use ensemble model to estimate epistemic uncertainty
            # The intrinsic_reward_penn method returns uncertainty estimates
            uncertainty_estimates = self.ensemble.intrinsic_reward_penn(state_action_input)
            
            # Extract the uncertainty values and move to CPU
            return uncertainty_estimates[:, 0].cpu().numpy()
 