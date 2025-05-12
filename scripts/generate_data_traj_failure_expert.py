import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ruamel.yaml as yaml
import argparse
import io
from PIL import Image
import numpy as np
import pickle
from pathlib import Path
import pathlib
import random

import sys
import os
dreamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model_based_irl_torch'))
sys.path.append(dreamer_dir)
import dreamer.tools as tools


import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
def load_gt(path):
	# Load the .mat file
	mat_data = sio.loadmat(path)

	# Access the BRT slice and grid data
	BRT_slice = mat_data['BRT_slice']  # The 2D BRT slice
	grid_x = mat_data['grid_x'].squeeze()  # Grid for x (px)
	grid_y = mat_data['grid_y'].squeeze()  # Grid for y (py)
	grid_theta = mat_data['grid_theta'].squeeze()
	return BRT_slice, grid_x, grid_y, grid_theta

def create_interpolator(BRT_slice, grid_x, grid_y, grid_theta):
	return RegularGridInterpolator((grid_x, grid_y, grid_theta), BRT_slice, bounds_error=False, fill_value=None)

def is_safe_state(state, interpolator):
	# Interpolate to determine if the given state is in the failure set
	px, py, theta = state[0], state[1], state[2]

	fixed_angle = theta
	if fixed_angle > 2 * np.pi: 
		fixed_angle -= 2*np.pi
	if fixed_angle < 0: 
		fixed_angle += 2*np.pi

	value = interpolator([px, py, fixed_angle])

	# If the interpolated value is greater than 0, it's safe
	return value > 0.001 #FIXME : deal with interpolation error!

def optimal_action(state, interpolator, u_max, dt, v):
	# Find the optimal action to avoid unsafe regions
	best_action = None
	max_safety_value = -float('inf')
	min_safety_value = float('inf')
	mapping = torch.tensor([-u_max, 0, u_max])

	possible_actions = []
	
	for action in mapping:
		# Predict next state
		next_state = torch.zeros(3)
		next_state[0] = state[0] + v * dt * torch.cos(torch.tensor(state[2]))
		next_state[1] = state[1] + v * dt * torch.sin(torch.tensor(state[2]))
		next_state[2] = state[2] + dt * action
		
		fixed_angle = next_state[2]
		if fixed_angle > 2 * np.pi: 
			fixed_angle -= 2*np.pi
		if fixed_angle < 0: 
			fixed_angle += 2*np.pi

		# Check safety value of the next state
		safety_value = interpolator([next_state[0], next_state[1], fixed_angle])

		if safety_value >= 0 :
			possible_action = action.item()
			possible_actions.append(possible_action)

		if safety_value > max_safety_value:
			max_safety_value = safety_value
			best_action = action.item()
	
	# Random when safe.
	if len(possible_actions) > 0 :
		random_integers = random.randint(0, len(possible_actions) - 1)
		best_action = possible_actions[random_integers]

	return best_action

def gen_one_traj_img_expert(interpolator, x_min, x_max, y_min, y_max, u_max, dt, v, dpi, rand=-1, save=False):
  x_max -= 0.1
  y_max -= 0.1
  x_min += 0.1
  y_min += 0.1
  
  states = torch.zeros(3)
  while True:
        states = torch.rand(3)
        states[0] *= x_max - x_min
        states[1] *= y_max - y_min
        states[0] += x_min
        states[1] += y_min
        states[2] = torch.atan2(-states[1], -states[0]) + np.random.normal(0, 1)
        if states[2] < 0: 
          states[2] += 2*np.pi
        if states[2] > 2*np.pi: 
          states[2] -= 2*np.pi

        if is_safe_state(states.numpy(), interpolator):
             break
  
  
  state_obs = []
  img_obs = []
  state_gt = []
  dones = []
  acs = []
  mapping = torch.tensor([-u_max, 0, u_max])
  center = (0.0, 0.0)
  radius = 0.5

  # END Initialization.

  for t in range(100):

    # Terminate if out of bounds
    if torch.abs(states[0]) > 1.0 or torch.abs(states[1]) > 1.0 :
        dones.append(1)
        break

    if rand == -1:
      random_integers = torch.randint(0, 3, (1,))
    else:
      random_integers = torch.tensor([rand])

    # Action: 0 (-|w| = -1), 1 (0) or 2 (|w| = 1)
    # Map 0 to -1, 1 to 0, and 2 to 1
    # ac = mapping[random_integers].item()
    ac = optimal_action(states.numpy(), interpolator, u_max, dt, v)

    states_next = torch.zeros(3)

    states_next[0] = states[0] + v*dt*torch.cos(states[2])
    states_next[1] = states[1] + v*dt*torch.sin(states[2])
    states_next[2] = states[2] + dt*ac
       
    state_obs.append(states[2].numpy()) # get to observe theta
    state_gt.append(states.numpy()) # gt state

    if t == 99:
      dones.append(1)
    else:
      dones.append(1)

    acs.append(ac) # Store the action without disturbances

    # Render the environment
    fig, ax = plt.subplots()
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.axis('off')
    fig.set_size_inches(1, 1)

    # Draw the circle
    # circle = patches.Circle(center, radius, edgecolor="red", facecolor='none', linewidth=2)
    circle = patches.Circle(center, radius, edgecolor="#b3b3b3", facecolor="#b3b3b3", linewidth=2)
    ax.add_patch(circle)

    # Draw the trajectory
    plt.quiver(
      states[0], states[1],
      dt * v * torch.cos(states[2]),
      dt * v * torch.sin(states[2]),
      angles='xy', scale_units='xy', minlength=0,
      width=0.1, scale=0.18, color='black', zorder=3
    )

    plt.scatter(
        states[0], states[1],
        s=20, color='black', zorder=3
    )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if save :
       path = "datasets"
       if not os.path.exists(path):
          os.mkdir(path)
       plt.savefig(path+"/vis.png")

    # Save the frame
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img_array = np.array(img)
    img_obs.append(img_array)
    plt.close()

    states = states_next
  
  return state_obs, acs, state_gt, img_obs, dones

def gen_one_traj_img(x_min, x_max, y_min, y_max, u_max, dt, v, dpi, rand=-1, save=False):
  x_max -= 0.1
  y_max -= 0.1
  x_min += 0.1
  y_min += 0.1
  
  states = torch.zeros(3)
  while True:
        states = torch.rand(3)
        states[0] *= x_max - x_min
        states[1] *= y_max - y_min
        states[0] += x_min
        states[1] += y_min

        if (torch.abs(states[0]) >= 0.5 or torch.abs(states[1]) >= 0.5):
            break
  
  states[2] = torch.atan2(-states[1], -states[0]) + np.random.normal(0, 1)
  if states[2] < 0: 
    states[2] += 2*np.pi
  if states[2] > 2*np.pi: 
    states[2] -= 2*np.pi
  
  state_obs = []
  img_obs = []
  state_gt = []
  dones = []
  acs = []
  mapping = torch.tensor([-u_max, 0, u_max])
  center = (0.0, 0.0)
  radius = 0.5

  # END Initialization.

  for t in range(100):

    # Terminate if out of bounds
    if torch.abs(states[0]) > 1.0 or torch.abs(states[1]) > 1.0 :
        dones.append(1)
        break

    if rand == -1:
      random_integers = torch.randint(0, 3, (1,))
    else:
      random_integers = torch.tensor([rand])

    # Action: 0 (-|w| = -1), 1 (0) or 2 (|w| = 1)
    # Map 0 to -1, 1 to 0, and 2 to 1
    ac = mapping[random_integers].item()
    states_next = torch.zeros(3)

    states_next[0] = states[0] + v*dt*torch.cos(states[2])
    states_next[1] = states[1] + v*dt*torch.sin(states[2])
    states_next[2] = states[2] + dt*ac
       
    state_obs.append(states[2].numpy()) # get to observe theta
    state_gt.append(states.numpy()) # gt state

    if t == 99:
      dones.append(1)
    else:
      dones.append(1)

    acs.append(ac) # Store the action without disturbances

    # Render the environment
    fig, ax = plt.subplots()
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.axis('off')
    fig.set_size_inches(1, 1)

    # Draw the circle
    circle = patches.Circle(center, radius, edgecolor="#b3b3b3", facecolor="#b3b3b3", linewidth=2)
    ax.add_patch(circle)

    # Draw the trajectory
    plt.quiver(
      states[0], states[1],
      dt * v * torch.cos(states[2]),
      dt * v * torch.sin(states[2]),
      angles='xy', scale_units='xy', minlength=0,
      width=0.1, scale=0.18, color='black', zorder=3
    )

    plt.scatter(
        states[0], states[1],
        s=20, color='black', zorder=3
    )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if save :
       path = "datasets"
       if not os.path.exists(path):
          os.mkdir(path)
       plt.savefig(path+"/vis.png")

    # Save the frame
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img_array = np.array(img)
    img_obs.append(img_array)
    plt.close()

    states = states_next
  
  return state_obs, acs, state_gt, img_obs, dones


import tqdm
def generate_trajs(interpolator, x_min, x_max, y_min, y_max, u_max, dt, v, num_pts, dpi, save_path, num_random):

  print(save_path)
  demos = []
  for i in tqdm.tqdm(range(num_pts), total=num_pts):
    # print('demo: ', i)
    state_obs, acs, state_gt, img_obs, dones = gen_one_traj_img_expert(interpolator, x_min, x_max, y_min, y_max, u_max, dt, v, dpi, save = False)
    demo = {}
    demo['obs'] = {'image': img_obs, 'state': state_obs, 'priv_state': state_gt}
    demo['actions'] = acs
    demo['dones'] = dones
    demos.append(demo)

  # Add normal trajectories
  for i in tqdm.tqdm(range(num_random), total=num_random):
    # print('demo: ', i)
    state_obs, acs, state_gt, img_obs, dones = gen_one_traj_img(x_min, x_max, y_min, y_max, u_max, dt, v, dpi, save = False)
    demo = {}
    demo['obs'] = {'image': img_obs, 'state': state_obs, 'priv_state': state_gt}
    demo['actions'] = acs
    demo['dones'] = dones
    demos.append(demo)
  
  with open(save_path, 'wb') as f:
    pickle.dump(demos, f)

def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

if __name__=='__main__':      
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--expt_name", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="datasets/demos")
    parser.add_argument("--num_random", type=int, default=50)
    config, remaining = parser.parse_known_args()

    
    yaml = yaml.YAML(typ="safe", pure=True)
    configs = yaml.load(
        (pathlib.Path(sys.argv[0]).parent / "../configs/config.yaml").read_text()
    )

    name_list = ["defaults", *config.configs] if config.configs else ["defaults"]

    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()

    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_args(remaining)

    BRT_slice, grid_x, grid_y, grid_theta = load_gt('logs/v_1_w_1.25_brt.mat')
    interpolator = create_interpolator(BRT_slice, grid_x, grid_y, grid_theta)

    num_pts =  final_config.num_pts
    x_min = final_config.x_min
    x_max = final_config.x_max
    y_min = final_config.y_min
    y_max = final_config.y_max
    u_max = final_config.u_max
    dt = final_config.dt
    v = final_config.speed
    dpi = 128
    demos = generate_trajs(interpolator, x_min, x_max, y_min, y_max, u_max, dt, v, num_pts, dpi, save_path=config.save_path, num_random=config.num_random)
