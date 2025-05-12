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

# Region to stop (green area)
stop_region_x_1 = (-1.0, 1.0)
stop_region_y_1 = (-1.0, -0.6)

# Region to stop (green area)
stop_region_x_2 = (-1.0, 1.0)
stop_region_y_2 = (0.6, 1.0)

def gen_one_traj_img(x_min, x_max, y_min, y_max, u_max, dt, v, dpi, rand=-1, save=False, ood=True):
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

        # Check if the state is NOT inside either stop region
        inside_region1 = (stop_region_x_1[0] <= states[0] <= stop_region_x_1[1] and 
                          stop_region_y_1[0] <= states[1] <= stop_region_y_1[1])
        inside_region2 = (stop_region_x_2[0] <= states[0] <= stop_region_x_2[1] and 
                          stop_region_y_2[0] <= states[1] <= stop_region_y_2[1])
        
        if not (inside_region1 or inside_region2):
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

    # END Initialization.
    for t in range(100):
        
        inside_region1 = (stop_region_x_1[0] <= states[0] <= stop_region_x_1[1] and 
                          stop_region_y_1[0] <= states[1] <= stop_region_y_1[1])
        inside_region2 = (stop_region_x_2[0] <= states[0] <= stop_region_x_2[1] and 
                          stop_region_y_2[0] <= states[1] <= stop_region_y_2[1])

        if ood and (inside_region1 or inside_region2):
          dones.append(1)
          break

        # Terminate if out of bounds
        if torch.abs(states[0]) > 1.0 or torch.abs(states[1]) > 1.0:
            dones.append(1)
            break

        if rand == -1:
            random_integers = torch.randint(0, 3, (1,))
        else:
            random_integers = torch.tensor([rand])

        # Action: 0 (-|w| = -1), 1 (0) or 2 (|w| = 1)
        ac = mapping[random_integers].item()
        states_next = torch.zeros(3)
        states_next[0] = states[0] + v * dt * torch.cos(states[2])
        states_next[1] = states[1] + v * dt * torch.sin(states[2])
        states_next[2] = states[2] + dt * ac
       
        state_obs.append(states[2].numpy())  # observe theta
        state_gt.append(states.numpy())       # ground truth state

        if t == 99:
            dones.append(1)
        else:
            dones.append(1)

        acs.append(ac)  # Store the action without disturbances

        # Render the environment
        fig, ax = plt.subplots()
        plt.xlim([-1.0, 1.0])
        plt.ylim([-1.0, 1.0])
        plt.axis('off')
        fig.set_size_inches(1, 1)

        rect1 = patches.Rectangle(
            (stop_region_x_1[0], stop_region_y_1[0]),
            stop_region_x_1[1] - stop_region_x_1[0],
            stop_region_y_1[1] - stop_region_y_1[0],
            linewidth=3, facecolor='#BB96EB', edgecolor="none"
        )

        ax.add_patch(rect1)

        rect2 = patches.Rectangle(
            (stop_region_x_2[0], stop_region_y_2[0]),
            stop_region_x_2[1] - stop_region_x_2[0],
            stop_region_y_2[1] - stop_region_y_2[0],
            linewidth=3, facecolor='#BB96EB', edgecolor="none"
        )
        ax.add_patch(rect2)

        # Draw the trajectory
        plt.quiver(
            states[0], states[1],
            dt * v * torch.cos(states[2]),
            dt * v * torch.sin(states[2]),
            angles='xy', scale_units='xy', minlength=0,
            width=0.1, scale=0.18, color="black", zorder=3
        )
        plt.scatter(
            states[0], states[1],
            s=20, color="black", zorder=3
        )
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        if save:
            path = "datasets"
            if not os.path.exists(path):
                os.mkdir(path)
            plt.savefig(path + "/vis_uncertainty.png")

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
def generate_trajs(x_min, x_max, y_min, y_max, u_max, dt, v, num_pts, dpi, save_path, ood):

  print(save_path)
  demos = []
  for i in tqdm.tqdm(range(num_pts), total=num_pts):
    # print('demo: ', i)
    state_obs, acs, state_gt, img_obs, dones = gen_one_traj_img(x_min, x_max, y_min, y_max, u_max, dt, v, dpi, save = False, ood=ood)
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
    parser.add_argument("--ood", action="store_false", default=True)
    parser.add_argument("--save_path", type=str, default="datasets/demos")
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

    num_pts =  final_config.num_pts
    x_min = final_config.x_min
    x_max = final_config.x_max
    y_min = final_config.y_min
    y_max = final_config.y_max
    u_max = final_config.u_max
    dt = final_config.dt
    v = final_config.speed
    dpi = 128
    demos = generate_trajs(x_min, x_max, y_min, y_max, u_max, dt, v, num_pts, dpi, save_path=config.save_path, ood=config.ood)
