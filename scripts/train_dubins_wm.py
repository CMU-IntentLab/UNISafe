import argparse
import collections
import functools
import pathlib
import sys
from datetime import datetime
import os 
import numpy as np
import ruamel.yaml as yaml
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import gym
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
sys.path.append('model_based_irl_torch')
sys.path.append('safety_rl')

from safety_rl.gym_reachability import gym_reachability  # Custom Gym env.
import model_based_irl_torch.dreamer.tools as tools
from model_based_irl_torch.common.constants import HORIZONS
from model_based_irl_torch.common.utils import (
    to_np,
)
from model_based_irl_torch.dreamer.dreamer import Dreamer

class Learner:
    def __init__(self, config):
        self.config = config
        tools.set_seed_everywhere(config.seed)
        if config.deterministic_run:
            tools.enable_deterministic_run()
        self.logdir = pathlib.Path(config.logdir).expanduser()
        self.config.steps //= config.action_repeat
        self.config.eval_every //= config.action_repeat
        self.config.log_every //= config.action_repeat
        self.config.time_limit //= config.action_repeat

        self.logger = self.setup_logging()
        self.train_envs = self.create_envs()
        self.setup_datasets()
        self.agent = self.create_agent()

        if config.resume :
            checkpoint = torch.load(config.resume, map_location=self.config.device)
            agent_state_dict = checkpoint['agent_state_dict']
            
            # All
            self.agent.load_state_dict(agent_state_dict)
            print("successfully loaded state dict from {}".format(config.resume))
            del checkpoint, agent_state_dict #, filtered_state_dict


    def setup_logging(self):
        print("Logdir", self.logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        with open(f"{self.logdir}/config.yaml", "w") as f:
            yaml.dump(vars(self.config), f)
        self.config.traindir = self.config.traindir or self.logdir / "train_eps"
        self.config.evaldir = self.config.evaldir or self.logdir / "eval_eps"
        self.config.traindir.mkdir(parents=True, exist_ok=True)
        self.config.evaldir.mkdir(parents=True, exist_ok=True)
        step = self.count_steps(self.config.traindir)
        logger = tools.DebugLogger(self.logdir, self.config.action_repeat * step) if self.config.debug else tools.Logger(self.logdir, self.config.action_repeat * step)
        logger.config(vars(self.config))
        logger.write()
        return logger

    def setup_datasets(self):
        expert_eps = collections.OrderedDict()
        tools.fill_expert_dataset_dubins(self.config, expert_eps, predefined_path=self.config.dataset_path)
        self.expert_dataset = self.make_dataset(expert_eps)

        expert_val_eps = collections.OrderedDict()
        tools.fill_expert_dataset_dubins(self.config, expert_val_eps, predefined_path=self.config.eval_dataset_path)
        self.eval_dataset = self.make_dataset(expert_val_eps)

        obs_train_eps, obs_eval_eps = collections.OrderedDict(), collections.OrderedDict()
        for i, (key, value) in enumerate(expert_eps.items()):
            if i < int(len(expert_eps) * 0.9):
                obs_train_eps[key] = value
            else:
                obs_eval_eps[key] = value

        self.obs_train_dataset = self.make_dataset(obs_train_eps)
        self.obs_eval_dataset = self.make_dataset(obs_eval_eps)
        
    def create_envs(self):
        env_name = "dubins_car_img-v1"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sample_inside_obs = self.config.doneType not in ['TF', 'fail']
        train_env = gym.make(env_name, config=self.config, device=device, mode=self.config.mode, doneType=self.config.doneType, sample_inside_obs=sample_inside_obs)
        return [train_env]

    def create_agent(self):
        agent = Dreamer(self.train_envs[0].observation_space, self.train_envs[0].action_space, self.config, self.logger, self.expert_dataset).to(self.config.device)

        agent.requires_grad_(requires_grad=False)
        return agent

    def train_eval(self):
        total_pretrain_steps = self.config.pretrain_joint_steps
        best_pretrain_success = float("inf")

        for step in trange(total_pretrain_steps, desc="Encoder + Actor pretraining", ncols=0, leave=False):
            if self.config.eval_num_seeds > 0 and ((step + 1) % self.config.eval_every) == 0:
                print('eval')
                self.evaluate(other_dataset=self.expert_dataset, eval_prefix="pretrain")
                tools.save_checkpoint("pretrain_joint", step=step, score=None, best_score=best_pretrain_success, agent=self.agent, logdir=self.logdir)
            
            exp_data = next(self.expert_dataset)

            if self.config.ensemble_only:
                self.agent.pretrain_disagreement_ensemble_only(exp_data, step)
            
            else:
                self.agent.pretrain_model_only(exp_data, step)

        self.close_envs(self.train_envs)

    def train_reward(self, ckpt_name='classifier'):
        print('training l(x)')
        recon_steps = 2501
        best_pretrain_success_classifier = float("inf")
        lx_mlp, lx_opt = self.agent._wm._init_lx_mlp(self.config, 1)
        train_loss = []
        eval_loss = []

        for i in tqdm(range(1, recon_steps), total=recon_steps):
            if i % 250 == 0:
                print('eval')
                new_loss, eval_plot = self.agent.train_lx(
                    next(self.obs_eval_dataset), lx_mlp, lx_opt, eval=True
                )
                eval_loss.append(new_loss)
                self.logger.image("classifier", np.transpose(eval_plot, (2, 0, 1)))
                self.logger.write(step=i+40000)
                best_pretrain_success_classifier = tools.save_checkpoint(
                    ckpt_name, i, new_loss, best_pretrain_success_classifier, lx_mlp, self.logdir
                )

            else:
                new_loss, _ = self.agent.train_lx(
                    next(self.obs_train_dataset), lx_mlp, lx_opt
                )
                train_loss.append(new_loss)

        self.log_plot("train_lx_loss", train_loss)
        self.log_plot("eval_lx_loss", eval_loss)
        self.logger.scalar("pretrain/train_lx_loss_min", np.min(train_loss))
        self.logger.scalar("pretrain/eval_lx_loss_min", np.min(eval_loss))
        self.logger.write(step=i)
        print(eval_loss)
        print('logged')
        return lx_mlp, lx_opt

    def close_envs(self, envs):
        for env in envs:
            try:
                env.close()
            except Exception:
                pass

    def log_plot(self, title, data):
        buf = BytesIO()
        plt.plot(np.arange(len(data)), data)
        plt.title(title)
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        plot = Image.open(buf).convert("RGB")
        plot_arr = np.array(plot)
        self.logger.image("pretrain/" + title, np.transpose(plot_arr, (2, 0, 1)))

    def evaluate(self, other_dataset=None, eval_prefix=""):
        self.agent.eval()
        if self.config.video_pred_log:
            video_pred = self.agent._wm.video_pred(next(self.eval_dataset))
            self.logger.video("eval_recon/openl_agent", to_np(video_pred))
            if other_dataset:
                video_pred = self.agent._wm.video_pred(next(other_dataset))
                self.logger.video("train_recon/openl_agent", to_np(video_pred))
        self.logger.scalar(f"{eval_prefix}/eval_episodes", self.config.eval_num_seeds * self.config.eval_per_seed)
        self.logger.write(step=self.logger.step)
        self.agent.train()
        return

    @staticmethod
    def count_steps(folder):
        return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))

    def make_dataset(self, episodes):
        generator = tools.sample_episodes(episodes, self.config.batch_length)
        dataset = tools.from_generator(generator, self.config.batch_size)
        return dataset

    @staticmethod
    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                Learner.recursive_update(base[key], value)
            else:
                base[key] = value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--expt_name", type=str, default=None)
    parser.add_argument("--remark", type=str, default="")
    parser.add_argument("--no_lx", action="store_false", default=True)
    parser.add_argument("--resume", type=str, default="")
    config, remaining = parser.parse_known_args()


    curr_time = datetime.now().strftime("%m%d/%H%M%S")
    config.expt_name = f"{curr_time}_{config.expt_name}" if config.expt_name else curr_time
    config.expt_name += ( "_" + config.remark)

    yaml = yaml.YAML(typ="safe", pure=True)
    configs = yaml.load((pathlib.Path(sys.argv[0]).parent / "../configs/config.yaml").read_text())
    name_list = ["defaults", *config.configs] if config.configs else ["defaults"]
    defaults = {}

    # Set initial argument values as defaults
    for key, value in vars(config).items():
        if value is not None:
            defaults[key] = value  # Merge initial parsed values into defaults

    for name in name_list:
        Learner.recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()


    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    final_config = parser.parse_args(remaining)
    final_config.logdir = f"{final_config.logdir}/{config.expt_name}"
    final_config.time_limit = HORIZONS[final_config.task.split("_")[-1]]

    learner = Learner(final_config)
    learner.train_eval()

    if final_config.no_lx:    
        learner.train_reward()
