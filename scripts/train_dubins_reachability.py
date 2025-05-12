import os
import sys
import argparse
import time
from datetime import datetime
from warnings import simplefilter
from pathlib import Path
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import ruamel.yaml as yaml
from termcolor import cprint
import wandb 
import collections

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
sys.path.append('model_based_irl_torch')
sys.path.append('safety_rl')

from safety_rl.RARL.DDQNSingle import DDQNSingle
from safety_rl.RARL.config import dqnConfig
from safety_rl.RARL.utils import save_obj
from safety_rl.gym_reachability import gym_reachability
import model_based_irl_torch.dreamer.models as models
import model_based_irl_torch.dreamer.tools as tools
import model_based_irl_torch.dreamer.exploration as expl

matplotlib.use('Agg')
simplefilter(action='ignore', category=FutureWarning)

timestr = time.strftime("%Y-%m-%d-%H_%M")

class RARLAgent:
    def __init__(self, config):
        self.config = config
        self.timestr = time.strftime("%Y-%m-%d-%H_%M")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_output_folders()
        self.initialize_environment()
        self.initialize_agent()

    def setup_output_folders(self):
        fn = f"{self.config.name}-{self.config.doneType}"
        if self.config.showTime:
            fn += f"-{self.timestr}"
        if self.config.learnedMargin:
            fn += "-lm"
        if self.config.learnedDyn:
            fn += "-ld"
        if self.config.image:
            fn += "-img"
        if self.config.gt_lx:
            fn += "-gtlx"

        self.outFolder = os.path.join(self.config.outFolder, self.config.remark, fn)
        self.figureFolder = os.path.join(self.outFolder, 'figure')
        os.makedirs(self.figureFolder, exist_ok=True)

    def initialize_environment(self):

        env_name = "dubins_car_img-v1"

        print("\n== Environment Information ==")
        sample_inside_obs = self.config.doneType not in ['TF', 'fail']
        self.env = gym.make(
            env_name, config=self.config, device=self.device, mode=self.config.mode,
            doneType=self.config.doneType, sample_inside_obs=sample_inside_obs
        )

        if self.config.wm:
            self.setup_world_model()

    def setup_world_model(self):
        wm = models.WorldModel(self.env.observation_space, self.env.action_space, 0, self.config)
        wm_checkpoint = torch.load(self.config.wm_ckpt)
        wm.dynamics.sample = False
        state_dict = {k[4:]: v for k, v in wm_checkpoint['agent_state_dict'].items() if '_wm' in k}
        wm.load_state_dict(state_dict)

        lx_mlp, _ = wm._init_lx_mlp(self.config, 1)
        lx_ckpt = torch.load(self.config.lx_ckpt)
        lx_mlp.load_state_dict(lx_ckpt['agent_state_dict'])
        self.env.car.set_wm(wm, lx_mlp, self.config)

        if self.config.use_ensemble:
            disag_ensemble = expl.OneStepPredictor(self.config, wm)
            state_dict = {k[16:]: v for k, v in wm_checkpoint['agent_state_dict'].items() if '_disag_ensemble' in k}
            disag_ensemble.load_state_dict(state_dict)
            self.env.car.set_ensemble(disag_ensemble, self.config)

    def make_dataset(self, episodes):
        generator = tools.sample_episodes(episodes, self.config.batch_length)
        dataset = tools.from_generator(generator, 16)
        return dataset

    def initialize_agent(self):
        maxUpdates = self.config.maxUpdates
        updatePeriod = int(maxUpdates / self.config.updateTimes)
        EPS_PERIOD = int(updatePeriod / 10)
        EPS_RESET_PERIOD = maxUpdates if not self.config.annealing else updatePeriod

        stateDim = self.env.state.shape[0] if not self.config.wm else (
            self.config.dyn_stoch + self.config.dyn_deter if not self.config.dyn_discrete else self.config.dyn_stoch * self.config.dyn_discrete + self.config.dyn_deter
        )
        actionNum = self.env.action_space.n
        

        CONFIG = dqnConfig(
        DEVICE=self.device, ENV_NAME="dubins_car_img-v1", SEED=self.config.randomSeed,
        MAX_UPDATES=maxUpdates, MAX_EP_STEPS=self.config.numT, BATCH_SIZE=self.config.reachability_batch_size,
        MEMORY_CAPACITY=self.config.memoryCapacity, ARCHITECTURE=self.config.architecture,
        ACTIVATION=self.config.actType, GAMMA=self.config.gamma, GAMMA_PERIOD=updatePeriod,
        GAMMA_END=self.config.gamma if not self.config.annealing else 0.9999,
        EPS_PERIOD=EPS_PERIOD, EPS_DECAY=0.7,
        EPS_RESET_PERIOD=EPS_RESET_PERIOD, LR_C=self.config.learningRate,
        LR_C_PERIOD=updatePeriod, LR_C_DECAY=0.8, MAX_MODEL=50
        )
        dimList = [stateDim] + list(CONFIG.ARCHITECTURE) + [actionNum]

        self.agent = DDQNSingle(
            CONFIG, actionNum, np.arange(actionNum), dimList=dimList, mode=self.config.mode,
            terminalType=self.config.terminalType
        )

    def warmup_agent(self):

        if hasattr(self.config, 'Q_pretrained') and self.config.Q_pretrained is not None :
            q_ckpt = torch.load(self.config.Q_pretrained)
            self.agent.Q_network.load_state_dict(q_ckpt)
            self.agent.target_network.load_state_dict(q_ckpt)
            self.agent.build_optimizer()
            print(f"Initial Model loaded from {self.config.Q_pretrained}")
            return
        
        if self.config.warmup:
            print("\n== Warmup Q ==")
            lossList = self.agent.initQ(
                self.env, self.config.warmupIter, self.outFolder, num_warmup_samples=200,
                vmin=-2, vmax=2, plotFigure=self.config.plotFigure, storeFigure=self.config.storeFigure
            )
            self.plot_warmup_loss(lossList)

    def plot_warmup_loss(self, lossList):
        tmp = np.arange(25, self.config.warmupIter)
        wandb.log({"initQ_Loss": {"Iteration": tmp, "Loss": lossList[tmp]}})

        if self.config.plotFigure or self.config.storeFigure:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            tmp = np.arange(25, self.config.warmupIter)
            ax.plot(tmp, lossList[tmp], 'b-')
            ax.set_xlabel('Iteration', fontsize=18)
            ax.set_ylabel('Loss', fontsize=18)
            plt.tight_layout()
            if self.config.storeFigure:
                wandb.log({"initQ_Loss": wandb.Image(fig)})
            if self.config.plotFigure:
                plt.show()
                plt.pause(0.001)
            plt.close()

    def train_agent(self):
        print("\n== Training Information ==")
        trainRecords, trainProgress = self.agent.learn(
            self.env, MAX_UPDATES=self.config.maxUpdates, MAX_EP_STEPS=self.config.numT,
            warmupBuffer=True, warmupQ=False, doneTerminate=True, vmin=-2, vmax=2, showBool=False,
            checkPeriod=self.config.checkPeriod, outFolder=self.outFolder,
            plotFigure=self.config.plotFigure, storeFigure=self.config.storeFigure
        )

    def execute(self):
        self.warmup_agent()
        self.train_agent()
        
def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

def custom_exclude_fn(path, root=None):
    """
    Excludes all non-.py files and any files within the 'logs/' subdirectory.
    """
    normalized_path = path.replace(os.sep, '/')
    
    # Exclude any files within the 'logs/' directory
    if normalized_path.startswith("logs/"):
        return True
    
    # Include all other .py files
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--expt_name", type=str, default=None)
    parser.add_argument("--resume_run", type=bool, default=False)
    parser.add_argument("--remark", type=str, default=None)
    parser.add_argument("--wm_ckpt", type=str, default=None)
    parser.add_argument("--lx_ckpt", type=str, default=None)
    config, remaining = parser.parse_known_args()

    if not config.resume_run:
        curr_time = datetime.now().strftime("%m%d/%H%M%S")
        config.expt_name = f"{curr_time}_{config.expt_name}" if config.expt_name else curr_time
        if config.remark is not None:
            config.expt_name += ("_" + config.remark)
    else:
        assert config.expt_name, "Need to provide experiment name to resume run."

    yaml = yaml.YAML(typ="safe", pure=True)
    configs = yaml.load((Path(sys.argv[0]).parent / "../configs/config.yaml").read_text())
    name_list = ["defaults", *config.configs] if config.configs else ["defaults"]

    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    # Set initial argument values as defaults
    for key, value in vars(config).items():
        if value is not None:
            defaults[key] = value  # Merge initial parsed values into defaults


    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
        
    final_config = parser.parse_args(remaining)
    final_config.outFolder = f"{final_config.outFolder}/{config.expt_name}"

    print("---------------------")
    cprint(f"Experiment name: {config.expt_name}", "red", attrs=["bold"])
    cprint(f"Task: {final_config.task}", "cyan", attrs=["bold"])
    cprint(f"Logging to: {final_config.outFolder}", "cyan", attrs=["bold"])
    print("---------------------")

    wandb.init(project="dubins_reachability", name=config.expt_name)
    wandb.run.log_code("../safety_rl", exclude_fn=custom_exclude_fn)
    # Changes for wandb: Log configuration values
    wandb.config.update(final_config)

    rarl_agent = RARLAgent(final_config)
    rarl_agent.execute()

    wandb.finish()  # Finish the wandb run