# 🏗️ Block Plucking with UNISafe

This repository provides the implementation of **Block Plucking** with [UNISafe](https://cmu-intentlab.github.io/UNISafe/).  

> **Note**: The code release is **in progress**. You may encounter incomplete features or runtime errors.
---

## 📦 Installation

1. **Install Isaac Lab**  
   Follow the official [Isaac Lab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

2. **Clone and Set Up the Environment**

```bash
# Clone the repository
git clone https://github.com/CMU-IntentLab/UNISafe.git
git checkout isaaclab
cd latent_safety

# Create and activate the conda environment
conda env create -f environment.yaml
conda activate isaaclab
````

---

## 🗂️ 1. Collecting Demonstrations (Optional)

- You can optionally collect demonstrations via teleoperation using a keyboard or a 3Dconnexion SpaceMouse.

  * Press **K** to save the current episode.
  * Press **L** to reset without saving.

  ```bash
  python latent_safety/takeoff/collect_demonstrations.py --headless --enable_camera
  ```

- Alternatively, you can generate trajectories automatically by running Dreamer training with online rollouts.

- You can also use the provided offline datasets, available at the following links: [dataset-all](https://drive.google.com/file/d/1gaLfQrR53Kiksd-uXRG-WqOSnPsipNya/view?usp=sharing) and [dataset-success_only](https://drive.google.com/file/d/14Ofq7gCEnPMZXY9K5lANNzxynyBfBHST/view?usp=sharing). These datasets contain either both successes and failures, or only successes without any failure demonstrations, respectively.

  - To use these datasets, set `model_only=True` in `configs.yaml`, and point `offline_traindir` in `dreamerv3_torch/configs.yaml` to the unzipped dataset directory.

---

## 🧠 2. Training the World Model (Dreamer)

Train the world model (Dreamer) either:

* **Offline** using collected demonstrations by specifying `offline_traindir` in `dreamerv3_torch/configs.yaml`, or
* **Online** through interactions with the environment.

Run the following command:

```bash
python latent_safety/train_dreamer.py --headless --enable_camera
```

Running this will save the online trajectories (failures and successes, excluding timeouts) to the log directory under `eval_eps` or `train_eps`, which can be used for training an ensemble or for reachability analysis.

- You can (optionally) finetune the ensemble using the same dataset by uncommenting `agent.train_uncertainty_only(training=True)` in `latent_safety/train_dreamer.py`, and commenting `agent.train_model_only(training=True)` in the same file. In the paper, we finetuned the ensemble for 200K iterations after the initial training.

---

## 🛡️ 3. Latent Reachability Analysis

- To perform latent reachability analysis, you should set `model_path` in `latent_safety/reachability/config.yaml` to the path of the trained Dreamer model and set `offline_traindir` to the directory containing the offline dataset. This offline dataset is used as for setting initial states for the reachability analysis, while reachability analysis is performed entirely in the imgaination of the world model.

- Run latent-space reachability based on SAC:

```bash
python latent_safety/reachability/train_reach_sac_env_failure.py --headless --enable_camera
```

- To learn success-only safety filter, run the following command with proper `model_path` and `offline_traindir` set in `latent_safety/reachability/config.yaml`. This defines the failure margin only based on the ensemble disagreement without a learned failure classifier:
```bash
python latent_safety/reachability/train_reach_sac_env.py --headless --enable_camera
```

---

## 🙏 Acknowledgements

This implementation builds on the following open-source projects:

1. [dreamerv3-pytorch](https://github.com/NM512/dreamerv3-torch)
2. [HJReachability](https://github.com/HJReachability/safety_rl/)
3. [PENN](https://github.com/tkkim-robot/online_adaptive_cbf/tree/main/nn_model/penn)


---

## 📄 Citation
If you build upon this work, please consider citing our research.


```bibtex
@article{seo2025uncertainty,
  title={Uncertainty-aware Latent Safety Filters for Avoiding Out-of-Distribution Failures},
  author={Seo, Junwon and Nakamura, Kensuke and Bajcsy, Andrea},
  journal={arXiv preprint arXiv:2505.00779},
  year={2025}
}
```


## ✅ TODO
* [ ] Release pretrained checkpoints.
* [x] Release offline datasets.
* [ ] Update the README for filtering a base policy with the learned safety filter.