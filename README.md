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

You can optionally collect demonstrations via teleoperation using a keyboard or a 3Dconnexion SpaceMouse.

* Press **K** to save the current episode.
* Press **L** to reset without saving.

```bash
python latent_safety/takeoff/collect_demonstrations.py --headless --enable_camera
```

Alternatively, you can generate trajectories automatically by running Dreamer training with online rollouts.

---

## 🧠 2. Training the World Model (Dreamer)

Train the world model (Dreamer) either:

* **Offline** using collected demonstrations by specifying `offline_traindir` in `dreamerv3_torch/configs.yaml`, or
* **Online** through interactions with the environment.

Run the following command:

```bash
python latent_safety/train_dreamer.py --headless --enable_camera
```

---

## 🛡️ 3. Latent Reachability Analysis

Run latent-space reachability based on SAC:

```bash
python latent_safety/reachability/train_reach_sac_env_failure.py --headless --enable_camera
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
