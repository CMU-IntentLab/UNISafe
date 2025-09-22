# 🛡️ Latent Safety with Reachability Analysis

This repository provides the implementation of **Uncertainty-aware Latent Safety Filters** for avoiding out-of-distribution failures in robotics tasks using [Isaac Lab](https://isaac-sim.github.io/IsaacLab/).

---

## 📦 Installation

1. **Install Isaac Lab**  
   Follow the official [Isaac Lab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). (This repo uses stale isaacsim version 4.2.0, while the latests version is 5.x.x. We are working on updating the code to the latest version, and it only requires changing some of the import paths.)

2. **Clone and Set Up the Environment**

```bash
# Clone the repository
git clone https://github.com/CMU-IntentLab/UNISafe.git
git checkout isaaclab
cd latent_safety

# Create and activate the conda environment
conda env create -f environment.yaml
conda activate isaaclab
```

---

## 📥 Quick Start: Download Pretrained Models

For immediate evaluation and experimentation, download our pretrained models:[pretrained models](https://drive.google.com/file/d/1RddRw3eVUhufuUdq_BAThjwvO1fsmTeM/view?usp=sharing)
```bash
# Download pretrained models (world model + reachability filter)
pip install gdown
gdown https://drive.google.com/uc?id=1RddRw3eVUhufuUdq_BAThjwvO1fsmTeM
unzip pretrained_models.zip

# This will create:
# - dreamer.pt (pretrained world model)
# - filter/ (reachability filter directory)
#   └── model/ (filter checkpoints at different training steps)
```

**Directory Structure After Download:**
```
latent_safety/
├── log/                          # Centralized log directory
│   ├── dreamer.pt               # Pretrained world model
│   ├── filter/                  # Pretrained reachability filter
│   │   └── model/
│   ├── dreamerv3/              # World model training logs
│   ├── reachability/           # Reachability training logs
└── ... (other files)
```

---

## 🚀 Quick Evaluation with Pretrained Models

Once you have the pretrained models, you can immediately start evaluating. **Note**: The evaluation uses the policy learned within the world model during training - there is no separate policy model file.

<!-- ### 1. 📊 Quantitative Evaluation

```bash
# Evaluate reachability filter performance
python latent_safety/reachability/evaluate_reachability_filter.py \
    --headless \
    --enable_cameras \
    --configs filter_eval \
    --model_path "latent_safety/log/dreamer.pt" \
    --policy_model_path "latent_safety/log/dreamer.pt" \
    --reachability_model_path "latent_safety/log/filter" \
    --is_filter true \
    --num_episodes 100
``` -->

### 🎮 Qualitative Evaluation with Teleoperation

Experience the safety filter interactively:

```bash
# Run teleoperation with safety filter
python latent_safety/teleop_dreamer/filter_with_dreamer_failure.py \
    --enable_cameras \
    --model_path "latent_safety/log/dreamer.pt" \
    --reachability_model_path "latent_safety/log/filter"

# Controls:
# - Use keyboard (WASD, QE, RF) or SpaceMouse for teleoperation
# - Press K to save current episode
# - Press L to reset without saving
# - Watch the filter intervene when detecting unsafe actions
```

---

## 🏗️ Full Training Pipeline

For training your own models from scratch:

### 1. 🗂️ Data Collection (Optional)

You can collect your own demonstrations or use our provided datasets.

#### Option A: Manual Teleoperation
```bash
python latent_safety/takeoff/collect_demonstrations.py --headless --enable_cameras
```
- Press **K** to save the current episode
- Press **L** to reset without saving

#### Option B: Use Provided Datasets
Download our curated datasets:
- [Complete Dataset](https://drive.google.com/file/d/1gaLfQrR53Kiksd-uXRG-WqOSnPsipNya/view?usp=sharing) (successes + failures)
- [Success-Only Dataset](https://drive.google.com/file/d/14Ofq7gCEnPMZXY9K5lANNzxynyBfBHST/view?usp=sharing) (successes only)

```bash
# Download and extract dataset
unzip dataset.zip -d datasets/
```

### 2. 🧠 World Model Training

Train the world model (Dreamer) with both dynamics and policy learning:

```bash
python latent_safety/train_dreamer.py --headless --enable_cameras
```

**Configuration**: Update `dreamerv3_torch/configs.yaml`:
```yaml
# For offline training (model + policy from demonstrations)
offline_traindir: ["path/to/your/dataset"]
model_only: true

# For online training (model + policy through environment interaction)
model_only: false
```

**Important**: The world model training learns both:
1. **World dynamics** (environment simulation)
2. **Base policy** (task execution through imagination)
3. **Failure prediction** (safety classification)
4. **Uncertainty estimation** (epistemic uncertainty through ensemble)

**Optional Ensemble Fine-tuning**: After world model training, fine-tune the uncertainty ensemble:
- Uncomment `agent.train_uncertainty_only(training=True)` in `train_dreamer.py`
- Comment out `agent.train_model_only(training=True)`
- Train for additional 200K iterations

### 3. 🛡️ Reachability Filter Training

Train safety filters using the learned world model:

#### Option A: Full Safety Filter (Uncertainty + Failure Prediction)
```bash
python latent_safety/reachability/train_reachability_sac_with_failure_prediction.py \
    --headless \
    --enable_cameras \
    --model_path "path/to/dreamer.pt" \
    --configs failure_filter
```

#### Option B: Uncertainty-Only Filter  
```bash
python latent_safety/reachability/train_reachability_sac_uncertainty_only.py \
    --headless \
    --enable_cameras \
    --model_path "path/to/dreamer.pt" \
    --configs uncertainty_filter
```

**Configuration**: Update `latent_safety/reachability/config.yaml`:
```yaml
# Paths
model_path: "path/to/your/dreamer.pt"
offline_traindir: ["path/to/your/dataset"]

# Training parameters
maxUpdates: 200000
checkPeriod: 10000
```

---

## 📊 Evaluation Pipeline

### Quantitative Metrics

The evaluation script provides comprehensive safety metrics. **Important**: The evaluation uses the policy learned during world model training, not a separate pretrained policy.

```bash
python latent_safety/reachability/evaluate_reachability_filter.py \
    --model_path "latent_safety/log/dreamer.pt" \
    --policy_model_path "learned_dreamer_policy_path" \
    --reachability_model_path "latent_safety/log/filter" \
    --num_episodes 100 \
    --is_filter true
```


### Qualitative Analysis

#### Interactive Teleoperation with Filter
```bash
python latent_safety/teleop_dreamer/filter_with_dreamer_failure.py \
    --enable_cameras \
    --model_path "latent_safety/log/dreamer.pt" \
    --reachability_model_path "latent_safety/log/filter"
```

#### Base Policy Rollouts (No Filter)
```bash
python latent_safety/teleop_dreamer/rollout_with_dreamer.py \
    --enable_cameras \
    --model_path "path/to/dreamer.pt"
```

---

## 🔧 Configuration Guide

### World Model Configuration (`dreamerv3_torch/configs.yaml`)
```yaml
defaults:
  # Training mode
  model_only: true              # true for offline, false for online
  
  # Data paths
  offline_traindir: ["path/to/dataset"]
  
  # Model architecture
  dyn_stoch: 32                 # Stochastic state size
  dyn_deter: 512                # Deterministic state size
  dyn_discrete: 32              # Discrete components
  
  # Training
  train_steps: 1000000          # Total training steps
  batch_size: 16                # Batch size
  batch_length: 64              # Sequence length
```

### Reachability Filter Configuration (`reachability/config.yaml`)
```yaml
defaults:
  # Model paths
  model_path: "path/to/dreamer.pt"
  offline_traindir: ["path/to/dataset"]
  
  # Network architecture
  control_net: [256, 256]       # Actor network
  critic_net: [256, 256]        # Critic network
  
  # Training parameters
  maxUpdates: 200000            # Training updates
  checkPeriod: 10000            # Checkpoint frequency
  
  # Safety parameters
  ood_threshold: 0.5            # Uncertainty threshold
  use_uq: true                  # Enable uncertainty quantification
```


## 🙏 Acknowledgements

This implementation builds on the following open-source projects:

1. [dreamerv3-pytorch](https://github.com/NM512/dreamerv3-torch) - World model implementation
2. [HJReachability](https://github.com/HJReachability/safety_rl/) - Reachability analysis
3. [PENN](https://github.com/tkkim-robot/online_adaptive_cbf/tree/main/nn_model/penn) - Uncertainty estimation
4. [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) - Robotics simulation platform

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{seo2025uncertainty,
  title={Uncertainty-aware Latent Safety Filters for Avoiding Out-of-Distribution Failures},
  author={Seo, Junwon and Nakamura, Kensuke and Bajcsy, Andrea},
  journal={Conference on Robot Learning (CoRL)},
  year={2025}
}
```
