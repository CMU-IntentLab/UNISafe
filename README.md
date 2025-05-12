# 🏎️ Dubins Car


## 📦 Installation

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage dependencies.

```bash
# Clone the repository
git clone https://github.com/your-username/dubins-latent-safety.git
cd dubins-latent-safety

# Create and activate conda environment
conda env create -f environment.yaml
conda activate dubins_latent_safety
```


## 🗂️ Dataset Generation

Generate synthetic expert and random trajectory data for training the world model.

```bash
bash scripts/generate_dataset.sh
```

This script internally runs:

```python
python scripts/generate_data_traj_failure_expert.py \
  --num_pts {num_expert_traj} \
  --num_random {num_random_traj} \
  --save_path {save_path}
```

Modify scripts/generate_dataset.sh to configure dataset parameters as needed.


## 🧠 World Model Training

Train the latent dynamics world model using the generated dataset.

```bash
bash scripts/train_wm.sh
```

Model checkpoints and logs will be saved in the specified logging directory.

## 🛡️ Latent Reachability Analysis

Run RL for latent-space reachability:

```bash
bash scripts/train_reachability.sh
```

## 🙏 Acknowledgements

This implementation builds on the following open-source projects:

1. [dreamerv3-pytorch](https://github.com/NM512/dreamerv3-torch)
2. [HJReachability](https://github.com/HJReachability/safety_rl/)
3. [PENN](https://github.com/tkkim-robot/online_adaptive_cbf/tree/main/nn_model/penn)

If you build upon this work, please consider citing our research.


📄 Citation

```
@article{seo2025uncertainty,
        title={Uncertainty-aware Latent Safety Filters for Avoiding Out-of-Distribution Failures},
        author={Seo, Junwon and Nakamura, Kensuke and Bajcsy, Andrea},
        journal={arXiv preprint arXiv:2505.00779},
        year={2025}
      }
```
