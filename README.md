<div align="center">
    <h1><span style="color: #ff9500; font-style: italic; font-weight: bold;">UNISafe:</span> Uncertainty-aware Latent Safety Filters for Avoiding Out-of-Distribution Failures
</h1>
    <a href="https://cmu-intentlab.github.io/UNISafe/">Homepage</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://www.arxiv.org/abs/2505.00779">Paper</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://youtu.be/Li9jCixTPXw">Video</a>
    <br />
</div>

---

This is a repository for [Uncertainty-aware Latent Safety Filters for Avoiding Out-of-Distribution Failures](https://cmu-intentlab.github.io/UNISafe/).   

<p align="center">
 <img width="1200" src="imgs/main_compressed.png" style="background-color:white;" alt="framework">
 <br />
 <em></em>
</p>

## 📂 Code Structure

```bash
git clone https://github.com/CMU-IntentLab/UNISafe.git
cd UNISafe
```

The project is organized into separate branches:

* **`dubins`**: 3D Dubins Car. [Link](https://github.com/CMU-IntentLab/UNISafe/tree/dubins)

```bash
git checkout dubins
```

* **`isaaclab`**: Block-plucking tasks implemented in NVIDIA IsaacLab. [Link](https://github.com/CMU-IntentLab/UNISafe/tree/isaaclab)

```bash
git checkout isaaclab
```


## 🙏 Acknowledgements

This implementation builds on the following open-source projects:

1. [dreamerv3-pytorch](https://github.com/NM512/dreamerv3-torch)
2. [HJReachability](https://github.com/HJReachability/safety_rl/)
3. [PENN](https://github.com/tkkim-robot/online_adaptive_cbf/tree/main/nn_model/penn)




## 📄 Citation
If you build upon this work, please consider citing our research.
```
@article{seo2025uncertainty,
        title={Uncertainty-aware Latent Safety Filters for Avoiding Out-of-Distribution Failures},
        author={Seo, Junwon and Nakamura, Kensuke and Bajcsy, Andrea},
        journal={arXiv preprint arXiv:2505.00779},
        year={2025}
      }
```