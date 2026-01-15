
<div align="center">

# 🚗 dVLM-AD

### Diffusion-based Vision-Language Models for Autonomous Driving

[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://dvlm-ad.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](https://arxiv.org/abs/2512.04459)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](./LICENSE)

</div>

**dVLM-AD** formulates autonomous driving decision-making as a **conditional diffusion process over actions**, enabling bidirectional context reasoning, improved robustness to uncertainty, and stronger reasoning–action consistency compared to autoregressive vision-language models.

For motivation, qualitative examples, and benchmark evaluations, please refer to the project website:  
👉 https://dvlm-ad.github.io/

---

## Environment Setup

We recommend using **conda** to manage the environment.

### Create and activate environment

```bash
conda create -n dvlm python=3.10 -y
bash init_env.sh
````

---

## Running Inference

### Prepare model checkpoint

Download the [checkpoint](https://huggingface.co/datasets/gray311/dVLM-AD/tree/main) and place it under:

```text
checkpoints/
└── dvlm_ad.ckpt
```

Checkpoint download links will be provided on the project website.

---

### Run inference

```bash
cd eval
python inference.py \
```

This script will generate:

* Driving action trajectories
* (Optional) reasoning traces associated with each trajectory

---



## Citation

If you find this work useful, please consider citing:

```bibtex
@article{ma2025dvlm,
  title={dVLM-AD: Enhance Diffusion Vision-Language-Model for Driving via Controllable Reasoning},
  author={Ma, Yingzi and Cao, Yulong and Ding, Wenhao and Zhang, Shuibai and Wang, Yan and Ivanovic, Boris and Jiang, Ming and Pavone, Marco and Xiao, Chaowei},
  journal={arXiv preprint arXiv:2512.04459},
  year={2025}
}
```

