
<div align="center">
  <h1>Dynamic Objects Relocalization in Changing Environments with Flow Matching</h1>

  <a href="https://www.linkedin.com/in/fra-arg/">Francesco Argenziano</a><sup>1,#</sup> ·
  <a href="https://mikes96.github.io/">Miguel Saavedra-Ruiz</a><sup>2,3,#</sup> ·
  <a href="https://sachamorin.github.io/">Sacha Morin</a><sup>2,3</sup> ·
  <a href="https://scholar.google.com/citations?user=xZwripcAAAAJ&hl=it&oi=ao">Daniele Nardi</a><sup>1</sup> ·
  <a href="https://liampaull.ca/">Liam Paull</a><sup>2,3</sup><br/>
  <sup>1</sup>Department of Computer, Control and Management Engineering, Sapienza University of Rome, Italy<br/>
  <sup>2</sup>DIRO, Université de Montréal, Montréal, QC, Canada · <sup>3</sup>Mila - Quebec AI Institute, Montréal, QC, Canada<br/>
  <sup>#</sup>Co‑first authors
  <br/><br/>

  <a href="https://arxiv.org/pdf/2509.16398"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-Here-red"></a>
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/License-MIT-yellow"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.13+-green">
  <img alt="Ubuntu 22.04" src="https://img.shields.io/badge/Ubuntu-22.04-E95420">
  <img alt="Ubuntu 24.04" src="https://img.shields.io/badge/Ubuntu-24.04-E95420">
</div>

---

Official code release for the paper *Dynamic Objects Relocalization in Changing Environments with Flow Matching* (code‑name: **FlowMaps**).

<!-- If you have a teaser image or GIF, put it here -->
<!-- <p align="center"><img src="assets/teaser.png" width="75%"/></p> -->

## Table of Contents
- [Installation](#installation)
- [Dataset Generation (FlowSim)](#dataset-generation-flowsim)
- [Configuration](#configuration)
- [Training & Logging](#training--logging)
- [Evaluation & Reproducing Results](#evaluation--reproducing-results)
- [Citation](#citation)
- [License](#license)

## Installation
```bash
# 1) Clone
git clone https://github.com/Fra-Tsuna/flowmaps
cd flowmaps

# 2) Create an environment (example with conda) and install deps
conda create -n flowmaps python=3.13 -y
conda activate flowmaps
pip install -r requirements.txt
```

## Dataset Generation (FlowSim)
To generate datasets with FlowSim:
```bash
python3 data.py
```
This creates training/validation data according to `config/config.yaml`.
We use a Hydra‑style syntax for overrides (`key=value`). You can edit `config/config.yaml` directly or pass overrides at the CLI.
```bash
python3 data.py n_env_train=200 n_env_val=20 max_timesteps=20 stochastic=true
```
Key parameters:
- `n_env_train`, `n_env_val`: number of training and validation environments (respectively).
- `max_timesteps`: time horizon of object trajectories.
- `stochastic`: enable stochastic object behavior.
- `size`: environment canvas size.
- `max_tables`, `min_each`: number of furniture instances and minimum per type.
- `display_scale`: **visualization only**; upscales the rendered image. Underlying data remain at **size×size**.


## Training & Logging
Our project supports **Weights & Biases** logging.
- **Online** (default below): 
  ```bash
  python3 main.py wandb.mode=online wandb.project=YOUR_PROJECT wandb.entity=YOUR_ENTITY wandb.tags="[flowmaps,cdit]" experiment=cdit
  ```
- **Offline**:
  ```bash
  python3 main.py wandb.mode=offline experiment=cdit
  ```

To train the MLP baseline, simply override `experiment=mlp`

## Evaluation & Reproducing Results
After training, place your checkpoint file in `ckpt/` and run:
```bash
python3 eval.py checkpoint_name=YOUR_CHECKPOINT_FILENAME
```

## Citation
If you find this work useful, please cite the paper (preprint coming soon):
```bibtex
@misc{argenziano2025flowmaps,
  title        = {Dynamic Objects Relocalization in Changing Environments with Flow Matching},
  author       = {Argenziano, Francesco and Saavedra-Ruiz, Miguel and Morin, Sacha and Nardi, Daniele and Paull, Liam},
  year         = {2025},
  eprint       = {to appear},
  archivePrefix= {arXiv}
}
```

## License
This repository is released under the **MIT License**. See [LICENSE](LICENSE) for details.
