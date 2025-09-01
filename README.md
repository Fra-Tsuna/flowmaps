<div align="center">
<!-- <h1 style="font-size: 50px">Context Matters!</h1>  -->
<h1>Dynamic Objects Relocalization in Changing Environments with Flow Matching</h1>

<a href="https://www.linkedin.com/in/fra-arg/">Francesco Argenziano</a><sup><span>1,#</span></sup>,
<a href="https://mikes96.github.io/">Miguel Saavedra-Ruiz</a><sup><span>2,3,#</span></sup>,
<a href="https://scholar.google.com/citations?user=sk3SpmUAAAAJ&hl=it&oi=ao">Sacha Morin</a><sup><span>2,3</span></sup>,
<a href="https://scholar.google.com/citations?user=xZwripcAAAAJ&hl=it&oi=ao">Daniele Nardi</a><sup><span>1</span></sup>,
<a href="https://liampaull.ca/">Liam Paull</a><sup><span>2,3</span></sup>
</br>

<sup>1</sup> Department of Computer, Control and Management Engineering, Sapienza University of Rome, Rome, Italy, <br>
<sup>2</sup> Department of Computer Science and Operations Research, Université de Montréal, Montréal, QC, Canada. <br>
<sup>3</sup> Mila - Quebec AI Institute, Montréal, QC, Canada. <br>
<sup>#</sup> First Co-Authors
<div>

[![arxiv paper](https://img.shields.io/badge/arXiv-SOON-red)](https://github.com/Fra-Tsuna/flowmaps)
[![license](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
![flat](https://img.shields.io/badge/python-3.13+-green)
![flat](https://img.shields.io/badge/Ubuntu-22.04-E95420)
![flat](https://img.shields.io/badge/Ubuntu-24.04-E95420)

# Description
Official release repository for the paper *Dynamic Objects Relocalization in Changing Environments with Flow Matching*

# Install

1. Clone this repo
```
git clone https://github.com/Fra-Tsuna/flowmaps
```

2. Setup a virtual environment (conda, venv, ...) and install the requirements.txt

```
conda create -n flowmaps python=3.13
pip install -r requirements.txt
```

# Generate dataset
To generate the dataset with FlowSim

```
python3 data.py
```

This will create training and validation data in according with the parameters specified in `config/config.yaml`. It is possible to change the parameters either by editing the file directly, or by modifying them inline while launching the command. 
E.g.:
```
python3 data.py n_env_train=200 n_env_val=20 max_timesteps=20 stochastic=true
```
- `n_env_train` and `n_env_val` set respectevely the number of training and validation environments
- `max_timesteps` defines the time-horizon of objects trajectories
- `stochastic` sets if we want a stochastic behavior of for the objects

Other parameters:
- `size` sets the environment's size
- `max_tables` and `min_each` define how many furnitures and how many of each type we can spawn in the env
- `display_scale` only for visualization, upscales the rendered image accordingly

# Run
Our project supports `wandb` logging.
To run with wandb, launch:

```
python3 main.py wandb.mode=online wandb.project=... wandb.entity=... wandb.tags=... experiment=cdit
```

To launch it locally, change `wandb.mode` to `offline`.

# Results

To reproduce the results of our paper, after training the model and moving the checkpoint to `ckpt/`

```
python3 eval.py checkpoint_name=...
```