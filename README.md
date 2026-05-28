# flow-sim
Landmark-based simulator for the Flow-Maps project

## Data layout

Datasets are organised by duration mode under `data/`:

```
data/
├── gaussian/          # original data (mode: gaussian)
│   ├── train/
│   │   ├── env0/
│   │   ├── env1/
│   │   └── ...
│   ├── val/
│   ├── minival/       # symlinks into val/ for envs with topdown renders
│   ├── latents/       # VAE checkpoint + latent statistics
│   │   ├── vae.pth
│   │   └── latent_statistics.npz
│   └── topdown/       # AI2-THOR top-down renders
│       ├── val/
│       │   ├── env0/
│       │   └── ...
│       └── minival/   # symlinks into topdown/val/
└── even/              # balanced data (mode: even)
    ├── train/
    ├── val/
    ├── minival/
    ├── latents/
    └── topdown/
        └── val/
```

Each env directory contains `scan_merged.parquet`, `config.yaml`, `pickupable_names.json`, `receptacle_names.json`, and `pickupable_to_receptacle.json`. Checkpoints live under `ckpt/{mode}/dit.pth` and `ckpt/{mode}/vae.pth`.

All Hydra commands accept `mode=<mode>` to select the dataset (`gaussian` or `even`). This derives `data_root` and `ckpt_root` automatically. Scripts that take a `--data_root` argument still accept it directly.

## Full pipeline: start to finish

All commands assume `MODE=gaussian` (replace as needed). `mode` defaults to `gaussian`. There is no longer an `env_name` parameter — all envs in the split directory are loaded automatically.

### 1. Merge scan parquets

Raw scans from SemiStaticSim arrive as `scan_0.parquet`, `scan_1.parquet`, … per environment.
Merge them into a single `scan_merged.parquet`:

```bash
python scripts/merge_scan_parquets.py --data-root ./data/$MODE
```

### 2. Pre-train the VAE

```bash
# Local
python slurm.py experiment=vae mode=$MODE wandb.mode=offline

# Cluster (SLURM)
python slurm.py --multirun hydra/launcher=remote +hydra/sweep=remote \
  experiment=vae mode=$MODE wandb.mode=online wandb.tags=["vae"]
```

The checkpoint is saved by Hydra to `outputs/<date>/<time>/checkpoints/best.pth` (best val loss) and `last.pth` (final iteration). Copy it manually:

```bash
cp outputs/<date>/<time>/checkpoints/best.pth ckpt/$MODE/vae.pth
```

### 3. Compute latent statistics

Required before DiT training. Encodes all training data through the VAE and saves mean/std:

```bash
python scripts/compute_statistics.py mode=$MODE
```

Writes `data/${MODE}/latents/latent_statistics.npz` and copies the VAE checkpoint to `data/${MODE}/latents/vae.pth`.

### 4. Train the DiT

```bash
# Local
python slurm.py experiment=dit mode=$MODE wandb.mode=offline

# Cluster (SLURM)
python slurm.py --multirun hydra/launcher=remote +hydra/sweep=remote \
  experiment=dit mode=$MODE wandb.mode=online wandb.tags=["dit"]
```

Same as VAE, copy from the Hydra output directory:

```bash
cp outputs/<date>/<time>/checkpoints/best.pth ckpt/$MODE/dit.pth
```

### 5. Run eval

```bash
python eval.py mode=$MODE
```

Loads `ckpt/${MODE}/dit.pth`, samples 5 scenes from the minival set, generates 25 predictions each, and saves BEV PNGs to `eval/png/`. Key overrides:

```bash
python eval.py mode=$MODE \
  nsamples=10 \
  npreds=50 \
  log_path=./eval/${MODE}
```