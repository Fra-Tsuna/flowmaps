# FlowMaps

**FlowMaps: Modeling Long-Term Multimodal Object Dynamics with Flow Matching**

FlowMaps is a latent flow matching (FM) model that recovers multimodal spatio-temporal
distributions over the future locations of dynamic household objects directly in continuous
3D space. Instead of predicting a single future position, it learns the latent regularities
induced by recurring human routines and predicts a *distribution* over plausible future
bounding boxes for a queried object at a future time, conditioned on the current scene.

FlowMaps is composed of two modules:

1. **VAE**: encodes each object token (a normalized 3D bounding box + semantic label) into a
   latent code, and decodes latents back into geometry and class predictions.
2. **Latent CDiT** (Conditional Diffusion Transformer): a flow matching network that
   transports a Gaussian latent to the latent of the queried object's future bounding box. A
   *map encoder* aggregates the scene context into tokens, and a stack of CDiT blocks refines
   the noisy query latent by cross-attending to that context.

## Habits

Training data is generated with [ProcTHOR](https://procthor.allenai.org/): object movements
are driven by predefined human-like routines (*habits*) that produce semantically consistent
patterns. We model three representative habits, and **a separate FlowMaps model is trained per
habit**:

| Mode     | Habit    | Behaviour |
|----------|----------|-----------|
| `habit1` | Habit #1 | **Location preferences** — the simulated human repeatedly returns to a small set of favoured places, spending more time there. |
| `habit2` | Habit #2 | **Balanced routine** — time is distributed approximately uniformly across the relevant locations. |
| `habit3` | Habit #3 | **Highly dynamic routine** — frequent transitions between locations, with short intervals spent at each. |

For each habit we generate 2706 training and 918 validation environments. Each scene contains
up to 15 dynamic objects moving between semantically compatible receptacles (e.g. a `Fork` can
appear on a `Sink` or `DiningTable`, but not on a `ShelvingUnit`), drawn from a closed set of
41 object classes and 17 receptacles. Each environment is simulated for 4 weeks at hourly
resolution (672 timesteps per scene), yielding over 1.8M training samples per habit.

> **Note on the provided data.** This repository ships only a **single sample environment per
> habit**, and the *same* environment is used for both the `train` and `val` splits. It is
> meant purely to exercise and test the code end-to-end, not to reproduce the paper results.
> The full dataset is not included here — we plan to release it in the future as a separate
> contribution.

## Data layout

Datasets are organised by habit under `data/`:

```
data/
├── habit1/            # Habit #1 (location preferences)
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
├── habit2/            # Habit #2 (balanced routine)
│   └── ...
└── habit3/            # Habit #3 (highly dynamic routine)
    └── ...
```

Each env directory contains `scan_merged.parquet`, `config.yaml`, `pickupable_names.json`,
`receptacle_names.json`, and `pickupable_to_receptacle.json`. Checkpoints live under
`ckpt/{mode}/cdit.pth` and `ckpt/{mode}/vae.pth`.

All Hydra commands accept `mode=<mode>` to select the dataset (`habit1`, `habit2`, or
`habit3`). This derives `data_root` and `ckpt_root` automatically. Scripts that take a
`--data-root` argument still accept it directly.

## Installation

```bash
pip install -r requirements.txt
```

## Full pipeline: start to finish

All commands below assume `MODE=habit1` (replace with `habit2` / `habit3` as needed). `mode`
defaults to `habit1`. There is no `env_name` parameter — all envs in the split directory are
loaded automatically.

```bash
export MODE=habit1
```

### 1. Merge scan parquets

Raw scans arrive as `scan_0.parquet`, `scan_1.parquet`, … per environment. Merge them into a
single `scan_merged.parquet`:

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

The checkpoint is saved by Hydra to `outputs/<date>/<time>/checkpoints/best.pth` (best val
loss) and `last.pth` (final iteration). Copy it manually:

```bash
cp outputs/<date>/<time>/checkpoints/best.pth ckpt/$MODE/vae.pth
```

### 3. Compute latent statistics

Required before CDiT training. Encodes all training data through the VAE and saves the latent
mean/std:

```bash
python scripts/compute_statistics.py mode=$MODE
```

Writes `data/${MODE}/latents/latent_statistics.npz` and copies the VAE checkpoint to
`data/${MODE}/latents/vae.pth`.

### 4. Train the CDiT

```bash
# Local
python slurm.py experiment=cdit mode=$MODE wandb.mode=offline

# Cluster (SLURM)
python slurm.py --multirun hydra/launcher=remote +hydra/sweep=remote \
  experiment=cdit mode=$MODE wandb.mode=online wandb.tags=["cdit"]
```

As with the VAE, copy from the Hydra output directory:

```bash
cp outputs/<date>/<time>/checkpoints/best.pth ckpt/$MODE/cdit.pth
```

### 5. Run eval

```bash
python eval.py mode=$MODE
```

Loads `ckpt/${MODE}/cdit.pth`, samples scenes from the minival set, generates multiple
predictions per query, and saves BEV PNGs to `eval/png/`. Key overrides:

```bash
python eval.py mode=$MODE \
  nsamples=10 \
  npreds=50 \
  log_path=./eval/${MODE}
```

To overlay predictions on AI2-THOR top-down renders instead of the BEV (requires the
`minival` + `topdown` renders described above):

```bash
python eval.py mode=$MODE display_mode=topdown
```

## Repository layout

```
config/          # Hydra configs (train, eval, compute_statistics, experiments, ...)
scripts/         # Data utilities (merge scans, compute latent statistics)
src/
├── dataset/     # SimDataset (CDiT) and VAE dataset
├── flowmaps/    # FlowMaps sampling pipeline + evaluation/visualisation
├── models/      # VAE, CDiT transformer, probability paths, ODE solver, embeddings
├── trainer/     # Training loops (CDiT + VAE), checkpoint manager
└── utils/       # Lookup tables, losses, logging, visualisation, torch helpers
eval.py          # Evaluation / qualitative sampling entrypoint
slurm.py         # Training entrypoint (local or SLURM via Hydra)
```
