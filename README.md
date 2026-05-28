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

### 6. Make GIFs

During DiT training, `display_results` saves one BEV PNG per scene at every validation checkpoint, named `<scene>_idx<iteration>.png`, under the Hydra output directory (`outputs/<date>/<time>/png/`). Copy those PNGs to a convenient location, then run:

```bash
python scripts/make_gifs.py outputs/<date>/<time>/png/ --fps 2
```

This groups files by scene name, sorts by iteration index, and saves one GIF per scene to `outputs/<date>/<time>/png/gifs/`. To write GIFs elsewhere:

```bash
python scripts/make_gifs.py outputs/<date>/<time>/png/ --out results_png/gifs/ --fps 2
```


## ChangeLog

### 2026-04-15

**Removed dataset-variant suffix from all paths and configs:**

- **Directory rename**: all env directories renamed from `env{N}_obj15_furns5_span4w` to `env{N}` across `train/`, `val/`, and `minival/` for all three modes. Symlinks in `minival/` and `topdown/minival/` recreated accordingly.

- **Flat latents**: `data/{mode}/latents/obj15_furns5_span4w/` flattened to `data/{mode}/latents/`. VAE checkpoint renamed from `vae_obj15_furns5_span4w.pth` to `vae.pth`.

- **Flat topdown**: `data/{mode}/topdown/val/obj15_furns5_span4w/env{N}/` flattened to `data/{mode}/topdown/val/env{N}/`.

- **Checkpoint rename**: `ckpt/{mode}/dit_obj15_furns5_span4w.pth` → `dit.pth`, same for `vae.pth`.

- **`env_name` removed everywhere**: no longer a config parameter. Datasets (`SimDataset`, `VaeDataset`) now automatically scan and load all `env{N}` directories present in the split root. `load_lookup()` similarly scans all envs when called without an argument. The `res_path` interpolation (which depended on `env_name`) is also gone.

- **Config/code simplification**: `config/experiment/{dit,vae}.yaml`, `config/compute_statistics.yaml`, and all dataset configs cleaned up. `latent_statistics_path` in VAE dataset configs now points to `${data_root}/latents/latent_statistics.npz`. `_pattern_requested` logic removed from `compute_statistics.py`.

### 2026-03-27

**Eval visualisation on AI2-THOR top-down renders:**

- **`display_results_topdown()` in `FMTester`**: new eval-only display method that overlays predicted and GT bounding boxes on the 1024×1024 AI2-THOR top-down renders instead of BEV images. Implements the full coordinate chain: BEV-normalised space → world coordinates → top-down pixel space using `camera.json` (world-to-pixel mapping saved alongside each render). Three-panel layout: environment at τ_c (GT box at current position), ground-truth at τ_q, and sampled predictions at τ_q.

- **`generate_eval_samples()`**: separate sampling method for eval (does not touch `generate_samples` used by the trainer). Adds displacement-based filtering (`only_moved`, `min_displacement` in metres) with re-shuffling loop so exactly `nsamples` qualifying scenes are collected. Always stores `groundtruth_current` (GT bbox at τ_c) alongside `groundtruth` (GT bbox at τ_q) so the two panels can show different positions.

- **`sim_minival` dataset config**: `config/dataloader/dataset/sim_minival.yaml` points to `data/minival/`, a symlinked subset of val environments that have completed top-down renders.

- **`eval.yaml`**: added `only_moved`, `min_displacement`, `nsamples`, `npreds` flags.

### 2026-03-26

- **`DiT.train()` override**: overrides `nn.Module.train()` to always keep `self.vae` in eval mode. The explicit `vf.vae.eval()` calls that followed every `vf.train()` call in `TrainingPipeline` are removed; the invariant is now enforced at the model level.

- **Minibatch OT coupling (experimental)**: added `use_coupling: false` flag to the trainer. When enabled, `prepare_batch_coupling` permutes only `x_0` via the OT plan (minimising `‖x_0 − x_1‖²`) while keeping `x_1` and all conditioning tensors in their original order, preserving semantic alignment. Requires `OTPlanSampler.sample_plan_with_indices`, which is also added. Off by default.

### 2026-03-25

**Codebase cleanup and correctness fixes:**

- **Lazy loading for `SimDataset`**: Mirrored `VaeDataset`'s lazy loading pattern. `load_from_path` now reads only parquet row counts and JSONs at init; full tensor data is loaded on first access per env via `_load_env_to_cache` and stored in a bounded LRU cache. Concurrent preloading via `ThreadPoolExecutor` warms the cache at startup. Reduces init time from ~4 minutes to ~30-60 seconds for 918 envs. `max_cached_envs` and `preload_workers` are configurable in `sim_train.yaml` / `sim_val.yaml`.

- **`eval.py` refactor**: Standalone DiT evaluation script. Loads checkpoint via `CheckpointManager` with EMA applied, runs the validation loop (identical to `TrainingPipeline.validate`), prints `val_loss` to stdout, and dumps PNGs. Uses `experiment=dit` config composition to guarantee parameter consistency with training. Checkpoint path defaults to `ckpt/dit_${env_name}.pth`.

- **Batched ODE inference in `FMTester`**: All objects from a scene are now batched into a single ODE solver call per prediction, reducing inference from `nsamples x No x npreds` calls to `nsamples x npreds` calls.

- **`model_kwargs()` on datasets**: `SimDataset` and `VaeDataset` now expose a `model_kwargs()` method returning the dict of dataset-derived model constructor arguments (`obj_classes`, `furn_classes`, `max_transitions` for DiT; `n_classes` for VAE). `run.py` calls `train_dataset.model_kwargs()` directly, removing the `hasattr` coupling.

- **`DiT.train()` override**: Overrides `nn.Module.train()` to always keep `self.vae` in eval mode, regardless of whether the DiT itself is set to train or eval. Prevents `vae.train()` from being called implicitly when `dit.train()` is invoked.

- **`CFGVectorField` renamed to `ModelWrapper`**: Removed dead CFG logic (commented-out guided/unguided interpolation). `linear.py` renamed to `wrapper.py`.

- **Dead code removal**: Deleted unused model files (`gaussians.py`, `aligner.py`, `linear_policy.py`, `transformer_policy.py`, `unet.py`, `encoders.py`, `MLP` class from `linear.py`), unused util files (`transforms.py`, `debug.py`), unused dataset (`env_dataset.py`), and stale configs (`model/linear.yaml`, `model/linear_policy.yaml`, `model/unet_policy.yaml`, `config/config.yaml`, `dataloader/dataset/env_train.yaml`, `dataloader/dataset/env_val.yaml`).

### 2026-03-21
SimDataset and DiT training updated for new dataset (3624 envs):
- **Normalization aligned**: `SimDataset` constants updated to match `VaeDataset`: `MIN_SCENE_SIZE=-0.5`, `MAX_SCENE_SIZE=34.0`, `SCENE_RANGE=34.5`.
- **`max_tokens` unified**: `sim_train.yaml` and `sim_val.yaml` updated to `max_tokens=32` (was 20), matching `vae.yaml` and `compute_statistics.yaml`.
- **Log size in DiT `embed_map`**: `DiT.embed_map` now applies `vae._size_to_model` to size dimensions before passing to `BB3DEmbedding`, consistent with how `VAE.embed_query` handles query object tokens. Padded tokens (size=0) are safe due to `clamp_min + eps` in `_size_to_model`.

### 2026-03-20
**BB3DEmbedding redesign** (breaking change - existing checkpoints incompatible): Rewrote to match the MapAnything paper. Previously, direction and magnitude were concatenated into a single 4D vector passed to one MLP pair. Now direction (3D unit vector) and magnitude (1D L2 norm scalar) are encoded by **separate** 4-layer MLP pairs and **summed**, treating direction and scale as independent quantities. This applies to both translation and size branches.

Differences from the MapAnything paper:
- **No log-transform inside the embedder**: MapAnything operates on raw world coordinates (meters), where translation magnitudes can span `[0.1, 50m]` and sizes vary wildly - hence the log-transform inside the scale MLP. Here, positions arrive pre-normalized to `[0, 1]` so `t_norm ∈ [0, sqrt(3)]` is already in a well-behaved range. Sizes arrive log-normalized via `use_log_size` in `embed_query`. No further log-transform is needed or applied inside the embedder.
- **`encode_yaw=False`**: MapAnything always encodes orientation. Yaw is intentionally discarded here.
- **Yaw-only quaternion**: When yaw encoding is enabled, the quaternion is constructed analytically as `[0, 0, sin(yaw/2), cos(yaw/2)]` (rotation around vertical axis only), unlike MapAnything's full 6-DOF quaternion.

Also added CIoU breakdown logging: `complete_box_iou_loss_3d` now returns a dict with `loss`, `iou_loss`, `distance_loss`, and `aspect_loss`, all logged separately to wandb as `train/ciou_iou`, `train/ciou_distance`, `train/ciou_aspect`.

### 2026-03-18
Split `l1_w` into `l1_center_w` and `l1_size_w` (both default `5.0`). The split allows independent monitoring and weighting of position vs size reconstruction. Sizes are supervised in log-normalized space (when `use_log_size: true`), positions in linear-normalized space. **Breaking change**: configs using `loss_weights.l1` must be updated to `loss_weights.l1_center` and `loss_weights.l1_size`.

### 2026-03-17
- **KL annealing**: Added `kl_warmup_steps` hyperparameter to `vae.yaml` (default 10000). Beta ramps linearly from 0 to `beta` over the first `kl_warmup_steps` iterations, preventing KL from competing with reconstruction early in training. `train/beta` is now logged to wandb. Set `kl_warmup_steps: 0` to disable.
- **Viz bug fix**: `save_bev_sequence` in `vae_dataset.py` was calling `render_bev` with global `MAX_SCENE_SIZE` and no `min_size`, causing incorrect BEV rendering. Fixed to use per-environment `scene_min`/`scene_max` from `_env_meta`.

### 2026-03-16 (3)
Fixed bounding box visualization in `VAETester.inference`. Predicted bboxes were converted to pixel coordinates by multiplying the global-normalized [0,1] output directly by `(img_size-1)`, but the BEV image is rendered with per-environment tight bounds `[scene_min, scene_max]`. This caused boxes to be drawn at incorrect positions whenever per-env bounds differed from the global bounds. Fix: denormalize to world coordinates first (`bbox * SCENE_RANGE + MIN_SCENE_SIZE`), then convert to per-env pixel space using `(world - scene_min) / (scene_max - scene_min) * (img_size - 1)`. Also changed bbox dtype from `int16` to `float32` to avoid truncation before `draw_box` rounds.

### 2026-03-16
Fixed visualization during VAE pretraining validation to scale correctly across 900+ environments.

`VAETester.generate_samples` now accepts a `max_envs` parameter (default `15` in `config/trainer/vae.yaml`) that caps the number of environments used for visualization to a fixed, deterministically-sampled subset, preventing 900×5 PNGs being dumped per validation checkpoint.

Replaced the global `MAX_SCENE_SIZE=34m` display range with a per-environment tight range. At dataset load time, `load_from_path` derives `scene_min = round(xz.min() - 0.5)` and `scene_max = round(xz.max() + 0.5)` from the receptacle cornerpoints (x and z only). These values are passed to `render_bev` (which now accepts a `min_size` argument) so each BEV image fills its full 256×256 frame, and threaded through the sample → result → `display_results` pipeline so axis tick labels reflect the per-env world-coordinate range.

### 2026-03-10
Migrated to the charlie dataset (`data/train/`, `data/val/`). Data was structured as `data/{split}/env{ID}_obj{N}_furns{K}_span{T}/` with the train/val split explicit in the directory structure rather than derived at runtime from env ordering. Dataset configs use `data_root: ${data_root}/${split}` and `split` is no longer used for runtime splitting in the dataset. (The `_obj{N}_furns{K}_span{T}` suffix was later removed in 2026-04-15.)

Switched lookup and dataset label maps from **full instance names** (e.g. `Fork|surface|6|46`) to **object category names** (e.g. `Fork`). Previously the lookup accumulated per-instance IDs, which was consistent within the old data (single procthor house, multiple configs) but breaks across the charlie data where each env ID is a different house with unique instance suffixes. Using categories gives a stable, house-agnostic vocabulary and is the correct granularity for generalization.

Rewrote `VaeDataset` data loading with lazy loading + LRU cache + concurrent preload. Previously all parquets were read and expanded into ~1.9M per-timestep Python dicts at init, causing OOM and a 12-minute wall before training started. Now `load_from_path` registers only lightweight metadata (JSON files + parquet row count from footer); tensor data is loaded on first access per env via `_load_env_to_cache` and stored in a bounded `OrderedDict` LRU cache (`max_cached_envs=1000`). At init time all envs are preloaded concurrently using a `ThreadPoolExecutor` (`preload_workers=16`) so DataLoader workers inherit a warm cache via fork, avoiding I/O contention during training. Both `max_cached_envs` and `preload_workers` are configurable via the dataset config.

### 2026-02-10
Fixed cross-env contamination bug in `SimDataset`: `get_sample` now scopes sampling within per-env boundaries using `env_ranges`, preventing cross-env timestamp pairing in multi-env runs. Added `overfit` support and pattern-based train/val splitting to `SimDataset` (aligned with `VaeDataset` logic). Wired `overfit` flag into `sim_train.yaml`/`sim_val.yaml`. Added `__main__` smoke tests with BEV image dumps for single-env and multi-env validation.
Fixed `compute_statistics.py`. No uses using deterministic `mu` instead of stochastic `VAE.reparameterize(mu, logvar)` for computing latent statistics.

### 2026-02-09
Updated latent statistics workflow to support pattern-based VAE checkpoints and centralized latent outputs. `compute_statistics.py` now writes stats (and copies the matching `vae_<pattern>.pth`) under `data/latents/<pattern>/`, the DiT experiment config reads VAE checkpoints and latent stats from that location, and the README documents the new usage. SimDataset now supports pattern-based train/val splits (all but last env for train, last env for val) and the sim dataloader configs pass split/env_name accordingly.
Added VAE overfit support: setting `overfit=true` makes train and val load the same full set of matched envs (no split), while default behavior remains unchanged. The flag is exposed in `config/train.yaml` and wired into `vae_train.yaml`/`vae_val.yaml`.

### 2026-01-30
Made some improvements in the image logging for VAE pretraining to show also GT just to be extra sure. Also, made the contribution of each loss term in the reconstruction loss configurable in hydra.


### 2026-01-29
Added multi-environment VAE pretraining support with a new dataset spanning three environments (`env0`, `env1`, `env2`). Each setup includes 5, 10, or 15 movable objects, up to 3 or 5 furnitures, and temporal spans of 3 days, 1 week, 3 weeks, or 1 month. Data follows the convention `env{ID}_obj{N}_furns{K}_span{T}` (for example `env0_obj5_furns3_span1m`). Training originally targeted environments via `env_name` (full name or pattern prefix). This mechanism was removed in 2026-04-15; all envs in the split directory are now loaded automatically.
Supporting both single- and multi-env workflows required refactoring the data pipeline. A global lookup now provides shared object and receptacle IDs, while receptacle geometry is sourced per environment. The previous global OOBB lookup from JSON was removed because identical receptacles may appear in different locations across environments, leading to incorrect geometry. The lookup is therefore geometry-free but maintains a consistent global ID space. Functionality was validated through smoke tests in `vae_dataset.py`. On the modeling side, bounding box prediction was stabilized by introducing log-scaling for size. Training visualization was also improved with clearer image logging.
