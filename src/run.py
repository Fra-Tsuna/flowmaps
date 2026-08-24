"""
To run in the cluster do:
python3 slurm.py --multirun hydra/launcher=remote +hydra/sweep=remote ...
"""

from typing import Union
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import os
from hydra.utils import instantiate

import torch

from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler
import wandb

from src.dataset.sim_dataset import SimDataset

from src.utils.torch_utils import seed_everything, model_size_b, MiB
from src.utils.mylogging import pretty_print_config
from src.utils.lookup import load_lookup, set_active_lookup
from src.trainer.trainer import TrainingPipeline
from src.trainer.vae_trainer import VAETrainer
from src.models.transformer import CDiT
from src.models.vae import VAE
from src.models.paths import CondOTProbPath

import logging

# A logger for this file
logger = logging.getLogger(__name__)


def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)
    pretty_print_config(cfg)

    lookup = load_lookup(data_root=Path(cfg.data_root) / "train")
    set_active_lookup(lookup)
    # Setup logger
    run = wandb.init(**cfg.wandb)
    wandb.config.update({"slurm_job_id": os.environ.get("SLURM_JOB_ID")})
    wandb.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    # Train dataloader
    train_loader: DataLoader = instantiate(cfg.train_dataloader)
    # Validation dataloader
    val_loader: DataLoader = instantiate(cfg.val_dataloader)

    train_dataset: SimDataset = train_loader.dataset

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
    }

    logger.info(f"Dataset size: {len(train_dataset)}")

    model_kwargs = train_dataset.model_kwargs()
    logger.info(f"Model kwargs from dataset: {model_kwargs}")
    model: Union[CDiT, VAE] = instantiate(cfg.model, **model_kwargs).to(device)
    
    
    # Instantiate probability path
    path: CondOTProbPath = instantiate(cfg.path)
    # init optimizer
    optim: Union[Adam, AdamW] = instantiate(cfg.optimizer, params=model.trainable_parameters())
    # init lr scheduler
    lr_scheduler = get_scheduler(**cfg.lr_scheduler, optimizer=optim)

    # Report model size
    size_b = model_size_b(model)
    logger.info(f"Model size: {size_b / MiB:.3f} MiB")
    
    trainer: Union[TrainingPipeline, VAETrainer] = instantiate(
        cfg.trainer,
        model=model,
        dataloaders=dataloaders,
        path=path,
        optimizer=optim,
        lr_scheduler=lr_scheduler,
        seed = cfg.seed,
    )
    
    trainer.train(run)
