from typing import Callable, Union

from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from src.utils.torch_utils import seed_everything
from src.models.transformer import CDiT
from src.utils.mylogging import pretty_print_config
from src.utils.viz import setup_savefig
from src.trainer.checkpoint_manager import CheckpointManager
from src.flowmaps.flowmaps import FlowMapsPipeline, FMTester

import logging

now = datetime.now()
date_dir = now.strftime("%Y-%m-%d")
time_dir = now.strftime("%H-%M-%S")

# A logger for this file
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="eval")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)
    pretty_print_config(cfg)

    result_dir = os.path.join(cfg.log_path, date_dir, time_dir)
    os.makedirs(result_dir, exist_ok=True)
    savefig: Callable = setup_savefig(
        res_path=result_dir,
        fig_fmt="png",
        dpi=100,
        transparent_png=False,
    )

    # Load checkpoint
    manager = CheckpointManager(cfg.checkpoint_path)
    model: CDiT = manager.load_checkpoint(f_name=cfg.checkpoint_name)
    model.to(device)
    model.eval()

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    val_loader: DataLoader = instantiate(cfg.val_dataloader, generator=g)
    val_dataset = val_loader.dataset
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    flowmaps: FlowMapsPipeline = instantiate(cfg.sampler)
    # TODO: make this an actual configurable option using hydra
    if type(model) is CDiT:
        tester_cls = FMTester
    else:
        raise NotImplementedError(f"Model type {type(model)} not supported yet.")
    tester = tester_cls(model=model, pipeline=flowmaps, dataset=val_dataset, savefig=savefig, seed=cfg.seed)

    samples = tester.generate_samples(nsamples=cfg.nsamples, npreds=cfg.npreds)
    result = tester.inference(samples)
    tester.display_results(result, iteration=-1)


if __name__ == "__main__":
    main()