import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
from pathlib import Path
import os
import logging

from src.utils.torch_utils import seed_everything
from src.utils.mylogging import pretty_print_config
from src.utils.viz import setup_savefig
from src.flowmaps.flowmaps import FlowMapsPipeline, FMTester
from src.trainer.checkpoint_manager import CheckpointManager
from src.utils.lookup import load_lookup, set_active_lookup

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="eval")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)
    pretty_print_config(cfg)

    result_dir = cfg.log_path
    os.makedirs(result_dir, exist_ok=True)
    savefig = setup_savefig(res_path=result_dir, fig_fmt="png", dpi=100, transparent_png=False)

    # Initialize lookup tables (same as run.py)
    lookup = load_lookup(data_root=Path(cfg.data_root) / "train")
    set_active_lookup(lookup)

    # Load CDiT checkpoint, applying EMA weights if available
    ckpt_dir, ckpt_name = os.path.split(cfg.checkpoint_path)
    manager = CheckpointManager(checkpoints_path=ckpt_dir)
    model = manager.load_checkpoint(f_name=ckpt_name, apply_ema=True)
    model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint from {cfg.checkpoint_path}")

    # Val dataloader
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    val_loader: DataLoader = instantiate(cfg.val_dataloader, generator=g)
    val_dataset = val_loader.dataset
    logger.info(f"Val dataset size: {len(val_dataset)}")

    sampler: FlowMapsPipeline = instantiate(cfg.trainer.sampler)
    tester = FMTester(model=model, pipeline=sampler, dataset=val_dataset, savefig=savefig, seed=cfg.seed)

    display_mode = cfg.get("display_mode", "bev")
    if display_mode == "topdown":
        samples = tester.generate_eval_samples(nsamples=cfg.nsamples, npreds=cfg.npreds,
                                               only_moved=cfg.only_moved, min_displacement=cfg.min_displacement)
        result = tester.inference(samples)
        topdown_root = Path(cfg.topdown_root)
        tester.display_results_topdown(result, iteration=0, topdown_root=topdown_root)
    else:
        samples = tester.generate_eval_samples(nsamples=cfg.nsamples, npreds=cfg.npreds,
                                               only_moved=cfg.only_moved, min_displacement=cfg.min_displacement)
        result = tester.inference(samples)
        tester.display_results(result, iteration=0)
    logger.info(f"PNGs saved to {os.path.join(result_dir, 'png')}/")


if __name__ == "__main__":
    main()
