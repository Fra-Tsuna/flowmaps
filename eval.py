import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
from functools import partial
from pathlib import Path
from typing import List
import os
import logging
from tqdm import tqdm

from src.utils.torch_utils import seed_everything, sample_logit_normal, sample_beta
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

    # Load DiT checkpoint, applying EMA weights if available
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

    # Path and t-sampler (same as training)
    path = instantiate(cfg.path)
    strategy = cfg.trainer.t_sampling_strategy
    if strategy == "uniform":
        t_sampler = torch.rand
    elif strategy == "logit_normal":
        t_sampler = sample_logit_normal
    elif strategy == "beta":
        t_sampler = partial(sample_beta, s=0.999)
    else:
        raise ValueError(f"Unknown t_sampling_strategy: {strategy}")

    mean = torch.tensor(val_dataset.latent_statistics["mean"], device=device)
    std = torch.tensor(val_dataset.latent_statistics["std"], device=device)

    val_loss: List[float] = []

    # pbar = tqdm(val_loader, desc="Validation", total=len(val_loader), dynamic_ncols=True)
    # for data in pbar:
    #     # Only move tensors to device; keep non-tensor metadata on CPU.
    #     data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}

    #     t_query = data["t_query"]  # (B, 1)
    #     t_current = data["t_current"]  # (B, 1)
    #     mask = data["mask"]  # (B, S)
    #     map = data["map"]  # (B, S, 8)
    #     obj_q = data["query_object"]  # (B, 1, 8)
    #     types = data["types"]  # (B, S)

    #     # Get query object class label
    #     obj_cls = obj_q[:, :, -1]  # (B, 1), label is at -1

    #     B, _, D = obj_q.shape

    #     with torch.inference_mode():
    #         # Encode and normalize query
    #         _mu, _logvar, query_latents = model.encode_query(obj_q.view(B, D))
    #         query_latents = (query_latents.view(B, 1, -1) - mean) / std

    #         x_0 = torch.randn_like(query_latents)  # (B, 1, latent_dim)
    #         x_1 = query_latents
    #         t = t_sampler(B).to(device)
    #         path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

    #         prediction = model(
    #             x=path_sample.x_t,
    #             t=t,
    #             obj=obj_cls,
    #             map=map,
    #             tau0=t_current,
    #             tau=t_query,
    #             types=types,
    #             key_padding_mask=mask,
    #         )  # (B, 1, out_dim)

    #         # Validation loss
    #         loss = torch.pow(prediction - path_sample.dx_t, 2).mean()
    #     val_loss.append(loss.item())

    #     pbar.set_postfix(val_loss=f"{loss.item():.3f}")

    # val_loss = torch.mean(torch.tensor(val_loss))
    # print(f"\nval_loss: {val_loss.item():.6f}")

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
