import os
import shutil
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from src.models.vae import VAE
from src.trainer.checkpoint_manager import CheckpointManager
from src.utils.lookup import load_lookup, set_active_lookup


def compute_stats(embedding_model, dataloader, device="cuda"):
    embedding_model.eval()
    embedding_model.to(device)

    embeddings = []

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            map = batch["map"].to(device)    # (B, S, 8)
            types = batch["types"].to(device)  # (B, S)

            obj_mask = types == 1
            obj_tokens = map[obj_mask]  # (B * S_obj, 8) 
            mu, logvar = embedding_model.encode(obj_tokens)
            embeddings.append(mu.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)  # (Total, D)

    # Compute a scalar mean and variance for the whole token
    stats = {
        "mean": embeddings.mean(axis=0),
        "std": embeddings.std(axis=0),
        "min": embeddings.min(axis=0),
        "max": embeddings.max(axis=0),
    }

    return stats

@hydra.main(config_path="../config", config_name="compute_statistics", version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats_path = cfg.stats_path
    checkpoint_name = cfg.checkpoint_name
    latents_root = cfg.latents_root

    lookup = load_lookup(data_root=Path(cfg.data_root) / "train")
    set_active_lookup(lookup)

    train_loader = instantiate(cfg.train_dataloader)

    manager = CheckpointManager(checkpoints_path=cfg.checkpoint_path)
    vae = manager.load_checkpoint(
        f_name=checkpoint_name,
        optimizer=None,
        ema=None,
        lr_scheduler=None,
    )
    if not isinstance(vae, VAE):
        raise FileNotFoundError(
            f"Checkpoint not found or invalid model: {os.path.join(cfg.checkpoint_path, checkpoint_name)}"
        )

    stats = compute_stats(vae, train_loader, device=device)
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    np.savez(stats_path, **stats)
    print("Finished computing statistics, saving to", stats_path)

    os.makedirs(latents_root, exist_ok=True)
    src_ckpt = os.path.join(cfg.checkpoint_path, checkpoint_name)
    dst_ckpt = os.path.join(latents_root, checkpoint_name)
    if os.path.isfile(src_ckpt):
        shutil.copy2(src_ckpt, dst_ckpt)


if __name__ == "__main__":
    main()
