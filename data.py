from omegaconf import DictConfig
import hydra
import os

import numpy as np

from src.env import FlowSimEnvironment
from src.utils.torch_utils import seed_everything
from tqdm import tqdm


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):

    seed_everything(cfg.seed)
    train_data = []
    val_data = []
    for id_env in tqdm(range(cfg.n_env_train), desc="Generating training data"):
        env = FlowSimEnvironment(
            height=cfg.size,
            width=cfg.size,
            max_tables=cfg.max_tables,
            min_each=cfg.min_each,
            stochastic=cfg.stochastic,
        )
        imgs, furn, objs_timeline = env.cycle(cfg.max_timesteps,
                                              icon_dir=cfg.icons_dir,
                                              display_scale=cfg.display_scale)
        data = {
            "id_env": id_env,
            "imgs": imgs,
            "furniture": furn,
            "object": objs_timeline,
            "scale": cfg.display_scale,
        }
        train_data.append(data)

    for id_env in tqdm(range(cfg.n_env_val), desc="Generating validation data"):
        env = FlowSimEnvironment(
            height=cfg.size,
            width=cfg.size,
            max_tables=cfg.max_tables,
            min_each=cfg.min_each,
            stochastic=cfg.stochastic,
        )
        imgs, furn, objs_timeline = env.cycle(cfg.max_timesteps,
                                              icon_dir=cfg.icons_dir,
                                              display_scale=cfg.display_scale)
        data = {
            "id_env": id_env,
            "imgs": imgs,
            "furniture": furn,
            "object": objs_timeline,
            "scale": cfg.display_scale,
        }
        val_data.append(data)

    os.makedirs(cfg.res_path, exist_ok=True)
    if train_data:
        np.savez(f"{cfg.res_path}/dataset_train.npz", data=train_data)
    if val_data:
        np.savez(f"{cfg.res_path}/dataset_val.npz", data=val_data)

if __name__ == "__main__":
    main()
