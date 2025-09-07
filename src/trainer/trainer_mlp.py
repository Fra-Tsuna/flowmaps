import os

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Normal, Independent

from tqdm import tqdm
from typing import Dict, List, Tuple

from omegaconf import DictConfig

import logging

from src.trainer.checkpoint_manager import CheckpointManager
from src.utils.torch_utils import cycle
from src.utils.statistics import compute_occurrences, build_gmms_from_stats

logger = logging.getLogger(__name__)


class TrainingPipelineMLP:
    """
    Training pipeline for the MLP baseline that regresses the mean of a unit-variance Gaussian.
    Matches your dataloader contract and logging pattern.

    Metrics at validation:
      - val_loss: MSE between predicted mean and gt bbox
      - nll/mlp: NLL of gt bbox under N(mu_pred, I)
      - kl/mlp:  E[log q(x)] - E[log p(x)] using your GMM q and our N(mu,I) p
    """

    def __init__(
        self,
        model: nn.Module,
        iterations: int,
        dataloaders: Dict[str, DataLoader],
        optimizer: Optimizer,
        lr_scheduler: LambdaLR,
        result_dir: str,
        validate_every: int = 100,
        patience: int = 10,
        use_ema: bool = False,
        ema: DictConfig = None,
    ):
        self.model = model
        self.train_dataloader = dataloaders["train"]
        self.val_dataloader = dataloaders["val"]

        self.optim = optimizer
        self.lr_scheduler = lr_scheduler
        self.iterations = iterations + 1
        self.validate_every = validate_every

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.result_dir = result_dir
        self.checkpoint_dir = os.path.join(self.result_dir, "checkpoints")
        self.manager = CheckpointManager(checkpoints_path=self.checkpoint_dir, patience=patience)

        self.gaussians_gt = None  # prepared on-demand for KL

    def train(self, run):
        self.model.train()
        iterator = cycle(self.train_dataloader)

        pbar = tqdm(range(self.iterations), desc="Training (MLP)", total=self.iterations,
                    dynamic_ncols=True, leave=True)
        metrics = {"train_loss": float("nan"), "val_loss": float("nan")}

        for i in pbar:
            data = {k: v.to(self.device) for k, v in next(iterator).items()}

            t_query    = data["t_query"]      # (B,)
            t_current  = data["t_current"]    # (B,)
            mask       = data["mask"]         # (B, S)
            map        = data["map"]          # (B, S, 5)
            obj_q      = data["query_object"] # (B, 1, 5)
            types      = data["types"]        # (B, S)
            x_1        = obj_q[:, :, :4]      # (B, 1, 4)
            x_1_q      = obj_q[:, :, 4]       # (B, 1)

            # Predict mean (B,1,4)
            mu = self.model(
                obj=x_1_q,
                map=map,
                tau0=t_current,
                tau=t_query,
                types=types,
                key_padding_mask=mask,
            )

            # Train loss: MSE to the GT bbox
            loss = torch.pow(mu - x_1, 2).mean()
            metrics["train_loss"] = float(loss.item())
            lr = self.lr_scheduler.get_last_lr()[0]

            # Step
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            self.lr_scheduler.step()

            pbar.set_postfix(train_loss=f"{metrics['train_loss']:.3f}", val_loss=f"{metrics.get('val_loss', float('nan')):.3f}")
            run.log({"train_loss": metrics["train_loss"], "lr": lr, "step": i}, step=i)

            # Validation
            if (i + 1) % self.validate_every == 0:
                stats = self.validate()
                metrics["val_loss"] = stats["val_loss"]

                run.log({
                    "val_loss": stats["val_loss"],
                    "nll/mlp": stats["nll/mlp"],
                    "kl/mlp": stats["kl/mlp"],
                    "step": i,
                }, step=i)

                # Save if best
                self.manager.save_if_best(
                    model=self.model,
                    optimizer=self.optim,
                    ema=None,
                    lr_scheduler=self.lr_scheduler,
                    val_loss=metrics["val_loss"],
                    iteration=i,
                    model_args=getattr(self.model, "model_args", {}),
                    run=run,
                )
                self.model.train()

        # Save last
        self.manager.save(
            model=self.model,
            optimizer=self.optim,
            ema=None,
            lr_scheduler=self.lr_scheduler,
            iteration=self.iterations,
            f_name="last.pth",
            model_args=getattr(self.model, "model_args", {}),
            run=run,
        )

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()

        mse_vals:     List[float] = []
        nll_mlp_list: List[Tensor] = []
        kl_list:      List[Tensor] = []
        
        pbar = tqdm(self.val_dataloader, desc="Validation (MLP)", total=len(self.val_dataloader),
                    dynamic_ncols=True, leave=False)

        for data in pbar:
            data = {k: v.to(self.device) for k, v in data.items()}

            t_query    = data["t_query"]      # (B,)
            t_current  = data["t_current"]    # (B,)
            mask       = data["mask"]         # (B, S)
            map        = data["map"]          # (B, S, 5)
            obj_q      = data["query_object"] # (B, 1, 5)
            types      = data["types"]        # (B, S)

            env_id     = data["env_id"]       # (B,)
            obj_id     = data["q_obj_id"]     # (B,)

            x_1   = obj_q[:, :, :4]  # (B,1,4)
            x_1_q = obj_q[:, :, 4]   # (B,1)

            # Predict mean
            mu = self.model(
                obj=x_1_q,
                map=map,
                tau0=t_current,
                tau=t_query,
                types=types,
                key_padding_mask=mask,
            )  # (B,1,4)

            # MSE val loss
            mse = torch.pow(mu - x_1, 2).mean()
            mse_vals.append(float(mse.item()))

            # NLL under N(mu, I)
            # log p(x) = -0.5 * (||x - mu||^2 + D * log(2*pi))
            X  = x_1.flatten(start_dim=1)   # (B,4)
            MU = mu.flatten(start_dim=1)    # (B,4)
            dist = Independent(Normal(loc=MU, scale=torch.ones_like(MU)), 1)  # event_dim=1
            log_p = dist.log_prob(X)  # (B,)
            nll_mlp_list.append(-log_p)

            # KL(q || p) approx. E[log q(x)] - E[log p(x)] at gt x
            keys = [(int(env_id[i]), int(t_query[i].item()), int(obj_id[i])) for i in range(env_id.shape[0])]
            log_q = self.compute_log_q(x_1=x_1, keys=keys)  # (B,)
            kl_list.append((log_q - log_p))

        # Aggregate like your trainer.py
        val_loss = float(torch.tensor(mse_vals).mean().item())
        nll_mlp  = float(torch.cat(nll_mlp_list).sum().item())
        kl_mlp   = float(torch.cat(kl_list).mean().item())

        statistics = {
            "val_loss": val_loss,
            "nll/mlp": nll_mlp,
            "kl/mlp": kl_mlp,
        }
        
        return statistics

    @torch.no_grad()
    def compute_log_q(self, x_1: Tensor, keys: List[Tuple[int, int, int]]) -> Tensor:
        """
        x_1: (B,1,4) in [y,x,h,w] -> transform to u=(cy,cx,h,w) and evaluate the GMM at the same points.
        """
        if self.gaussians_gt is None:
            self.gaussians_gt = self._prepare_gaussians()
        gmms = self.gaussians_gt

        y, x, h, w = x_1[:, 0, :].unbind(-1)                    # (B,)
        x = torch.stack([y + 0.5*h, x + 0.5*w, h, w], dim=-1)   # (B, 4)
        
        logs = []
        for i, key in enumerate(keys):
            g = gmms[key]
            logs.append(g["dist"].log_prob(x[i]))
            
        return torch.stack(logs, dim=0)  # (B,)
    
    
    def _prepare_gaussians(self):
        statistics = compute_occurrences(self.val_dataloader.dataset)
        gmms = build_gmms_from_stats(statistics, 
                                     beta_pos=0.35, 
                                     beta_size=0.02,
                                     return_dists_params=True, 
                                     device=self.device)
        return gmms
import math
