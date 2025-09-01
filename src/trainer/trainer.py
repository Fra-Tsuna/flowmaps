import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from diffusers.training_utils import EMAModel

import os
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from functools import partial
from collections import defaultdict

from omegaconf import DictConfig

from src.models.paths import ProbPath
from src.trainer.checkpoint_manager import CheckpointManager
from src.utils.viz import setup_savefig
from src.utils.torch_utils import sample_logit_normal, sample_beta, cycle
from src.utils.statistics import compute_occurrences, build_gmms_from_stats
from src.flowmaps.flowmaps import FlowMapsPipeline, FMTester

import logging
import wandb

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Training pipeline for the model.
    This class handles the training process.
    Args:
        model (MLP): The model to be trained.
        iterations (int): Number of training iterations.
        dataloader (DataLoader): DataLoader for the training data.
        path (ProbPath): ProbPath for the Flow Matching process.
        optimizer (Optimizer): Optimizer for the training process.
        device (torch.device): Device to run the training on.
        logger (logging.Logger): Logger for logging messages.
        result_dir (str): Directory to save the results.
    """

    def __init__(
        self,
        model: nn.Module,
        sampler: FlowMapsPipeline,
        iterations: int,
        dataloaders: Dict[str, DataLoader],
        path: ProbPath,
        optimizer: Optimizer,
        lr_scheduler: LambdaLR,
        seed: int, 
        t_sampling_strategy: str,
        result_dir: str,
        validate_every: int = 100,
        patience: int = 10,
        max_tokens: int = 128,
        use_ema: bool = False,
        tester: DictConfig = None,
        ema: DictConfig = None,
    ):

        self.vf = model
        self.sampler = sampler
        self.sampler.cfg_solver.return_intermediates = False
        self.train_dataloader = dataloaders["train"]
        self.val_dataloader = dataloaders["val"]
        self.path = path
        self.iterations = iterations

        self.optim = optimizer
        self.lr_scheduler = lr_scheduler
        self.iterations = iterations + 1
        # self.guidance_scale = self.sampler.guidance_scale
        self.use_ema = use_ema

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.result_dir = result_dir
        self.checkpoint_dir = os.path.join(self.result_dir, "checkpoints")

        self.savefig = setup_savefig(res_path=result_dir, fig_fmt="png", dpi=300, transparent_png=False)
        self.manager = CheckpointManager(checkpoints_path=self.checkpoint_dir, patience=patience)
        # Printing configuration
        self.validate_every = validate_every
        # Get t sampler
        if t_sampling_strategy == "uniform":
            self.t_sampler = torch.rand
        elif t_sampling_strategy == "logit_normal":
            self.t_sampler = sample_logit_normal
        elif t_sampling_strategy == "beta":
            self.t_sampler = partial(sample_beta, s=0.999)
        else:
            raise ValueError(f"Unknown sampling strategy: {t_sampling_strategy}")
        
        self.max_tokens = max_tokens
        self.cfg_tester = tester
        self.seed = seed

        # If EMA model is provided, wrap the model with EMA
        self.ema: Optional[EMAModel] = None
        if self.use_ema:
            self.ema = EMAModel(
                parameters=self.vf.parameters(),
                **ema)

        self.logged_imgs = set()  # To keep track of logged images
        self.gaussians_gt = None  # For KL computation

    def train(self, run):
        
        self.vf.train()

        # Run the code for num_iterations of iterations instead of epochs, gives a bit more control
        iterator = cycle(self.train_dataloader)

        pbar = tqdm(range(self.iterations), desc="Training", total=self.iterations, dynamic_ncols=True, leave=True)
        metrics = {"train_loss": torch.nan, "val_loss": torch.nan, "nll": torch.nan}

        for i in pbar:
            data = {k: v.to(self.device) for k, v in next(iterator).items()}
            
            t_query    = data["t_query"]      # (B,)
            t_current  = data["t_current"]    # (B,)
            mask       = data["mask"]         # (B, S)
            map        = data["map"]          # (B, S, 5)
            obj_q      = data["query_object"] # (B, 1, 5)
            types      = data["types"]        # (B, S)
        
            B  = t_query.shape[0]  # Batch size
            
            #Fetching the Q for the cross attention
            tgt_obj_bb = obj_q[:,:,:4]  # (B, 1, 4)
            tgt_obj_col = obj_q[:,:,4] # (B, 1,)
            
            x_1 = tgt_obj_bb # (B, 1, 4)
            x_1_q = tgt_obj_col # (B, 1,)

            x_0 = torch.randn_like(x_1)  # (B, 1, 4)
            t = self.t_sampler(B).to(self.device)         
            path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
            
            x_t = path_sample.x_t  # (B, 1, 4)
            
            prediction = self.vf(x = x_t, 
                        t = t,
                        obj = x_1_q,
                        map = map, 
                        tau0 = t_current,
                        tau = t_query,
                        types = types,
                        key_padding_mask = mask) # (B, 1, out_dim)
            
            # flow matching l2 loss
            loss = torch.pow(prediction - path_sample.dx_t, 2).mean()
            metrics["train_loss"] = loss.item()
            lr = self.lr_scheduler.get_last_lr()[0]

            # optimizer step
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            self.lr_scheduler.step()
            if self.use_ema:
                self.ema.step(self.vf.parameters())

            pbar.set_postfix(train_loss=f"{metrics['train_loss']:.3f}", val_loss=f"{metrics['val_loss']:.3f}")
            run.log({"train_loss": metrics["train_loss"], "lr": lr, "step": i}, step=i)

            if (i + 1) % self.validate_every == 0:
                statistics = self.validate(i)
                metrics["val_loss"] = statistics["val_loss"]
                metrics["nll/flowmaps"] = statistics["nll/flowmaps"]
                metrics["nll/zero"] = statistics["nll/zero"]
                metrics["nll/diag"] = statistics["nll/diag"]
                metrics["nll/full"] = statistics["nll/full"]
                metrics["kl"] = statistics["kl"]
                
                run.log({"val_loss": metrics["val_loss"],
                        "nll/flowmaps": metrics["nll/flowmaps"],
                        "nll/zero": metrics["nll/zero"],
                        "nll/diag": metrics["nll/diag"],
                        "nll/full": metrics["nll/full"],
                        "kl": metrics["kl"],
                        "step": i,
                        }, step=i)

                if self.use_ema:
                    # Temporarily store and copy EMA weights to vf
                    self.ema.store(self.vf.parameters())
                    self.ema.copy_to(self.vf.parameters())
                self.manager.save_if_best(
                    model=self.vf,
                    optimizer=self.optim,
                    ema=self.ema,
                    lr_scheduler=self.lr_scheduler,
                    val_loss=metrics["val_loss"],
                    iteration=i,
                    model_args=self.vf.model_args,
                    run=run,
                )
                if self.use_ema:
                    # Restore original weights to avoid affecting training
                    self.ema.restore(self.vf.parameters())
                self.vf.train()

        if self.use_ema:
            # Temporarily store and copy EMA weights to vf
            self.ema.store(self.vf.parameters())
            self.ema.copy_to(self.vf.parameters())
        self.manager.save(
            model=self.vf,
            optimizer=self.optim,
            ema=self.ema,
            lr_scheduler=self.lr_scheduler,
            iteration=self.iterations,
            f_name="last.pth",
            model_args=self.vf.model_args,
            run=run,
        )
        if self.use_ema:
            # Restore original weights to avoid affecting training
            self.ema.restore(self.vf.parameters())

    def validate(self, iteration: int) -> Dict[str, float]:
        
        # Start validation
        if self.use_ema:
            self.ema.store(self.vf.parameters())    # Save current weights
            self.ema.copy_to(self.vf.parameters())  # Overwrite with EMA weights
            
        # Set model to eval mode
        self.vf.eval()
        
        val_loss: List[float] = []
        kl:       List[float] = []
        nll:      List[float] = []
        nll_zero: List[float] = []
        nll_diag: List[float] = []
        nll_full: List[float] = []

        pbar = tqdm(self.val_dataloader, desc="Validation", total=len(self.val_dataloader), dynamic_ncols=True)
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
            
            B  = t_query.shape[0]  # Batch size
            
            #Fetching the Q for the cross attention
            tgt_obj_bb = obj_q[:,:,:4]  # (B, 1, 4)
            tgt_obj_col = obj_q[:,:,4] # (B, 1,)
            
            x_1 = tgt_obj_bb # (B, 1, 4)
            x_1_q = tgt_obj_col # (B, 1,)

            x_0 = torch.randn_like(x_1)  # (B, 1, 4)
            t = self.t_sampler(B).to(self.device)  

            # Disable gradient computation for loss calculation
            with torch.inference_mode():
                path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
                x_t = path_sample.x_t  # (B, 1, 4)
                prediction = self.vf(x = x_t, 
                            t = t,
                            obj = x_1_q,
                            map = map, 
                            tau0 = t_current,
                            tau = t_query,
                            types = types,
                            key_padding_mask = mask) # (B, 1, out_dim)
                
                # Validation loss
                loss = torch.pow(prediction - path_sample.dx_t, 2).mean()
            val_loss.append(loss.item())
                
            # Negative log likelihood computation
            model_extras = {
                "obj": x_1_q,
                "map": map,
                "tau0": t_current,
                "tau": t_query,
                "types": types,
                "key_padding_mask": mask
            }
            # We need gradient enabled to compute log likelihood
            log_p = self.compute_log_p(x_1=x_1, model_extras=model_extras)
            nll.append((-log_p).detach())
            
            # Compute KL divergence
            keys = [(int(env_id[i]), int(t_query[i]), int(obj_id[i])) for i in range(env_id.shape[0])]
            log_q = self.compute_log_q(x_1=x_1, keys=keys)
            kl.append((log_q - log_p).detach())

            with torch.no_grad():
                # Computing other NLL baselines
                X = x_1.flatten(start_dim=1)  # (B, D)
                B, D = X.shape
                
                # Zero-field / identity baseline
                nll_zero_ = 0.5 * (X.pow(2).sum(dim=1) + D * math.log(2*math.pi))
                nll_zero.append(nll_zero_)
                
                # Diagonal Gaussian baseline
                mu  = X.mean(dim=0, keepdim=True)
                var = X.var(dim=0, unbiased=False, keepdim=True) + 1e-8
                diag  = ((X - mu)**2 / var).sum(dim=1)
                const_diag = D * math.log(2*math.pi) + torch.log(var).sum(dim=1)
                nll_diag_batch = 0.5 * (diag + const_diag)
                nll_diag.append(nll_diag_batch)
                
                # Full Gaussian baseline
                Xc = X - mu
                Sigma = (Xc.T @ Xc) / B + 1e-6 * torch.eye(D, device=X.device, dtype=X.dtype)
                L = torch.linalg.cholesky(Sigma)
                Z = torch.linalg.solve_triangular(L, Xc.T, upper=False)
                m2 = (Z**2).sum(dim=0)
                logdet = 2.0 * torch.log(torch.diag(L)).sum()
                nll_full_batch = 0.5 * (m2 + D * math.log(2*math.pi) + logdet)
                nll_full.append(nll_full_batch)
            
        val_loss = torch.mean(torch.tensor(val_loss))
        kl       = torch.mean(torch.cat(kl))
        nll      = torch.sum(torch.cat(nll))
        nll_zero = torch.sum(torch.cat(nll_zero))
        nll_diag = torch.sum(torch.cat(nll_diag))
        nll_full = torch.sum(torch.cat(nll_full))

        self.log_images(iteration)
        
        # Resume training
        if self.use_ema:
            self.ema.restore(self.vf.parameters())  # Restore original weights
        self.vf.train()
        
        statistics = {
            "val_loss": val_loss.item(),
            "nll/flowmaps":  nll.item(),
            "nll/zero": nll_zero.item(),
            "nll/diag": nll_diag.item(),
            "nll/full": nll_full.item(),
            "kl": kl.item()
        }
        
        return statistics
    
    
    def log_images(self, iteration: int):
        
        tester = FMTester(
            model=self.vf,
            pipeline=self.sampler,
            dataset=self.val_dataloader.dataset,
            savefig=self.savefig,
            seed=self.seed
        )

        samples = tester.generate_samples(nsamples=self.cfg_tester.nsamples, npreds=self.cfg_tester.npreds)
        result = tester.inference(samples)
        tester.display_results(result, iteration, self.cfg_tester.scale)
        images_path = os.path.join(self.result_dir, "png")
        pngs = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
        logger.info(f"Logging images to wandb")
        for png in pngs:
            if png in self.logged_imgs:
                continue
            env, obj, _ = png.split("_")
            wandb.log({f"{env}_{obj}": wandb.Image(os.path.join(images_path, png))}, step=iteration)
            self.logged_imgs.add(png)     


    def compute_log_p(self, x_1: Tensor, model_extras: Dict[str, Tensor]) -> Tensor:

        _, log_likelihood = self.sampler.compute_likelihood(
            model=self.vf,
            x_1=x_1,
            model_extras=model_extras
        )
        
        return log_likelihood
    
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
