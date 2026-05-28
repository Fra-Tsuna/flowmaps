import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from diffusers.training_utils import EMAModel

import os
from tqdm import tqdm
from typing import Dict, List, Optional
from functools import partial

from omegaconf import DictConfig

from src.models.paths import ProbPath
from src.trainer.checkpoint_manager import CheckpointManager
from src.models.optimal_transport import OTPlanSampler
from src.utils.viz import setup_savefig
from src.utils.torch_utils import sample_logit_normal, sample_beta, cycle
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
        use_coupling: bool = False,
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
        # self.guidance_scale = self.sampler.guidance_scale
        self.use_ema = use_ema

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.result_dir = result_dir
        self.checkpoint_dir = os.path.join(self.result_dir, "checkpoints")

        self.savefig = setup_savefig(res_path=result_dir, fig_fmt="png", dpi=300, transparent_png=False)
        self.ot_sampler = OTPlanSampler(method="exact")
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
            self.ema = EMAModel(parameters=self.vf.trainable_parameters(), **ema)

        self.use_coupling = use_coupling
        self.logged_imgs = set()  # To keep track of logged images

    def train(self, run):

        self.vf.train()  # DiT.train() override keeps vae in eval
        mean = torch.tensor(self.train_dataloader.dataset.latent_statistics["mean"], device=self.device)
        std = torch.tensor(self.train_dataloader.dataset.latent_statistics["std"], device=self.device)

        # Run the code for num_itereations of iterations instead of epochs, gives a bit more control
        iterator = cycle(self.train_dataloader)

        pbar = tqdm(range(self.iterations), desc="Training", total=self.iterations, dynamic_ncols=True, leave=True)
        metrics = {"train_loss": torch.nan, "val_loss": torch.nan, "val_nll": torch.nan}

        for i in pbar:
            # Only move tensors to device; keep non-tensor metadata on CPU.
            data = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in next(iterator).items()}

            t_query = data["t_query"]  # (B,)
            t_current = data["t_current"]  # (B,)
            mask = data["mask"]  # (B, S)
            map = data["map"]  # (B, S, 8)
            obj_q = data["query_object"]  # (B, 1, 8)
            types = data["types"]  # (B, S)

            # Get query object class label
            obj_cls = obj_q[:, :, -1]  # (B, 1), label is at -1
            B, _, D = obj_q.shape

            # Encode and normalize query
            # README: use mean?
            _mu, _logvar, query_latents = self.vf.encode_query(obj_q.view(B, D))
            query_latents = (query_latents.view(B, 1, -1) - mean) / std

            x_0 = torch.randn_like(query_latents)  # (B, 1, latent_dim)
            x_1 = query_latents
            if self.use_coupling:
                x_0, x_1, map, obj_cls, t_current, t_query, types, mask = self.prepare_batch_coupling(
                    x_0, x_1, map, obj_cls, t_current, t_query, types, mask
                )
            t = self.t_sampler(B).to(self.device)
            path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)

            prediction = self.vf(
                x=path_sample.x_t,
                t=t,
                obj=obj_cls,
                map=map,
                tau0=t_current,
                tau=t_query,
                types=types,
                key_padding_mask=mask,
            )  # (B, 1, latent_dim)

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
                self.ema.step(self.vf.trainable_parameters())

            pbar.set_postfix(train_loss=f"{metrics['train_loss']:.3f}", val_loss=f"{metrics['val_loss']:.3f}")
            run.log({"train/loss": metrics["train_loss"], "lr": lr}, step=i)

            if (i + 1) % self.validate_every == 0:
                val_metrics = self.validate(i)
                metrics["val_loss"] = val_metrics["val_loss"]
                metrics["nll"] = val_metrics["nll"]
                run.log({"val/loss": metrics["val_loss"], "val/nll": metrics["nll"]}, step=i)
                if self.use_ema:
                    # Temporarily store and copy EMA weights to vf
                    self.ema.store(self.vf.trainable_parameters())
                    self.ema.copy_to(self.vf.trainable_parameters())
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
                    self.ema.restore(self.vf.trainable_parameters())
                self.vf.train()  # DiT.train() override keeps vae in eval
                # Early stopping: TODO: fix this, it;s not working
                # if self.manager.early_stop:
                #     return

        if self.use_ema:
            # Temporarily store and copy EMA weights to vf
            self.ema.store(self.vf.trainable_parameters())
            self.ema.copy_to(self.vf.trainable_parameters())
        self.manager.save(
            model=self.vf,
            optimizer=self.optim,
            ema=self.ema,
            lr_scheduler=self.lr_scheduler,
            iteration=self.iterations - 1,
            f_name="last.pth",
            model_args=self.vf.model_args,
            run=run,
        )
        if self.use_ema:
            # Restore original weights to avoid affecting training
            self.ema.restore(self.vf.trainable_parameters())

    def prepare_batch_coupling(self, x_0, x_1, map, obj_cls, t_current, t_query, types, mask):
        """Apply minibatch OT coupling: only x_0 is permuted to minimise ||x_0 - x_1||².
        x_1 and all conditioning stay in their original order so semantic alignment is preserved.

        See: Tong et al. 2023 (https://arxiv.org/abs/2302.00482)
             Pooladian et al. 2023 (https://arxiv.org/abs/2304.14772)
        """
        B, _, latent_dim = x_0.shape
        x_0_coupled, _, i, _ = self.ot_sampler.sample_plan_with_indices(
            x_0.view(B, -1), x_1.view(B, -1)
        )
        return (
            x_0_coupled.view(B, 1, latent_dim),
            x_1,
            map,
            obj_cls,
            t_current,
            t_query,
            types,
            mask,
        )

    def validate(self, iteration: int) -> float:

        # Start validation
        if self.use_ema:
            self.ema.store(self.vf.trainable_parameters())  # Save current weights
            self.ema.copy_to(self.vf.trainable_parameters())  # Overwrite with EMA weights
        # Set model to eval mode
        self.vf.eval()
        mean = torch.tensor(self.val_dataloader.dataset.latent_statistics["mean"], device=self.device)
        std = torch.tensor(self.val_dataloader.dataset.latent_statistics["std"], device=self.device)

        val_loss: List[float] = []
        # val_nll: List[float] = []

        pbar = tqdm(self.val_dataloader, desc="Validation", total=len(self.val_dataloader), dynamic_ncols=True)
        for data in pbar:
            # Only move tensors to device; keep non-tensor metadata on CPU.
            data = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in data.items()}

            t_query = data["t_query"]  # (B,)
            t_current = data["t_current"]  # (B,)
            mask = data["mask"]  # (B, S)
            map = data["map"]  # (B, S, 8)
            obj_q = data["query_object"]  # (B, 1, 8)
            types = data["types"]  # (B, S)

            # Get query object class label
            obj_cls = obj_q[:, :, -1]  # (B, 1), label is at -1

            B, _, D = obj_q.shape

            with torch.inference_mode():
                # Encode and normalize query
                _mu, _logvar, query_latents = self.vf.encode_query(obj_q.view(B, D))
                query_latents = (query_latents.view(B, 1, -1) - mean) / std

                x_0 = torch.randn_like(query_latents)  # (B, 1, latent_dim)
                x_1 = query_latents
                t = self.t_sampler(B).to(self.device)
                path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)

                prediction = self.vf(
                    x=path_sample.x_t,
                    t=t,
                    obj=obj_cls,
                    map=map,
                    tau0=t_current,
                    tau=t_query,
                    types=types,
                    key_padding_mask=mask,
                )  # (B, 1, out_dim)

                # Validation loss
                loss = torch.pow(prediction - path_sample.dx_t, 2).mean()
            val_loss.append(loss.item())

            # # Negative log likelihood computation
            # model_extras = {
            #     "obj": obj_cls,
            #     "map": map,
            #     "tau0": t_current,
            #     "tau": t_query,
            #     "types": types,
            #     "key_padding_mask": mask,
            # }
            # # We need gradient enabled to compute log likelihood
            # log_p = self.compute_log_p(x_1=x_1.detach().clone().requires_grad_(True), model_extras=model_extras)
            # val_nll.append(log_p.detach())

            pbar.set_postfix(val_loss=f"{loss.item():.3f}")

        val_loss = torch.mean(torch.tensor(val_loss))
        # val_nll = -torch.sum(torch.cat(val_nll))

        self.log_images(iteration)

        # Resume training
        if self.use_ema:
            self.ema.restore(self.vf.trainable_parameters())  # Restore original weights
        self.vf.train()  # DiT.train() override keeps vae in eval
        return {"val_loss": val_loss.item(), "nll": float("nan")}

    def log_images(self, iteration: int):

        tester = FMTester(
            model=self.vf,
            pipeline=self.sampler,
            dataset=self.val_dataloader.dataset,
            savefig=self.savefig,
            seed=self.seed,
        )

        samples = tester.generate_samples(nsamples=self.cfg_tester.nsamples, npreds=self.cfg_tester.npreds)
        result = tester.inference(samples)
        tester.display_results(result, iteration)
        images_path = os.path.join(self.result_dir, "png")
        pngs = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
        logger.info(f"Logging images to wandb")
        for png in pngs:
            if png in self.logged_imgs:
                continue
            # Filename format: {env_name}_object{j}_idx{iteration}.png
            # env_name may contain underscores, so split from the right on "_object"
            stem = png.rsplit(".png", 1)[0]  # remove extension
            key = stem.rsplit("_idx", 1)[0]  # remove _idx{iteration} suffix
            wandb.log({key: wandb.Image(os.path.join(images_path, png))}, step=iteration)
            self.logged_imgs.add(png)

    def compute_log_p(self, x_1: torch.Tensor, model_extras: Dict[str, torch.Tensor]) -> torch.Tensor:

        _, log_likelihood = self.sampler.compute_likelihood(model=self.vf, x_1=x_1, model_extras=model_extras)

        return log_likelihood
