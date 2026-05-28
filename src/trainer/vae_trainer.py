import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from diffusers.training_utils import EMAModel

import os
from tqdm import tqdm
from typing import Dict, Optional

from omegaconf import DictConfig

from src.trainer.checkpoint_manager import CheckpointManager
from src.models.optimal_transport import OTPlanSampler
from src.utils.viz import setup_savefig
from src.utils.torch_utils import cycle
from src.flowmaps.flowmaps import VAETester

import logging
import wandb

logger = logging.getLogger(__name__)


class VAETrainer:
    """
    Training pipeline for the VAE model.
    """

    def __init__(
        self,
        model: nn.Module,
        iterations: int,
        dataloaders: Dict[str, DataLoader],
        optimizer: Optimizer,
        lr_scheduler: LambdaLR,
        seed: int,
        result_dir: str,
        hyperparams: DictConfig,
        validate_every: int = 100,
        patience: int = 10,
        max_tokens: int = 128,
        use_ema: bool = False,
        ema: DictConfig = None,
        viz_args: DictConfig = None,
        **kwargs,
    ):

        self.vae = model
        self.train_dataloader = dataloaders["train"]
        self.val_dataloader = dataloaders["val"]
        self.iterations = iterations

        self.optim = optimizer
        self.lr_scheduler = lr_scheduler
        self.use_ema = use_ema

        self.hyperparams = hyperparams

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.result_dir = result_dir
        self.checkpoint_dir = os.path.join(self.result_dir, "checkpoints")

        self.savefig = setup_savefig(res_path=result_dir, fig_fmt="png", dpi=300, transparent_png=False)
        self.ot_sampler = OTPlanSampler(method="exact")
        self.manager = CheckpointManager(checkpoints_path=self.checkpoint_dir, patience=patience)
        # Printing configuration
        self.validate_every = validate_every

        self.max_tokens = max_tokens
        self.seed = seed

        # If EMA model is provided, wrap the model with EMA
        self.ema: Optional[EMAModel] = None
        if self.use_ema:
            self.ema = EMAModel(parameters=self.vae.parameters(), **ema)

        self.viz_args = viz_args
        self.logged_imgs = set()  # To keep track of logged images

    def train(self, run):

        self.vae.train()

        # Run the code for num_itereations of iterations instead of epochs, gives a bit more control
        iterator = cycle(self.train_dataloader)

        pbar = tqdm(range(self.iterations), desc="Training", total=self.iterations, dynamic_ncols=True, leave=True)
        val_loss = float("inf")

        kl_warmup_steps = self.hyperparams.get("kl_warmup_steps", 0)
        beta_target = self.hyperparams["beta"]

        for i in pbar:
            # Only move tensors to device; keep non-tensor metadata on CPU.
            data = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in next(iterator).items()}

            map = data["map"]  # (B, S, 8), or 5 in the 2d case
            types = data["types"]  # (B, S)

            obj_mask = types == 1  # (B, S)
            obj_tokens = map[obj_mask]  # (B * Nobj, 8)

            gt_bbox = obj_tokens[:, :7]  # (B * Nobj, 7) # or 4 in 2d case
            gt_yaw = obj_tokens[:, 6]  # (B * Nobj,)
            gt_cls = obj_tokens[:, 7].long()  # (B * Nobj,)
            mu, logvar, bbox, cls_logits = self.vae(x=obj_tokens, sample=True)  #(B * Nobj, latent_dim), (B * Nobj, latent_dim), (B * Nobj, 6), (B * Nobj, n_classes)

            # KL annealing: linearly ramp beta from 0 to beta_target over kl_warmup_steps
            if kl_warmup_steps > 0:
                beta = beta_target * min(1.0, i / kl_warmup_steps)
            else:
                beta = beta_target
            annealed_hyperparams = {**self.hyperparams, "beta": beta}

            gt_bbox = gt_bbox[:, :6]  #FIXME for now ignore the yaw
            train_loss_dict = self.vae.compute_loss(
                mu=mu,
                logvar=logvar,
                bbox=bbox,
                cls_logits=cls_logits,
                target_bbox=gt_bbox,
                target_cls=gt_cls,
                hyperparams=annealed_hyperparams,
            )
            
            loss = train_loss_dict["loss"]
            train_dict = {f"train/{k}": v.detach().item() for k, v in train_loss_dict.items()}
            train_dict["train/beta"] = beta

            lr = self.lr_scheduler.get_last_lr()[0]

            # optimizer step
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            self.lr_scheduler.step()
            if self.use_ema:
                self.ema.step(self.vae.parameters())

            pbar.set_postfix(train_loss=f"{loss:.3f}", val_loss=f"{val_loss:.3f}")
            run.log({**train_dict, "lr": lr}, step=i)

            if (i + 1) % self.validate_every == 0:
                val_loss = self.validate(run, i)
                if self.use_ema:
                    # Temporarily store and copy EMA weights to vf
                    self.ema.store(self.vae.parameters())
                    self.ema.copy_to(self.vae.parameters())
                self.manager.save_if_best(
                    model=self.vae,
                    optimizer=self.optim,
                    ema=self.ema,
                    lr_scheduler=self.lr_scheduler,
                    val_loss=val_loss,
                    iteration=i,
                    model_args=self.vae.model_args,
                    run=run,
                )
                if self.use_ema:
                    # Restore original weights to avoid affecting training
                    self.ema.restore(self.vae.parameters())
                self.vae.train()

        if self.use_ema:
            # Temporarily store and copy EMA weights to vf
            self.ema.store(self.vae.parameters())
            self.ema.copy_to(self.vae.parameters())
        self.manager.save(
            model=self.vae,
            optimizer=self.optim,
            ema=self.ema,
            lr_scheduler=self.lr_scheduler,
            iteration=self.iterations - 1,
            f_name="last.pth",
            model_args=self.vae.model_args,
            run=run,
        )
        if self.use_ema:
            # Restore original weights to avoid affecting training
            self.ema.restore(self.vae.parameters())

    def validate(self, run, iteration: int) -> float:

        # Start validation
        if self.use_ema:
            self.ema.store(self.vae.parameters())  # Save current weights
            self.ema.copy_to(self.vae.parameters())  # Overwrite with EMA weights
        # Set model to eval mode
        self.vae.eval()
        n_batches = len(self.val_dataloader)

        with torch.inference_mode():
            val_metrics: Dict[str, float] = {}

            pbar = tqdm(self.val_dataloader, desc="Validation", dynamic_ncols=True)
            for data in pbar:
                # Only move tensors to device; keep non-tensor metadata on CPU.
                data = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in data.items()}

                map = data["map"]  # (B, S, 8), or 5 in the 2d case 
                types = data["types"]  # (B, S)

                obj_mask = types == 1 
                obj_tokens = map[obj_mask]  # (B * Nobj, 8)        
                
                gt_bbox = obj_tokens[:, :7]  # (B * Nobj, 7) # or 4 in 2d case
                gt_yaw = obj_tokens[:, 6]  # (B * Nobj,)
                gt_cls = obj_tokens[:, 7].long()  # (B * Nobj,)
                
                mu, logvar, bbox, cls_logits = self.vae(x=obj_tokens, sample=False)  # (B, S, out_dim)
                
                gt_bbox = gt_bbox[:, :6]  #FIXME for now ignore the yaw
                
                val_loss_dict = self.vae.compute_loss(
                    mu=mu,
                    logvar=logvar,
                    bbox=bbox,
                    cls_logits=cls_logits,
                    target_bbox=gt_bbox,
                    target_cls=gt_cls,
                    hyperparams=self.hyperparams,
                )
                loss = val_loss_dict["loss"]
                for k, v in val_loss_dict.items():
                    if k not in val_metrics:
                        val_metrics[k] = v.item() / n_batches
                    else:
                        val_metrics[k] += v.item() / n_batches 
                pbar.set_postfix(val_loss=f"{loss:.3f}")

            val_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
            run.log(val_metrics, step=iteration)

        self.log_images(iteration)
        # Resume training
        if self.use_ema:
            self.ema.restore(self.vae.parameters())  # Restore original weights
        self.vae.train()
        return val_metrics['val/loss']

    def log_images(self, iteration: int):

        tester = VAETester(
            model=self.vae, dataset=self.val_dataloader.dataset, savefig=self.savefig, seed=self.seed
        )

        samples = tester.generate_samples(nsamples=self.viz_args.samples, max_envs=self.viz_args.get("max_envs"))
        result = tester.inference(samples, img_size=self.viz_args.img_size)
        tester.display_results(result, iteration)
        images_path = os.path.join(self.result_dir, "png")
        pngs = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
        logger.info(f"Logging images to wandb")
        for png in pngs:
            if png in self.logged_imgs:
                continue
            # Parse {env}_sample{idx}_idx{iter} to keep W&B keys stable.
            stem = os.path.splitext(png)[0]
            if "_sample" in stem and "_idx" in stem:
                env_part, sample_part = stem.split("_sample", 1)
                env_name = env_part
                sample_tag = "sample" + sample_part.split("_idx", 1)[0]
                key = f"{env_name}_{sample_tag}"
            else:
                key = stem
            wandb.log({key: wandb.Image(os.path.join(images_path, png))}, step=iteration)
            self.logged_imgs.add(png)
