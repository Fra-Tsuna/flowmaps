import torch
import os
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, checkpoints_path, patience=None):
        """
        Args:
            checkpoint_path (str): Path to save the best checkpoint.
            patience (int or None): Number of epochs to wait without improvement before stopping.
        """
        self.checkpoints_path = checkpoints_path
        self.best_val_loss = float("inf")
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        os.makedirs(checkpoints_path, exist_ok=True)

    def save(self, model, optimizer, ema, lr_scheduler, iteration, f_name, model_args, run=None):
        """
        Save the model checkpoint.

        Args:
            model (torch.nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer to save.
            val_loss (float): The validation loss.
            iteration (int): The current iteration number.
        """
        path = os.path.join(self.checkpoints_path, f_name)
        torch.save(
            {
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "ema_state_dict": ema.state_dict() if ema else None,
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "model_args": model_args,
            },
            path,
        )
        logger.info(f"[Checkpoint] Saved at {path}")
        # Log the checkpoint
        if run:
            # Get parent dir before "checkpoints" to improve logging
            base_path = os.path.dirname(os.path.dirname(path))
            run.save(
                path,
                base_path,
            )
        return True

    def save_if_best(self, model, optimizer, ema, lr_scheduler, val_loss, iteration, model_args, run=None):
        """
        Save checkpoint if validation loss improves.

        Returns:
            bool: True if a new best model was saved.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
            self.save(
                model=model, 
                optimizer=optimizer, 
                ema=ema, 
                lr_scheduler=lr_scheduler, 
                iteration=iteration, 
                f_name="best.pth", 
                model_args=model_args,
                run=run)
        else:
            self.counter += 1
            logger.info(f"[Checkpoint] Validation loss did not improve. Counter: {self.counter}/{self.patience}")
            self.save(
                model=model,
                optimizer=optimizer,
                ema=ema,
                lr_scheduler=lr_scheduler,
                iteration=iteration,
                f_name="last.pth",
                model_args=model_args,
                run=run)
            if self.patience and self.counter >= self.patience:
                logger.info("[Checkpoint] Early stopping triggered.")
                self.early_stop = True

    def load_checkpoint(self, f_name: str, optimizer=None, ema=None, lr_scheduler=None, apply_ema: bool = False):
        """
        Load the checkpoint into model and optionally optimizer/ema/lr_scheduler.

        Args:
            apply_ema: if True, copy EMA weights onto the model (for eval). Mutually
                       exclusive with passing an ema object (which is for training resume).
        Returns:
            model
        """
        path = os.path.join(self.checkpoints_path, f_name)
        if not os.path.isfile(path):
            logger.warning(f"[Checkpoint] No checkpoint found at {path}, returning current model.")
            return 0, float("inf")

        import inspect
        checkpoint = torch.load(path, weights_only=False)
        logger.info(f"[Checkpoint] Loading model from {path} at iteration {checkpoint['iteration']}")
        # Filter stale args so checkpoints saved with older model versions remain loadable.
        model_args = dict(checkpoint["model_args"])
        model_cls = model_args.pop("__class__")
        valid_keys = inspect.signature(model_cls.__init__).parameters.keys() - {"self"}
        model_args = {k: v for k, v in model_args.items() if k in valid_keys}

        # Backward-compat: older checkpoints may store a legacy VAE path
        # (e.g. .../latents/<pattern>/vae_<pattern>.pth) that was later flattened
        # to .../latents/vae.pth. Resolve before model construction.
        if "vae_path" in model_args:
            model_args["vae_path"] = self._resolve_vae_path(model_args["vae_path"])

        model = model_cls(**model_args)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if ema and "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        if lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        if apply_ema and checkpoint.get("ema_state_dict") is not None:
            from diffusers.training_utils import EMAModel
            ema_model = EMAModel(parameters=model.trainable_parameters())
            ema_model.load_state_dict(checkpoint["ema_state_dict"])
            ema_model.copy_to(model.trainable_parameters())
            logger.info("[Checkpoint] Applied EMA weights to model.")

        return model

    def _infer_mode_from_checkpoints_path(self):
        ckpt_path = Path(self.checkpoints_path).resolve()
        parts = ckpt_path.parts

        # Typical layout: <project>/ckpt/<mode>
        if "ckpt" in parts:
            idx = parts.index("ckpt")
            if idx + 1 < len(parts):
                mode = parts[idx + 1]
                if mode != "checkpoints":
                    return mode

        # Fallback for known modes appearing elsewhere in the path.
        for mode in ("habit1", "habit2", "habit3"):
            if mode in parts:
                return mode

        return None

    def _resolve_vae_path(self, vae_path: str) -> str:
        original = Path(vae_path).expanduser()
        if original.is_file():
            return str(original)

        candidates = []

        # In-place flatten fallback:
        # .../latents/<pattern>/vae_<pattern>.pth -> .../latents/vae.pth
        candidates.append(original.parent / "vae.pth")
        if original.parent.parent.name == "latents":
            candidates.append(original.parent.parent / "vae.pth")

        # Derive from current checkpoint mode:
        # <project>/ckpt/<mode> + model flattening -> <project>/data/<mode>/latents/vae.pth
        mode = self._infer_mode_from_checkpoints_path()
        ckpt_path = Path(self.checkpoints_path).resolve()
        if mode is not None and "ckpt" in ckpt_path.parts:
            idx = ckpt_path.parts.index("ckpt")
            project_root = Path(*ckpt_path.parts[:idx])
            candidates.append(project_root / "data" / mode / "latents" / "vae.pth")

        # De-duplicate while preserving order.
        seen = set()
        deduped = []
        for c in candidates:
            s = str(c)
            if s in seen:
                continue
            seen.add(s)
            deduped.append(c)

        for candidate in deduped:
            if candidate.is_file():
                logger.warning(
                    "[Checkpoint] Resolved missing legacy vae_path '%s' -> '%s'",
                    original,
                    candidate,
                )
                return str(candidate)

        attempted = "\n".join(f"  - {c}" for c in deduped)
        raise FileNotFoundError(
            "[Checkpoint] VAE file not found.\n"
            f"  checkpoint vae_path: {original}\n"
            "  attempted fallbacks:\n"
            f"{attempted if attempted else '  (none)'}"
        )
