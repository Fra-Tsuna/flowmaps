import torch
import os

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

    def load_checkpoint(self, f_name: str, optimizer=None, ema=None, lr_scheduler=None):
        """
        Load the checkpoint into model and optionally optimizer.

        Returns:
            int: epoch to resume from
            float: best_val_loss
        """
        path = os.path.join(self.checkpoints_path, f_name)
        # Check if file exists
        if not os.path.isfile(path):
            logger.warning(f"[Checkpoint] No checkpoint found at {path}, returning current model.")
            return 0, float("inf")

        checkpoint = torch.load(path, weights_only=False)
        logger.info(f"[Checkpoint] Loading model from {path} at iteration {checkpoint['iteration']}")
        # Load model class and instantiate
        model_args = checkpoint["model_args"]
        model_cls = model_args.pop("__class__")
        model = model_cls(**model_args)

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if ema and "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        if lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        return model