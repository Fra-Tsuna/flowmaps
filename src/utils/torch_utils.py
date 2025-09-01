import torch
import numpy as np
import os
import random
import itertools

from torch import Tensor
from torch import nn

MiB = 1024 ** 2

def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    Args:
    - model: self-explanatory
    Returns:
    - size: model size in bytes
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size

def expand_tensor_like(input_tensor: Tensor, expand_to: Tensor) -> Tensor:
    """`input_tensor` is a 1d vector of length equal to the batch size of `expand_to`,
    expand `input_tensor` to have the same shape as `expand_to` along all remaining dimensions.
    
    Code taken from: https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/utils/utils.py

    Args:
        input_tensor (Tensor): (batch_size,).
        expand_to (Tensor): (batch_size, ...).

    Returns:
        Tensor: (batch_size, ...).
    """
    assert input_tensor.ndim == 1, "Input tensor must be a 1d vector."
    assert (
        input_tensor.shape[0] == expand_to.shape[0]
    ), f"The first (batch_size) dimension must match. Got shape {input_tensor.shape} and {expand_to.shape}."

    dim_diff = expand_to.ndim - input_tensor.ndim

    t_expanded = input_tensor.clone()
    t_expanded = t_expanded.reshape(-1, *([1] * dim_diff))

    return t_expanded.expand_as(expand_to)

def seed_everything(seed=42):
    """
    Taken from https://github.com/Lightning-AI/pytorch-lightning/issues/1565
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

def sample_logit_normal(size: int) -> Tensor:
    """
    Sample from a logit-normal distribution
    
    Parameters:
        size (int): Number of samples to generate.
    
    Returns:
        torch.Tensor: Samples from the logit-normal distribution.
    """
    normal_samples = torch.randn(size)
    logit_samples = torch.sigmoid(normal_samples)
    return logit_samples

def sample_beta(size: int, s = 0.999) -> Tensor:
    """
    Samples τ ~ p(τ) defined by τ = s * (1 - u), where u ~ Beta(1.5, 1)
    """
    u = torch.distributions.Beta(torch.tensor([1.5]), torch.tensor([1.0])).sample(torch.tensor([size]))
    tau = s * (1.0 - u)
    return tau.squeeze() 