from typing import Union
from dataclasses import dataclass, field

from abc import ABC, abstractmethod

from torch import Tensor
import torch


@dataclass
class SchedulerOutput:
    r"""Represents a sample of a conditional-flow generated probability path.

    Attributes:
        alpha_t (Tensor): :math:`\alpha_t`, shape (...).
        sigma_t (Tensor): :math:`\sigma_t`, shape (...).
        d_alpha_t (Tensor): :math:`\frac{\partial}{\partial t}\alpha_t`, shape (...).
        d_sigma_t (Tensor): :math:`\frac{\partial}{\partial t}\sigma_t`, shape (...).

    """

    alpha_t: Tensor = field(metadata={"help": "alpha_t"})
    beta_t: Tensor = field(metadata={"help": "beta_t"})
    d_alpha_t: Tensor = field(metadata={"help": "Derivative of alpha_t."})
    d_beta_t: Tensor = field(metadata={"help": "Derivative of beta_t."})
    sigma_t: Tensor = field(metadata={"help": "Standard deviation of prob. path."})

class Scheduler(ABC):
    """Base Scheduler class."""

    @abstractmethod
    def __call__(self, t: Tensor) -> SchedulerOutput:
        r"""
        Args:
            t (Tensor): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: :math:`\alpha_t,\sigma_t,\frac{\partial}{\partial t}\alpha_t,\frac{\partial}{\partial t}\sigma_t`
        """
        pass

class CondOTScheduler(Scheduler):
    """Conditional Optimal Transport Scheduler.

    Adapted from https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/path/scheduler/scheduler.py"""

    def __init__(self, sigma: Union[float, int] = 0.0):
        """_summary_

        Args:
            sigma (Union[float, int], optional): standard deviation of the conditional path. Defaults to 0.0.
        """
        self.sigma = sigma

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            beta_t=1 - t,
            d_alpha_t=torch.ones_like(t),
            d_beta_t=-torch.ones_like(t),
            sigma_t=torch.ones_like(t) * self.sigma,
        )