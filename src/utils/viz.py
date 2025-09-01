from typing import Optional, Union, List, Tuple, Callable

from functools import partial

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib import cm
from matplotlib.figure import Figure

import re
import imageio

from torch import Tensor
from numpy import ndarray

import logging

from src.utils.lookup import *

logger = logging.getLogger(__name__)

def save_fig(
        fig: Figure, 
        fig_name: str, 
        fig_dir: str, 
        fig_fmt: str,
        fig_size: Tuple[float, float] = [6.4, 4], 
        save: bool = True, 
        dpi: int = 300,
        transparent_png = True,
    ):
    """
    Code adapted from https://zhauniarovich.com/post/2022/2022-09-matplotlib-graphs-in-research-papers/
    This procedure stores the generated matplotlib figure to the specified 
    directory with the specified name and format.

    Parameters
    ----------
    fig : [type]
        Matplotlib figure instance
    fig_name : str
        File name where the figure is saved
    fig_dir : str
        Path to the directory where the figure is saved
    fig_fmt : str
        Format of the figure, the format should be supported by matplotlib 
        (additional logic only for pdf and png formats)
    fig_size : Tuple[float, float]
        Size of the figure in inches, by default [6.4, 4] 
    save : bool, optional
        If the figure should be saved, by default True. Set it to False if you 
        do not want to override already produced figures.
    dpi : int, optional
        Dots per inch - the density for rasterized format (png), by default 300
    transparent_png : bool, optional
        If the background should be transparent for png, by default True
    """
    if not save:
        return
    
    fig.set_size_inches(fig_size, forward=False)
    fig_fmt = fig_fmt.lower()
    fig_dir = os.path.join(fig_dir, fig_fmt)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    pth = os.path.join(
        fig_dir,
        '{}.{}'.format(fig_name, fig_fmt.lower())
    )
    if fig_fmt == 'pdf':
        metadata={
            'Creator' : '',
            'Producer': '',
            'CreationDate': None
        }
        fig.savefig(pth, bbox_inches='tight', metadata=metadata)
    elif fig_fmt == 'png':
        alpha = 0 if transparent_png else 1
        axes = fig.get_axes()
        fig.patch.set_alpha(alpha)
        for ax in axes:
            ax.patch.set_alpha(alpha)
        fig.savefig(
            pth, 
            bbox_inches='tight',
            dpi=dpi,
        )
    else:
        try:
            fig.savefig(pth, bbox_inches='tight')
        except Exception as e:
            logger.error("Cannot save figure: {}".format(e)) 

def setup_savefig(res_path: str, fig_fmt: str = "pdf", dpi: int = 300, transparent_png = True) -> Callable:
    """
    Set up the figure saving directory and format.

    Parameters
    ----------
    res_path : str
        Path to the directory where the figure is saved
    fig_fmt : str, optional
        Format of the figure, by default "pdf"
    dpi : int, optional
        Dots per inch - the density for rasterized format (png), by default 300
    transparent_png : bool, optional
        If the background should be transparent for png, by default True
    """
    if res_path is None:
        res_path = os.path.join(os.path.abspath(os.curdir), "assets", datetime.datetime.now().isoformat(), exp_name)
    logger.info(f"Saving results to {res_path}")
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    return partial(save_fig, fig_dir=res_path, fig_fmt=fig_fmt, transparent_png=transparent_png, dpi=dpi)