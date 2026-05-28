from typing import Optional, Union, List, Tuple, Callable, Dict

from functools import partial

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.axes import Axes
from matplotlib import cm
from matplotlib.figure import Figure

from torch import Tensor
from numpy import ndarray

import logging

from src.utils.lookup import get_active_lookup

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

# Several plotting utility functions
def hist2d_samples(samples: Union[Tensor, ndarray], ax: Optional[Axes] = None, bins: int = 200, range: List[List[float]] = [[1., 1.], [1., 1.]], percentile: int = 99, **kwargs):
    if type(samples) == torch.Tensor:
        samples = samples.cpu().numpy()
    assert samples.shape[1] == 2, "samples should be of shape (n_samples, 2)"
    if ax is None:
        ax = plt.gca()
    H, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins, range=range)

    # Determine color normalization based on the 99th percentile
    nonzero_H = H[H > 0]
    if len(nonzero_H) > 0:
        cmax = np.percentile(nonzero_H, percentile)
    else:
        cmax = 1  # Default value if everything is zero
    # cmax = np.percentile(H, percentile)
    cmin = 0.0
    norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)

    # Plot using imshow for more control
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(H.T, extent=extent, origin='lower', norm=norm, **kwargs)


def render_bev(
    max_size: float,
    sample: Dict,
    receptacle_colors: Dict[int, tuple],
    object_colors: Dict[int, tuple],
    img_size: int = 256,
    t: Optional[int] = None,
    min_size: float = 0.0,
):
    """
    Build a BEV image (H x W x 3, float32 in [0, 1]) for time step t.

    World:
        - x, z in [min_size, max_size]
        - origin (min_size, min_size) is bottom-left of the image
        - x -> right, z -> forward/up
    """

    # White background
    img = np.ones((img_size, img_size, 3), dtype=np.float32)
    scene_range = max_size - min_size

    def world_to_pixel(x: float, z: float):
        """
        Map world coordinates (x, z) in [min_size, max_size] to image pixels
        with origin at bottom-left and z increasing upward.

        No flips here because imshow uses origin="lower".
        """
        # clamp to valid range
        x = max(min_size, min(max_size, x))
        z = max(min_size, min(max_size, z))

        px = (x - min_size) / scene_range * (img_size - 1)
        pz = (z - min_size) / scene_range * (img_size - 1)
        col = int(round(px))
        row = int(round(pz))  # no inversion

        # clamp pixel indices
        col = max(0, min(img_size - 1, col))
        row = max(0, min(img_size - 1, row))
        return row, col


    def draw_box(bbox_corners, color):
        """
        bbox_corners: tensor [8, 3]
        color: (r, g, b) in [0, 1]
        """
        xz = bbox_corners[:, [0, 2]].detach().cpu().numpy()  # [8, 2]

        rows = []
        cols = []
        for x, z in xz:
            r, c = world_to_pixel(x, z)
            rows.append(r)
            cols.append(c)

        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)

        img[r0 : r1 + 1, c0 : c1 + 1, :] = np.array(color, dtype=np.float32)[None, None, :]

    # Draw receptacles (static)
    for rec in sample["receptacles"]:
        label = int(rec["label"])
        color = receptacle_colors.get(label, (0.6, 0.6, 0.6))  # fallback gray
        draw_box(rec["bbox_corners"], color)

    
    # Draw objects at time t
    if t is not None:
        sample_object = sample["objects"][t]
    else:
        sample_object = sample["objects"]
    for obj in sample_object:
        label = int(obj["label"])
        color = object_colors.get(label, (0.0, 0.0, 0.0))  # fallback black
        draw_box(obj["bbox_corners"], color)
    
    return torch.from_numpy(img)



def save_bev(
    img,
    receptacle_colors,
    object_colors,
    max_size,
    path,
    figsize=(10, 6),
    title="BEV View",
    legend_sample: Optional[Dict] = None,
    legend_receptacle_ids: Optional[List[int]] = None,
    legend_object_ids: Optional[List[int]] = None,
):
    """
    Save a BEV image with axes + legend to a file instead of showing it.

    img: (H, W, 3) numpy array in [0,1]
    path: output file, e.g. 'bev.png'
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Render BEV image
    img = img.numpy() if isinstance(img, torch.Tensor) else img
    ax.imshow(img, origin="lower")

    # Axis labeling
    ax.set_xlabel("x")
    ax.set_ylabel("z")

    # Ticks in world coordinates
    ax.set_xticks(np.linspace(0, img.shape[1] - 1, 6))
    ax.set_yticks(np.linspace(0, img.shape[0] - 1, 6))
    ax.set_xticklabels(np.linspace(0, max_size, 6).astype(int))
    ax.set_yticklabels(np.linspace(0, max_size, 6).astype(int))

    lookup = get_active_lookup()

    # Decide which labels to show: explicit ids > ids from sample > all known ids.
    rec_ids = legend_receptacle_ids
    obj_ids = legend_object_ids
    if legend_sample is not None and (rec_ids is None or obj_ids is None):
        # Derive labels from the provided sample to keep the legend env-specific.
        rec_labels = set()
        for rec in legend_sample.get("receptacles", []):
            rec_labels.add(int(rec["label"]))

        obj_labels = set()
        obj_entries = legend_sample.get("objects", [])
        if obj_entries:
            if isinstance(obj_entries[0], dict):
                for obj in obj_entries:
                    obj_labels.add(int(obj["label"]))
            else:
                for step in obj_entries:
                    for obj in step:
                        obj_labels.add(int(obj["label"]))

        if rec_ids is None:
            rec_ids = sorted(rec_labels)
        if obj_ids is None:
            obj_ids = sorted(obj_labels)

    if rec_ids is None:
        rec_ids = sorted(receptacle_colors)
    if obj_ids is None:
        obj_ids = sorted(object_colors)

    # Build legend patches for the chosen labels only.
    rec_patches = [
        patches.Patch(color=receptacle_colors[k], label=f"Rec {lookup.id2receptacle[k].split('|')[0]}")
        for k in rec_ids
    ]
    obj_patches = [
        patches.Patch(color=object_colors[k], label=f"Obj {lookup.id2pickupable[k].split('|')[0]}")
        for k in obj_ids
    ]

    # Put legend outside plot
    ax.legend(
        handles=rec_patches + obj_patches,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        title="Legend",
    )

    ax.set_title(title)

    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)   # IMPORTANT: avoid memory leaks in long loops
