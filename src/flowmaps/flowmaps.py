from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from omegaconf import DictConfig
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset

from typing import Optional, Dict, Callable, Tuple, Union
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image as PILImage

from torch.distributions import Normal, Independent
from src.models.wrapper import ModelWrapper
from src.models.ode_solver import ODESolver

from src.utils.lookup import get_active_lookup

logger = logging.getLogger(__name__)

@dataclass
class FlowMapsOutput():
    
    samples: torch.Tensor = field(metadata={"help": "Predicted samples"})
    time_grid: torch.Tensor = field(metadata={"help": "Time grid"})


class FlowMapsPipeline():
    
    def __init__(self, solver_args: DictConfig):
        self.solver = ODESolver(ModelWrapper(None))  # net is set per-call in __call__
        self.cfg_solver = solver_args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @torch.no_grad()
    def __call__(self, model: nn.Module, sample: Tensor, model_extras: Optional[Dict]) -> FlowMapsOutput:

        # Load Model
        model.eval()
        self.solver.velocity_model.net = model.to(self.device)

        samples = sample.to(self.device)
        
        new_result = self.solver.sample(x_init=samples, 
                                        step_size=self.cfg_solver.step_size, 
                                        time_grid=torch.linspace(0, 1, self.cfg_solver.steps).to(self.device), 
                                        return_intermediates=self.cfg_solver.return_intermediates,
                                        method=self.cfg_solver.method, 
                                        **model_extras)
        
        output: FlowMapsOutput = FlowMapsOutput(
            samples=new_result,
            time_grid=torch.linspace(0, 1, self.cfg_solver.steps)
        )

        return output

    def compute_likelihood(self, model: nn.Module, x_1: Tensor, model_extras: Optional[Dict]) -> Tuple[Tensor, Tensor]:
        
        # Load Model
        self.solver.velocity_model.net = model.to(self.device)
        self.solver.velocity_model.eval()

        B, *dims = x_1.shape
        gaussian_log_density = Independent(Normal(torch.zeros(dims, device=self.device),
                                    torch.ones(dims, device=self.device)
                                    ), len(dims)).log_prob
        
        x_0, log_p1 = self.solver.compute_likelihood(
            x_1=x_1,
            step_size=self.cfg_solver.step_size,
            method=self.cfg_solver.method,
            log_p0=gaussian_log_density,
            enable_grad=False,
            exact_divergence=True,
            **model_extras
        )
        return x_0, log_p1
    

class FMTester():
    
    def __init__(self,
                model: nn.Module,
                pipeline: FlowMapsPipeline,
                dataset: Dataset,
                savefig: Optional[Callable] = None,
                seed: int = 2025,
                **kwargs
                ):
        
        self.vf = model
        self.vf.eval()
        self.pipeline = pipeline
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.savefig: Callable = savefig
        self.seed: int = seed
        
    def generate_samples(self, nsamples, npreds, **kwargs):
        inference_samples = []

        # Fetch a subset of the val set
        g = torch.Generator()
        g.manual_seed(self.seed)
        indices = torch.randperm(len(self.dataset), generator=g)[:nsamples].tolist()
        latent_dim = self.vf.vae.latent_dim

        for idx in tqdm(indices, desc="Generating samples", dynamic_ncols=True, leave=False):
            batch = self.dataset.get_sample(idx, mode="viz")
            # Only move tensors to device; keep metadata (e.g., env_name) as-is.
            data = {k: (v.to(self.device).unsqueeze(0) if torch.is_tensor(v) else v) for k, v in batch.items()}

            image      = data["image"]
            gt_image_q = data["gt_image_query"]
            t_query    = data["t_query"]                 # (B, 1)
            t_current  = data["t_current"]               # (B, 1)
            mask       = data["mask"]                    # (B, Nobj*S)
            map        = data["map"]                     # (B, S, 8)
            obj_q      = data["query_object"]            # (B, Nobj, 8)
            types      = data["types"]                   # (B, S)

            # Extract per-env legend info: which receptacle/object IDs are in this scene
            labels = batch["map"][:, -1].long()
            type_ids = batch["types"]
            valid = type_ids >= 0
            labels_valid = labels[valid]
            type_ids_valid = type_ids[valid]
            legend_receptacle_ids = torch.unique(labels_valid[type_ids_valid == 0]).tolist()
            legend_object_ids = torch.unique(labels_valid[type_ids_valid == 1]).tolist()

            # Get env_name for this sample
            env_name = self.dataset.datapoints[idx].get("env_name", f"env{idx}")

            S = types.shape[1]  # Sequence length
            queries = [obj_q[:,i,:].unsqueeze(1) for i in range(obj_q.shape[1])]
            masks = [mask[:,S*i:S*(i+1)] for i in range(obj_q.shape[1])]

            points = []
            for obj_q, mask in zip(queries, masks):
                point = {}
                B, S, _ = obj_q.shape   # (B, 1, 8)
                x_1_q = obj_q[:,:,-1]    # (B, 1), label is at -1

                x_0s = []
                for _ in range(npreds):
                    x_0s.append(torch.randn(B, S, latent_dim, device=x_1_q.device))  # (B, S, D)

                # sample
                point["x_0s"] = x_0s
                # model extras
                point["obj"] = x_1_q
                point["map"] = map
                point["tau_query"] = t_query
                point["tau_current"] = t_current
                point["types"] = types
                point["key_padding_mask"] = mask
                # viz
                point["image"] = image
                point["gt_image_q"] = gt_image_q
                point["groundtruth"] = obj_q
                # per-env metadata
                point["env_name"] = env_name
                point["legend_receptacle_ids"] = legend_receptacle_ids
                point["legend_object_ids"] = legend_object_ids
                point["query_object_label"] = int(x_1_q.item())
                point["scene_min"] = batch.get("scene_min", self.dataset.MIN_SCENE_SIZE)
                point["scene_max"] = batch.get("scene_max", self.dataset.MAX_SCENE_SIZE)

                points.append(point)
            inference_samples.append(points)

        return inference_samples

    def generate_eval_samples(self, nsamples, npreds, only_moved=False, min_displacement=1.0, **kwargs):
        """
        Like generate_samples, but for eval:
        - Iterates with re-shuffling until nsamples scenes with qualifying objects are found.
        - When only_moved=True, skips objects whose XZ displacement between tau_current and
          tau_query is below min_displacement metres; scenes with no qualifying objects are
          dropped and the search continues.
        - Always stores groundtruth_current (GT bbox at tau_c) so display_results_topdown
          can show where the object was when the prediction was made.
        """
        inference_samples = []

        g = torch.Generator()
        g.manual_seed(self.seed)
        latent_dim = self.vf.vae.latent_dim

        index_pool = torch.randperm(len(self.dataset), generator=g).tolist()
        pool_pos   = 0
        max_iters  = nsamples * 100
        iters      = 0

        pbar = tqdm(total=nsamples, desc="Generating eval samples", dynamic_ncols=True, leave=False)
        while len(inference_samples) < nsamples and iters < max_iters:
            if pool_pos >= len(index_pool):
                index_pool = torch.randperm(len(self.dataset), generator=g).tolist()
                pool_pos   = 0
            idx      = index_pool[pool_pos]
            pool_pos += 1
            iters    += 1

            batch = self.dataset.get_sample(idx, mode="viz")
            data = {k: (v.to(self.device).unsqueeze(0) if torch.is_tensor(v) else v) for k, v in batch.items()}

            image      = data["image"]
            gt_image_q = data["gt_image_query"]
            t_query    = data["t_query"]
            t_current  = data["t_current"]
            mask       = data["mask"]
            map        = data["map"]
            obj_q      = data["query_object"]
            types      = data["types"]

            labels = batch["map"][:, -1].long()
            type_ids = batch["types"]
            valid = type_ids >= 0
            labels_valid = labels[valid]
            type_ids_valid = type_ids[valid]
            legend_receptacle_ids = torch.unique(labels_valid[type_ids_valid == 0]).tolist()
            legend_object_ids = torch.unique(labels_valid[type_ids_valid == 1]).tolist()

            env_name = self.dataset.datapoints[idx].get("env_name", f"env{idx}")

            # Per-object positions and bboxes at t_current for filtering and display.
            obj_bboxes_current = batch["map"][batch["types"] == 1]  # (No, 8) normalised
            obj_pos_current    = obj_bboxes_current[:, :3]           # (No, 3)

            S = types.shape[1]
            queries = [obj_q[:,i,:].unsqueeze(1) for i in range(obj_q.shape[1])]
            masks = [mask[:,S*i:S*(i+1)] for i in range(obj_q.shape[1])]

            points = []
            for i, (obj_q_i, mask_i) in enumerate(zip(queries, masks)):
                if only_moved and i < obj_pos_current.shape[0]:
                    pos_current  = obj_pos_current[i]
                    pos_query    = obj_q_i[0, 0, :3].cpu()
                    xz_dist_m    = (pos_query[[0, 2]] - pos_current[[0, 2]]).norm().item() * self.dataset.SCENE_RANGE
                    if xz_dist_m < min_displacement:
                        continue

                point = {}
                B, S, _ = obj_q_i.shape
                x_1_q = obj_q_i[:,:,-1]

                x_0s = []
                for _ in range(npreds):
                    x_0s.append(torch.randn(B, S, latent_dim, device=x_1_q.device))

                point["x_0s"] = x_0s
                point["obj"] = x_1_q
                point["map"] = map
                point["tau_query"] = t_query
                point["tau_current"] = t_current
                point["types"] = types
                point["key_padding_mask"] = mask_i
                point["image"] = image
                point["gt_image_q"] = gt_image_q
                point["groundtruth"] = obj_q_i
                point["groundtruth_current"] = (
                    obj_bboxes_current[i] if i < obj_bboxes_current.shape[0] else None
                )
                point["env_name"] = env_name
                point["legend_receptacle_ids"] = legend_receptacle_ids
                point["legend_object_ids"] = legend_object_ids
                point["query_object_label"] = int(x_1_q.item())
                point["scene_min"] = batch.get("scene_min", self.dataset.MIN_SCENE_SIZE)
                point["scene_max"] = batch.get("scene_max", self.dataset.MAX_SCENE_SIZE)

                points.append(point)

            if points:
                inference_samples.append(points)
                pbar.update(1)

        pbar.close()
        if len(inference_samples) < nsamples:
            logger.warning(
                f"generate_eval_samples: only collected {len(inference_samples)}/{nsamples} "
                f"scenes after {iters} iterations (only_moved={only_moved})"
            )
        return inference_samples
    
    def inference(self, samples_, img_size=256):
        results = []
        mean = torch.tensor(self.dataset.latent_statistics['mean'], device=self.device)
        std = torch.tensor(self.dataset.latent_statistics['std'], device=self.device)

        def _to_numpy(v):
            return v if isinstance(v, np.ndarray) else v.cpu().numpy().squeeze()

        for samples in tqdm(samples_, desc="Inference", dynamic_ncols=True, leave=False):
            No = len(samples)
            npreds = len(samples[0]["x_0s"])

            # Scene-level metadata is the same for every object in this scene.
            scene_min = samples[0]["scene_min"]
            scene_max = samples[0]["scene_max"]
            scene_range_env = scene_max - scene_min
            MIN_SCENE_SIZE = self.dataset.MIN_SCENE_SIZE
            SCENE_RANGE = self.dataset.SCENE_RANGE

            def _to_pixel(bbox_norm):
                bbox_world = bbox_norm.copy()
                bbox_world[:3] = bbox_norm[:3] * SCENE_RANGE + MIN_SCENE_SIZE
                bbox_world[3:6] = bbox_norm[3:6] * SCENE_RANGE
                bbox_px = bbox_world.copy()
                bbox_px[:3] = (bbox_world[:3] - scene_min) / scene_range_env * (img_size - 1)
                bbox_px[3:6] = bbox_world[3:6] / scene_range_env * (img_size - 1)
                return bbox_px.astype(np.float32)

            # Batch model_extras across all No objects (same scene context, different obj labels).
            batched_extras = {
                "obj":              torch.cat([s["obj"]              for s in samples], dim=0),  # (No, 1)
                "map":              torch.cat([s["map"]              for s in samples], dim=0),  # (No, S, 8)
                "tau0":             torch.cat([s["tau_current"]      for s in samples], dim=0),  # (No, 1)
                "tau":              torch.cat([s["tau_query"]        for s in samples], dim=0),  # (No, 1)
                "types":            torch.cat([s["types"]            for s in samples], dim=0),  # (No, S)
                "key_padding_mask": torch.cat([s["key_padding_mask"] for s in samples], dim=0),  # (No, S)
            }

            # One ODE call per pred, batched over all No objects.
            output_samples_per_obj = [[] for _ in range(No)]
            for pred_idx in range(npreds):
                x_0_batch = torch.cat([s["x_0s"][pred_idx] for s in samples], dim=0)  # (No, 1, D)
                output: FlowMapsOutput = self.pipeline(self.vf, x_0_batch, batched_extras)
                latent = output.samples * std + mean  # (No, 1, D)
                bbox_batch, cls_logits_batch = self.vf.vae.decode_latents(latent)
                bbox_batch = torch.cat([
                    bbox_batch[..., :3],
                    self.dataset.size_from_model(bbox_batch[..., 3:6]),
                    bbox_batch[..., 6:],
                ], dim=-1)  # (No, 1, 7)
                for obj_idx in range(No):
                    bbox_np = bbox_batch[obj_idx].cpu().numpy().squeeze()  # (7,)
                    out = {
                        "bbox": _to_pixel(bbox_np),
                        "cls_logits": cls_logits_batch[obj_idx].cpu().numpy().squeeze(),
                    }
                    output_samples_per_obj[obj_idx].append(out)

            # Build per-object result dicts (same structure as before for display_results).
            results_ = []
            for obj_idx, sample in enumerate(samples):
                gt_tensor = sample["groundtruth"].squeeze().squeeze()
                gt_tensor = torch.cat([gt_tensor[..., :3], self.dataset.size_from_model(gt_tensor[..., 3:6]), gt_tensor[..., 6:]], dim=-1)
                gt_raw = gt_tensor.cpu().numpy()
                gt = gt_raw.copy()
                if gt.shape[0] >= 6:
                    gt[:6] = _to_pixel(gt_raw[:6])

                gt_current = None
                if sample.get("groundtruth_current") is not None:
                    gt_c_tensor = sample["groundtruth_current"].float().squeeze()
                    gt_c_tensor = torch.cat([gt_c_tensor[:3], self.dataset.size_from_model(gt_c_tensor[3:6]), gt_c_tensor[6:]], dim=-1)
                    gt_c_raw = gt_c_tensor.cpu().numpy()
                    gt_current = gt_c_raw.copy()
                    if gt_current.shape[0] >= 6:
                        gt_current[:6] = _to_pixel(gt_c_raw[:6])

                result = {
                    "image": _to_numpy(sample["image"]),
                    "gt_image_q": _to_numpy(sample["gt_image_q"]),
                    "tau_query": sample["tau_query"].cpu().numpy().squeeze(),
                    "tau_current": sample["tau_current"].cpu().numpy().squeeze(),
                    "groundtruth": gt,
                    "groundtruth_current": gt_current,
                    "samples": output_samples_per_obj[obj_idx],
                    "env_name": sample.get("env_name"),
                    "legend_receptacle_ids": sample.get("legend_receptacle_ids"),
                    "legend_object_ids": sample.get("legend_object_ids"),
                    "query_object_label": sample.get("query_object_label"),
                    "scene_min": scene_min,
                    "scene_max": scene_max,
                }
                results_.append(result)
            results.append(results_)

        return results
            
    def display_results(self, results, iteration, max_size=None):
        lookup = get_active_lookup()

        def draw_box(image, bbox, color, line_width=2, halo_width=2):
            """
            bbox: (cx, cy, cz, dx, dy, dz) in pixel coords already
            color: (r, g, b) in [0, 1]
            """
            H, W = image.shape[:2]
            cx, _, cz, dx, _, dz = bbox

            # compute x,z min/max from center + size
            x_min = cx - dx / 2.0
            x_max = cx + dx / 2.0
            z_min = cz - dz / 2.0
            z_max = cz + dz / 2.0

            # clamp to valid image bounds
            x_min = max(0, min(W - 1, x_min))
            x_max = max(0, min(W - 1, x_max))
            z_min = max(0, min(H - 1, z_min))
            z_max = max(0, min(H - 1, z_max))

            rows = []
            cols = []
            for x, z in [(x_min, z_min), (x_min, z_max),
                        (x_max, z_min), (x_max, z_max)]:
                rows.append(int(round(z)))
                cols.append(int(round(x)))

            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)

            color_arr = np.array(color, dtype=np.float32)[None, None, :]
            halo_arr = np.array((0.0, 0.0, 0.0), dtype=np.float32)[None, None, :]

            def draw_outline(r0, r1, c0, c1, width, col):
                for w in range(width):
                    rr0 = max(0, r0 - w)
                    rr1 = min(H - 1, r1 + w)
                    cc0 = max(0, c0 - w)
                    cc1 = min(W - 1, c1 + w)

                    image[rr0, cc0:cc1 + 1, :] = col
                    image[rr1, cc0:cc1 + 1, :] = col
                    image[rr0:rr1 + 1, cc0, :] = col
                    image[rr0:rr1 + 1, cc1, :] = col

            # draw halo first
            draw_outline(r0, r1, c0, c1, line_width + halo_width, halo_arr)
            # draw colored outline on top
            draw_outline(r0, r1, c0, c1, line_width, color_arr)

        def format_axes(ax, img, title, scene_min, scene_max):
            ax.imshow(img, origin="lower")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            ax.set_xticks(np.linspace(0, img.shape[1] - 1, 6))
            ax.set_yticks(np.linspace(0, img.shape[0] - 1, 6))
            ax.set_xticklabels(np.linspace(scene_min, scene_max, 6).astype(int))
            ax.set_yticklabels(np.linspace(scene_min, scene_max, 6).astype(int))

        for i, result_ in enumerate(results):
            for j, result in enumerate(result_):
                image = result["image"]
                gt_image_q = result["gt_image_q"]
                tau_query = result["tau_query"]
                tau_current = result["tau_current"]
                gt = result["groundtruth"]
                samps = result["samples"]
                scene_min = result.get("scene_min", self.dataset.MIN_SCENE_SIZE)
                scene_max = result.get("scene_max", self.dataset.MAX_SCENE_SIZE)

                # Per-env legend: only show receptacles/objects present in this env
                rec_ids = result.get("legend_receptacle_ids") or sorted(lookup.receptacle_colors)
                obj_ids = result.get("legend_object_ids") or sorted(lookup.pickupable_colors)
                rec_patches = [
                    mpatches.Patch(color=lookup.receptacle_colors[k], label=f"Rec {lookup.id2receptacle[k].split('|')[0]}")
                    for k in rec_ids if k in lookup.receptacle_colors
                ]
                obj_patches = [
                    mpatches.Patch(color=lookup.pickupable_colors[k], label=f"Obj {lookup.id2pickupable[k].split('|')[0]}")
                    for k in obj_ids if k in lookup.pickupable_colors
                ]

                # Query object name for the title
                query_label = result.get("query_object_label")
                query_obj_name = lookup.id2pickupable.get(query_label, f"id={query_label}").split("|")[0] if query_label is not None else f"obj{j}"
                env_name = result.get("env_name", f"env{i}")

                if image.shape[0] == 3:
                    image = image.transpose(1, 2, 0).astype(np.uint8)
                if gt_image_q.shape[0] == 3:
                    gt_image_q = gt_image_q.transpose(1, 2, 0).astype(np.uint8)

                img_current = image.copy()
                img_gt = gt_image_q.copy()
                img_samples = gt_image_q.copy()

                if gt.shape[0] >= 6:
                    gt_label = int(gt[7]) if gt.shape[0] > 7 else None
                    gt_color = lookup.pickupable_colors.get(gt_label, (0.0, 0.0, 0.0))
                    draw_box(img_gt, gt[:6], gt_color)

                for samp in samps:
                    class_id = samp["cls_logits"].argmax(-1).item()
                    samp_color = lookup.pickupable_colors.get(class_id, (0.0, 0.0, 0.0))
                    draw_box(img_samples, samp["bbox"][:6], samp_color)

                fig, ax = plt.subplots(1, 3, figsize=(30, 10))
                fig.suptitle(f"{env_name} — Predicting: {query_obj_name}", fontsize=14, fontweight="bold")
                format_axes(ax[0], img_current, rf"Environment at $\tau_c={tau_current}$", scene_min, scene_max)
                format_axes(ax[1], img_gt, rf"Ground-truth at $\tau_q={tau_query}$", scene_min, scene_max)
                format_axes(ax[2], img_samples, rf"Samples at $\tau_q={tau_query}$", scene_min, scene_max)

                ax[2].legend(
                    handles=rec_patches + obj_patches,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    frameon=True,
                    title="Legend",
                )

                plt.tight_layout()
                self.savefig(fig, f"{env_name}_object{j}_idx{iteration}")
                plt.close(fig)

    def display_results_topdown(self, results, iteration, topdown_root: Union[str, Path]):
        """
        Like display_results, but overlays bboxes on AI2-THOR top-down renders
        instead of on the BEV.  Only called from eval.py (never from the trainer).

        topdown_root: path to data/topdown/minival (or data/topdown/val),
                      expected structure: {topdown_root}/{env_id}/{t}/image.png
                                          {topdown_root}/{env_id}/camera.json
        """
        topdown_root = Path(topdown_root)
        lookup = get_active_lookup()
        BEV_SIZE = 256  # pixel size used in inference()._to_pixel

        def _parse_env_name(env_name):
            parts = env_name.split("_", 1)
            return parts[0], parts[1] if len(parts) > 1 else ""

        def _bbox_bev_to_world(bbox_px, scene_min, scene_range_env):
            """Invert inference()._to_pixel: BEV pixel coords to world coords."""
            w = bbox_px.copy().astype(np.float64)
            w[0] = bbox_px[0] / (BEV_SIZE - 1) * scene_range_env + scene_min  # cx
            w[1] = bbox_px[1] / (BEV_SIZE - 1) * scene_range_env + scene_min  # cy
            w[2] = bbox_px[2] / (BEV_SIZE - 1) * scene_range_env + scene_min  # cz
            w[3] = bbox_px[3] / (BEV_SIZE - 1) * scene_range_env              # dx
            w[4] = bbox_px[4] / (BEV_SIZE - 1) * scene_range_env              # dy
            w[5] = bbox_px[5] / (BEV_SIZE - 1) * scene_range_env              # dz
            return w

        def _world_to_topdown(wx, wz, dx, dz, cam):
            """Project world (wx, wz, dx, dz) to topdown image (col, row, dcol, drow).
            Row 0 is the top of the image (large z), consistent with AI2-THOR output."""
            col  = (wx - cam["world_x_min"]) / cam["world_size"] * cam["img_w"]
            row  = (cam["world_z_max"] - wz) / cam["world_size"] * cam["img_h"]
            dcol = dx / cam["world_size"] * cam["img_w"]
            drow = dz / cam["world_size"] * cam["img_h"]
            return col, row, dcol, drow

        def _make_rect(col, row, dcol, drow, color, lw, ls, alpha, fill=False):
            """Return (halo_patch, color_patch) for a bbox in topdown image pixel coords.
            The black halo underneath makes the box readable on any background."""
            xy = (col - dcol / 2, row - drow / 2)
            facecolor = (*color, 0.08) if fill else "none"
            halo = mpatches.Rectangle(
                xy, dcol, drow,
                linewidth=lw + 4, edgecolor="black", facecolor="none",
                linestyle=ls, alpha=alpha, zorder=4,
            )
            rect = mpatches.Rectangle(
                xy, dcol, drow,
                linewidth=lw, edgecolor=color, facecolor=facecolor,
                linestyle=ls, alpha=alpha, zorder=5,
            )
            return halo, rect

        for i, result_ in enumerate(results):
            for j, result in enumerate(result_):
                env_name = result.get("env_name", f"env{i}")
                env_id, pattern = _parse_env_name(env_name)
                tau_current = int(result["tau_current"])
                tau_query   = int(result["tau_query"])
                scene_min   = float(result.get("scene_min", self.dataset.MIN_SCENE_SIZE))
                scene_max   = float(result.get("scene_max", self.dataset.MAX_SCENE_SIZE))
                scene_range_env = scene_max - scene_min

                # Load camera.json for this env
                cam_path = topdown_root / pattern / env_id / "camera.json"
                if not cam_path.exists():
                    logger.warning(f"No topdown camera.json for {env_name} ({cam_path}), skipping")
                    continue
                with open(cam_path) as f:
                    cam = json.load(f)

                # Load topdown images as float32 [0,1]
                def _load_td(t):
                    p = topdown_root / pattern / env_id / str(t) / "image.png"
                    if not p.exists():
                        return None
                    return np.array(PILImage.open(p).convert("RGB")).astype(np.float32) / 255.0

                img_current = _load_td(tau_current)
                img_query   = _load_td(tau_query)
                if img_current is None or img_query is None:
                    logger.warning(f"Missing topdown image for {env_name} at t={tau_current} or t={tau_query}")
                    continue

                # Store GT box params so we can build a fresh patch per axes that needs it
                # (a patch can only belong to one axes)
                gt_box_params = None
                gt_current_box_params = None
                samp_box_params = []

                gt = result["groundtruth"]
                if gt.shape[0] >= 6:
                    gt_world = _bbox_bev_to_world(gt[:6], scene_min, scene_range_env)
                    c, r, dc, dr = _world_to_topdown(
                        gt_world[0], gt_world[2], gt_world[3], gt_world[5], cam)
                    gt_label = int(gt[7]) if gt.shape[0] > 7 else None
                    gt_color = lookup.pickupable_colors.get(gt_label, (0.0, 0.0, 0.0))
                    gt_box_params = (c, r, dc, dr, gt_color)

                gt_c = result.get("groundtruth_current")
                if gt_c is not None and gt_c.shape[0] >= 6:
                    gt_c_world = _bbox_bev_to_world(gt_c[:6], scene_min, scene_range_env)
                    c, r, dc, dr = _world_to_topdown(
                        gt_c_world[0], gt_c_world[2], gt_c_world[3], gt_c_world[5], cam)
                    gt_c_label = int(gt_c[7]) if gt_c.shape[0] > 7 else None
                    gt_c_color = lookup.pickupable_colors.get(gt_c_label, (0.0, 0.0, 0.0))
                    gt_current_box_params = (c, r, dc, dr, gt_c_color)

                for samp in result["samples"]:
                    bbox_world = _bbox_bev_to_world(samp["bbox"][:6], scene_min, scene_range_env)
                    c, r, dc, dr = _world_to_topdown(
                        bbox_world[0], bbox_world[2], bbox_world[3], bbox_world[5], cam)
                    class_id   = samp["cls_logits"].argmax(-1).item()
                    samp_color = lookup.pickupable_colors.get(class_id, (0.0, 0.0, 0.0))
                    samp_box_params.append((c, r, dc, dr, samp_color))

                def _gt_rects():
                    if gt_box_params is None:
                        return []
                    c, r, dc, dr, color = gt_box_params
                    return list(_make_rect(c, r, dc, dr, color, lw=2, ls="-", alpha=1.0))

                def _gt_current_rects():
                    params = gt_current_box_params if gt_current_box_params is not None else gt_box_params
                    if params is None:
                        return []
                    c, r, dc, dr, color = params
                    return list(_make_rect(c, r, dc, dr, color, lw=2, ls="-", alpha=1.0))

                def _samp_rects():
                    patches = []
                    for c, r, dc, dr, color in samp_box_params:
                        patches.extend(_make_rect(c, r, dc, dr, color, lw=4, ls="--", alpha=0.9, fill=True))
                    return patches

                # Query object name for the main title
                query_label    = result.get("query_object_label")
                query_obj_name = (
                    lookup.id2pickupable.get(query_label, f"id={query_label}").split("|")[0]
                    if query_label is not None else f"obj{j}"
                )

                # World-coord tick labels — integer meters, no decimals
                n_ticks  = 6
                x_ticks  = np.linspace(0, cam["img_w"] - 1, n_ticks)
                x_labels = [f"{cam['world_x_min'] + k / (n_ticks - 1) * cam['world_size']:.0f}"
                            for k in range(n_ticks)]
                y_ticks  = np.linspace(0, cam["img_h"] - 1, n_ticks)
                y_labels = [f"{cam['world_z_max'] - k / (n_ticks - 1) * cam['world_size']:.0f}"
                            for k in range(n_ticks)]

                # --- Layout ---
                fig, axes = plt.subplots(1, 3, figsize=(45, 16))
                fig.suptitle(f"{env_name}  |  Predicting: {query_obj_name}",
                             fontsize=22, fontweight="bold", y=0.99)

                # Panel 0: current env with GT box shown as context
                # Panels 1 & 2: query timestamp (same image), GT box vs sampled boxes
                panel_info = [
                    (axes[0], img_current, _gt_current_rects(), rf"Environment  $\tau_c={tau_current}$"),
                    (axes[1], img_query,   _gt_rects(),         rf"Ground-truth  $\tau_q={tau_query}$"),
                    (axes[2], img_query,   _samp_rects(),       rf"Samples  $\tau_q={tau_query}$"),
                ]

                for ax, img, rects, caption in panel_info:
                    ax.imshow(img)
                    ax.set_xlabel("x [m]", fontsize=20, labelpad=10)
                    ax.set_ylabel("z [m]", fontsize=20, labelpad=10)
                    ax.tick_params(axis="both", labelsize=18, length=7, width=1.5)
                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels(x_labels)
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels(y_labels)
                    ax.spines[["top", "right"]].set_visible(False)
                    # Caption below the subplot
                    ax.text(0.5, -0.08, caption, transform=ax.transAxes,
                            ha="center", va="top", fontsize=20, fontweight="bold")
                    for rect in rects:
                        ax.add_patch(rect)

                plt.tight_layout(rect=[0, 0.0, 1, 0.96])
                self.savefig(fig, f"{env_name}_object{j}_idx{iteration}", fig_size=(45, 16))
                plt.close(fig)


class VAETester():
    
    def __init__(self,
                 model: nn.Module,
                 dataset,
                 savefig: Optional[Callable] = None,
                 seed: int = 2025,
                 **kwargs
                 ):
        
        self.vae = model
        self.vae.eval()
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.savefig: Callable = savefig
        self.seed: int = seed
        
    def generate_samples(self, nsamples, max_envs=None, **kwargs):
        inference_samples = []

        # Prefer per-env sampling when env_name metadata exists.
        env_groups = {}
        if hasattr(self.dataset, "datapoints"):
            for idx, point in enumerate(self.dataset.datapoints):
                env_name = point.get("env_name") if isinstance(point, dict) else None
                if env_name is None:
                    env_groups = {}
                    break
                env_groups.setdefault(env_name, []).append(idx)

        if env_groups:
            # Subsample a fixed set of environments when max_envs is specified.
            all_envs = sorted(env_groups.keys())
            if max_envs is not None and max_envs < len(all_envs):
                g = torch.Generator()
                g.manual_seed(self.seed)
                perm = torch.randperm(len(all_envs), generator=g)[:max_envs].tolist()
                all_envs = [all_envs[i] for i in sorted(perm)]
            # Sample nsamples per env with deterministic seeds.
            for env_idx, env_name in enumerate(all_envs):
                indices = env_groups[env_name]
                g = torch.Generator()
                g.manual_seed(self.seed + env_idx)
                if len(indices) <= nsamples:
                    chosen = indices
                else:
                    perm = torch.randperm(len(indices), generator=g)[:nsamples].tolist()
                    chosen = [indices[i] for i in perm]
                for sample_idx, idx in enumerate(chosen):
                    batch = self.dataset.get_sample(idx, mode="viz")
                    # Only move tensors to device; keep metadata (e.g., env_name) on CPU.
                    data = {
                        k: (v.to(self.device).unsqueeze(0) if torch.is_tensor(v) else v)
                        for k, v in batch.items()
                    }

                    image       = data["image"]
                    map         = data["map"]           # (B, S, 8)
                    types       = data["types"]         # (B, S)
                    tau_current = data["t_current"]     # (B, 1)

                    # Store sample
                    point = {}
                    point["map"] = map
                    point["types"] = types
                    point["image"] = image
                    point["t_current"] = tau_current
                    point["env_name"] = env_name
                    point["env_sample_idx"] = sample_idx
                    point["scene_min"] = batch["scene_min"]
                    point["scene_max"] = batch["scene_max"]
                    # Capture per-sample labels so legends only show present classes.
                    if "map" in batch and "types" in batch:
                        labels = batch["map"][:, -1].long()
                        type_ids = batch["types"]
                        valid = type_ids >= 0
                        labels = labels[valid]
                        type_ids = type_ids[valid]
                        point["legend_receptacle_ids"] = torch.unique(labels[type_ids == 0]).tolist()
                        point["legend_object_ids"] = torch.unique(labels[type_ids == 1]).tolist()
                    inference_samples.append(point)
            return inference_samples

        # Fallback: sample nsamples total when env grouping is unavailable.
        g = torch.Generator()
        g.manual_seed(self.seed)
        indices = torch.randperm(len(self.dataset), generator=g)[:nsamples].tolist()

        for idx in tqdm(indices, desc="Generating samples", dynamic_ncols=True, leave=False):
            batch = self.dataset.get_sample(idx, mode="viz")
            # Only move tensors to device; keep metadata (e.g., env_name) on CPU.
            data = {
                k: (v.to(self.device).unsqueeze(0) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }

            image       = data["image"]
            map         = data["map"]           # (B, S, 8)
            types       = data["types"]         # (B, S)
            tau_current = data["t_current"]     # (B, 1)

            # Store sample
            point = {}
            point["map"] = map
            point["types"] = types
            point["image"] = image
            point["t_current"] = tau_current
            # Preserve env_name for more informative logging/filenames.
            if "env_name" in data:
                point["env_name"] = data["env_name"]
            # Capture per-sample labels so legends only show present classes.
            if "map" in batch and "types" in batch:
                labels = batch["map"][:, -1].long()
                type_ids = batch["types"]
                valid = type_ids >= 0
                labels = labels[valid]
                type_ids = type_ids[valid]
                point["legend_receptacle_ids"] = torch.unique(labels[type_ids == 0]).tolist()
                point["legend_object_ids"] = torch.unique(labels[type_ids == 1]).tolist()
            inference_samples.append(point)
        return inference_samples
    
    def inference(self, samples, img_size=256):
        results = []

        with torch.inference_mode():
            for sample in tqdm(samples, desc="Inference", dynamic_ncols=True, leave=False):
                
                map = sample["map"]
                types = sample["types"]
                
                obj_mask = types == 1
                obj_tokens = map[obj_mask]  # (B * S_obj, 8) 

                _, _, bbox, cls_logits = self.vae(x=obj_tokens, sample=False)
                obj_cls = cls_logits.argmax(-1)  # (S_obj, 1)

                # bbox sizes are in log-size model space; invert before pixel conversion.
                bbox = torch.cat([bbox[..., :3], self.dataset.size_from_model(bbox[..., 3:6]), bbox[..., 6:]], dim=-1)

                # bbox is in global-normalized [0,1] space (MIN_SCENE_SIZE=-0.5, SCENE_RANGE=34.5).
                # The BEV image is rendered with per-env tight bounds [scene_min, scene_max].
                # We must denormalize to world coords first, then convert to per-env pixel space.
                scene_min = sample["scene_min"]
                scene_max = sample["scene_max"]
                scene_range_env = scene_max - scene_min
                MIN_SCENE_SIZE = self.dataset.MIN_SCENE_SIZE
                SCENE_RANGE = self.dataset.SCENE_RANGE
                bbox_np = bbox.cpu().numpy().squeeze()  # (N_obj, 6)
                bbox_world = bbox_np.copy()
                bbox_world[:, :3] = bbox_np[:, :3] * SCENE_RANGE + MIN_SCENE_SIZE  # centers to world
                bbox_world[:, 3:6] = bbox_np[:, 3:6] * SCENE_RANGE                # sizes to world
                bbox_px = bbox_world.copy()
                bbox_px[:, :3] = (bbox_world[:, :3] - scene_min) / scene_range_env * (img_size - 1)
                bbox_px[:, 3:6] = bbox_world[:, 3:6] / scene_range_env * (img_size - 1)

                img = sample["image"]
                result = {
                    "image": img if isinstance(img, np.ndarray) else img.cpu().numpy().squeeze(),
                    "tau_current": sample["t_current"].cpu().numpy().squeeze(),
                    "bbox": bbox_px.astype(np.float32),
                    "cls": obj_cls.cpu().numpy().squeeze(),
                }
                if "env_name" in sample:
                    result["env_name"] = sample["env_name"]
                if "scene_min" in sample:
                    result["scene_min"] = sample["scene_min"]
                    result["scene_max"] = sample["scene_max"]
                if "legend_receptacle_ids" in sample:
                    result["legend_receptacle_ids"] = sample["legend_receptacle_ids"]
                if "legend_object_ids" in sample:
                    result["legend_object_ids"] = sample["legend_object_ids"]
                results.append(result)
            
        return results
            
    def display_results(self, results, iteration):
        for i, result in enumerate(results):
            base_image = result["image"]
            base_img = base_image.numpy() if isinstance(base_image, torch.Tensor) else base_image
            H, W = base_img.shape[:2]
            
            def draw_box(image, bbox, color, line_width=2, halo_width=2):
                """
                bbox: tensor with (cx, cy, cz, dx, dy, dz) in pixel coords already
                color: (r, g, b) in [0, 1]
                line_width: thickness of the colored outline
                halo_width: extra thickness for the black halo
                """
                cx, _, cz, dx, _, dz = bbox

                # compute x,z min/max from center + size
                x_min = cx - dx / 2.0
                x_max = cx + dx / 2.0
                z_min = cz - dz / 2.0
                z_max = cz + dz / 2.0

                # clamp to valid image bounds
                x_min = max(0, min(W - 1, x_min))
                x_max = max(0, min(W - 1, x_max))
                z_min = max(0, min(H - 1, z_min))
                z_max = max(0, min(H - 1, z_max))

                rows = []
                cols = []
                for x, z in [(x_min, z_min), (x_min, z_max),
                            (x_max, z_min), (x_max, z_max)]:
                    rows.append(int(round(z)))
                    cols.append(int(round(x)))

                r0, r1 = min(rows), max(rows)
                c0, c1 = min(cols), max(cols)

                color_arr = np.array(color, dtype=np.float32)[None, None, :]
                halo_arr = np.array((0.0, 0.0, 0.0), dtype=np.float32)[None, None, :]

                def draw_outline(r0, r1, c0, c1, width, col):
                    for w in range(width):
                        rr0 = max(0, r0 - w)
                        rr1 = min(H - 1, r1 + w)
                        cc0 = max(0, c0 - w)
                        cc1 = min(W - 1, c1 + w)

                        image[rr0, cc0:cc1 + 1, :] = col
                        image[rr1, cc0:cc1 + 1, :] = col
                        image[rr0:rr1 + 1, cc0, :] = col
                        image[rr0:rr1 + 1, cc1, :] = col

                # draw halo first
                draw_outline(r0, r1, c0, c1, line_width + halo_width, halo_arr)

                # draw colored outline on top
                draw_outline(r0, r1, c0, c1, line_width, color_arr)


            tau_current = result["tau_current"]
            obj_bbox = result["bbox"]
            obj_cls = result["cls"]
            scene_min = result["scene_min"]
            scene_max = result["scene_max"]

            lookup = get_active_lookup()
            overlay_img = base_img.copy()
            for obj in range(obj_bbox.shape[0]):
                label = int(obj_cls[obj])
                color = lookup.pickupable_colors.get(label, (0.0, 0.0, 0.0))  # fallback black
                draw_box(overlay_img, result["bbox"][obj], color)

            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

            def format_axes(ax, img, title):
                ax.imshow(img, origin="lower")
                ax.set_xlabel("x")
                ax.set_ylabel("z")
                ax.set_xticks(np.linspace(0, img.shape[1] - 1, 6))
                ax.set_yticks(np.linspace(0, img.shape[0] - 1, 6))
                ax.set_xticklabels(np.linspace(scene_min, scene_max, 6).astype(int))
                ax.set_yticklabels(np.linspace(scene_min, scene_max, 6).astype(int))
                ax.set_title(title, fontsize=10)

            format_axes(ax_left, base_img, "GT environment (no preds)")
            format_axes(ax_right, overlay_img, rf"Environment at $\tau={tau_current}$")

            # Build legend patches for labels present in this sample only.
            rec_ids = result.get("legend_receptacle_ids", sorted(lookup.receptacle_colors))
            obj_ids = result.get("legend_object_ids", sorted(lookup.pickupable_colors))
            rec_patches = [
                mpatches.Patch(color=lookup.receptacle_colors[k], label=f"Rec {lookup.id2receptacle[k].split('|')[0]}")
                for k in rec_ids
            ]
            obj_patches = [
                mpatches.Patch(color=lookup.pickupable_colors[k], label=f"Obj {lookup.id2pickupable[k].split('|')[0]}")
                for k in obj_ids
            ]

            # Put legend outside plot
            ax_right.legend(
                handles=rec_patches + obj_patches,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                frameon=True,
                title="Legend",
            )
            
            plt.tight_layout()
            # Use only the env{id} prefix in filenames (e.g., env0_sample3_idx999).
            env_name = result.get("env_name")
            env_str = "env" if env_name is None else str(env_name)
            short_env = env_str.split("_", 1)[0]
            if not short_env.startswith("env"):
                short_env = "env"
            sample_idx = result.get("env_sample_idx", i)
            self.savefig(fig, f"{short_env}_sample{sample_idx}_idx{iteration}")
            
            plt.close(fig)
