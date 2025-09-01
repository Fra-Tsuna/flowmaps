from dataclasses import dataclass, field
from omegaconf import DictConfig
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset
from torch.distributions import Normal, Independent

from typing import Optional, Dict, Callable, Tuple
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.models.cfg_vf import CFGVectorField
from src.models.ode_solver import ODESolver

@dataclass
class FlowMapsOutput():
    
    samples: torch.Tensor = field(metadata={"help": "Predicted samples"})
    time_grid: torch.Tensor = field(metadata={"help": "Time grid"})

class FlowMapsPipeline():
    
    def __init__(self, 
                 solver_args: DictConfig,
                 guidance_scale: Optional[float] = 1.0,
                 null_label: Optional[int] = None,
                 ):
        
        self.solver = ODESolver(CFGVectorField(None, # Placeholder
                                               guidance_scale=guidance_scale, 
                                               null_label=null_label))
    
        self.guidance_scale = guidance_scale
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
        model.eval()
        self.solver.velocity_model.net = model.to(self.device)

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
            exact_divergence=False,
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
        
        for idx in tqdm(indices, desc="Generating samples", dynamic_ncols=True, leave=False):
            batch = self.dataset.get_sample(idx, mode="viz")
            data = {k: v.to(self.device).unsqueeze(0) for k, v in batch.items()}
            
            image      = data["image"]
            gt_image_q = data["gt_image_query"]  
            t_query    = data["t_query"]                 # (B,)
            t_current  = data["t_current"]               # (B,) 
            mask       = data["mask"]                    # (B, 3*S)
            map        = data["map"]                     # (B, S, 5)
            obj_q      = data["query_object"]            # (B, 3, 5)
            types      = data["types"]                   # (B, S)
            
            S = types.shape[1]  # Sequence length 
            queries = [obj_q[:,i,:].unsqueeze(1) for i in range(obj_q.shape[1])]
            masks = [mask[:,S*i:S*(i+1)] for i in range(obj_q.shape[1])]

            points = []
            for obj_q, mask in zip(queries, masks):
                point = {}
                tgt_obj_bb = obj_q[:,:,:4]   # (B, 1, 4)
                tgt_obj_col = obj_q[:,:,4]   # (B, 1)
                x_1 = tgt_obj_bb             # (B, 1, 4)
                x_1_q = tgt_obj_col          # (B, 1)
                
                x_0s = []
                for _ in range(npreds):
                    x_0s.append(torch.randn_like(x_1))
                
                # sample
                point["x_0s"] = x_0s
                # model extras
                point["obj"] = x_1_q              # (B, 1)
                point["map"] = map                # (B, S, 5)
                point["tau_query"] = t_query      # (B,)
                point["tau_current"] = t_current  # (B,)
                point["types"] = types            # (B, S)
                point["key_padding_mask"] = mask  # (B, S)
                # viz
                point["image"] = image
                point["gt_image_q"] = gt_image_q
                point["groundtruth"] = x_1
                
                points.append(point)
            inference_samples.append(points)
            
        return inference_samples
    
    def inference(self, samples_):
        results = []
        
        for samples in tqdm(samples_, desc="Inference", dynamic_ncols=True, leave=False):
            results_ = []
            for sample in samples:
                x_0s = sample["x_0s"]
                model_extras = {
                    "obj": sample["obj"],
                    "map": sample["map"],
                    "tau0": sample["tau_current"],
                    "tau": sample["tau_query"],
                    "types": sample["types"],
                    "key_padding_mask": sample["key_padding_mask"]
                }
                
                output_samples = []
                for x_0 in x_0s:
                    output: FlowMapsOutput = self.pipeline(self.vf, x_0, model_extras)
                    output_samples.append((output.samples.cpu().numpy().squeeze().squeeze() * 255).astype(np.int16))
                result = {
                    "image": sample["image"].cpu().numpy().squeeze(),
                    "gt_image_q": sample["gt_image_q"].cpu().numpy().squeeze(),
                    "tau_query": sample["tau_query"].cpu().numpy().squeeze(),
                    "tau_current": sample["tau_current"].cpu().numpy().squeeze(),
                    "groundtruth": (sample["groundtruth"].cpu().numpy().squeeze().squeeze() * 255).astype(np.int16),
                    "samples": output_samples,
                    "model_extras": model_extras,
                }
                results_.append(result)
                
            results.append(results_)
            
        return results
            
    def display_results(self, results, iteration, scale):
        
        for i, result_ in enumerate(results):
            for j, result in enumerate(result_):
                image = result["image"]
                gt_image_q = result["gt_image_q"]
                tau_query = result["tau_query"]
                tau_current = result["tau_current"]
                gt = result["groundtruth"]
                samps = result["samples"]
                y1, x1, h, w = gt[:4]
                y1 = int(y1 * scale)
                x1 = int(x1 * scale)
                h = int(h * scale)  
                w = int(w * scale)
                color = "cyan"   # normalize RGB to [0,1]

                fig, ax = plt.subplots(1, 3, figsize=(30, 10))
                
                if image.shape[0] == 3:
                    image = image.transpose(1, 2, 0).astype(np.uint8)
                if gt_image_q.shape[0] == 3:
                    gt_image_q = gt_image_q.transpose(1, 2, 0).astype(np.uint8)

                ax[0].imshow(image)
                ax[0].set_title(rf"Environment at $\tau_c={tau_current}$", fontsize=10)
                ax[0].tick_params(axis='both', labelsize=6)
                
                ax[1].imshow(gt_image_q)
                ax[1].set_title(rf"Ground-truth at $\tau_q={tau_query}$", fontsize=10)
                rect_gt = patches.Rectangle(
                    (x1, y1),
                    w,
                    h,
                    linewidth=1,
                    edgecolor=color,
                    facecolor="none"
                )
                ax[1].add_patch(rect_gt)
                ax[1].tick_params(axis='both', labelsize=6)

                ax[2].imshow(gt_image_q)
                ax[2].set_title(rf"Samples at $\tau_q={tau_query}$", fontsize=10)
                for samp in samps:
                    ys1, xs1, hs, ws = samp
                    ys1 = int(ys1 * scale)
                    xs1 = int(xs1 * scale)
                    hs = int(hs * scale)
                    ws = int(ws * scale)
                    rect_samp = patches.Rectangle(
                        (xs1, ys1),
                        ws,
                        hs,
                        linewidth=1,
                        edgecolor=color,
                        facecolor="none"
                    )
                    ax[2].add_patch(rect_samp)
                ax[2].tick_params(axis='both', labelsize=6)

                self.savefig(fig, f"env{i}_object{j}_idx{iteration}")
                
                plt.close(fig)
                
    def compute_log_likelihood(self, results):

        log_likelihoods = []
        
        for result_ in results:
            for result in result_:
                
                x_1 = result["groundtruth"] / 255.0  # Normalize to [0,1]
                x_1 = torch.tensor(x_1, dtype=torch.float32).to(self.device).unsqueeze(0)  # Add batch dimension
                model_extras = result["model_extras"]
                x_0, log_p1 = self.pipeline.compute_likelihood(self.vf, x_1, model_extras)
                log_likelihoods.append(log_p1.cpu().numpy())

        return log_likelihoods