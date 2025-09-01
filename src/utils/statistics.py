import torch
from torch.utils.data import Dataset
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

from typing import Dict, List


def compute_occurrences(dataset: Dataset) -> Dict:
    """
    results[env_id][timestamp][obj_id] = { (y1,x1,h,w): count, ... }

    For each (env_id, obj_id), we compute K = number of distinct on_furniture_id
    that object visits in that scene. For a given t, we look at all frames tau
    with tau % K == t % K and count identical (rounded) bboxes.
    """
    
    results = {}

    def _yxyx_to_yxhw(bboxes):
            b = torch.as_tensor(bboxes)
            if b.shape[-1] != 4:
                raise ValueError(f"_yxyx_to_yxhw expects last dim=4, got {tuple(b.shape)}")
            y1, x1, y2, x2 = b.unbind(-1)  # use last dim, not dim=1
            h = y2 - y1
            w = x2 - x1
            return torch.stack((y1, x1, h, w), dim=-1)

    # Compute statistics
    for point in dataset.datapoints:
            T = int(point["seq_length"])
            env_id = int(point["env_id"])
            env_out = results.setdefault(env_id, {})

            # collect object IDs present in this scene
            def _to_int(x):  # tensor or int
                return int(x.item()) if isinstance(x, torch.Tensor) else int(x)

            obj_ids = sorted({
                _to_int(o["id"])
                for tau in range(T)
                for o in point["objects"][tau]
                if "id" in o
            })

            for oid in obj_ids:
                # gather this object's track: (tau, bbox_yxyx) and the furniture ids it visited
                track: List[tuple] = []
                visited_furn = set()
                for tau in range(T):
                    matches = [o for o in point["objects"][tau] if _to_int(o["id"]) == oid]
                    if not matches:
                        continue
                    o = matches[0]
                    visited_furn.add(_to_int(o["on_furniture_id"]))
                    track.append((tau, torch.as_tensor(o["bbox"], dtype=torch.float32)))  # y1,x1,y2,x2

                if not track:
                    continue

                K = max(1, len(visited_furn))  # cycle length

                # group by residue r = tau % K
                by_res = {}
                for tau, bb_yxyx in track:
                    r = tau % K
                    by_res.setdefault(r, []).append(bb_yxyx)

                # for each absolute timestamp t, tally boxes from the matching residue
                for t in range(T):
                    r = t % K
                    if r not in by_res:
                        continue
                    t_out = env_out.setdefault(t, {})
                    counts = t_out.setdefault(oid, {})
                    for bb_yxyx in by_res[r]:
                        bb_yxhw = _yxyx_to_yxhw(bb_yxyx)              # (4,)
                        key = tuple(round(v, 4) for v in bb_yxhw.tolist())
                        counts[key] = counts.get(key, 0) + 1

    return results




def build_gmms_from_stats(
    results: Dict,
    beta_pos: float = 0.35,
    beta_size: float = 0.02,
    min_sigma: float = 1e-4,
    return_dists_params: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict:
    """
    Input:
      results[env_id][timestamp][obj_id] = { (y1,x1,h,w): count, ... }

    Output (flat dict):
      gmms[(env_id, timestamp, obj_id)] = {
          "dist": MixtureSameFamily over R^2 (y,x),
          # optionally also:
          "weights": Tensor[K],
          "means":   Tensor[K, 2],      # (cy, cx)
          "cov":     Tensor[K, 2, 2],   # diag [[sigma_y^2, 0], [0, sigma_x^2]]
      }

    Covariance is proportional to box size:
      sigma_y = max(beta * h, min_sigma),  sigma_x = max(beta * w, min_sigma)
    """
    gmms: Dict = {}

    for env_id, per_t in results.items():
        for t, per_obj in per_t.items():
            for obj_id, bb_counts in per_obj.items():
                if not bb_counts:
                    continue

                means, covs, weights = [], [], []
                for (y, x, h, w), cnt in bb_counts.items():
                    y = float(y); x = float(x); h = float(h); w = float(w)

                    cy = y + 0.5 * h
                    cx = x + 0.5 * w
                    sy = max(beta_pos * h, min_sigma)
                    sx = max(beta_pos * w, min_sigma)
                    sh = max(beta_size * h, min_sigma)
                    sw = max(beta_size * w, min_sigma)

                    means.append([cy, cx, h, w])
                    covs.append([[sy**2, 0.0, 0.0, 0.0],
                                 [0.0, sx**2, 0.0, 0.0],
                                 [0.0, 0.0, sh**2, 0.0],
                                 [0.0, 0.0, 0.0, sw**2]])
                    weights.append(float(cnt))

                weights = torch.tensor(weights, dtype=dtype, device=device)
                weights = weights / weights.sum()
                means   = torch.tensor(means,  dtype=dtype, device=device)   # [K,2]
                covs    = torch.tensor(covs,   dtype=dtype, device=device)   # [K,2,2]

                components = MultivariateNormal(loc=means, covariance_matrix=covs)  # batch=[K], event=[2]
                mix = Categorical(probs=weights)
                dist = MixtureSameFamily(mix, components)

                key = (int(env_id), int(t), int(obj_id))
                gmms[key] = {"dist": dist}
                if return_dists_params:
                    gmms[key].update({"weights": weights, "means": means, "cov": covs})

    return gmms