from typing import Dict, Optional, Sequence, Union
import os
import json
import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars
import torch
import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset
from src.utils.lookup import get_active_lookup, LookupTables, resolve_env_names
from src.utils.viz import render_bev, save_bev

class SimDataset(Dataset):

    def __init__(
        self,
        datapath: Optional[Union[str, Sequence[str]]] = None,
        latent_statistics_path: str = None,
        max_tokens: int = 20,
        seed: int = 42,
        bev_size: int = 256,
        lookup: Optional[LookupTables] = None,
        env_name: Optional[Union[str, Sequence[str]]] = None,
        data_root: Optional[str] = None,
        scan_name: str = "scan_merged.parquet",
        overfit: bool = False,
        use_log_size: bool = True,
        size_log_eps: float = 1e-4,
        max_cached_envs: int = 3000,
        preload_workers: int = 16,
    ):
        # Lightweight index: list of {"env_name": str, "timestamp": int}
        self.datapoints = []
        self.env_ranges = []  # (start_idx, end_idx) per loaded env
        # Per-env metadata (JSONs + row count), populated at init
        self._env_meta: Dict[str, dict] = {}
        # Per-env tensor cache, populated lazily on first access (LRU eviction)
        self._env_cache: OrderedDict[str, dict] = OrderedDict()
        self._max_cached_envs = max_cached_envs
        # FIXME: For multi-env runs, a single latent_statistics_path may be invalid; provide a merged stats file.
        self.latent_statistics = np.load(latent_statistics_path, allow_pickle=True) if latent_statistics_path else None
        self.max_tokens = max_tokens
        self.seed = seed
        self.bev_size = bev_size
        self.lookup = lookup or get_active_lookup()
        data_root = Path(data_root) if data_root else Path(__file__).resolve().parents[2] / "data"
        if overfit and data_root.name == "val":
            data_root = data_root.parent / "train"
        self.data_root = data_root
        self.env_names = []
        self.n_classes_objs = len(self.lookup.id2pickupable)
        self.n_classes_recpt = len(self.lookup.id2receptacle)
        self.MIN_SCENE_SIZE = -0.5  # From data stats
        self.MAX_SCENE_SIZE = 34.0  # From data stats
        self.SCENE_RANGE = self.MAX_SCENE_SIZE - self.MIN_SCENE_SIZE
        self.use_log_size = use_log_size
        self.size_log_eps = float(size_log_eps)
        if self.use_log_size:
            if self.size_log_eps <= 0.0:
                raise ValueError("size_log_eps must be > 0")
            self._log_size_min = math.log(self.size_log_eps)
            self._log_size_range = math.log(1.0 + self.size_log_eps) - self._log_size_min

        self.load_from_spec(datapath, env_name, scan_name)
        self._preload_cache(preload_workers)

    def load_from_spec(
        self,
        datapath: Optional[Union[str, Sequence[str]]],
        env_name: Optional[Union[str, Sequence[str]]],
        scan_name: str,
    ):
        if datapath is None:
            for env in resolve_env_names(env_name, data_root=self.data_root):
                env_dir = self.data_root / env
                self.env_names.append(env)
                self.load_from_path(env_dir / scan_name, env_dir)
            return

        if isinstance(datapath, Sequence) and not isinstance(datapath, (str, Path)):
            paths = [Path(p) for p in datapath]
        else:
            paths = [Path(datapath)]
        for path in paths:
            env_dir = path.parent
            self.env_names.append(env_dir.name)
            self.load_from_path(path, env_dir)

    def _load_json_list(self, path: Path):
        with open(path, "r") as f:
            return json.load(f)

    def _build_label_maps(self, env_dir: Path):
        pickupable_names = self._load_json_list(env_dir / "pickupable_names.json")
        receptacle_names = self._load_json_list(env_dir / "receptacle_names.json")

        pickupable_id_map = {}
        for i, name in enumerate(pickupable_names):
            category = name.split("|")[0]
            if category not in self.lookup.pickupable2id:
                raise ValueError(f"Pickupable '{name}' (category '{category}') missing from lookup.")
            pickupable_id_map[i] = self.lookup.pickupable2id[category]

        receptacle_id_map = {}
        for i, name in enumerate(receptacle_names):
            category = name.split("|")[0]
            if category not in self.lookup.receptacle2id:
                raise ValueError(f"Receptacle '{name}' (category '{category}') missing from lookup.")
            receptacle_id_map[i] = self.lookup.receptacle2id[category]

        return pickupable_names, receptacle_names, pickupable_id_map, receptacle_id_map

    def _load_receptacles_locations(self, env_dir: Path):
        return self._load_json_list(env_dir / "receptacles_oobb.json")

    def load_from_path(self, path: Path, env_dir: Optional[Path] = None):
        """Register an env for lazy loading: reads only JSONs and row count."""
        datapath = Path(path)
        env_dir = env_dir or datapath.parent
        if datapath.suffix != ".parquet":
            return

        # Row count from parquet metadata only — no data read.
        T = polars.scan_parquet(datapath).select(polars.len()).collect().item()

        pickupable_names, receptacle_names, pickupable_id_map, receptacle_id_map = self._build_label_maps(env_dir)

        receptacles_locations = self._load_receptacles_locations(env_dir)
        receptacles_cornerpoints = torch.stack(
            [
                torch.tensor(receptacles_locations[rid]["cornerPoints"], dtype=torch.float32)
                for rid in receptacle_names
            ],
            dim=0,
        )  # [N_recepts, 8, 3]

        # Per-env display range: padded by 0.5 m on each side, rounded to int.
        xz = receptacles_cornerpoints[:, :, [0, 2]]  # [N_recepts, 8, 2]
        scene_min = round(float(xz.min().item()) - 0.5)
        scene_max = round(float(xz.max().item()) + 0.5)

        env_name = env_dir.name
        self._env_meta[env_name] = {
            "parquet_path": datapath,
            "pickupable_id_map": pickupable_id_map,
            "receptacle_id_map": receptacle_id_map,
            "receptacles_cornerpoints": receptacles_cornerpoints,
            "N_objs": len(pickupable_names),
            "N_recepts": len(receptacle_names),
            "T": T,
            "scene_min": scene_min,
            "scene_max": scene_max,
        }

        start_idx = len(self.datapoints)
        for t in range(T):
            self.datapoints.append({"env_name": env_name, "timestamp": t})
        self.env_ranges.append((start_idx, start_idx + T))

    def _load_env_to_cache(self, env_name: str):
        """Read parquet for env_name and store tensors in _env_cache."""
        meta = self._env_meta[env_name]
        data = polars.read_parquet(meta["parquet_path"])
        # Polars may return a non-writable NumPy view; copy before torch conversion.
        assignments = torch.from_numpy(np.array(data["assignment"].to_numpy(), copy=True)).long()  # [T, N_objs, N_recepts]
        N_objs_parquet    = assignments.shape[1]
        N_recepts_parquet = assignments.shape[2]
        if N_objs_parquet != meta["N_objs"]:
            raise ValueError(
                f"Pickupable count mismatch for {env_name}: "
                f"{meta['N_objs']} names vs {N_objs_parquet} in parquet."
            )
        if N_recepts_parquet != meta["N_recepts"]:
            raise ValueError(
                f"Receptacle count mismatch for {env_name}: "
                f"{meta['N_recepts']} names vs {N_recepts_parquet} in parquet."
            )
        self._env_cache[env_name] = {
            "rotations":         torch.from_numpy(np.array(np.stack(data["rotation"].to_numpy()), copy=True)).float(),  # [T, N_objs]
            "aabb_cornerpoints": torch.from_numpy(np.array(np.stack(data["aabb_cornerPoints"].to_numpy()), copy=True)).float(),  # [T, N_objs, 8, 3]
        }
        # Evict least recently used env if over budget
        while len(self._env_cache) > self._max_cached_envs:
            self._env_cache.popitem(last=False)

    def _preload_cache(self, num_workers: int):
        """Load all env parquets concurrently in the main process before workers are forked."""
        env_names = list(self._env_meta.keys())
        if not env_names:
            return
        import logging
        log = logging.getLogger(__name__)
        log.info(f"Preloading {len(env_names)} envs with {num_workers} threads...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._load_env_to_cache, name): name for name in env_names}
            for future in as_completed(futures):
                future.result()  # re-raise any exceptions
        log.info(f"Preload complete. {len(self._env_cache)} envs cached.")

    def _build_sample_dict(self, env_name: str, t: int) -> dict:
        """Reconstruct the per-timestep sample dict from cache (loads env if needed)."""
        if env_name not in self._env_cache:
            self._load_env_to_cache(env_name)
        else:
            # Mark as recently used
            self._env_cache.move_to_end(env_name)
        cached = self._env_cache[env_name]
        meta = self._env_meta[env_name]
        N_objs = meta["N_objs"]
        N_recepts = meta["N_recepts"]
        pickupable_id_map = meta["pickupable_id_map"]
        receptacle_id_map = meta["receptacle_id_map"]
        receptacles_cornerpoints = meta["receptacles_cornerpoints"]
        return {
            "timestamp": t,
            "env_name": env_name,
            "objects": [
                {
                    "label": pickupable_id_map[o],
                    "bbox_corners": cached["aabb_cornerpoints"][t, o],
                    "rotation": torch.zeros_like(cached["rotations"][t, o]),  # FIXME
                }
                for o in range(N_objs)
            ],
            "receptacles": [
                {
                    "label": receptacle_id_map[f],
                    "bbox_corners": receptacles_cornerpoints[f],
                    "rotation": torch.tensor(0.0),  # FIXME
                }
                for f in range(N_recepts)
            ],
        }

    @property
    def max_transitions(self):
        if not self.env_ranges:
            return 0
        return max(end - start for start, end in self.env_ranges)

    def model_kwargs(self) -> dict:
        return {
            "obj_classes":      self.n_classes_objs,
            "furn_classes":     self.n_classes_recpt,
            "max_transitions":  self.max_transitions,
        }

    def _env_for_idx(self, idx: int):
        for env_idx, (start, end) in enumerate(self.env_ranges):
            if start <= idx < end:
                return env_idx, start, end
        raise IndexError(f"Index {idx} out of range for env_ranges {self.env_ranges}")

    def __len__(self):
        return len(self.datapoints)

    def get_sample(self, idx: int, mode: str = "normal"):
        env_idx, env_start, env_end = self._env_for_idx(idx)
        T = env_end - env_start
        if T < 2:
            raise ValueError("SimDataset requires at least 2 timesteps to sample t_current and t_query.")

        if mode != "viz":
            t_current = torch.randint(0, T - 1, (1,)).item()
            t_query = torch.randint(t_current + 1, T, (1,)).item()
        else:
            g = torch.Generator()
            g.manual_seed(self.seed + idx)
            t_current = torch.randint(0, T - 1, (1,), generator=g).item()
            t_query = torch.randint(t_current + 1, T, (1,), generator=g).item()

        entry = self.datapoints[env_start]
        env_name = entry["env_name"]
        sample_current = self._build_sample_dict(env_name, t_current)
        sample_query   = self._build_sample_dict(env_name, t_query)

        scene_min = self._env_meta[env_name]["scene_min"]
        scene_max = self._env_meta[env_name]["scene_max"]

        new_sample = {"env_name": env_name, "scene_min": scene_min, "scene_max": scene_max}

        if mode == "viz":
            bev_current = render_bev(
                min_size=scene_min,
                max_size=scene_max,
                sample=sample_current,
                receptacle_colors=self.lookup.receptacle_colors,
                object_colors=self.lookup.pickupable_colors,
                img_size=self.bev_size,
            )
            bev_query = render_bev(
                min_size=scene_min,
                max_size=scene_max,
                sample=sample_query,
                receptacle_colors=self.lookup.receptacle_colors,
                object_colors=self.lookup.pickupable_colors,
                img_size=self.bev_size,
            )
            # Shape: (H, W, 3), values in [0, 1]
            new_sample["image"] = bev_current
            new_sample["gt_image_query"] = bev_query

        receptacle_bbox, receptacle_labels, Nr = self.prepare_tensor_data("receptacles", sample_current) # (Nr, 7), (Nr, 1)
        obj_bbox, obj_labels, No = self.prepare_tensor_data("objects", sample_current) # (No, 7), (No, 1)
        obj_q_bbox, obj_q_labels, _ = self.prepare_tensor_data("objects", sample_query) # (No, 7), (No, 1)

        new_sample["t_current"] = torch.tensor(t_current, dtype=torch.long)
        new_sample["t_query"] = torch.tensor(t_query, dtype=torch.long)

        receptacle = torch.cat([receptacle_bbox, receptacle_labels], dim=1) # (Nr, 8), 8 is (cx, cy, cz, sx, sy, sz, yaw, label)
        obj = torch.cat([obj_bbox, obj_labels], dim=1) # (No, 8), 8 is (cx, cy, cz, sx, sy, sz, yaw, label)
        obj_q = torch.cat([obj_q_bbox, obj_q_labels], dim=1) # (No, 8), 8 is (cx, cy, cz, sx, sy, sz, yaw, label)
        map_ = torch.cat([receptacle, obj], dim=0) # (Nr + No, 8)

        # clamping the number of objects to max_tokens
        trunc_total = min(Nr + No, self.max_tokens)
        map_ = map_[:trunc_total]
        Nr_trunc = min(Nr, trunc_total)
        No_trunc = max(0, trunc_total - Nr)
        Nr, No = Nr_trunc, No_trunc
        total = Nr + No

        # now we pad the map to max_tokens
        map_padded = F.pad(map_, (0, 0, 0, self.max_tokens - map_.shape[0]), value=0.0) # (max_tokens, 8)
        new_sample["map"] = map_padded

        mask = torch.zeros(self.max_tokens, dtype=torch.bool)
        mask[total:] = True # True indicates positions to be ignored, we are using torch Transformers

        if mode != "viz":
            i = torch.randint(0, No, (1,)).item()
            new_sample["mask"] = mask
            new_sample["query_object"] = obj_q[i].unsqueeze(0)
        else:
            queries = []
            masks = []
            for i in range(No):
                val_mask = mask.clone()
                queries.append(obj_q[i].unsqueeze(0))
                masks.append(val_mask)
            new_sample["mask"] = torch.cat(masks, dim=0)
            new_sample["query_object"] = torch.cat(queries, dim=0)

        new_sample["types"] = -1 * torch.ones(self.max_tokens, dtype=torch.int16)
        new_sample["types"][:Nr] = 0
        new_sample["types"][Nr : Nr + No] = 1

        return new_sample

    def __getitem__(self, idx: int):
        return self.get_sample(idx)

    def prepare_tensor_data(self, type: str, sample: Dict):
        """
        Prepares tensor data for objects or furnitures.
        """

        data = sample[type]

        N = len(data)

        centers, sizes = self.corners_to_center_size(
            torch.stack(
                [data[i]["bbox_corners"] for i in range(N)],  # (N, 8, 3)
                dim=0,
            )
        )  # (N, 3), (N, 3)
        rotations = torch.stack(
            [data[i]["rotation"] for i in range(N)],  # (N,)
            dim=0,
        )  # (N,)

        rotations = rotations.unsqueeze(1) % (2 * np.pi)  # Ensure rotations are within [0, 2pi], (N, 1)

        centers_norm, sizes_norm = self.normalize_center_size(centers, sizes)
        assert (
            centers_norm.min() >= 0.0 and centers_norm.max() <= 1.0
            and sizes_norm.min() >= 0.0 and sizes_norm.max() <= 1.0
        ), f"{type} bounding boxes must be normalized between 0 and 1. \
            Got centers in [{centers_norm.min().item()}, {centers_norm.max().item()}] \
            and sizes in [{sizes_norm.min().item()}, {sizes_norm.max().item()}]"

        sizes_norm = self.size_to_model(sizes_norm)
        bbox = torch.cat([centers_norm, sizes_norm, rotations], dim=1)  # (N, 7)

        labels = [data[i]["label"] for i in range(N)]
        labels = torch.tensor(labels).unsqueeze(1)           # (N, 1)

        return bbox, labels, N

    def size_to_model(self, sizes: torch.Tensor) -> torch.Tensor:
        if self.use_log_size:
            log_sizes = torch.log(sizes.clamp_min(0.0) + self.size_log_eps)
            return (log_sizes - self._log_size_min) / self._log_size_range
        return sizes

    def size_from_model(self, sizes_model: torch.Tensor) -> torch.Tensor:
        if self.use_log_size:
            log_sizes = sizes_model * self._log_size_range + self._log_size_min
            return (torch.exp(log_sizes) - self.size_log_eps).clamp(0.0, 1.0)
        return sizes_model

    def normalize_center_size(self, center: torch.Tensor, size: torch.Tensor):
        """
        Normalize 3D box center and size.
        Outputs normalized center and size in range [0, 1].
        """
        center_normalized = (center - self.MIN_SCENE_SIZE) / self.SCENE_RANGE
        size_normalized = size / self.SCENE_RANGE

        return center_normalized, size_normalized


    def denormalize_center_size(self, center_norm: torch.Tensor, size_norm: torch.Tensor):
        """
        Denormalize 3D box center and size from [0, 1] back to world coordinates.
        """
        center = center_norm * self.SCENE_RANGE + self.MIN_SCENE_SIZE
        size   = size_norm * self.SCENE_RANGE

        return center, size


    def corners_to_center_size(self, corners: torch.Tensor):
        """
        Convert 3D box corners to center and size.
        """
        xyz_min = corners.min(dim=-2).values   # (..., 3)
        xyz_max = corners.max(dim=-2).values   # (..., 3)

        center = (xyz_min + xyz_max) / 2.0
        sizes = xyz_max - xyz_min

        return center, sizes

    def save_bev_sequence(
        self,
        output_dir: str,
        prefix: str = "bev",
        start: int = 0,
        end: Optional[int] = None,
    ):
        """
        Save BEV images for all timesteps in [start, end) to output_dir.
        """
        os.makedirs(output_dir, exist_ok=True)
        if end is None:
            end = len(self.datapoints)
        for t in range(start, min(end, len(self.datapoints))):
            entry = self.datapoints[t]
            sample = self._build_sample_dict(entry["env_name"], entry["timestamp"])
            timestamp = sample.get("timestamp", t)
            scene_min = self._env_meta[entry["env_name"]]["scene_min"]
            scene_max = self._env_meta[entry["env_name"]]["scene_max"]
            bev_img = render_bev(
                min_size=scene_min,
                max_size=scene_max,
                sample=sample,
                receptacle_colors=self.lookup.receptacle_colors,
                object_colors=self.lookup.pickupable_colors,
                img_size=self.bev_size,
            )
            save_path = os.path.join(output_dir, f"{prefix}_{timestamp:06d}.png")
            save_bev(
                bev_img,
                self.lookup.receptacle_colors,
                self.lookup.pickupable_colors,
                self.MAX_SCENE_SIZE,
                save_path,
                title=f"BEV at t={timestamp}",
                legend_sample=sample,
            )



if __name__ == "__main__":
    from src.utils.lookup import load_lookup, set_active_lookup

    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data"

    env_dirs = [
        p for p in sorted(data_root.iterdir())
        if p.is_dir() and p.name.startswith("env") and "_obj" in p.name
    ]
    if not env_dirs:
        raise SystemExit(f"No env directories found under {data_root}")

    single_env = env_dirs[0].name
    pattern = single_env.split("_", 1)[1]
    if not pattern.startswith("obj"):
        raise SystemExit(f"Derived pattern '{pattern}' does not start with 'obj'.")


    # Test 1: Single-env loading
    print(f"[single] env_name={single_env}")
    lookup = load_lookup(single_env, data_root=data_root)
    set_active_lookup(lookup)
    dataset = SimDataset(env_name=single_env, data_root=str(data_root), scan_name="scan_merged.parquet")
    print(f"[single] loaded {len(dataset)} datapoints, env_ranges={dataset.env_ranges}")

    assert len(dataset.env_ranges) == 1, f"Expected 1 env_range, got {len(dataset.env_ranges)}"
    start, end = dataset.env_ranges[0]
    assert start == 0 and end == len(dataset), "env_range should span all datapoints"
    print("[single] OK: env_ranges correct")

    sample = dataset.get_sample(0)
    assert dataset.datapoints[0].get("env_name") == single_env
    print(f"[single] sample keys: {list(sample.keys())}")

    # Test 2: No cross-env contamination (single env validates the path)
    for i in range(50):
        s = dataset.get_sample(i % len(dataset))
        t_c = s["t_current"].item()
        t_q = s["t_query"].item()
        assert t_c < t_q, f"t_current={t_c} >= t_query={t_q}"
        assert 0 <= t_c < end - start, f"t_current={t_c} out of range [0, {end - start})"
        assert 0 <= t_q < end - start, f"t_query={t_q} out of range [0, {end - start})"
    print("[single] OK: 50 random samples have valid t_current < t_query within env range")

    # Visual sanity check: render BEV pair for single env
    sample_viz = dataset.get_sample(0, "viz")
    legend_sample = dataset._build_sample_dict(dataset.datapoints[0]["env_name"], dataset.datapoints[0]["timestamp"])
    bev_out_dir = project_root / "tmp" / "sim_bev_single"
    os.makedirs(bev_out_dir, exist_ok=True)
    save_bev(
        sample_viz["image"],
        dataset.lookup.receptacle_colors,
        dataset.lookup.pickupable_colors,
        dataset.MAX_SCENE_SIZE,
        str(bev_out_dir / f"{single_env}_current.png"),
        title=rf"BEV Current at $\tau$={sample_viz['t_current'].item()}",
        legend_sample=legend_sample,
    )
    save_bev(
        sample_viz["gt_image_query"],
        dataset.lookup.receptacle_colors,
        dataset.lookup.pickupable_colors,
        dataset.MAX_SCENE_SIZE,
        str(bev_out_dir / f"{single_env}_query.png"),
        title=rf"BEV Query at $\tau$={sample_viz['t_query'].item()}",
        legend_sample=legend_sample,
    )
    print(f"[single] saved BEV images to {bev_out_dir}")

    # Test 3: Multi-env loading (pattern)
    print(f"\n[multi] pattern={pattern}")
    lookup = load_lookup(pattern, data_root=data_root)
    set_active_lookup(lookup)
    dataset_all = SimDataset(env_name=pattern, data_root=str(data_root), scan_name="scan_merged.parquet")
    print(f"[multi] all={len(dataset_all)} (envs={len(dataset_all.env_names)})")

    # Test 4: env_ranges integrity and no cross-env contamination
    print(f"[multi] env_ranges={dataset_all.env_ranges}")
    for env_idx, (s, e) in enumerate(dataset_all.env_ranges):
        env_name_expected = dataset_all.env_names[env_idx]
        for dp_idx in range(s, e):
            dp_env = dataset_all.datapoints[dp_idx].get("env_name")
            assert dp_env == env_name_expected, \
                f"datapoint[{dp_idx}] env_name={dp_env} != expected {env_name_expected}"
    print("[multi] OK: all datapoints have correct env_name matching env_ranges")

    for i in range(min(100, len(dataset_all))):
        s = dataset_all.get_sample(i)
        env_idx, env_start, env_end = dataset_all._env_for_idx(i)
        t_c = s["t_current"].item()
        t_q = s["t_query"].item()
        T = env_end - env_start
        assert 0 <= t_c < T, f"idx={i}: t_current={t_c} out of env range T={T}"
        assert t_c < t_q <= T - 1, f"idx={i}: t_query={t_q} out of env range T={T} (t_current={t_c})"
    print("[multi] OK: 100 samples stay within their env boundaries")

    # Visual sanity check: save a BEV current/query pair for first sample of each env
    bev_out_dir = project_root / "tmp" / "sim_bev_multi"
    os.makedirs(bev_out_dir, exist_ok=True)
    for env_idx, (s, e) in enumerate(dataset_all.env_ranges):
        env_name = dataset_all.env_names[env_idx]
        sample_viz = dataset_all.get_sample(s, "viz")
        legend_sample = dataset_all._build_sample_dict(dataset_all.datapoints[s]["env_name"], dataset_all.datapoints[s]["timestamp"])
        safe_env = env_name.replace(os.sep, "_")
        save_bev(
            sample_viz["image"],
            dataset_all.lookup.receptacle_colors,
            dataset_all.lookup.pickupable_colors,
            dataset_all.MAX_SCENE_SIZE,
            str(bev_out_dir / f"{safe_env}_current.png"),
            title=rf"{env_name} BEV Current at $\tau$={sample_viz['t_current'].item()}",
            legend_sample=legend_sample,
        )
        save_bev(
            sample_viz["gt_image_query"],
            dataset_all.lookup.receptacle_colors,
            dataset_all.lookup.pickupable_colors,
            dataset_all.MAX_SCENE_SIZE,
            str(bev_out_dir / f"{safe_env}_query.png"),
            title=rf"{env_name} BEV Query at $\tau$={sample_viz['t_query'].item()}",
            legend_sample=legend_sample,
        )
    print(f"[multi] saved BEV images for {len(dataset_all.env_ranges)} envs to {bev_out_dir}")

    # Test 5: max_transitions property reflects largest env
    expected_max = max(e - s for s, e in dataset_all.env_ranges)
    assert dataset_all.max_transitions == expected_max, \
        f"max_transitions={dataset_all.max_transitions} != expected {expected_max}"
    print(f"[multi] OK: max_transitions={dataset_all.max_transitions} (largest env)")

    print("\nAll tests passed.")
