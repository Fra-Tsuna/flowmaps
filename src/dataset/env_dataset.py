import torch
import numpy as np
import torch.nn.functional as F
import random
from einops import rearrange

from torch.utils.data import Dataset
from src.utils.lookup import COLOR2ID, ID2COLOR


class EnvDataset(Dataset):

    def __init__(
        self,
        datapath: str,
        max_tokens: int = 20,
        seed: int = 42,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.datapoints = []
        self.max_tokens = max_tokens
        self.seed = seed

        self.load_from_path(datapath)

    def load_from_path(self, path):
        datapath = path
        if datapath.endswith(".npz"):
            data = np.load(datapath, allow_pickle=True)
            data = data["data"]
            datapoints = data.tolist()
            print(f"Loaded {len(datapoints)} datapoints from {datapath}")

            scale = datapoints[0]["scale"]
            self.H = datapoints[0]["imgs"][0].shape[1] / scale

            for datapoint in datapoints:

                T = len(datapoint["imgs"])
                env_id = datapoint["id_env"]
                point = {
                    "env_id": env_id,
                    "seq_length": T,
                    "imgs": datapoint["imgs"],
                    "furnitures": [],  # 'color', 'bbox'
                    "objects": [],  # for timestamp ['color', 'bbox', 'timestamp']
                }

                point["furnitures"] = [
                    {
                        "color": torch.as_tensor(COLOR2ID[f["color"]], dtype=torch.long),
                        "bbox": torch.as_tensor(f["bbox"], dtype=torch.float32) / (self.H - 1),
                        "id": torch.as_tensor(f["id"], dtype=torch.long),
                    }
                    for f in datapoint["furniture"]
                ]

                for t in range(T):
                    point["objects"].append(
                        [
                            {
                                "color": torch.as_tensor(COLOR2ID[obj["color"]], dtype=torch.long),
                                "bbox": torch.as_tensor(obj["bbox"], dtype=torch.float32) / (self.H - 1),
                                "on_furniture_id": torch.as_tensor(obj["on_furniture_id"], dtype=torch.long),
                                "id": torch.as_tensor(obj["id"], dtype=torch.long),
                            }
                            for obj in datapoint["object"][t]
                        ]
                    )

                self.datapoints.append(point)

        else:
            raise ValueError(f"Unsupported file format: {datapath}")

    def __len__(self):
        return len(self.datapoints)

    def get_sample(self, idx: int, mode: str = "normal"):
        sample = self.datapoints[idx]

        T = sample["seq_length"]
        if mode != "viz":
            t_current = torch.randint(0, T, (1,)).item()
            t_query = torch.randint(t_current + 1, T, (1,)).item() if t_current != T - 1 else T - 1
        else:
            g = torch.Generator()
            g.manual_seed(self.seed + idx)
            t_current = torch.randint(0, T, (1,), generator=g).item()
            t_query = torch.randint(t_current + 1, T, (1,), generator=g).item() if t_current != T - 1 else T - 1

        new_sample = {}

        if mode == "viz":
            new_sample["image"] = torch.tensor(sample["imgs"][t_current], dtype=torch.uint8)
            new_sample["gt_image_query"] = torch.tensor(sample["imgs"][t_query], dtype=torch.uint8)

        Nf = len(sample["furnitures"])
        furn_bbox = [sample["furnitures"][i]["bbox"] for i in range(Nf)]
        furn_bbox = torch.stack(furn_bbox, dim=0)  # (Nf, 4)
        assert (
            furn_bbox.min() >= 0.0 and furn_bbox.max() <= 1.0
        ), f"Furniture bounding boxes must be normalized between 0 and 1"
        furn_bbox = self.yxyx_to_yxhw(furn_bbox)  # (Nf, 4)

        furn_color = [sample["furnitures"][i]["color"] for i in range(Nf)]
        furn_color = rearrange(torch.stack(furn_color, dim=0), "Nf -> Nf 1")  # (Nf, 1)

        No = len(sample["objects"][t_current])
        obj_bbox = [sample["objects"][t_current][i]["bbox"] for i in range(No)]
        obj_bbox = torch.stack(obj_bbox, dim=0)  # (No, 4)
        assert (
            obj_bbox.min() >= 0.0 and obj_bbox.max() <= 1.0
        ), "object bounding boxes must be normalized between 0 and 1"
        obj_bbox = self.yxyx_to_yxhw(obj_bbox)  # (No, 4)

        obj_q_bbox = [sample["objects"][t_query][i]["bbox"] for i in range(No)]
        obj_q_bbox = torch.stack(obj_q_bbox, dim=0)  # (No, 4)
        assert (
            obj_q_bbox.min() >= 0.0 and obj_q_bbox.max() <= 1.0
        ), "query object bounding boxes must be normalized between 0 and 1"
        obj_q_bbox = self.yxyx_to_yxhw(obj_q_bbox)  # (No, 4)

        obj_color = [sample["objects"][t_current][i]["color"] for i in range(No)]
        obj_color = rearrange(torch.stack(obj_color, dim=0), "No -> No 1")  # (No, 1)
        obj_q_color = [sample["objects"][t_query][i]["color"] for i in range(No)]
        obj_q_color = rearrange(torch.stack(obj_q_color, dim=0), "No -> No 1")  # (No, 1)
        
        obj_ids_q = [sample["objects"][t_query][i]["id"] for i in range(No)]
        obj_ids_q = torch.stack(obj_ids_q, dim=0)

        new_sample["t_current"] = torch.tensor(t_current, dtype=torch.long)
        new_sample["t_query"] = torch.tensor(t_query, dtype=torch.long)
        furn = torch.cat([furn_bbox, furn_color], dim=1)
        obj = torch.cat([obj_bbox, obj_color], dim=1)
        obj_q = torch.cat([obj_q_bbox, obj_q_color], dim=1)
        map_ = torch.cat([furn, obj], dim=0)

        # clamping the number of objects to max_tokens
        trunc_total = min(Nf + No, self.max_tokens)
        map_ = map_[:min(Nf + No, self.max_tokens)]
        Nf_trunc = min(Nf, trunc_total)
        No_trunc = max(0, trunc_total - Nf)
        Nf, No = Nf_trunc, No_trunc
        total = Nf + No

        # now we pad the map to max_tokens
        map_padded = F.pad(map_, (0, 0, 0, self.max_tokens - map_.shape[0]), value=0.0)
        new_sample["map"] = map_padded

        mask = torch.zeros(self.max_tokens, dtype=torch.bool)
        mask[total:] = True

        if mode != "viz":
            i = random.sample(range(0, No), k=1)[0]
            new_sample["mask"] = mask
            new_sample["query_object"] = obj_q[i].unsqueeze(0)
            new_sample["q_obj_id"] = obj_ids_q[i]
        else:
            queries = []
            masks = []
            for i in range(No):
                val_mask = mask.clone()
                queries.append(obj_q[i].unsqueeze(0))
                masks.append(val_mask)
            new_sample["query_object"] = torch.cat(queries, dim=0)
            new_sample["mask"] = torch.cat(masks, dim=0)

        new_sample["types"] = -1 * torch.ones(self.max_tokens, dtype=torch.int16)
        new_sample["types"][:Nf] = 0
        new_sample["types"][Nf : Nf + No] = 1

        new_sample["env_id"] = torch.tensor(sample["env_id"], dtype=torch.long)

        return new_sample
    
    def __getitem__(self, idx: int):
        return self.get_sample(idx)

    def yxyx_to_yxhw(self, bboxes: torch.Tensor) -> torch.Tensor:
        """
        Convert bounding boxes from (y1, x1, y2, x2) to (y1, x1, h, w).
        """

        y1, x1, y2, x2 = bboxes.unbind(dim=1)
        h = y2 - y1
        w = x2 - x1

        return torch.stack((y1, x1, h, w), dim=1)


if __name__ == "__main__":

    dataset = EnvDataset("DummyPath")
    
    for element in dataset:
        print(f"t_current: {element['t_current']}, t_query: {element['t_query']}")
        print(f"map: {element['map']}")
        print(f"query_object: {element['query_object']}")
        print("\n")