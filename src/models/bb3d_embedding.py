import torch
import torch.nn as nn

from typing import Optional, Callable
from torch import Tensor

def yaw_to_quat(yaw):
    if not torch.is_tensor(yaw):
        yaw = torch.as_tensor(yaw)
    half = yaw / 2
    s = torch.sin(half)
    c = torch.cos(half)
    zeros = torch.zeros_like(s)
    return torch.cat((zeros, zeros, s, c), dim=-1)



class Mlp(nn.Module):
    '''
    Reference: https://github.com/facebookresearch/map-anything/blob/main/mapanything/models/external/vggt/layers/mlp.py
    '''
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BB3DEmbedding(nn.Module):
    """
    This class takes a 3D Bounding Box (3DBB) as input and produces an embedding feature vector.
    In particular, the 3DBB is represented as a 7D vector, which contains:
    - 3D coordinates of the center of the box (x, y, z)
    - 3D dimensions of the box (width, height, depth)
    - yaw rotation of the box (around the vertical axis)

    Inspired by MapAnything https://arxiv.org/pdf/2509.13414, we use a 4-layer MLP with GeLU activation functions to embed the
    three sources of information separatedly. In particular, we embed the translation, rotation, and size information
    and then merge them together with layer norm, summation, and layer norm again.
    """
    
    def __init__(self, hidden_dim: int = 1024, drop: float = 0.0, eps: float = 1e-6, encode_yaw: bool = True):
        super().__init__()
                
        self.encode_yaw = encode_yaw
        
        # Per MapAnything: direction and log-scale are independent, encoded by separate MLPs and summed.
        # translation: unit direction (3) and log-magnitude (1) — separate 4-layer MLPs, summed
        self.t_dir_mlp1 = Mlp(in_features=3, hidden_features=hidden_dim, out_features=hidden_dim, drop=drop)
        self.t_dir_mlp2 = Mlp(in_features=hidden_dim, hidden_features=hidden_dim, out_features=hidden_dim, drop=drop)
        self.t_scale_mlp1 = Mlp(in_features=1, hidden_features=hidden_dim, out_features=hidden_dim, drop=drop)
        self.t_scale_mlp2 = Mlp(in_features=hidden_dim, hidden_features=hidden_dim, out_features=hidden_dim, drop=drop)
        # size: unit direction (3) and log-magnitude (1) — separate 4-layer MLPs, summed
        self.s_dir_mlp1 = Mlp(in_features=3, hidden_features=hidden_dim, out_features=hidden_dim, drop=drop)
        self.s_dir_mlp2 = Mlp(in_features=hidden_dim, hidden_features=hidden_dim, out_features=hidden_dim, drop=drop)
        self.s_scale_mlp1 = Mlp(in_features=1, hidden_features=hidden_dim, out_features=hidden_dim, drop=drop)
        self.s_scale_mlp2 = Mlp(in_features=hidden_dim, hidden_features=hidden_dim, out_features=hidden_dim, drop=drop)

        self.ln_t = nn.LayerNorm(hidden_dim, eps=eps)
        self.ln_s = nn.LayerNorm(hidden_dim, eps=eps)
        
        # orientation
        if self.encode_yaw:
            self.q_mlp1 = Mlp(in_features=4, hidden_features=hidden_dim, out_features=hidden_dim, drop=drop)
            self.q_mlp2 = Mlp(in_features=hidden_dim, hidden_features=hidden_dim, out_features=hidden_dim, drop=drop)
            self.ln_q = nn.LayerNorm(hidden_dim, eps=eps)

        self.ln_sum = nn.LayerNorm(hidden_dim, eps=eps)

        self.eps = eps
        
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _safe_l2_normalize(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
        return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)

    def forward(self, x: Tensor):
        """
        Args:
            x: 
            - if self.encode_yaw: (B, 7), [tx, ty, tz, h, w, l, yaw]
            - else: (B, 6), [tx, ty, tz, h, w, l]
            
        Returns:
            F_sum: (B, D)
        """
        
        trasl  = x[:, 0:3]
        size   = x[:, 3:6]

        t_norm = trasl.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        t_unit = trasl / t_norm

        s_norm = size.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        s_unit = size / s_norm

        # direction and scale encoded separately, then summed (per MapAnything)
        # scales are not log-transformed here — log scaling is applied by the dataset before tokenization
        feats_t = self.t_dir_mlp2(self.t_dir_mlp1(t_unit)) + self.t_scale_mlp2(self.t_scale_mlp1(t_norm))
        feats_s = self.s_dir_mlp2(self.s_dir_mlp1(s_unit)) + self.s_scale_mlp2(self.s_scale_mlp1(s_norm))
        
        feats_t_norm = self.ln_t(feats_t)
        feats_s_norm = self.ln_s(feats_s)
        
        feats_sum = feats_t_norm + feats_s_norm
        
        if self.encode_yaw:
            angles = x[:, 6].unsqueeze(-1) 
            quats  = yaw_to_quat(angles)
            
            assert quats.shape[-1] == 4 and trasl.shape[-1] == 3 and size.shape[-1] == 3
            assert quats.shape[0] == trasl.shape[0] and quats.shape[0] == size.shape[0]
            
            q_unit = self._safe_l2_normalize(quats, dim=-1, eps=self.eps)
            feats_q = self.q_mlp2(self.q_mlp1(q_unit))  # 4-layer total
            feats_q_norm = self.ln_q(feats_q)
            
            feats_sum = feats_sum + feats_q_norm
            
        
        return self.ln_sum(feats_sum)




if __name__ == "__main__":
    # Test the BB3DEmbedding class
    model = BB3DEmbedding(hidden_dim=256)
    x = torch.randn(10, 7)  # Batch of 10 3D bounding boxes with (tx, ty, tz, h, w, l, yaw)
    out = model(x)
    print(out.shape)  # Should be (10, 256)