import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Tuple, List

from src.models.transformer import TransformerEncoder, ColorEmbedding, BBoxEmbedding

class MLP(nn.Module):
    """
    MLP baseline that regresses the mean of a 4-D Gaussian with unit variance
    for the target bbox [y, x, h, w].

    Hyperparameters adapted to your DiT config (dit.yaml):
      - hidden_size (mapped to per-layer width)
      - depth (number of hidden layers)
      - mlp_ratio (kept for parity/logging; not critical here)
      - max_transitions (used to embed tau0 and tau like in DiT)
      - dropout

    Output: mean with shape (B, 1, 4).

    Checkpoint compatibility:
      - self.model_args includes __class__ and all resolved sizes.
    """

    def __init__(
        self,
        hidden_size: int = 256,     # layer width
        depth: int = 8,             # # of hidden layers 
        num_heads: int = 8,         # for the map transformer encoder
        mlp_ratio: float = 4.0,     # kept for parity/logging  
        max_transitions: int = 20,  # for time embeddings of tau0/tau 
        dropout: float = 0.0,     
        ):
        super().__init__()

        # Vocabulary sizes
        n_obj_ids: int = 6        # object-id vocabulary size
        n_type_ids: int = 2       # token-type vocabulary size

        # Embedding dimensions
        self.hidden_size = hidden_size

        self.depth = depth
        self.max_transitions = max_transitions
        self.dropout = dropout

        # Embeddings
        self.obj_emb = ColorEmbedding(hidden_dim=hidden_size, n_classes=n_obj_ids)
        self.bb_chunk_emb = BBoxEmbedding(hidden_dim=hidden_size)
        self.type_emb = nn.Parameter(torch.zeros(n_type_ids, hidden_size), requires_grad=True) 

        self.tau0_emb = nn.Embedding(max_transitions, self.hidden_size)
        self.tau_emb = nn.Embedding(max_transitions, self.hidden_size)

        # Input feature dimension
        time_feat_dim = (self.hidden_size * 2)
        in_dim = self.hidden_size + self.hidden_size + time_feat_dim

        # MLP: width = hidden_size, depth = num layers
        layers: List[nn.Module] = []
        last = in_dim
        for _ in range(depth):
            layers += [nn.Linear(last, hidden_size), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            last = hidden_size
        layers += [nn.Linear(last, 4)]  # output mean [y, x, h, w]
        self.mlp = nn.Sequential(*layers)

        self.map_encoder = TransformerEncoder(
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

        self.model_args = {k: v for k, v in locals().items() if k != "self"}

        # Weight init (xavier) for linear layers
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_encoder(self, map, types, tau0, key_padding_mask: Optional[torch.Tensor] = None):
        map_emb = self._embed_map(map, types, tau0)  # [B, S, D]
        
        return self.map_encoder(
            map_emb, 
            src_key_padding_mask=key_padding_mask
        ) # [B, S, D]

    def _embed_map(self, map_tokens, types, tau0):
        """
        - map_tokens: (B, S, 5) tensor of map tokens
        """
        bb_chunk_emb = self.bb_chunk_emb(map_tokens[:, :, :4])  # (B, S, D)
        color_chunk_emb = self.obj_emb(map_tokens[:, :, 4])  # (B, S, D)
        tau0 = self.tau0_emb(tau0).unsqueeze(1)  # [B, 1, D]

        type_emb = torch.zeros_like(bb_chunk_emb)
        
        mask0 = types == 0  # Furniture
        mask1 = types == 1  # Object
        type_emb[mask0] = self.type_emb[0]  # Furniture
        type_emb[mask1] = self.type_emb[1]  # Object
        
        map_emb = bb_chunk_emb + color_chunk_emb + type_emb + tau0  # (B, S, D)
        return map_emb  # (B, S, D)

    def _masked_mean(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        """
        x: (B, S, D)
        mask: (B, S) bool with True -> padding (ignore)
        returns: (B, D)
        """
        if mask is None:
            return x.mean(dim=1)
        valid = (~mask.bool()).float().unsqueeze(-1)  # 1 where valid
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (x * valid).sum(dim=1) / denom

    def forward(
        self,
        obj: Optional[Tensor] = None,        # (B,1) integer object id
        map: Optional[Tensor] = None,        # (B,S,5)
        tau0: Optional[Tensor] = None,       # (B,)
        tau: Optional[Tensor] = None,        # (B,)
        types: Optional[Tensor] = None,      # (B,S) int type ids
        key_padding_mask: Optional[Tensor] = None,  # (B,S) bool
    ):
        
        assert map is not None, "map (B,S,5) is required"
        B = map.shape[0]

        # Map pooled features
        map = self.forward_encoder(map, types, tau0, key_padding_mask)
        map_pooled = self._masked_mean(map.float(), key_padding_mask)

        # Object features
        obj_id = obj.squeeze(-1).long().clamp_min(0)
        obj_emb = self.obj_emb(obj_id)

        # Time features
        t0 = self.tau0_emb(tau0.long())  
        tt = self.tau_emb(tau.long())    
        times = torch.cat([t0, tt], dim=-1)

        # Assemble features
        feats = torch.cat([map_pooled, obj_emb, times], dim=-1)  # (B, in_dim)

        # MLP to predict mean
        mu = self.mlp(feats).view(B, 1, 4)
        
        return mu
