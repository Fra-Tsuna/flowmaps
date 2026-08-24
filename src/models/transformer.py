# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# CDiT (Conditional DiT), based on DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp
from typing import Optional

from src.models.vae import VAE
from src.models.embeddings import LabelEmbedding, TimestepEmbedder
from src.models.bb3d_embedding import BB3DEmbedding

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

##################################################################
#                  Transformer Encoder                           #
##################################################################

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size: int, depth: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        ff_dim = int(hidden_size * mlp_ratio)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            norm_first=True,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Transformer encoder.
          - x: Input tensor of shape (B, S, D) where B is batch size, S is sequence length, and D is hidden size.
          - mask: Optional attention mask of shape (S, S) for self-attention.
          - src_key_padding_mask: Optional key padding mask of shape (B, S) where True indicates positions to be ignored.
        """
        return self.encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask)


##################################################################
#                  Core CDiT Model                                #
##################################################################

class CDiTBlock(nn.Module):
    """
    A CDiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        # self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cttn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, add_bias_kv=True, bias=True, batch_first=True, **block_kwargs)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 8 * hidden_size, bias=True)
        )
        
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, x_cond, c, key_padding_mask: Optional[torch.Tensor] = None):
        '''
        Forward pass of CDiTBlock.
        Extended to support cross attention.
        - x: query tokens [B, 1, D]
        - x_cond: key and value tokens (e.g. map tokens) [B, S, D]
        - c: conditioning vector [B, 1, D]
        - key_padding_mask: optional mask for the key tokens [B, S]
        https://arxiv.org/pdf/2412.03572 Fig. 2
        '''
        c = c.squeeze(1)  # [B, D]
        shift_ca_xcond, scale_ca_xcond, shift_ca_x, scale_ca_x, gate_ca_x, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(8, dim=-1)
        # README: This part makes sense ONLY if we difuse the whole scene at once.
        # x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x_cond_norm = modulate(self.norm_cond(x_cond), shift_ca_xcond, scale_ca_xcond)
        query = modulate(self.norm2(x), shift_ca_x, scale_ca_x)
        x = x + gate_ca_x.unsqueeze(1) * self.cttn(query=query, 
                                                   key=x_cond_norm, 
                                                   value=x_cond_norm, 
                                                   key_padding_mask=key_padding_mask,
                                                   need_weights=False)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of CDiT.
    """
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        c = c.squeeze(1)  # [B, D]
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    

class CDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        vae_path: str,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        max_transitions=10,
        furn_classes=3,
        obj_classes=3,
        dropout=0.0,
        encode_yaw=False,
    ):
        super().__init__()
        self.model_args = {k: v for k, v in locals().items() if k != "self"}

        # See 3.2 https://arxiv.org/pdf/2412.03572
        # Maybe we can use a similar sin-cos emb for the bbox? and even color?
        #############
        ## Encoder ##
        #############
        self.encode_yaw = encode_yaw
        self.bbox_embedder = BB3DEmbedding(hidden_dim=hidden_size, encode_yaw=encode_yaw)
        self.obj_c_embedder = LabelEmbedding(hidden_dim=hidden_size, n_classes=obj_classes) 
        self.furn_c_embedder = LabelEmbedding(hidden_dim=hidden_size, n_classes=furn_classes) 
        self.type_embedder = nn.Parameter(torch.zeros(2, hidden_size), requires_grad=True) 
        
        self.map_encoder = TransformerEncoder(
            hidden_size = hidden_size,
            depth = depth, 
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            dropout = dropout
        )

        #########
        ## VAE ##
        #########
        self.vae = VAE.from_pretrained(vae_path)  # Load the VAE model
        self.vae.freeze_model()
        self.vae.eval()

        self.latent_dim = self.vae.latent_dim
        self.hidden_size = hidden_size
        
        #############
        ##   CDiT   ##
        #############
        self.input_embedder = nn.Sequential( # Embedding for the input (noise) 
            nn.Linear(self.latent_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.t_embedder = TimestepEmbedder(hidden_size)  # Embedding for the diffusion timesteps
        self.tau_embedder = nn.Embedding(max_transitions, hidden_size) # Embedding for the number of transitions
        self.tau0_embedder = nn.Embedding(max_transitions, hidden_size) # Embedding for the number of transitions
        self.query_embedder = LabelEmbedding(hidden_dim=hidden_size, n_classes=obj_classes) # Embedding for the query object

        # CDiT
        self.blocks = nn.ModuleList([
            CDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(hidden_size, out_size=self.latent_dim)
        self.initialize_weights()

    def initialize_weights(self):
        
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # self.apply(_basic_init)
        for name, module in self.named_children():
            if name == "vae":
                continue  # skip VAE
            module.apply(_basic_init)

        # Initialize bbox embedding layers:
        self.bbox_embedder.apply(self.bbox_embedder._init_weights)

        # Initialize furniture color embedding layers:
        nn.init.normal_(self.furn_c_embedder.emb.weight, std=0.02)
        nn.init.normal_(self.furn_c_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.furn_c_embedder.mlp[2].weight, std=0.02)

        # Initialize object color embedding layers:
        nn.init.normal_(self.obj_c_embedder.emb.weight, std=0.02)
        nn.init.normal_(self.obj_c_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.obj_c_embedder.mlp[2].weight, std=0.02)

        # Initialize query embedding layers:
        nn.init.normal_(self.query_embedder.emb.weight, std=0.02)
        nn.init.normal_(self.query_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.query_embedder.mlp[2].weight, std=0.02)

        # Initialize input embedding layers:
        nn.init.normal_(self.input_embedder[0].weight, std=0.02)
        nn.init.normal_(self.input_embedder[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize tau embedding table:
        nn.init.normal_(self.tau_embedder.weight, std=0.02)

        # Initialize tau0 embedding table:
        nn.init.normal_(self.tau0_embedder.weight, std=0.02)

        # Zero-out adaLN modulation layers in CDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()  # VAE is always frozen in eval mode
        return self

    def trainable_parameters(self):
        """Exclude vae parameters"""
        return (
            p for n, p in self.named_parameters()
            if p.requires_grad and not n.startswith("vae.")
        )

    def encode_query(self, query: torch.Tensor):
        with torch.no_grad():
            mu, logvar = self.vae.encode(query)
            latents = VAE.reparameterize(mu, logvar)
        return mu, logvar, latents

    def embed_map(self, map_tokens, types, tau0):
        """
        - map_tokens: (B, S, 8) tensor of map tokens
        """
        B, S, L = map_tokens.shape
        assert tau0.ndim == 1, f"tau0 must be 1-D (B,), got {tuple(tau0.shape)}"
        assert tau0.numel() == B, f"tau0 batch size mismatch: got {tau0.numel()} for B={B}"
        assert tau0.min() >= 0 and tau0.max() < self.tau0_embedder.num_embeddings, (
            f"tau0 out of range for max_transitions={self.tau0_embedder.num_embeddings}"
        )
        
        # Embed bbox
        if self.encode_yaw:
            bb_tokens = map_tokens[:, :, :7].view(B * S, -1)  # (B * S, 7)
        else:
            bb_tokens = map_tokens[:, :, :6].view(B * S, -1)  # (B * S, 6)
        bb_chunk_emb = self.bbox_embedder(bb_tokens).view(B, S, -1)  # (B, S, D)

        # Furniture and object masks
        map_flat = map_tokens[:, :, -1].view(B * S, 1).long()    # [B*S, 1]
        assert map_flat.dtype in (torch.int16, torch.int32, torch.int64), (
            f"map token labels must be integer dtype, got {map_flat.dtype}"
        )
        assert map_flat.min() >= 0 and map_flat.max() < max(
            self.furn_c_embedder.n_classes, self.obj_c_embedder.n_classes
        ), (
            "map token labels out of range for class embedders"
        )
        furn_mask = (types == 0).view(-1)  # [B*S]
        obj_mask = (types == 1).view(-1)  # [B*S]

        # Embed cls
        class_chunk_emb = torch.zeros(B * S, 1, self.hidden_size, device=map_tokens.device, dtype=map_tokens.dtype)
        class_chunk_emb[furn_mask] = self.furn_c_embedder(map_flat[furn_mask])
        class_chunk_emb[obj_mask] = self.obj_c_embedder(map_flat[obj_mask])
        class_chunk_emb = class_chunk_emb.view(B, S, self.hidden_size)  # [B, S, D]

        # README: Consider adding tau via FiLM too?
        tau0 = self.tau0_embedder(tau0).unsqueeze(1)  # [B, 1, D]

        type_emb = torch.zeros_like(bb_chunk_emb)
        
        mask0 = types == 0  # Furniture
        mask1 = types == 1  # Object
        type_emb[mask0] = self.type_embedder[0]  # Furniture
        type_emb[mask1] = self.type_embedder[1]  # Object
        
        map_emb = bb_chunk_emb + class_chunk_emb + type_emb + tau0  # (B, S, D)
        return map_emb  # (B, S, D)
    
    def forward_encoder(self, map, types, tau0, key_padding_mask: Optional[torch.Tensor] = None):
        map_emb = self.embed_map(map, types, tau0)  # [B, S, D]

        return self.map_encoder(
            map_emb, 
            src_key_padding_mask=key_padding_mask
        )  # [B, S, D]

    def forward(self, x, t, obj, map, tau0, tau, types, key_padding_mask: Optional[torch.Tensor] = None):
        """
        Forward pass of CDiT.
        - x:   (B, 1, D) x_t
        - map: (B, S, D) tensor of map tokens
        - obj: (B, 1, D) x_1
        - t:   (B,) tensor of diffusion timesteps
        - tau: (B,) tensor of number of transitions
        """
        if t.ndim < 1:
            t = t.expand(x.shape[0]) # [B]
        assert t.ndim == 1, f"t must be 1-D (B,), got {tuple(t.shape)}"
        assert tau.ndim == 1, f"tau must be 1-D (B,), got {tuple(tau.shape)}"
        assert t.numel() == x.shape[0], f"t batch size mismatch: got {t.numel()} for B={x.shape[0]}"
        assert tau.numel() == x.shape[0], f"tau batch size mismatch: got {tau.numel()} for B={x.shape[0]}"
        assert tau.min() >= 0 and tau.max() < self.tau_embedder.num_embeddings, (
            f"tau out of range for max_transitions={self.tau_embedder.num_embeddings}"
        )

        map = self.forward_encoder(map, types, tau0, key_padding_mask)  # [B, S, D]
        
        # README: For the query, let's maybe consider using a different embedding layer (either way in the future this should be a text embedding)
        # See Fig. 3 here https://arxiv.org/pdf/2112.10752        
        obj = obj.long()
        assert obj.dtype in (torch.int16, torch.int32, torch.int64), (
            f"obj labels must be integer dtype, got {obj.dtype}"
        )
        assert obj.min() >= 0 and obj.max() < self.query_embedder.n_classes, (
            "obj labels out of range for query embedder"
        )
        obj = self.query_embedder(obj)  # [B, 1, D]
        
        t = self.t_embedder(t).unsqueeze(1) # [B, 1, D]        
        tau = self.tau_embedder(tau).unsqueeze(1) # [B, 1, D]

        # README: this could potentially be other encoder
        x = self.input_embedder(x)
        
        c = t + tau + obj # [B, 1, D]
        
        for block in self.blocks:
            x = block(x=x, x_cond=map, c=c, key_padding_mask=key_padding_mask)       # [B, S, D]
        x = self.final_layer(x, c) 

        return x # [B, 1, out_dim] 
