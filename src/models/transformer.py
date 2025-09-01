# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------
import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Attention, Mlp
from einops import rearrange
from diffusers import AutoencoderKL
from typing import Optional

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



################################################################
#               Embedding Layers for Timesteps                 #
################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                            These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    
################################################################
#               Embedding Layers for Bounding Boxes            #
################################################################

class BBoxEmbedding(nn.Module):
    """
    Sinusoidal embedding for bounding boxes normalized in [0, 1].
    """
    def __init__(self, hidden_dim, frequency_embedding_size=256//4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.frequency_embedding_size = frequency_embedding_size  # There are four coordinates in a bbox
        self.input_dim = 4 * self.frequency_embedding_size

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def timestep_embedding(pos, dim, max_period=10000):
        """
        Apply sinusoidal embedding to a scalar position (float).
        pos: tensor of shape (...), values in [0, 1]
        returns: (..., dim)
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        ).to(device=pos.device)
        angles = rearrange(pos, 'b s -> b s 1') * rearrange(freqs, 'd -> 1 1 d')  # (..., half)
        emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)  # (..., dim)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb

    def forward(self, bboxes):
        """
        bboxes: (B, S, 4) in normalized [0,1] coordinates: [x0, y0, w, h]
        """
        x0, y0, x1, y1 = bboxes.unbind(dim=-1)  # Each of shape (B, S)

        emb_x0 = self.timestep_embedding(x0, self.frequency_embedding_size)
        emb_y0 = self.timestep_embedding(y0, self.frequency_embedding_size)
        emb_x1 = self.timestep_embedding(x1, self.frequency_embedding_size)
        emb_y1 = self.timestep_embedding(y1, self.frequency_embedding_size)

        emb = torch.cat([emb_x0, emb_y0, emb_x1, emb_y1], dim=-1)  # (B, S, frequency_embedding_size)

        return self.mlp(emb)  # Project to hidden_dim

    
########################################################
#               Embedding Layers for Colors            #
########################################################

class ColorEmbedding(nn.Module):
    def __init__(self, hidden_dim, n_classes: int = 6):
        super().__init__()
        
        self.n_classes = n_classes
        self.emb = nn.Embedding(self.n_classes, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

    def forward(self, x):
        x = self.emb(x.long())  # shape (batch, seq_len, hidden_dim)
        x = self.mlp(x)
        return x  # shape (batch, seq_len, hidden_dim)

###################################################################
#                   Transformer Encoder                           #
###################################################################

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
        # `mask` here should be a square BoolTensor of shape (S, S)
        # 'src_key_padding_mask` is a BoolTensor of shape (B, S) where S is the sequence length.
        # with True where you want to block attention.
        return self.encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask)
    
class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size: int, depth: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        ff_dim = int(hidden_size * mlp_ratio)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            norm_first=True,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)

    def forward(self, target: torch.Tensor, memory: torch.Tensor,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        return self.decoder(
            target, # Target sequence (B, T, D)
            memory, # Encoder output for cross-attention (B, T_enc, D)
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )    

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

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
    Conditional Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=4,
        output_size=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        max_transitions=10,
        dropout=0.0,
    ):
        super().__init__()
        self.model_args = {k: v for k, v in locals().items() if k != "self"}

        # See 3.2 https://arxiv.org/pdf/2412.03572

        #############
        ## Encoder ##
        #############
        self.bbox_embedder   = BBoxEmbedding(hidden_dim=hidden_size, frequency_embedding_size=256//4)
        self.colors_embedder = ColorEmbedding(hidden_dim=hidden_size, n_classes=6)  # 6 colors: 3 for furniture and 3 for objects
        self.type_embedder = nn.Parameter(torch.zeros(2, hidden_size), requires_grad=True) 
        
        self.map_encoder = TransformerEncoder(
            hidden_size = hidden_size,
            depth = depth, 
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            dropout = dropout
        )

        #############
        ##   DiT   ##
        #############
        self.input_embedder = nn.Sequential( # Embedding for the input (noise) 
            nn.Linear(input_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.t_embedder = TimestepEmbedder(hidden_size)  # Embedding for the diffusion timesteps
        self.tau_embedder = nn.Embedding(max_transitions, hidden_size) # Embedding for the number of transitions
        self.tau0_embedder = nn.Embedding(max_transitions, hidden_size) # Embedding for the number of transitions
        self.query_embedder = ColorEmbedding(hidden_dim=hidden_size, n_classes=3) # Embedding for the query object (e.g. color of the object)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)  # Positional encoding for the input sequence
        
        # DiT
        self.blocks = nn.ModuleList([
            CDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(hidden_size, out_size=output_size)
        self.initialize_weights()

    def initialize_weights(self):
        
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize bbox embedding layers:
        nn.init.normal_(self.bbox_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.bbox_embedder.mlp[2].weight, std=0.02)

        # Initialize color embedding layers:
        nn.init.normal_(self.colors_embedder.emb.weight, std=0.02)
        nn.init.normal_(self.colors_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.colors_embedder.mlp[2].weight, std=0.02)

        # Initialize query embedding layers:
        nn.init.normal_(self.query_embedder.emb.weight, std=0.02)
        nn.init.normal_(self.query_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.query_embedder.mlp[2].weight, std=0.02)

        # Initialize input embedding layers:
        nn.init.normal_(self.input_embedder[0].weight, std=0.02)
        nn.init.normal_(self.input_embedder[2].weight, std=0.02)

        # Initialize positional encoding:
        nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize type embedding layer:
        nn.init.normal_(self.type_embedder, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize tau embedding table:
        nn.init.normal_(self.tau_embedder.weight, std=0.02)

        # Initialize tau0 embedding table:
        nn.init.normal_(self.tau0_embedder.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @property
    def get_bbox_embedder(self):
        return self.bbox_embedder
    
    @property
    def get_colors_embedder(self):
        return self.colors_embedder
    
    @property
    def get_type_embedder(self):
        return self.type_embedder
    
    @property
    def get_map_encoder(self):
        return self.map_encoder
    
    def _embed_map(self, map_tokens, types, tau0):
        """
        - map_tokens: (B, S, 5) tensor of map tokens
        """
        bb_chunk_emb = self.bbox_embedder(map_tokens[:, :, :4])  # (B, S, D)
        color_chunk_emb = self.colors_embedder(map_tokens[:, :, 4])  # (B, S, D)
        tau0 = self.tau0_embedder(tau0).unsqueeze(1)  # [B, 1, D]

        type_emb = torch.zeros_like(bb_chunk_emb)
        
        mask0 = types == 0  # Furniture
        mask1 = types == 1  # Object
        type_emb[mask0] = self.type_embedder[0]  # Furniture
        type_emb[mask1] = self.type_embedder[1]  # Object
        
        map_emb = bb_chunk_emb + color_chunk_emb + type_emb + tau0  # (B, S, D)
        return map_emb  # (B, S, D)

    def forward_encoder(self, map, types, tau0, key_padding_mask: Optional[torch.Tensor] = None):
        map_emb = self._embed_map(map, types, tau0)  # [B, S, D]
        
        return self.map_encoder(
            map_emb, 
            src_key_padding_mask=key_padding_mask
        ) # [B, S, D]

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
        
        # README: the encoder does not need pos encoding, on theory it already has it from the bbox embedding
        map = self.forward_encoder(map, types, tau0, key_padding_mask)  # [B, S, D]
        
        obj = self.query_embedder(obj)  # [B, 1, D]
        
        t = self.t_embedder(t).unsqueeze(1) # [B, 1, D]
        tau = self.tau_embedder(tau).unsqueeze(1) # [B, 1, D]

        x = self.input_embedder(x) + self.pos_embed # [B, 1, D]
        
        c = t + tau + obj # [B, 1, D]
        
        for block in self.blocks:
            x = block(x=x, x_cond=map, c=c, key_padding_mask=key_padding_mask)       # [B, S, D]
        x = self.final_layer(x, c) 

        return x # [B, 1, out_dim] 