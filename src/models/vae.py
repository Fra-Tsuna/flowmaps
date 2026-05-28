from typing import Dict
import math

import torch
import torch.nn as nn
from torch.functional import F

_SIZE_LOG_EPS = 1e-4
_LOG_SIZE_MIN = math.log(_SIZE_LOG_EPS)
_LOG_SIZE_RANGE = math.log(1.0 + _SIZE_LOG_EPS) - _LOG_SIZE_MIN

from src.models.embeddings import LabelEmbedding
from src.models.bb3d_embedding import BB3DEmbedding

from src.utils.losses import complete_box_iou_loss_3d

def to_corners_from_center_and_size(bbox):
    """
    Convert bounding boxes from center format (x, y, z, dx, dy, dz) to corner format (x1, y1, z1, x2, y2, z2).
    :param bbox: Tensor of shape (..., 6) in center format.
    :return: Tensor of shape (..., 6) in corner format.
    """
    x, y, z, dx, dy, dz = bbox.unbind(-1)
    x1 = x - 0.5 * dx
    y1 = y - 0.5 * dy
    z1 = z - 0.5 * dz
    y2 = y + 0.5 * dy
    x2 = x + 0.5 * dx
    z2 = z + 0.5 * dz
    return torch.stack((x1, y1, z1, x2, y2, z2), dim=-1)
    

class VAEEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        latent_dim: int,
        depth: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        self.input_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=True), nn.LayerNorm(hidden_size), nn.SiLU())
        mlp_hidden = int(hidden_size * mlp_ratio)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, mlp_hidden, bias=True),
                    nn.LayerNorm(mlp_hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_hidden, hidden_size, bias=True),
                    nn.LayerNorm(hidden_size),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(depth)
            ]
        )

        self.out_proj = nn.Linear(hidden_size, 2 * latent_dim, bias=True)  # Output for mu and logvar

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, input_dim]
        Returns:
            mu:     Tensor of shape [B, latent_dim]
            logvar: Tensor of shape [B, latent_dim]
        """
        h = self.input_proj(x)
        residual = h
        for block in self.blocks:
            h = block(h) + residual
            residual = h

        out = self.out_proj(h)
        mu, logvar = out.chunk(2, dim=-1)  # Split into mu and logvar

        return mu, logvar


class VAEDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        depth: int,
        n_classes: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        encode_yaw: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.encode_yaw = encode_yaw

        self.input_proj = nn.Sequential(nn.Linear(latent_dim, hidden_size, bias=True), nn.LayerNorm(hidden_size), nn.SiLU())

        mlp_hidden = int(hidden_size * mlp_ratio)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, mlp_hidden, bias=True),
                    nn.LayerNorm(mlp_hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_hidden, hidden_size, bias=True),
                    nn.LayerNorm(hidden_size),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(depth)
            ]
        )

        # Output heads for bounding box and classification
        self.bbox_head = nn.Linear(hidden_size, 6, bias=True)  # 6 for bounding box (x, y, z, dx, dy, dz)
        # self.angle_head = nn.Linear(hidden_size, 1, bias=True)  # 1 for yaw angle
        self.cls_head = nn.Linear(hidden_size, n_classes, bias=True)  # n_classes for classification
        self.bbox_activation = nn.Sigmoid()

    def forward(self, x):
        h = self.input_proj(x)
        residual = h
        for block in self.blocks:
            h = block(h) + residual
            residual = h

        bbox = self.bbox_activation(self.bbox_head(h))
        logits = self.cls_head(h)
        return bbox, logits


class VAE(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        depth: int,
        n_classes: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        encode_yaw: bool = False,
    ):
        super().__init__()
        self.model_args = {k: v for k, v in locals().items() if k != "self"}
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.encode_yaw = encode_yaw
        self.bbox_embedder = BB3DEmbedding(
            hidden_dim=hidden_size,
            encode_yaw=self.encode_yaw,
        )
        self.label_emb = LabelEmbedding(hidden_dim=hidden_size, n_classes=n_classes)
        self.encoder = VAEEncoder(hidden_size, latent_dim, depth, mlp_ratio, dropout)
        self.decoder = VAEDecoder(latent_dim, hidden_size, depth, n_classes, mlp_ratio, dropout, encode_yaw=self.encode_yaw)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        

    def trainable_parameters(self):
        """
        Return the trainable parameters of the VAE model.
        """
        return self.parameters()

    @classmethod
    def from_pretrained(cls, path: str):
        """
        Load weights from a pretrained VAE model.
        :param path: Path to the pretrained model checkpoint.
        """
        import inspect
        checkpoint = torch.load(path, weights_only=False)
        model_args = checkpoint['model_args']
        model_args.pop('__class__', None)  # Remove class reference if present
        # Filter to only args accepted by the current __init__, so that checkpoints saved
        # with older model versions (e.g. with use_log_size/size_log_eps) remain loadable.
        valid_keys = inspect.signature(cls.__init__).parameters.keys() - {'self'}
        model_args = {k: v for k, v in model_args.items() if k in valid_keys}
        model = cls(**model_args)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def freeze_model(self):
        """
        Freeze the model parameters to prevent further training.
        """
        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def reparameterize(mu, log_var) -> torch.Tensor:
        """
        Reparameterization method for variational autoencoder's.
        :param latent: dataclass for mu, logvar tensors.
        :return: combined latent tensor.
        """
        assert mu.shape == log_var.shape
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_loss(self, mu, logvar):
        """
        Compute the KL divergence loss for the VAE.
        :param mu: Mean of the latent distribution.
        :param logvar: Log variance of the latent distribution.
        :return: KL divergence loss (unscaled).
        """
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, D]
        return kl.sum(dim=-1).mean()


    def reconstruction_loss(self, bbox, cls_logits, target_bbox, target_cls, loss_weights):
        ###############
        ## CIoU LOSS ##
        ###############
        bbox_xyzxyz = to_corners_from_center_and_size(bbox)
        target_xyzxyz = to_corners_from_center_and_size(target_bbox)
        raw_s = (torch.exp(bbox[..., 3:6] * _LOG_SIZE_RANGE + _LOG_SIZE_MIN) - _SIZE_LOG_EPS).clamp(0., 1.)
        raw_t = (torch.exp(target_bbox[..., 3:6] * _LOG_SIZE_RANGE + _LOG_SIZE_MIN) - _SIZE_LOG_EPS).clamp(0., 1.)
        bbox_xyzxyz = to_corners_from_center_and_size(torch.cat([bbox[..., :3], raw_s], dim=-1))
        target_xyzxyz = to_corners_from_center_and_size(torch.cat([target_bbox[..., :3], raw_t], dim=-1))
        ciou_components = complete_box_iou_loss_3d(bbox_xyzxyz, target_xyzxyz, reduction="mean")
        ciou = ciou_components["loss"]

        #######################################
        ## Smooth L1 loss for bounding boxes ##
        #######################################
        bbox_l1 = bbox
        target_l1 = target_bbox
        l1_per_dim = F.smooth_l1_loss(bbox_l1, target_l1, reduction="none")  # (B, 6)
        l1_center = l1_per_dim[:, :3].sum(dim=-1).mean()
        l1_size = l1_per_dim[:, 3:6].sum(dim=-1).mean()

        #############
        ## CE LOSS ##
        #############
        ce_loss = F.cross_entropy(cls_logits, target_cls, reduction="mean")  # [1]

        ciou_w = loss_weights["ciou"]
        l1_center_w = loss_weights["l1_center"]
        l1_size_w = loss_weights["l1_size"]
        ce_w = loss_weights["ce"]

        loss = ciou_w * ciou + l1_center_w * l1_center + l1_size_w * l1_size + ce_w * ce_loss
        return {
            "ciou": ciou,
            "ciou_iou": ciou_components["iou_loss"],
            "ciou_distance": ciou_components["distance_loss"],
            "ciou_aspect": ciou_components["aspect_loss"],
            "l1_center": l1_center,
            "l1_size": l1_size,
            "ce": ce_loss,
            "total_mse": loss,
        }

    def compute_loss(self, mu, logvar, bbox, cls_logits, target_bbox, target_cls, hyperparams: Dict[str, float]):
        kl = self.kl_loss(mu, logvar)
        reconstruction_dict = self.reconstruction_loss(
            bbox,
            cls_logits,
            target_bbox,
            target_cls,
            loss_weights=hyperparams["loss_weights"],
        )
        metrics = {"kl": kl, **reconstruction_dict}
        metrics["loss"] = reconstruction_dict["total_mse"] + hyperparams["beta"] * kl
        return metrics

    def embed_query(self, map_tokens):
        """
        - map_tokens: (B, D) tensor of map tokens
        """
        
        B, _ = map_tokens.shape

        bbox_tokens = map_tokens[:, :7] if self.encode_yaw else map_tokens[:, :6]
        bb_chunk_emb = self.bbox_embedder(bbox_tokens).view(B, self.hidden_size)  # (B, D)
            
        label_chunk_emb = self.label_emb(map_tokens[:, 7]).view(B, self.hidden_size) # (B, D)
        map_emb = bb_chunk_emb + label_chunk_emb
        return map_emb # (B, D)

    def encode(self, x):
        """
        Encode the input map tokens into latent space.
        :param x: Input map tokens of shape (B * S, D).
        :return: mu and logvar tensors of shape (B * S, D).
        """
        x = self.embed_query(x)
        
        mu, logvar = self.encoder(x)
        return mu, logvar

    def forward(self, x, sample=False):
        mu, logvar = self.encode(x) # [B*S, latent_dim], [B*S, latent_dim]
        z = self.reparameterize(mu, logvar) if sample else mu
        bbox, cls_logits = self.decode_latents(z)
        return mu, logvar, bbox, cls_logits  # Return bounding boxes and class logits

    def decode_latents(self, z: torch.Tensor):
        bbox, cls_logits = self.decoder(z) # bbox [..., 6], cls_logits [..., n_classes]
        return bbox, cls_logits


# if __name__ == "__main__":
#     import torch

# model = VAEEncoder(input_dim=5, hidden_size=512, latent_dim=128, depth=6)
# x = torch.randn(10, 5) + 5  # (B, S, D)
#     token_type = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1, -1]  # Random token types
#     token_type = torch.tensor(token_type, dtype=torch.int64)  # Convert to tensor
#     token_type = repeat(token_type, "S -> B S", B=10)  # Res
#     src_key_padding_mask = torch.zeros(10, 15, dtype=torch.bool)  # No padding
#     src_key_padding_mask[:, 10:] = True  # Example padding mask
# output = model(x)

# decoder = VAEDecoder(hidden_size=512, latent_dim=128, depth=6, num_heads=8, furn_classes=10, obj_classes=20)
# output = decoder(output, token_type, src_key_padding_mask=src_key_padding_mask)
# vae = VAE(latent_dim=128, hidden_size=512, depth=6, n_classes=10)
# bbox, cls = vae(x, sample=True)  # Sample from the VAE

#     print("Output shape:", bbox.shape)  # Should be (B, S, 4)
#     print("Log variance shape:", cls.shape)  # Should be (B, S, D)
