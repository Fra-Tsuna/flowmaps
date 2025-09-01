
import torch
from torch import nn

class CFGVectorField(nn.Module):
    def __init__(self, net: nn.Module, guidance_scale: float = 1.0, null_label: int = 4):
        super(CFGVectorField, self).__init__()
        self.net = net
        self.w = guidance_scale
        self.null_label = null_label

    # FIXME: QUICK FIX TO DO MINIMAL CHANGES TO THE CODE
    # FOR THE EVAL PART, TO ADJUST IT LATER
    # def forward(self, x: Tensor, t: Tensor, label: Tensor, landmark: Tensor, map: Tensor, source_sample: Tensor) -> torch.Tensor:
    #     """
    #     Args:
    #     - x: (bs, c, h, w)
    #     - t: (bs, 1, 1, 1)
    #     - y: (bs,)
    #     """
    #     guided_vector_field = self.net(x, t, label, landmark, map, source_sample)
    #     if self.w == 1:
    #         return guided_vector_field
    #     unguided_y = torch.ones_like(label) * self.null_label
    #     unguided_vector_field = self.net(x, t, unguided_y, landmark, map, source_sample)
    #     return (1 - self.w) * unguided_vector_field + self.w * guided_vector_field
    
    def forward(self, x, t, obj, map, tau0, tau, types, key_padding_mask) -> torch.Tensor:
        return self.net(x, t, obj, map, tau0, tau, types, key_padding_mask)