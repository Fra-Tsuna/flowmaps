from torch import nn


class ModelWrapper(nn.Module):
    """Wraps a model to satisfy the torchdiffeq ODE solver interface."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x, t, obj, map, tau0, tau, types, key_padding_mask):
        return self.net(x, t, obj, map, tau0, tau, types, key_padding_mask)
