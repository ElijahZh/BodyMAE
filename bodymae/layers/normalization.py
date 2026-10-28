import torch
import torch.nn as nn


class BatchNormal1d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__()
        self.normal = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype)

    def forward(self, x):
        return self.normal(x)


class BatchNormal2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__()
        self.normal = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype)

    def forward(self, x):
        return self.normal(x)


class PointBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__()
        self.normal = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine,
                                     track_running_stats=track_running_stats, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            return self.normal(x.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        elif x.dim() == 2:
            return self.normal(x)
        else:
            raise NotImplementedError(f"Dimension size `{x.dim()}` is not supported.")


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None):
        super().__init__()
        self.normal = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias, device=device, dtype=dtype)

    def forward(self, x):
        return self.normal(x)
