import torch.nn as nn


class ReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return self.relu(x)


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.GELU = nn.GELU()

    def forward(self, x):
        return self.GELU(x)


class Tanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(x)


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)

    def forward(self, x):
        return self.leaky_relu(x)


class SoftMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.soft_max = nn.Softmax(dim=dim)

    def forward(self, x):
        return self.soft_max(x)
