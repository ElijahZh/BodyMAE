import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Any


class GeneralRun:
    @classmethod
    def run(cls, func, value, *args, **kwargs):
        if isinstance(value, Tensor):
            func(value, *args, **kwargs)
        else:
            if hasattr(value, "parameters"):
                for v in value.parameters():
                    cls.run(func, v, *args, **kwargs)


def uniform(value: Any, low: float=0, high: float=1):
    GeneralRun.run(nn.init.uniform_, value, low, high)


def default(value: Any, size: float = None):
    def _uniform(_value, _size):
        _size = _value.size(1) if _size is None else _size
        stdv = 1. / math.sqrt(_size)
        nn.init.uniform_(_value, -stdv, stdv)
    GeneralRun.run(_uniform, value, size)


def constant(value: Any, fill_value: float):
    GeneralRun.run(nn.init.constant_, value, fill_value)


def normal(value: Any, mean: float = 0., std: float = 1.0):
    GeneralRun.run(nn.init.normal_, value, mean, std)


def kaiming_uniform(value: Any, a: float = 0.):
    GeneralRun.run(nn.init.kaiming_uniform_, value, a)


def kaiming_normal(value: Any, a: float = 0.):
    GeneralRun.run(nn.init.kaiming_normal_, value, a)


def xavier_uniform(value: Any, gain: float = 1.0):
    GeneralRun.run(nn.init.xavier_uniform_, value, gain)


def xavier_normal(value: Any, gain: float = 1.0):
    GeneralRun.run(nn.init.xavier_normal_, value, gain)


def zeros(value: Any):
    constant(value, fill_value=0.)


def ones(value: Any):
    constant(value, fill_value=1.0)


def reset(value: Any):
    if hasattr(value, "reset_parameters"):
        value.reset_parameters()
    else:
        if hasattr(value, "children"):
            for child in value.children():
                reset(child)
