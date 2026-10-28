import torch
import torch.nn as nn
from torch import Tensor, jit
from collections.abc import Callable, Sequence
from typing import Type
from collections import defaultdict
import io
from timm.models.layers import DropPath, trunc_normal_
import inspect

from bodymae.layers.initialization_registry import InitFunction
from bodymae.layers.activation_registry import ActiveClass
from bodymae.layers.normalization_registry import NormalClass
from functools import partial


class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.string_to_init_function = InitFunction()


        self.weight_init_method = {"method": "kaiming_normal"}
        self.bias_init_method = {"method": "zeros"}
        self.weight_init_func = self.string_to_init_function(**self.weight_init_method)
        self.bias_init_func = self.string_to_init_function(**self.bias_init_method)

    def initialization(self,
                       weight_method: dict | None | Callable = None,
                       bias_method: dict | None | Callable = None,
                       ):
        if isinstance(weight_method, Callable):
            self.weight_init_method = {"method": "Customized"}
            self.weight_init_func = weight_method
        else:
            if weight_method is not None:
                assert isinstance(weight_method, dict) and "method" in weight_method
                self.weight_init_method = weight_method
                self.weight_init_func = self.string_to_init_function(**self.weight_init_method)

        if isinstance(bias_method, Callable):
            self.bias_init_method = {"method": "Customized"}
            self.bias_init_func = bias_method
        else:
            if bias_method is not None:
                assert isinstance(bias_method, dict) and "method" in bias_method
                self.bias_init_method = bias_method
                self.bias_init_func = self.string_to_init_function(**self.bias_init_method)

        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                self.weight_init_func(layer.weight)
                if layer.bias is not None:
                    self.bias_init_func(layer.bias)

    def _init_weight(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0., std=.02, a=-2., b=2.)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, mean=0., std=.02, a=-2., b=2.)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_init_information(self):
        return {"weight": self.weight_init_method, "bias": self.bias_init_method}

    def get_parameters_information(self):
        dict = {}
        all_parameters = []
        trainable_parameters = []
        for p in self.parameters():
            size = p.numel()
            if p.requires_grad:
                trainable_parameters.append(size)
            all_parameters.append(size)

        dict["total_parameter_size"] = sum(all_parameters)
        dict["trainable_parameter_size"] = sum(trainable_parameters)
        return dict

    def get_model_size(self):
        param_size = 0
        buffer_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb

    def get_layer_categories(self):
        res = defaultdict(int)
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                res["nn.Conv1d"] += 1
            elif isinstance(layer, nn.Conv2d):
                res["nn.Conv2d"] += 1
            elif isinstance(layer, nn.Conv3d):
                res["nn.Conv3d"] += 1
            elif isinstance(layer, nn.Linear):
                res["nn.Linear"] += 1
        return f"Layer Number information:\n{res}"

    @staticmethod
    def load_model_jit(file_name: str, device: str = "cpu") -> nn.Module:
        with open(file_name, 'rb') as f:
            buffer = io.BytesIO(f.read())
        buffer.seek(0)
        model = jit.load(buffer, map_location=torch.device(device))
        return model

    def save_model_jit(self, file_name: str) -> None:
        complied_model = torch.jit.script(self)
        torch.jit.save(complied_model, file_name)


class MLPNet(BaseModule):
    def __init__(self,
                 mlp_list: Sequence[int],
                 normalization: bool | Sequence[bool] | None = True,
                 normalization_type: str | nn.Module | Type[nn.Module] = "BatchNormal1d",
                 normal_param: dict | None = None,
                 active: bool | Sequence[bool] | None = True,
                 active_type: str | nn.Module | Type[nn.Module] = "relu",
                 bias: bool | Sequence[bool] = True,
                 dropout_path: int | float | Sequence[float] = None,
                 dropout: int | float | Sequence[float] = None,
                 dropout_last: bool = False,
                 plain_last: bool = False,
                 return_layer: bool = False):

        super().__init__()
        self.plain_last = plain_last
        self.return_layer = return_layer

        self.string_to_active_class = ActiveClass()
        self.string_to_normal_class = NormalClass()

        layer_size = len(mlp_list) - 1
        valid_size = layer_size - 1 if plain_last else layer_size

        assert isinstance(mlp_list, Sequence)
        assert len(mlp_list) > 1
        assert all(isinstance(x, int) for x in mlp_list)

        self.mlp_bns = nn.ModuleList()
        if normalization is None:
            normalization = False
        if isinstance(normalization, bool):
            normalization = [normalization] * valid_size
        if normal_param is None:
            normal_param = {}
        assert len(normalization) == valid_size, f"Size of normalization doesn't match the valid linear layers size"
        assert all(isinstance(x, bool) for x in normalization), f"Parameter `normalization` should be bool or Sequence[bool]"
        iterator_mlp = mlp_list[1: -1] if plain_last else mlp_list[1:]
        iterator = zip(iterator_mlp, normalization)
        for channel, norm in iterator:
            if norm:
                if isinstance(normalization_type, nn.Module):
                    self.mlp_bns.append(normalization_type)
                elif self.is_module_class(normalization_type):
                    self.mlp_bns.append(normalization_type(channel))
                elif isinstance(normalization_type, str):
                    self.mlp_bns.append(self.string_to_normal_class(normalization_type, channel, **normal_param))
                else:
                    raise ValueError(
                        f"Normalization type `{type(normalization_type)}` is not supported in parameter `normalization_type`.")
            else:
                self.mlp_bns.append(nn.Identity())


        self.mlp_actives = nn.ModuleList()
        if active is None:
            active = False
        if isinstance(active, str):
            active = True
        if isinstance(active, bool):
            active = [active] * valid_size
        assert len(active) == valid_size, f"Size of active doesn't match the valid linear layers size"
        assert all(isinstance(x, bool) for x in active), f"Parameter `active` should be bool or Sequence[bool]"
        for act in active:
            if act:
                if isinstance(active_type, str):
                    self.mlp_actives.append(self.string_to_active_class(active_type))
                elif isinstance(active_type, nn.Module):
                    self.mlp_actives.append(active_type)
                elif self.is_module_class(active_type):
                    self.mlp_actives.append(active_type())
                else:
                    raise ValueError(f"Active type `{type(active_type)}` is not supported in parameter `active_type`.")
            else:
                self.mlp_actives.append(nn.Identity())



        self.list_bias = []
        if isinstance(bias, bool):
            self.list_bias = [bias] * layer_size
        elif isinstance(bias, Sequence):
            assert len(bias) == layer_size
            assert all(isinstance(x, bool) for x in bias)
            self.list_bias = bias


        self.mlp_lins = nn.ModuleList()
        iterator = zip(mlp_list[:-1], mlp_list[1:], self.list_bias)
        for in_channel, out_channel, _bias in iterator:
            self.mlp_lins.append(nn.Linear(in_channel, out_channel, bias=_bias))

        self.mlp_dropout = None
        self.mlp_dropout = [nn.Identity()] * valid_size

        if dropout_path is not None:
            if isinstance(dropout_path, float) or isinstance(dropout_path, int):
                if dropout_last:
                    dropout_path_list = [0.] * valid_size
                    dropout_path_list[-1] = dropout_path
                else:
                    dropout_path_list = [float(dropout_path)] * valid_size
            elif isinstance(dropout_path, Sequence):
                assert len(dropout_path) == valid_size
                assert all(isinstance(x, float) for x in dropout_path)
                dropout_path_list = dropout_path
            else:
                raise ValueError(f"type `{dropout_path}` is not supported in parameter `dropout_path`.")
            self.mlp_dropout = nn.ModuleList()
            for val in dropout_path_list:
                self.mlp_dropout.append(DropPath(val))
        elif dropout is not None:

            if isinstance(dropout, float) or isinstance(dropout, int):
                if dropout_last:
                    dropout_list = [0.] * valid_size
                    dropout_list[-1] = dropout
                else:
                    dropout_list = [float(dropout)] * valid_size
            elif isinstance(dropout, Sequence):
                assert len(dropout) == valid_size
                assert all(isinstance(x, float) for x in dropout)
                dropout_list = dropout
            else:
                raise ValueError(f"type `{dropout}` is not supported in parameter `dropout`.")

            self.mlp_dropout = nn.ModuleList()
            for val in dropout_list:
                self.mlp_dropout.append(nn.Dropout(val))


        assert len(self.mlp_bns) == len(self.mlp_dropout) == len(self.mlp_actives) == valid_size
        add_one = 1 if plain_last else 0
        assert len(self.mlp_lins) == len(self.mlp_bns) + add_one

        self.apply(self._init_weight)

    @staticmethod
    def is_module_class(x):

        if isinstance(x, partial):
            x = x.func
        return inspect.isclass(x) and issubclass(x, nn.Module)

    def forward(self, x: Tensor):
        each_layer_res = []
        if self.plain_last:
            for i, (lin, norm, active, dp) in enumerate(zip(self.mlp_lins[:-1], self.mlp_bns, self.mlp_actives, self.mlp_dropout)):
                x = active(norm(lin(x)))
                x = dp(x)
                each_layer_res.append(x)
            x = self.mlp_lins[-1](x)
            each_layer_res.append(x)
        else:
            for i, (lin, norm, active, dp) in enumerate(zip(self.mlp_lins, self.mlp_bns, self.mlp_actives, self.mlp_dropout)):
                x = active(norm(lin(x)))
                x = dp(x)
                each_layer_res.append(x)

        if self.return_layer:
            return x, each_layer_res
        return x
