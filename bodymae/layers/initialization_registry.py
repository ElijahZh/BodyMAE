import torch

from inspect import getmembers, isfunction
from collections.abc import Callable
from functools import partial

from bodymae.layers import initializers as init_function
from bodymae.utils.string_registry import StringInterface


class InitFunction(StringInterface):
    def __init__(self):
        self._funcs_dict = dict(getmembers(init_function, isfunction))
        super().__init__(string_list=list(self._funcs_dict.keys()))


    @torch.no_grad()
    def __call__(self, method: str, *args, **kwargs) -> Callable:
        init_key = self.get_key(method, "Initialization function")
        func = self._funcs_dict[init_key]
        inits_func = partial(func, *args, **kwargs)
        return inits_func
