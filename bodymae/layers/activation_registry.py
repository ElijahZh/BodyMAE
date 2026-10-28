import sys
import inspect

from bodymae.utils.string_registry import StringInterface
from bodymae.layers import activations as active_class


class ActiveClass(StringInterface):
    def __init__(self):
        self.active_class_list = inspect.getmembers(sys.modules[active_class.__name__], lambda m: inspect.isclass(m) and m.__module__ == active_class.__name__)
        self.active_dict = dict(self.active_class_list)
        super().__init__(list(self.active_dict.keys()))

    def __call__(self, name: str, *args, **kwargs):
        active_key = self.get_key(name, "Active Class")
        active = self.active_dict[active_key](*args, **kwargs)
        return active
