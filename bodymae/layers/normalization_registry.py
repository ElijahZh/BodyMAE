import sys
import inspect

from bodymae.utils.string_registry import StringInterface
from bodymae.layers import normalization as normal_class


class NormalClass(StringInterface):
    def __init__(self):
        self.normal_class_list = inspect.getmembers(sys.modules[normal_class.__name__], lambda m: inspect.isclass(m) and m.__module__ == normal_class.__name__)
        self.normal_dict = dict(self.normal_class_list)
        super().__init__(list(self.normal_dict.keys()))

    def __call__(self, name: str, *args, **kwargs):
        normal_key = self.get_key(name, "Normal Class")
        normal = self.normal_dict[normal_key](*args, **kwargs)
        return normal
