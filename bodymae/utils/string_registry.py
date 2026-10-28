import numpy as np
from difflib import SequenceMatcher
from itertools import repeat
from abc import ABC, abstractmethod
import collections
from typing import Sequence, Callable
import weakref


class StringChecker:
    _pool = weakref.WeakValueDictionary()

    def __new__(cls, method="sequencematcher"):

        assert method in ["levenshtein", "sequencematcher"]
        obj = cls._pool.get(method)
        if obj is None:
            obj = object.__new__(StringChecker)
            cls._pool[method] = obj
            obj.similarity_method = method
        return obj

    @staticmethod
    def process_string(parameter_string: str):
        processed_string = parameter_string.lower().strip()
        processed_string = processed_string.replace("-", "_")
        return processed_string

    @staticmethod
    def process_string_keys(string_dict: dict):
        res = {}
        for key, val in string_dict.items():
            res[StringChecker.process_string(key)] = val
        return res

    @staticmethod
    def _sequence_matcher(token1, token2):
        res = SequenceMatcher(None, token1, token2)
        return res.ratio()

    @staticmethod
    def _levenshtein_distance_dp(token1, token2):
        distances = np.zeros((len(token1) + 1, len(token2) + 1))
        for t1 in range(len(token1) + 1):
            distances[t1][0] = t1
        for t2 in range(len(token2) + 1):
            distances[0][t2] = t2

        for t1 in range(1, len(token1) + 1):
            for t2 in range(1, len(token2) + 1):
                if (token1[t1 - 1] == token2[t2 - 1]):
                    distances[t1][t2] = distances[t1 - 1][t2 - 1]
                else:
                    a = distances[t1][t2 - 1]
                    b = distances[t1 - 1][t2]
                    c = distances[t1 - 1][t2 - 1]

                    if a <= b and a <= c:
                        distances[t1][t2] = a + 1
                    elif b <= a and b <= c:
                        distances[t1][t2] = b + 1
                    else:
                        distances[t1][t2] = c + 1
            return distances[len(token1)][len(token2)]

    def get_closest_item_index(self, string, compared_list):
        recommendation_index = -1
        if self.similarity_method == "levenshtein":
            method = self._levenshtein_distance_dp
            operation_times = list(map(method, repeat(string), compared_list))
            if np.min(operation_times) <= len(string) // 3:
                recommendation_index = np.argmin(operation_times)
        elif self.similarity_method == "sequencematcher":
            method = self._sequence_matcher
            similarity_score = list(map(method, repeat(string), compared_list))
            if np.max(similarity_score) >= 0.8:
                recommendation_index = np.argmax(similarity_score)
        else:
            raise NotImplementedError(f"similarity method `{self.similarity_method}` is not supported")

        return recommendation_index

    def __repr__(self):
        return f"{type(self).__name__}(similarity method: {self.similarity_method})"


class StringInterface(ABC):
    def __init__(self,
                 string_list: Sequence[str] = None,
                 compare_method: str = "sequencematcher"):
        assert isinstance(string_list, collections.abc.Sequence)
        assert all(isinstance(item, str) for item in string_list)
        self.string_list = string_list
        self.string_checker = StringChecker(method=compare_method)
        self.processed_string_to_string_list = {self.string_checker.process_string(val): val for val in self.string_list}
        self.string_list_to_processed_string = {v: k for k, v in self.processed_string_to_string_list.items()}

    def get_key(self, string_parameter: str, string_information: str) -> str:
        processed_string = self.string_checker.process_string(string_parameter)
        if processed_string not in self.processed_string_to_string_list:
            self._raise_error(string_parameter, string_information)
        return self.processed_string_to_string_list[processed_string]

    @property
    def _option_size(self):
        return len(self.string_list)

    def _raise_error(self, string: str, string_information: str) -> None:
        recommendation_index = self.string_checker.get_closest_item_index(string, self.string_list)
        if recommendation_index == -1:
            raise ValueError(f"{string_information} `{string}` is not supported, "
                             f"please consider the following valid choice:\n"
                             f"{self.string_list}")
        else:
            raise ValueError(f"{string_information} `{string}` is not supported, "
                             f"do you mean `{self.string_list[recommendation_index]}`?")

    def __call__(self, *args, **kwargs):
        pass

    def __repr__(self):
        class_name = type(self).__name__
        return f"{class_name}\n string list: {self.string_list}\n string checker: {self.string_checker!r}"

    def __str__(self):
        class_name = type(self).__name__
        return f"{class_name} has {self._option_size} valid options, they are:\n {self.string_list}"
