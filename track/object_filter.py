from .data_struct import ObjectBox, SourceDetect
from utils.util_logger import UtilLogger
import warnings


class ObjectFilter:
    def __init__(self, **kwargs) -> None:
        self._init_logger(**kwargs)
        self._out_objects = None

    def _init_logger(self, **kwargs):
        self._logger = getattr(kwargs, 'logger', None)
        if self._logger is not None and not isinstance(self._logger, UtilLogger):
            warnings.warn("logger should be UtilLogger instance.")
            self._logger = None

    def input_objects(self, objects: list[ObjectBox]):
        self._out_objects = []
        for obj in objects:
            if not self._filter_objects(obj):
                tmp = obj.deep_copy_object()
                if tmp.source_detect == SourceDetect.CL:
                    tmp.set_width(max(tmp.width, 0.1))
                    tmp.set_length(max(tmp.length, 0.1))
                self._out_objects.append(tmp)

    def _filter_objects(self, obj):
        if not self._is_character_valid(obj):
            return True
        return False

    def _is_character_valid(self, obj: ObjectBox):
        if obj.time_idx == 0:
            return False
        if obj.type not in SourceDetect.values():
            return False
        return True

    @property
    def filtered_objects(self):
        return self._out_objects
