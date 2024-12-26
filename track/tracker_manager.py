"""
坐标系 2个
单个object的坐标系，已中心点为原点，前进方向为z轴，右方向为y轴，上方向为x轴
全局坐标系，(0,0,0)为原点，正北方向为z轴，正东方向为y轴，上方向为x轴

object_source可以作为目标检测可信度的权重表示，AI > CL
"""


from object_filter import ObjectFilter
from data_struct import ObjectBox, ObjectType
from objects_rematch import ObjectsRematch


class TrackerManager:
    def __init__(self):
        self._filter_manager = ObjectFilter()
        self._pre_time_s = 0
        self._cur_time_s = 0
        self._cache_objects = {}
        self._analyzer_velo_ = {}
        self._objects_rematch = ObjectsRematch()

    def input(self, objects: list[ObjectBox], time_s):
        self._cur_time_s = time_s
        self._process(objects)

    def _process(self, objects):
        # Filter
        self._filter_manager.input_objects(objects)
        measure_valid = self._filter_manager.filtered_objects
        # predict
        predict_objects = self._predict(self._cur_time_s - self._pre_time_s)
        # match
        self._objects_rematch.input_data(predict_objects, measure_valid)
        measure_matched, measure_unmatch = self._objects_rematch.output_data()

    def _maintain_all_objects(self, measures):
        for mea in measures:
            if mea.id in self._cache_objects:
                pass

    def _maintain_object(self, measure: ObjectBox):
        if measure.id not in self._cache_objects:
            return
        cache_object = self._cache_objects[measure.id]
        if (cache_object.type == ObjectType.PEDESTRIAN or cache_object.type == ObjectType.CYCLIST) and \
                measure.type >= ObjectType.CAR:
            return

    def _predict(self, time_s):
        predict_objects = []
        for k, v in self._cache_objects.items():
            new_box = v.deep_copy_object()
            if k in self._analyzer_velo_:
                self._analyzer_velo_[k].predict_only(time_s)
                position = self._analyzer_velo_[k].position_predict
                new_box.set_position_center(
                    position[0], position[1], position[2])
                new_box.set_covariance(
                    self._analyzer_velo_[k].covariance_predict
                )
            predict_objects.append(new_box)
        return predict_objects

    @property
    def results(self):
        return self._cache_objects.values.tolist()
