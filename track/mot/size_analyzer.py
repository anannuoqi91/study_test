from data_struct import ObjectBox, PointXYZ, ObjectType
from utils.util_angle import diff_angle_radius
from utils.util_tools import projection_bev
import math


class SizeAnalyzer:
    def __init__(self):
        self._min_l_pcl = 0
        self._min_w_pcl = 0
        self._min_h_pcl = 0

        self._min_l_box = 0
        self._min_w_box = 0
        self._min_h_box = 0

    def _size_threshold_ge_car(self):
        self._min_l_box = 4.0
        self._min_w_box = 1.6
        self._min_h_box = 1.2

    def _size_threshold_pedestrian_cyclist(self):
        self._min_l_box = 0.4
        self._min_w_box = 0.4
        self._min_h_box = 1.2

    def _init_params(self):
        self._exp_measure_f = 0
        self._exp_measure_b = 0
        self._exp_measure_l = 0
        self._exp_measure_r = 0

        self._exp_predict_f = 0
        self._exp_predict_b = 0
        self._exp_predict_l = 0
        self._exp_predict_r = 0

    def input_data(self, predict: ObjectBox, measure: ObjectBox, points_3d: list[PointXYZ]):
        # front right up
        if diff_angle_radius(predict.yaw, measure.yaw) < 0.1 and points_3d:
            self._points = points_3d
            self._cal_pcl(measure.yaw)

        if predict.type >= ObjectType.CAR:
            self._size_threshold_ge_car()
        elif predict.type == ObjectType.PEDESTRIAN or predict.type == ObjectType.CYCLIST:
            self._size_threshold_pedestrian_cyclist()

        if measure.length > predict.length + 0.5:
            arrow = [measure.z - predict.z, measure.y - predict.y]
            comps = projection_bev(arrow, predict.yaw)
            dist_f = comps[0] + 0.5 * measure.length - 0.5 * predict.length
            dist_b = comps[0] - 0.5 * measure.length + 0.5 * predict.length
            dist_l = comps[1] + 0.5 * measure.width - 0.5 * predict.width
            dist_r = comps[1] - 0.5 * measure.width + 0.5 * predict.width

            new_info_l = min(0.5, 0.1 * (measure.length - predict.length))
            new_info_w = min(0.5, 0.1 * (measure.width - predict.width))

    def _cal_pcl(self, yaw):
        cosv = math.cos(yaw)
        sinv = math.sin(yaw)

        max_f = self._points[0].z * cosv - self._points[0].y * sinv
        min_f = max_f
        max_r = -self._points[0].y * cosv - self._points[0].z.z * sinv
        min_r = max_r
        max_up = self._points[0].x
        min_up = max_up

        for p in self._points:
            f = p.z * cosv - p.y * sinv
            r = -p.y * cosv - p.z * sinv
            max_up = max(max_up, p.x)
            min_up = min(min_up, p.x)
            max_f = max(max_f, f)
            min_f = min(min_f, f)
            max_r = max(max_r, r)
            min_r = min(min_r, r)

        self._min_l_pcl = max(self._min_l_pcl, (max_f - min_f) * 0.8)
        self._min_w_pcl = max(self._min_w_pcl, (max_r - min_r) * 0.8)
        self._min_h_pcl = max(self._min_h_pcl, (max_up - min_up) * 0.8)
