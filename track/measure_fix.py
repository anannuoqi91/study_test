import numpy as np
import copy
from mot.mot_types import Trajectory, Measurement
from modules.box_pb2 import Boxes, BoxType
from modules.seyond_pb2_c import PointCloud2


class MeasureFix:
    def __init__(self, **kwargs):
        self.org_box = kwargs['msg_box']
        self.org_pts = kwargs['msg_pts']

        self._thresh_trust_yaw_diff = 0.1 * np.pi
        self._thresh_trust_hori_dist = 0.3
        self._thresh_trust_vert_dist = 0.3
        self._msg_box: list[Boxes] = None
        self._msg_pts: list[PointCloud2] = None
        self._cache_obj: list[Trajectory] = []
        self._fixed_matched_measure: list[Measurement] = []
        self._fixed_unmatch_measure: list[Measurement] = []

    def input_data(self, msg_bbox: list[Boxes],
                   msg_pcl2: list[PointCloud2],
                   vec_predict_object: list[Trajectory],
                   vec_matched_measure: list[Measurement],
                   vec_unmatch_measure: list[Measurement]):
        self._msg_box = copy.deepcopy(msg_bbox)
        self._msg_pts = copy.deepcopy(msg_pcl2)
        self._fixed_matched_measure = copy.deepcopy(vec_matched_measure)
        self._fixed_unmatch_measure = copy.deepcopy(vec_unmatch_measure)
        for measure in self._fixed_matched_measure:
            for obj in vec_predict_object:
                if obj.track_id == measure.match_id:
                    self.fix_axis_error(obj, measure)

        cache_fusioned = []
        for obj in vec_predict_object:
            pass

    def recalc_splited_measure(self, obj: Trajectory):
        if obj.type == BoxType.PEDESTRIAN or obj.type == BoxType.CYCLIST:
            return False
        if obj.box_l < 1.0 and obj.box_w < 1.0:
            return False

        comp_f = 0
        comp_b = 0
        comp_l = 0
        comp_r = 0
        max_z = 0
        min_z = 0
        index_used = []
        for i in range(len(self._fixed_unmatch_measure)):
            tmp = self._fixed_unmatch_measure[i]
            if tmp.type == BoxType.PEDESTRIAN or tmp.type == BoxType.CYCLIST:
                continue
            if abs(tmp.position_lidar_z - obj.position_lidar_z()) > max(2.0, 0.5 * (obj.box_h + tmp.box_h)):
                continue
            if self.calc_shared_area(self.expand(object, 0.3, 0.3, 0.1, 0.1), tmp) < 0.3 * min(obj.box_l * obj.box_w, tmp.box_l * tmp.box_w):
                continue
            if not tmp.indices:
                continue
            elif len(self.org_box[tmp.indices[0]].point_index_size) < 3:
                continue
            bbox = self.org_box[tmp.indices[0]]
            comps_pcl = self.calc_points_comp(
                bbox, obj.heading_lidar, obj.position_lidar)
            if not index_used:
                comp_f = comps_pcl[0]
                comp_b = comps_pcl[1]
                comp_l = comps_pcl[2]
                comp_r = comps_pcl[3]
                max_z = tmp.position_lidar_z + 0.5 * tmp.box_h
                min_z = tmp.position_lidar_z - 0.5 * tmp.box_h
            else:
                comp_f = max(comp_f, comps_pcl[0])
                comp_b = min(comp_b, comps_pcl[1])
                comp_l = max(comp_l, comps_pcl[2])
                comp_r = min(comp_r, comps_pcl[3])
                max_z = max(max_z, tmp.position_lidar_z + 0.5 * tmp.box_h)
                min_z = min(min_z, tmp.position_lidar_z - 0.5 * tmp.box_h)
            index_used.append(i)
        if not index_used:
            return False

    def calc_points_comp(self, box, heading_lidar, position_lidar):
        pass

    def calc_shared_area(self, obj):
        pass

    def expand(self, obj, mea):
        pass

    def fix_axis_error(self, obj: Trajectory, mea: Measurement):
        diff = self.heading_diff(obj.heading_lidar, mea.heading_lidar)
        if diff < self._thresh_trust_yaw_diff:
            return True

        if diff > np.pi / 2:
            if mea.heading_lidar > np.pi:
                mea.heading_lidar -= np.pi
            else:
                mea.heading_lidar += np.pi
            diff = self.heading_diff(obj.heading_lidar, mea.heading_lidar)
            return True
        return False

    def heading_diff(self, heading1: float, heading2: float):
        h1 = [np.cos(heading1), np.sin(heading1), 0, 1]
        h2 = [np.cos(heading2), np.sin(heading2), 0, 1]
        dot_product = h1[0] * h2[0] + h1[1] * h2[1]
        h1_l = np.sqrt(h1[0] * h1[0] + h1[1] * h1[1])
        h2_l = np.sqrt(h2[0] * h2[0] + h2[1] * h2[1])
        return abs(np.acos(dot_product / (h1_l * h2_l + 1e-6)))
