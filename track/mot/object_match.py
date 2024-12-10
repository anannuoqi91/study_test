from ..data_struct import ObjectBox
import numpy as np
from utils.util_angle import norm_angle_radius
from utils.util_shapely_geom import create_polygon, union_area
from utils.util_tools import projection_bev


M_PI_2 = np.pi / 2
M_PI_4 = np.pi / 4


class ObjectMatch:
    def __init__(self):
        self._thresh_prob = 0.2
        self._prob_detect = 0.9
        self._prob_valid = 0.99
        self._prob_noise = 0.05
        self._delta = self._prob_detect * self._prob_valid
        self._beta = self._prob_detect / self._prob_noise

        self._map_predict_detect = {}
        self._map_detect_predict = {}
        self._map_match_prob = []
        self._map_match_relation = {}

        self._cache_predict = []
        self._cache_detect = []

    def _cache_clear(self):
        self._map_predict_detect = {}
        self._map_detect_predict = {}
        self._map_match_prob = []
        self._map_match_relation = {}

    def input_data(self, predictions, measurements):
        self._cache_clear()
        self._cache_predict = predictions
        self._cache_detect = measurements
        self._select_best_match()
        self._iJIPDA0()
        self._select_match_relation()

    def matched_objects(self):
        out = []
        for k, v in self._map_match_relation.items():
            tmp = self._cache_detect[v[0]].deep_copy_object()
            tmp.id = self._cache_predict[k].id
            tmp.set_prob_match(v[1])
            out.append(tmp)
        return out

    def unmatched_objects(self):
        out = []
        for i in range(len(self._cache_detect)):
            if i not in self._map_detect_predict:
                out.append(self._cache_detect[i].deep_copy_object())
        return out

    def gate_true_unmatched(self):
        out = []
        used = []
        for k, v in self._map_match_relation.items():
            used.append(v[0])
        for i in range(len(self._cache_predict)):
            if i not in used and i in self._map_detect_predict:
                out.append(self._cache_predict[i].deep_copy_object())
        return out

    def _select_match_relation(self):
        pre_used = set()
        det_used = set()
        tmp = np.array(self._map_match_prob)
        sorted_indices = np.lexsort((tmp[:, 2], tmp[:, 0]))
        sorted_array = tmp[sorted_indices]
        for row in sorted_array:
            if row[0] in pre_used or row[1] in det_used:
                continue
            pre_used.add(row[0])
            det_used.add(row[1])
            self._map_match_relation[row[0]] = row[1:]

    def _gate(self, prediction: ObjectBox, detection: ObjectBox):
        dealta_h = abs(
            prediction.position_center.x - detection.position_center.x)
        if dealta_h > max(1.5, 0.5 * (prediction.height + detection.height)):
            return False

        comp = projection_bev([prediction.position_center.z - detection.position_center.z,
                               prediction.position_center.y - detection.position_center.y])
        if abs(comp[0]) > max(0.5 * (prediction.length + detection.length), 2.5) or \
                abs(comp[1]) > max(0.5 * (prediction.width + detection.width), 1.6):
            return False

        if abs(comp[0]) < 2 and abs(comp[1]) < 0.8:
            return False

        if (self._diff_yaw(self._diff_yaw(prediction.heading_rad, detection.heading_rad), M_PI_2) < M_PI_4):
            delta = 0.5 * prediction.width - 0.5 * detection.length
            if abs(comp[0]) < 2 and min(abs(comp[1] + delta), abs(comp[1] - delta)) < 0.6:
                return True
        else:
            delta = 0.5 * prediction.width - 0.5 * detection.width
            if abs(comp[0]) < 2 and min(abs(comp[1] + delta), abs(comp[1] - delta)) < 0.6:
                return True
        if self._shared_area_bev(prediction, detection) > 0.01:
            return True

        return False

    def _shared_area_bev(self, box1, box2):
        polygon1 = create_polygon(box1.bev_polygon())
        polygon2 = create_polygon(box2.bev_polygon())

        return union_area(polygon1, polygon2)

    def _diff_yaw(self, yaw1, yaw2):
        diff = norm_angle_radius(yaw1 - yaw2)
        return min(diff, 2 * np.pi - diff)

    def _select_best_match(self):
        self._map_predict_detect = {}
        self._map_detect_predict = {}
        for i in range(len(self._cache_predict)):
            for j in range(len(self._cache_detect)):
                if self._gate(self._cache_predict[i], self._cache_detect[j]):
                    if i not in self._map_predict_detect:
                        self._map_predict_detect[i] = [j]
                    else:
                        self._map_predict_detect[i].append(j)
                    if j not in self._map_detect_predict:
                        self._map_detect_predict[j] = [i]
                    else:
                        self._map_detect_predict[j].append(i)

    def _iJIPDA0(self):
        for i in range(len(self._cache_predict)):
            delta = self._delta
            sum_beta = 1 - delta
            cache_index_beta = {}
            if i not in self._map_predict_detect:
                continue
            map_pre_det = self._map_predict_detect[i]
            for j in map_pre_det:
                prob_gauss = self._gauss_prob(
                    self._cache_predict[i], self._cache_detect[j])
                beta = prob_gauss * self._beta
                delta -= beta
                sum_beta += beta
                cache_index_beta[j] = beta
            for k, v in cache_index_beta.items():
                association_prob = v / sum_beta
                if association_prob > self._thresh_prob:
                    self._map_match_prob.append([i, k, association_prob])

    def _gauss_prob(self, prediction: ObjectBox, detection: ObjectBox):
        residual = np.array([prediction.position_center.z - detection.position_center.z,
                             prediction.position_center.y - detection.position_center.y,
                             prediction.position_center.x - detection.position_center.x])
        covariance = np.array(prediction.covariance)[:3, :3]
        dist = residual.T @ np.linalg.inv(covariance) @ residual
        prob = np.exp(-0.5 * dist) / (np.power(2 * np.pi, 1.5)
                                      * np.sqrt(np.linalg.det(covariance)))
        return prob
