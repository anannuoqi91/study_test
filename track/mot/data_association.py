from enum import Enum
import copy
import numpy as np
from validate_gate import ValidateGate


class DataAssociationAlgo(Enum):
    iJIPDA0 = 0
    iJIPDA1 = 1
    iJIPDA2 = 2


class DataAssociation:
    def __init__(self, **kwargs):
        self._algo = DataAssociationAlgo.iJIPDA0
        self._gauss_pdf_consider_size = False
        self._gauss_pdf_consider_heading = False
        self._reselect_dense_echo = False
        self._maximum_echo_in_gate = 3
        self._P_D = 0.9
        self._P_G = 0.99
        self._rho = 0.05
        self._assoc_threshold = 0.2

        self._legal_input = False
        # [[index_track, index_measure]]
        self._gate_relation1 = np.empty((0, 2))
        # [[index_measure, index_track]]
        self._gate_relation2 = np.empty((0, 2))
        # [[index_track, index_measure, probability]]
        self._match_relation = np.empty((0, 3))
        self._vec_match_relation = np.empty(
            (0, 2))  # [[index_track, index_measure]]
        # [[index_track,  index_measure, beta]]
        self._match_relation_beta = np.empty((0, 3))
        self._buffer_trajectory = []
        self._buffer_measurement = []
        self._map_psi_update = np.empty((0, 2))  # [[index_track, psi_update]]

        self._gate = ValidateGate()

    def input_data(self, trajcetory, measure):
        self._vec_match_relation.clear()
        self._match_relation.clear()
        self._match_relation_beta.clear()
        if not trajcetory or not measure:
            self._legal_input = False
            self._buffer_trajectory.clear()
            self._buffer_measurement.clear()
            self._vec_match_relation.clear()
            return

        self._buffer_trajectory = copy.deepcopy(trajcetory)
        self._buffer_measurement = copy.deepcopy(measure)
        self._legal_input = True
        self._gate_relation1.clear()
        self._gate_relation2.clear()
        self._map_psi_update.clear()

        self.select_echo_by_gate()
        self.reselect_dense_echo()
        if self._algo == DataAssociationAlgo.iJIPDA0:
            self.ijipda_0()
        elif self._algo == DataAssociationAlgo.iJIPDA1:
            self.ijipda_1()
        elif self._algo == DataAssociationAlgo.iJIPDA2:
            self.ijipda_2()
        else:
            self.ijipda_0()

        self.select_match_relation()

    def select_echo_by_gate(self):
        for i in range(len(self._buffer_trajectory)):
            for j in range(len(self._buffer_measurement)):
                self._gate.config_validate_gate([self._buffer_trajectory[i].box_l * 0.5,
                                                 self._buffer_trajectory[i].box_l * 0.5,
                                                 self._buffer_trajectory[i].box_w * 0.5,
                                                 self._buffer_trajectory[i].box_h * 0.5])
                if self._gate.in_gate(self._buffer_trajectory[i], self._buffer_measurement[j]):
                    self._gate_relation1 = np.append(
                        self._gate_relation1, [[i, j]], axis=0)
                    self._gate_relation1.append([i, j])
                    self._gate_relation2 = np.append(
                        self._gate_relation2, [[j, i]], axis=0)

    def reselect_dense_echo(self):
        pass

    def select_match_relation(self):
        set_meas_matched = set()
        set_traj_matched = set()
        sorted_indices = np.argsort(self._match_relation[:, 2])[
            ::-1]  # 获取第二列的排序索引，并反转顺序
        sorted_match_relation = self._match_relation[sorted_indices]
        for i_r in sorted_match_relation:
            if i_r[0] in set_traj_matched:
                continue
            if i_r[1] in set_meas_matched:
                continue
            self._vec_match_relation = np.append(
                self._vec_match_relation, [i_r[:2]], axis=0)
            set_meas_matched.append(i_r[1])
            set_traj_matched.append(i_r[0])

    def ijipda_0(self):
        if not self._legal_input:
            return
        for i in range(len(self._buffer_trajectory)):
            delta = self._P_D * self._P_G
            sum_beta = 1 - delta
            mea_l = self._gate_relation1[self._gate_relation1[:, 0] == i][: 1]
            for j in mea_l:
                gauss_prob = self.gauss_pdf(i, j)
                beta = self._P_D * gauss_prob / self.rho
                delta -= beta
                sum_beta += beta
                self._match_relation_beta = np.append(
                    self._match_relation_beta, [[i, j, beta]], axis=0)
            mea_l = self._match_relation_beta[self._match_relation_beta[:, 0] == i]
            for j_r in mea_l:
                association_prob = j_r[2] / sum_beta
                if association_prob > self._assoc_threshold:
                    self._match_relation = np.append(
                        self._match_relation, [[i, j_r[1], association_prob]], axis=0)
            psi_update = (1 - delta) * self._buffer_trajectory[i].prob_exist / (
                1 - delta * self._buffer_trajectory[i].prob_exist)
            self._map_psi_update = np.append(
                self._map_psi_update, [[i, psi_update]], axis=0)

    def gauss_prob(self, traj, mea):
        delta = [self._buffer_measurement[mea].position_lidar_x - self._buffer_trajectory[traj].position_lidar_x,
                 self._buffer_measurement[mea].position_lidar_y -
                 self._buffer_trajectory[traj].position_lidar_y,
                 self._buffer_measurement[mea].position_lidar_z - self._buffer_trajectory[traj].position_lidar_z]
        cov_new_info = self._buffer_trajectory[traj].cov_new_info
        dist = delta.dot(np.linalg.inv(cov_new_info) @ delta)
        prob = np.exp(-0.5 * dist) / (np.power(2 * np.pi, 1.5)
                                      * np.sqrt(np.linalg.det(cov_new_info)))

        return prob

    def ijipda_1(self):
        pass

    def ijipda_2(self):
        pass

    def matched_measure(self):
        vec_measure = []
        for i_r in self._vec_match_relation:
            tmp = copy.deepcopy(self._buffer_measurement[i_r[1]])
            tmp.match_id = self._buffer_trajectory[i_r[0]].track_id
            tmp.prob_exist = self._map_psi_update[i_r[0]]
            vec_measure.append(tmp)
        return vec_measure

    def unmatched_measure(self):
        vec_measure = []
        for i in range(len(self._buffer_measurement)):
            if i not in self._gate_relation2[:, 0]:
                tmp = copy.deepcopy(self._buffer_measurement[i])
                vec_measure.append(tmp)

        return vec_measure
