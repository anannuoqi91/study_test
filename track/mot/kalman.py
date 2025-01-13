import numpy as np
from enum import Enum
from utils.util_angle import norm_angle_radius


class MotionType(Enum):
    CV = 0,  # Constant Velocity - CV
    CTRV = 1


class VelocityAnalyzer:
    # up z front x right y
    def __init__(self, **kwargs):
        self._enable_velo = getattr(kwargs, 'enable_velo', False)
        self._enable_yaw = getattr(kwargs, 'enable_yaw', False)
        self._motion_model = getattr(kwargs, 'motion_model', MotionType.CV)

        self._init_params()

    def _init_params(self):
        self._x = np.zeros([0, 0, 0, 0, 0, 0])  # x y z speed_x speed_y speed_z
        self._position = np.asarray([0, 0, 0, 1])
        self._velocity = np.asarray([0, 0, 0, 1])
        self._yaw = 0
        self._yaw_rate = 0

        self._x_predict = np.zeros([0, 0, 0, 0, 0, 0])
        self._position_predict = np.asarray([0, 0, 0, 1])
        self._velocity_predict = np.asarray([0, 0, 0, 1])
        self._yaw_predict = 0
        self._yaw_rate_predict = 0
        self._state_matrix_predict = self._init_state_matrix()

        self._measure_matrix = self._init_measure_matrix()  # m_H_
        self._measure_noise_matrix = self._init_measure_noise_matrix()  # m_R_
        self._state_matrix = self._init_state_matrix()  # m_P_
        self._predict_matrix = self._init_predict_matrix()  # m_F_
        self._process_noise_matrix = self._init_process_noise_matrix()  # m_Q_

        self._covariance_predict = None

    def _init_measure_matrix(self):
        if self._motion_model == MotionType.CV:
            if self._enable_velo:
                out = np.eye(6)
            else:
                out = np.zeros((3, 6))
                out[0, 0] = 1.0
                out[1, 1] = 1.0
                out[2, 2] = 1.0
        else:
            out = np.zeros((4, 6))
            out[0, 0] = 1.0
            out[1, 1] = 1.0
            out[2, 2] = 1.0
            out[3, 4] = 1.0
        return out

    def _init_measure_noise_matrix(self):
        if self._motion_model == MotionType.CV:
            if self._enable_cv:
                out = np.eye(6)
                out[0, 0] = 0.09
                out[1, 1] = 0.09
                out[2, 2] = 0.04
                out[3, 3] = 4.0
                out[4, 4] = 4.0
                out[5, 5] = 1.0
            else:
                out = np.zeros((3, 6))
                out[0, 0] = 0.09
                out[1, 1] = 0.09
                out[2, 2] = 0.04
        else:
            out = np.eye(4)
            out[0, 0] = 0.25
            out[1, 1] = 0.25
            out[2, 2] = 0.16
            out[3, 3] = 0.09
        return out

    def _init_predict_matrix(self, delta_t=0.1):
        if self._motion_model == MotionType.CV:
            out = np.eye(6)
            out[0, 3] = delta_t
            out[1, 4] = delta_t
            out[2, 5] = delta_t
        else:
            out = np.eye(6)
            out[0, 3] = delta_t * np.cos(self._x[4])
            out[1, 3] = delta_t * np.sin(self._x[4])
            out[0, 4] = -1 * delta_t * self._x[3] * np.sin(self._x[4])
            out[1, 4] = delta_t * self._x[3] * np.cos(self._x[4])
            out[4, 5] = delta_t
        return out

    def _init_state_matrix(self):
        if self._motion_model == MotionType.CV:
            out = np.zeros((6, 6))
            out = np.eye(6)
            out[0, 0] = 0.36
            out[1, 1] = 0.36
            out[2, 2] = 0.16
            out[3, 3] = 4.0
            out[4, 4] = 4.0
            out[5, 5] = 4.0
        else:
            out = np.eye(6)
            out[0, 0] = 0.36
            out[1, 1] = 0.36
            out[2, 2] = 0.16
            out[3, 3] = 9.0
            out[4, 4] = 0.09
            out[5, 5] = 0.09
        return out

    def _init_process_noise_matrix(self):
        if self._motion_model == MotionType.CV:
            out = np.eye(6)
            out[0, 0] = 0.01
            out[1, 1] = 0.01
            out[2, 2] = 0.01
            out[3, 3] = 0.01
            out[4, 4] = 0.01
            out[5, 5] = 0.01
        else:
            out = np.eye(3)
            out[0, 0] = 25.0
            out[1, 1] = 16.0
            out[2, 2] = 0.09
        return out

    def predict_only(self, delta_t=0.1):
        if self._motion_model == MotionType.CV:
            self._predict_cv(delta_t)
        elif self._motion_model == MotionType.CTRV:
            self._predict_ctrv(delta_t)

    def _predict_cv(self, delta_t=0.1):
        self._predict_matrix = self._init_predict_matrix(delta_t)
        self._x_predict = np.dot(self._predict_matrix, self._x)
        self._state_matrix_predict = np.dot(
            self._predict_matrix, self._state_matrix)
        self._state_matrix_predict = np.dot(
            self._state_matrix_predict, self._predict_matrix.T)
        self._state_matrix_predict += self._process_noise_matrix
        self._covariance_predict = np.dot(
            self._measure_matrix, self._state_matrix_predict)
        self._covariance_predict = np.dot(
            self._covariance_predict, self._measure_matrix.T)
        self._covariance_predict += self._measure_noise_matrix
        self._position_predict[:3] = self._x_predict[:3]
        self._velocity_predict[:3] = self._x_predict[3:]
        self._yaw_predict = self._vector_yaw()
        self._yaw_rate_predict = 0

    def _predict_ctrv(self, delta_t=0.1):
        self._predict_matrix = self._init_predict_matrix(delta_t)
        tmp_g = np.zeros((6, 3))
        tmp_g[0, 0] = 0.5 * pow(delta_t, 2) * np.cos(self._x[4])
        tmp_g[1, 0] = 0.5 * pow(delta_t, 2) * np.sin(self._x[4])
        tmp_g[2, 1] = 0.5 * pow(delta_t, 2)
        tmp_g[3, 0] = delta_t
        tmp_g[4, 2] = 0.5 * pow(delta_t, 2)
        tmp_g[5, 2] = delta_t
        self._x_predict[0] = self._x[0] + \
            self._x[3] * np.cos(self._x[4]) * delta_t
        self._x_predict[1] = self._x[1] + \
            self._x[3] * np.sin(self._x[4]) * delta_t
        self._x_predict[2] = self._x[2]
        self._x_predict[3] = self._x[3]
        self._x_predict[4] = self._x[4] + self._x[5] * delta_t
        self._x_predict[5] = self._x[5]
        self._state_matrix_predict = np.dot(
            self._predict_matrix, self._state_matrix)
        self._state_matrix_predict = np.dot(
            self._state_matrix_predict, self._predict_matrix.T)
        self._state_matrix_predict += np.dot(
            np.dot(tmp_g, self._process_noise_matrix), tmp_g.T)
        self._covariance_predict = np.dot(
            self._measure_matrix, self._state_matrix_predict)
        self._covariance_predict = np.dot(
            self._covariance_predict, self._measure_matrix.T)
        self._covariance_predict += self._measure_noise_matrix
        self._position_predict[:3] = self._x_predict[:3]
        self._velocity_predict[0] = self._x_predict[3] * \
            np.cos(self._x_predict[4])
        self._velocity_predict[1] = self._x_predict[3] * \
            np.sin(self._x_predict[4])
        self._velocity_predict[2] = 0
        self._velocity_predict[3] = 1
        self._yaw_predict = self._x_predict[4]
        self._yaw_rate_predict = self._x_predict[5]

    def _vector_yaw(self):
        return norm_angle_radius(np.arctan2(self._velocity_predict[1], self._velocity_predict[0]))

    def set_position_now(self, front, right, up):
        self._position = np.asarray([front, right, up, 1])
        self._x[:3] = self._position[:3]

    def set_velocity_now(self, front, right, up):
        self._velocity = np.asarray([front, right, up, 1])
        if self._motion_model == MotionType.CV:
            self._x[3] = self._velocity[0]
            self._x[4] = self._velocity[1]
            self._x[5] = self._velocity[2]
        elif self._motion_model == MotionType.CTRV:
            self._x[3] = np.linalg.norm(self._velocity[:2])
            self._x[4] = norm_angle_radius(np.arctan2(
                self._velocity[0], self._velocity[1]))

    def set_yaw_now(self, yaw):
        self._yaw = yaw
        if self._motion_model == MotionType.CV:
            velo = np.linalg.norm(self._velocity[:2])
            self._x[3] = velo * np.cos(self._yaw)
            self._x[4] = velo * np.sin(self._yaw)
        elif self._motion_model == MotionType.CTRV:
            self._x[4] = yaw

    def set_motion_model(self, motion_model):
        self._motion_model = motion_model
        self._init_params()

    @property
    def position_predict(self):
        return self._position_predict.tolist()

    @property
    def covariance_predict(self):
        return self._covariance_predict.tolist()
