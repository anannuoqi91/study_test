from enum import Enum
import copy
import numpy as np
from mot_types import Trajectory, Measurement


class ValidateGateType(Enum):
    RECTANGULAR_GATE = 0
    NORM_SQU_DIST_GATE = 1
    SPHERICAL_GATE = 2
    CUSTOM_DEFINED = 3


class ValidateGate:
    def __init__(self, gate_type):
        self._gate_type = gate_type

        self._consider_heading = False

        # Rectangular Gate
        self._init_rectangular_gate()

        # Normalized Square Distance Gate
        self._init_square_distance_gate()

        # Spherical Gate
        self._init_spherical_gate()

        self._gate_set_name = {
            ValidateGateType.RECTANGULAR_GATE: 'set_rectangular_gate',
            ValidateGateType.NORM_SQU_DIST_GATE: 'set_square_distance_gate',
            ValidateGateType.SPHERICAL_GATE: 'set_spherical_gate',
            ValidateGateType.CUSTOM_DEFINED: 'set_custom_gate',
        }

        self._in_gate_name = {
            ValidateGateType.RECTANGULAR_GATE: 'in_rectangular_gate',
            ValidateGateType.CUSTOM_DEFINED: 'custom_defined_gate',
            ValidateGateType.SPHERICAL_GATE: 'in_spherical_gate',
            ValidateGateType.NORM_SQU_DIST_GATE: 'in_square_distance_gate',
        }

    def _init_spherical_gate(self):
        self._s = np.eye(3) * 0.64
        self._gamma_gate = 9

    def _init_rectangular_gate(self):
        self._half_size_font = 4
        self._half_size_back = 4
        self._half_size_hori = 2
        self._half_size_vert = 2

    def _init_square_distance_gate(self):
        self._r = np.eye(3) * 0.25
        self._vx_max = 40
        self._vy_max = 10
        self._vz_max = 5
        self._vx_min = 0.0
        self._vy_min = 0.0
        self._vz_min = 0.0
        self._norm_squ_dist_max = 1.0

    def set_spherical_gate(self, value):
        if len(value) != 10:
            raise ValueError(f"Invalid spherical gate config.  {value}")

        self._s.flat[:9] = value[0:9]
        self._gamma_gate = value[9]

    def set_square_distance_gate(self, value):
        if len(value) != 10:
            raise ValueError(f"Invalid square distance gate config.  {value}")
        self._r[0, 0] = value[0]
        self._r[1, 1] = value[1]
        self._r[2, 2] = value[2]
        self._vx_max = value[3]
        self._vy_max = value[4]
        self._vz_max = value[5]
        self._vx_min = value[6]
        self._vy_min = value[7]
        self._vz_min = value[8]
        self._norm_squ_dist_max = value[9]

    def set_rectangular_gate(self, value):
        if len(value) != 4:
            raise ValueError(f"Invalid rectangular gate config.  {value}")
        self._half_size_font_ = value[0]
        self._half_size_back_ = value[1]
        self._half_size_hori_ = value[2]
        self._half_size_vert_ = value[3]

    def config_validate_gate(self, params):
        if self._gate_type not in self._gate_set_name:
            raise ValueError(f"{self._gate_type} is not ValidateGateType.")

        if hasattr(self, self._gate_set_name[self._gate_type]):
            method = getattr(self, self._gate_set_name[self._gate_type])
            if callable(method):
                return method(value=params)
            else:
                raise ValueError(f"method_name is not callable.")
        else:
            raise ValueError(f"{self._gate_type} is not supported.")

    def in_gate(self, traj: Trajectory, mea: Measurement):
        if self._gate_type not in self._in_gate_name:
            raise ValueError(f"{self._gate_type} is not ValidateGateType.")

        in_params = {
            'traj': traj.position_lidar,
            'measurement': mea.position_lidar,
            'conf': traj.heading_lidar,
        }
        if self._gate_type == ValidateGateType.SPHERICAL_GATE:
            in_params['conf'] = traj.cov_new_info

        if hasattr(self, self._in_gate_name[self._gate_type]):
            method = getattr(self, self._in_gate_name[self._gate_type])
            if callable(method):
                return method(**in_params)
            else:
                raise ValueError(f"method_name is not callable.")
        else:
            print(f"{self._gate_type} is not supported.")
            return False

    def in_rectangular_gate(self, traj: np.array, measurement: np.array, conf: float):
        diff = traj - measurement
        if (self._consider_heading):
            mat = np.eye(4)
            mat[0, 0] = np.cos(conf)
            mat[0, 1] = np.sin(conf)
            mat[1, 0] = 0.0 - np.sin(conf)
            mat[1, 1] = np.cos(conf)
            diff = mat * diff

        if diff[0] > self._half_size_font or diff[0] < 0.0 - self._half_size_back:
            return False

        if abs(diff[1]) > self._half_size_hori or abs(diff[2]) > self._half_size_vert:
            return False

        return True

    def in_spherical_gate(self, traj: np.array, measurement: np.array, conf: np.array):
        diff = traj - measurement
        diff = diff[0:3]
        v_gamma = diff.dot(np.linalg.inv(conf) @ diff)

        if v_gamma > self._gamma_gate:
            return False

        return True
