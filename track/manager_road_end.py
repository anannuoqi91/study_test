import numpy as np
from mot.pose import PoseManager, Pose
from config import Config


class EstimatorType:
    CV = 0   # 6 dimension state space [x x' y y' z z']
    SIMPLIFY_CTRV = 1,
    EXT_CTRV = 2,
    RM = 3


class TrajectoryManagerRoadEnd:
    def __init__(self):
        self._motion_model_default = EstimatorType.CV

        self._frame_ns = 1e8

        self._h_default = np.zeros((3, 6))
        self._h_default[0, 0] = 1.0
        self._h_default[1, 2] = 1.0
        self._h_default[2, 4] = 1.0

        self._f_default = np.eye(6)
        self._f_default[0, 1] = self._frame_ns * 1e-9
        self._f_default[2, 3] = self._frame_ns * 1e-9
        self._f_default[4, 5] = self._frame_ns * 1e-9

        self._enable_check_missing_time = True

        self._nu = 0
        self._idx_now = 0
        self._idx_duration = 0

        self._pose_manager = PoseManager()  # You need to implement this class
        self._pose_manager.activate = True
        self._pose_manager.capacity = 64

        self._vec_missing_time = [(0, 10),
                                  (300, 5),
                                  (600, 3)]  # Uncomment the others if needed

        self._trans_mat_pose = np.eye(4)
        self._read_parallel_matrix_done = False

        self._ck_config = Config()

        self._type_match_rules = []
        self._map_trajectory = {}
        self._lidar_location = np.array([0, 0, 0, 1])

    def set_config(self, conf: Config):
        self._ck_config = conf.model_copy()

    def input_pose(self, poses: list[Pose]):
        for i in poses:
            self._pose_manager.add_pose(i)

    def predict_trajectory(self, idx):
        for k, v in self._map_trajectory.items():
            delta_t = (idx - self._dx_now) * self._frame_ns / 1e9
            timestamp_pre = v.timestamp + delta_t * 1e9

    def reset_variable(self, idx):
        if self._idx_now > idx:
            self._map_trajectory = []
            self._map_estimator = []
            self._map_type_history = []
            self._map_vehicle_count = []
            self._map_measure_heading_history = []
            self._map_had_high_speed_motion = []
            self._nu = 0
            self._map_history = []
            self._map_check_points = []
            self._map_cache_new_id = []

    @property
    def lidar_location(self):
        return self._lidar_location.copy()

    @lidar_location.setter
    def lidar_location(self, v: np.array):
        self._lidar_location = v.copy()

    def output_predict_trajectory(self, l):
        pass
