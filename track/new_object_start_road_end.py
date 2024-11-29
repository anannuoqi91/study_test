import numpy as np
from config import Config


class NewObjectStartRoadEnd:
    def __init__(self) -> None:
        self._lidar_location = np.array([0, 0, 0, 1])

    def _set_configure_params(self, conf: Config):
        self._ck_config = conf.model_copy()

    @property
    def lidar_location(self):
        return self._lidar_location.copy()

    @lidar_location.setter
    def lidar_location(self, v: np.array):
        self._lidar_location = v.copy()
