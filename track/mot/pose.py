import numpy as np
from pydantic import BaseModel


class Pose(BaseModel):
    timestamp: int
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float


class PoseManager:
    def __init__(self):
        self._capacity = 64
        self.activate = True
        self.mat_base = np.eye(4)
        self._pose_list = []

    def trans_matrix(self, timestamp):
        if not self.activate:
            return self.mat_base

        if len(self._pose_list) == 0:
            return self.mat_base

    def add_pose(self, pose):
        if len(self._pose_list) >= self._capacity:
            self.__pose_list.pop(0)
        if len(self._pose_list) == 0 or self._pose_list[-1].timestamp < pose.timestamp:
            self._pose_list.append(pose)
