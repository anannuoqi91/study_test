import numpy as np
from enum import Enum
from pydantic import BaseModel, field_validator
from typing import Optional
from ...utils.util_angle import norm_angle


class Measurement(BaseModel):
    type: Optional[int] = 0
    box_l: Optional[float] = 0.0
    box_w: Optional[float] = 0.0
    box_h: Optional[float] = 0.0
    heading_lidar: Optional[float] = 0.0
    heading_world: Optional[float] = 0.0
    timestamp: Optional[int] = 0
    detection_source: Optional[int] = 0
    position_lidar: Optional[np.array] = np.zeros(4)
    lidar2world: Optional[np.array] = np.eye(4)

    idx: Optional[int] = 0
    status: Optional[int] = 0
    point_num: Optional[int] = 0
    extend_key: Optional[int] = 0
    match_id: Optional[int] = 0
    prob_exist: Optional[float] = 0.0
    has_extend_key: Optional[bool] = False

    indices: Optional[list[int]] = []

    @field_validator('position_lildar')
    def check_length(cls, v):
        if len(v) != 4:
            raise ValueError('position_lildar must be an array of length 4')
        return v

    @field_validator('lidar2world')
    def check_shape(cls, v):
        if v.shape != (4, 4):
            raise ValueError('lidar2world must be a 4x4 array')
        return v

    @staticmethod
    def create_cl_from_box(box, cur_idx, pose):
        heading = norm_angle(box.spindle / 18000.0 * np.pi)
        if pose:
            lidar2world = pose.trans_matrix(box.timestamp)
        else:
            lidar2world = np.eye(4)
        tmp = {
            'box_l': box.length,
            'box_w': box.width,
            'box_h': box.height,
            'heading_lidar': heading,
            'heading_world': heading,
            'timestamp': box.timestamp,
            'position_lidar': np.array([box.position_z, 0.0 - box.position_y,
                                        box.position_x, 1.0]),
            'lidar2world': lidar2world,
            'idx': cur_idx,
            'status': 0,
            'point_num': len(box.point_index),
            'extend_key': box.cluster_id,
            'match_id': 0,
            'type': box.object_type,
            'prob_exist': 0.0,
            'detection_source': box.detection_source
        }
        return Measurement(**tmp)

    @property
    def position_lidar_z(self):
        return self.position_lidar[2]


class DetectionSource(Enum):
    Source_AI = 1
    Source_Cluster = 2


class Trajectory(BaseModel):
    type: Optional[int] = 0
    track_id: Optional[int] = 0
    box_l: Optional[float] = 0.0
    box_w: Optional[float] = 0.0
    box_h: Optional[float] = 0.0
    heading_lidar: Optional[float] = 0.0
    heading_world: Optional[float] = 0.0
    timestamp: Optional[int] = 0
    detection_source: Optional[int] = 0
    position_lidar: Optional[np.array] = np.zeros(4)
    lidar2world: Optional[np.array] = np.eye(4)

    def position_lidar_z(self):
        return self.position_lidar[2]

    def position_lidar_y(self):
        return self.position_lidar[1]

    def position_lidar_x(self):
        return self.position_lidar[0]


class Estimator(BaseModel):
    pass
