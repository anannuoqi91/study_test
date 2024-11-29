from pydantic import BaseModel
from typing import Optional
import numpy as np


class Config(BaseModel):
    box_in_channel: str = "omnisense/cluster/01/boxes"
    assoc_threshold: Optional[float] = 0.1
    set_detection_source_type: Optional[int] = 0
    stationary_state_is_adjustable: Optional[bool] = False
    static_velocity_threshold3: Optional[float] = 1.0
    static_velocity_threshold4: Optional[float] = 1.0
    static_velocity_threshold5: Optional[float] = 1.0
    static_velocity_threshold6: Optional[float] = 1.0
    type_size: Optional[int] = 9
    type_unknown: int = BoxType.default
    type_pedestrian: int = BoxType.default
    type_light_vehicle: int = BoxType.default
    type_heavy_vehicle: int = BoxType.default
    type_cyclist: int = BoxType.default
    type_debris: int = BoxType.default
    type_car: int = BoxType.default
    type_truck: int = BoxType.default
    type_bus: int = BoxType.default
    # mirror_reflection para
    enable_check_mirror_reflection: Optional[bool] = True
    lidar_code: str
    pointcloud_in_channel: str = "omnisense/preprocess/01/parallel_up_dynamic_points"
    # Field-of-view (FOV) - Vertical (V), the upper edge is negative
    polar_vert_angle_range: list
    # Field-of-view (FOV) - Horizontal (H), the left edge is negative
    polar_hori_angle_range: list
    polar_vert_angle_resolution: float
    polar_hori_angle_resolution: float
    mirror_reflection_threshold: Optional[float] = 0.1
    mirror_reflection_heigth: Optional[float] = 2.5
    mirror_reflection_distance: Optional[float] = 40.0
    enable_static_false_positive_filter: Optional[bool] = False
    overlap_threshold_post: Optional[float] = 0.3
    p_d: Optional[float] = 0.9
    p_d: Optional[float] = 0.99
    angle_threshold: Optional[float] = 0.15
    velocity_esti_heading_thresh: Optional[float] = 3.0
    use_gpu_filter_boxes_points:  Optional[bool] = False
    heading_adjust_weight: Optional[float] = 0.5
    maintain_size_weight: Optional[float] = 0.75
    enable_wwd: Optional[bool] = False
    enable_wwd_start_delay: Optional[bool] = False
    wwd_disp_thresh: Optional[float] = 8
    enable_filter_ai_unknown_box: bool = True
    static_points_send_freq: Optional[float] = 1
    time_consumption_trajectory_init_all_: float = 0.0
    elapsed_time_all_: float = 0.0
    preprocess_time_all_: float = 0.0
    start_tracking_time_all_: float = 0.0
    postprocess_time_all_: float = 0.0
    checkoverlap_box_all_: float = 0.0
    checkoverlap_trajectory_all_: float = 0.0
    convert_trajectory_to_box_all_: float = 0.0
    m_msg_count_: int = 0
    m_read_merge_matrix_done_: bool = False
    m_read_time_interval_done_: bool = False
    prob_exist_default: Optional[float] = 0.1
    prob_exist_calc: Optional[int] = 0
    missingtime_from0to300: Optional[int] = 10
    missingtime_from300to600: Optional[int] = 5
    missingtime_larger600: Optional[int] = 3
    merge_matrix: Optional[np.array] = np.eye(4)
    q_default: Optional[np.array] = np.zeros((6, 6))
    r_default: Optional[np.array] = np.zeros((3, 3))
    p_default: Optional[np.array] = np.eye(6) * 9
    background_roi_mode: Optional[int] = 0
    traffic_flow_sample_num: Optional[int] = 50

    def __init__(self, **data):
        super().__init__(**data)

        self._init_q_default()
        self._init_r_default()

    def _init_q_default(self):
        self.q_default[0, 0] = 0.25
        self.q_default[1, 1] = 1.0
        self.q_default[2, 2] = 0.5
        self.q_default[3, 3] = 1.0
        self.q_default[4, 4] = 0.5
        self.q_default[5, 5] = 1.0

    def _init_r_default(self):
        self.r_default[0, 0] = 0.25
        self.r_default[1, 1] = 0.25
        self.r_default[2, 2] = 0.25

    def _init_p_default(self):
        self.p_default[0, 0] = 9
        self.p_default[1, 1] = 9
        self.p_default[2, 2] = 9
        self.p_default[3, 3] = 9
        self.p_default[4, 4] = 9
        self.p_default[5, 5] = 9
