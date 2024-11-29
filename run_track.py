import numpy as np
from track.config import Config
from track.new_object_start_road_end import NewObjectStartRoadEnd
from track.manager_road_end import TrajectoryManagerRoadEnd
from track.mot.mot_types import Measurement, DetectionSource, Trajectory
from track.selector import Selector
from track.mot.data_association import DataAssociation


def pre_process(box_msg):
    vec_measure = []
    for box in box_msg.box:
        measure = Measurement.create_cl_from_box(box, box_msg.idx, None)
        if (measure.position_lidar_z > 1000):
            continue
        if measure.box_w > 4.0:
            continue
        if measure.detection_source == DetectionSource.Source_Cluster and \
                measure.box_l > 0.3:
            measure.box_w = max(measure.box_w, 0.1)
            measure.box_h = max(measure.box_h, 0.1)

        if measure.detection_source == DetectionSource.Source_Cluster and \
                measure.box_w > 0.3:
            measure.box_l = max(measure.box_l, 0.1)
            measure.box_h = max(measure.box_h, 0.1)
        vec_measure.append(measure)
    return vec_measure


def process(box_msg, idx_now, last_frame_idx, last_frame_timestamp):
    m_new_object_start_ = NewObjectStartRoadEnd()
    m_new_object_start_.set_lidar_location(np.array([0, 0, 0, 1]))
    vec_input = pre_process(box_msg)
    m_selector_ = Selector()
    m_selector_.set_current_idx(idx_now)
    m_selector_.set_current_measure(vec_input)
    vec_measure = m_selector_.selected_measure


def start_tracking(measures: list[Measurement],
                   poses,
                   trajectory_manager: TrajectoryManagerRoadEnd,
                   vec_trajectory: list[Trajectory],
                   data_association: DataAssociation):
    trajectory_manager.input_pose(poses)
    trajectory_manager.output_predict_trajectory(vec_trajectory)

    data_association.input_data(vec_trajectory, measures)
    vec_measure_matched = data_association.matched_measure()
    vec_measure_unmatched = data_association.unmatched_measure()
    vec_measure_anti_match = []
    for mea in measures:
        is_matched = False
        for matched in vec_measure_matched:
            if mea.position_lidar == matched.position_lidar:
                is_matched = True
                break
        if not is_matched:
            vec_measure_anti_match.append(mea)


def proc(box_msg):
    trajectory_manager = TrajectoryManagerRoadEnd()
    new_obj_start = NewObjectStartRoadEnd()
    data_association = DataAssociation()
    vec_input = pre_process(box_msg)
    lidar_location = [0, 0, 0, 1]  # x, y, z 1  点云坐标系
    # 坐标系转换
    trajectory_manager.lidar_location = np.array(
        [lidar_location[2], -lidar_location[1], lidar_location[0], 1])
    new_obj_start.lidar_location = np.array(
        [lidar_location[2], -lidar_location[1], lidar_location[0], 1])
    vec_trajectory = []


if __name__ == '__main__':
    conf = Config()
