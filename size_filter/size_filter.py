from modules.seyond_pb2_c import PointCloud2, PointXYZI, PointXYZ
import math
import sys
import numpy as np
import cv2
from cyber_record.record import Record
import pandas as pd
from itertools import combinations
import cv2
import open3d as o3d
import time
import copy
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


VALID_LENGTH_CMP_PT = 1
INVALID_LENGTH_CMP_PT = 0

MIN_LENGTH_TO_TRIM = 4.0
RATIO_FOR_FAR_AWAY = 0.9

MIN_BOX_LENGTH_TO_SLICE = 2.0


class Line:
    def __init__(self, start, end):
        self.start = np.array(start)
        self.end = np.array(end)
        self.vec = self.end - self.start
        self.vec = self.vec / np.linalg.norm(self.vec)
        self.x = self.start[0]
        self.y = self.start[1]

    def calc_dist_with_pt(self, x, y):
        tmp = np.array([x - self.x, y - self.y])
        tmp = tmp / np.linalg.norm(tmp)
        length = np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)
        dot = np.dot(tmp, self.vec)
        # 处理 dot == ±1 的情况
        if np.isclose(dot, 1.0) or np.isclose(dot, -1.0):
            return 0
        return np.sin(np.arccos(dot)) * length


def calc_two_points_dist(lhs, rhs):
    dist = math.sqrt(pow(lhs[0] - rhs[0], 2) + pow(lhs[1] - rhs[1], 2) +
                     pow(lhs[2] - rhs[2], 2))
    return dist


def calc_dist_of_point_to_line(point, line, line_length=0.0):
    if len(line) != 2:
        raise ValueError(f"Line's point size {len(line)} not equal to 2!")

    # 将点转换为 NumPy 数组
    point = np.array(point)
    line_start = np.array(line[0])
    line_end = np.array(line[1])

    # 计算线的方向向量
    vec_line = line_start - line_end

    # 计算点到线端点的向量
    vec_pt = point - line_end

    # 计算投影的点积
    proj_dot = np.dot(vec_line, vec_pt)

    # 根据输入线段长度计算投影
    if line_length != 0.0:
        projection = proj_dot / line_length
    else:
        len_vec_line = calc_two_points_dist(line_start, line_end)
        projection = proj_dot / len_vec_line

    # 计算点到线段一端的距离
    len_p1_pt = calc_two_points_dist(point, line_end)

    # 返回点到线段的垂直距离
    return np.sqrt(len_p1_pt**2 - projection**2)


def calc_slice_endpoint_dist_to_edge(points, slice_max_min_pt_index, max_min_lines, line_length, slice_max_min_y):
    for k, v in slice_max_min_pt_index.items():
        min_pt_index = v[0]
        max_pt_index = v[1]
        min_pt = points[min_pt_index]
        max_pt = points[max_pt_index]
        dist_to_min_line = calc_dist_of_point_to_line(
            min_pt, max_min_lines[0], line_length)
        dist_to_max_line = calc_dist_of_point_to_line(
            max_pt, max_min_lines[1], line_length)
        slice_max_min_y[k].append(dist_to_min_line)
        slice_max_min_y[k].append(dist_to_max_line)
    return slice_max_min_y


def update_with_multi_slice_width(slice_map, global_width, global_length, filter_radius):
    ret = global_width
    if len(slice_map) == 0:
        return ret

    slice_widths_good = []
    slice_widths_better = []
    slice_widths_normal = []
    slice_widths_backup = []

    GOOD_DIST_THRESH = 0.05
    NORMAL_DIST_THRESH = 0.11
    BCK_DIST_THRESH = 0.25

    for k, v in slice_map.items():
        w = v[1] - v[0]
        if w >= global_width - filter_radius:
            if v[2] <= NORMAL_DIST_THRESH and v[3] <= NORMAL_DIST_THRESH:
                if v[2] <= GOOD_DIST_THRESH and v[3] <= GOOD_DIST_THRESH:
                    slice_widths_good.append(w)
                elif (v[2] <= GOOD_DIST_THRESH and v[3] <= NORMAL_DIST_THRESH) or (v[3] <= GOOD_DIST_THRESH and v[2] <= NORMAL_DIST_THRESH):
                    slice_widths_better.append(w)
                else:
                    slice_widths_normal.append(w)
            elif (v[2] <= GOOD_DIST_THRESH and v[3] <= BCK_DIST_THRESH) or (v[3] <= GOOD_DIST_THRESH and v[2] <= BCK_DIST_THRESH):
                slice_widths_backup.append(w)

    mean_width_good = np.average(np.array(slice_widths_good))
    mean_width_better = np.average(np.array(slice_widths_better))
    mean_width_normal = np.average(np.array(slice_widths_normal))

    ratio_local = 0.9
    ratio_global = 0.1

    if global_length <= 5:
        factor = 1.05
        ratio_local = 0.85 * factor
        ratio_global = 0.15 * factor
    if len(slice_widths_good) < 3 and len(slice_widths_better) == 0:
        ratio_local = 0.9
        ratio_local = 0.1

    if len(slice_widths_good) == 0 and len(slice_widths_better) == 0 and len(slice_widths_normal) == 0:
        if len(slice_widths_backup) == 0:
            return ret
        else:
            ret = ratio_local * \
                np.average(np.array(slice_widths_backup)) + \
                ratio_global * global_width
            return ret

    ratio = [4.0, 3.0, 2.0]
    if global_length <= 5:
        ratio[1] = 2.5

    if not slice_widths_good:
        ratio[0] = 0
        mean_width_good = 0

    if not slice_widths_normal:
        ratio[2] = 0
        mean_width_normal = 0

    if not mean_width_better:
        ratio[1] = 0
        mean_width_better = 0

    ratio_sum = sum(ratio)

    ret = ratio_local * (mean_width_good * ratio[0] + mean_width_better * ratio[1] +
                         mean_width_normal * ratio[2]) / ratio_sum + ratio_global * global_width
    return ret


def get_slice_start_pos(start, end, slice_num, slice_width):
    ret = []

    # 输入参数检查
    if slice_width <= 0:
        print(
            f"Error: input slice width {slice_width} invalid, less or equal to 0")
        return ret

    if slice_num <= 0:
        print(f"Error: input slice num {slice_num}, less than or equal to 0")
        return ret

    # 确保 start 小于或等于 end
    start_pos = min(start, end)
    end_pos = max(start, end)

    # 保留头尾的切片宽度
    if end_pos - start_pos < (slice_num + 2) * slice_width:
        print(f"Debug: not enough width to slice {slice_num} parts")
        return ret

    total_length = end_pos - start_pos - 2 * slice_width
    interval = total_length / slice_num

    for i in range(slice_num + 1):
        pos = start_pos + slice_width + i * interval
        ret.append(pos)

    return ret


def is_dual_lidar(points):
    return points.frame_ns_start == points.frame_ns_end


def point_in_bbox(contour, point):
    """检查点是否在多边形内部"""
    tmp_c = np.array(contour, dtype=np.float32)
    return cv2.pointPolygonTest(tmp_c, (np.float32(point[0]), np.float32(point[1])), False) >= 0


def get_inside_cloud(points, removed_indices, box_corner_pts):
    cloud_out = []

    if len(points) == 0 or len(removed_indices) == 0:
        return cloud_out

    # 计算轮廓点
    contour = []
    for i in range(len(box_corner_pts)):
        point2d = [box_corner_pts[i][1] *
                   10000, -box_corner_pts[i][2] * 10000]
        contour.append(point2d)

    n = len(points)

    # 遍历被移除的索引并检查每个点是否在边界框内
    for index in removed_indices:
        # 检查索引是否在有效范围内
        if index < 0 or index >= n:
            print(f"Error! Out of boundary: {index}")
            return cloud_out

        point_temp = [
            points[index][1] * 10000,
            -points[index][2] * 10000
        ]

        if point_in_bbox(contour, point_temp):
            cloud_out.append(points[index])

    return np.array(cloud_out)


def compute_boxes(box_points):
    if len(box_points) == 0:
        box_size_vec = [0, 0, 0]
        return [0, 0, 0], None

    # 转换到 OpenCV 坐标系并缩放
    pointcloud_in_cv = []
    for pt in box_points:
        p = (pt[1] * 10000, -pt[2] * 10000)
        pointcloud_in_cv.append(p)

    # 计算最小外接矩形
    rect = cv2.minAreaRect(np.array(pointcloud_in_cv, dtype=np.float32))

    # 计算角点
    box_corner_pts = np.zeros((4, 3), dtype=np.float32)
    corner_pts = cv2.boxPoints(rect)
    for i in range(4):
        box_corner_pts[i] = [0.0, corner_pts[i][0] /
                             10000.0, -corner_pts[i][1] / 10000.0]

    # 计算长宽高
    length = max(rect[1]) / 10000.0
    width = min(rect[1]) / 10000.0

    zmin = float('inf')
    zmax = float('-inf')

    for pt in box_points:
        zmin = min(zmin, pt[0])
        zmax = max(zmax, pt[0])

    height = zmax - zmin

    box_size_vec = [length, width, height]
    box_center_pt = [0.5 * (zmin + zmax), rect[0][0] /
                     10000.0, -rect[0][1] / 10000.0]
    return box_size_vec, box_corner_pts, box_center_pt


def get_rearview_removed_width(box, points, box_length, box_width, box_center_pt, filter_radius, box_width_with_all_pc):
    if len(points) < 100:
        return box_width_with_all_pc
    if box.width < 1.0:
        return box_width_with_all_pc

    trimed_length = 0.0
    if box_length < 3.0:
        trimed_length = box_length * 0.5
    elif box_length < 6:
        trimed_length = box_length * 2.0
    elif box_length < 12:
        trimed_length = 3.0
    else:
        trimed_length = 4.0

    trimed_max_z_in_box_coord = box_length * 0.5 - trimed_length

    angle = np.pi / 180.0 * (360.0 - box.spindle / 100.0)

    cos_t = np.cos(angle)
    sin_t = np.sin(angle)
    center_y = box_center_pt[1]
    center_z = box_center_pt[2]
    SLICE_NUM = 10
    SLICE_WIDTH = 0.2

    if box_length >= MIN_BOX_LENGTH_TO_SLICE and box_length <= 12:
        SLICE_NUM = 6
    elif box_length > 12 and box_length <= 15:
        SLICE_NUM = 10
    elif box_length > 15:
        SLICE_NUM = 15

    start_slice_pos = []

    if box_length >= MIN_BOX_LENGTH_TO_SLICE:
        start_slice_pos = get_slice_start_pos(
            -box_length * 0.5, trimed_max_z_in_box_coord, SLICE_NUM, SLICE_WIDTH)
    slice_max_min_y = {}
    slice_max_min_pt_index = {}
    y_in_box_coord = 0.0
    z_in_box_coord = 0.0
    const_temp_y = -cos_t * center_y - sin_t * center_z
    const_temp_z = sin_t * center_y - cos_t * center_z
    pc_trimed = []
    for i in range(len(points)):
        pt = points[i]
        # 坐标变换
        y_in_box_coord = cos_t * pt[1] + sin_t * pt[2] + const_temp_y
        z_in_box_coord = -sin_t * pt[1] + cos_t * pt[2] + const_temp_z

        if z_in_box_coord <= trimed_max_z_in_box_coord:
            pc_trimed.append(pt)
            if start_slice_pos:
                for j in range(len(start_slice_pos)):
                    if start_slice_pos[j] <= z_in_box_coord <= start_slice_pos[j] + SLICE_WIDTH:
                        if j in slice_max_min_y:
                            # 存在于字典中，仅更新 min_z 和 max_z
                            if y_in_box_coord < slice_max_min_y[j][0]:
                                slice_max_min_y[j][0] = y_in_box_coord
                                slice_max_min_pt_index[j][0] = i

                            if y_in_box_coord > slice_max_min_y[j][1]:
                                slice_max_min_y[j][1] = y_in_box_coord
                                slice_max_min_pt_index[j][1] = i
                        else:
                            # 初始化第一个元素
                            slice_max_min_y[j] = [
                                y_in_box_coord, y_in_box_coord]
                            slice_max_min_pt_index[j] = [i, i]
    if len(pc_trimed) < 100:
        return box_width_with_all_pc

    box_size_vec, width_box_corner_pts_, box_center_pt_ = compute_boxes(
        pc_trimed)
    line_1 = []
    line_2 = []
    max_min_lines = []
    if len(slice_max_min_y) > 0:
        dist1 = calc_two_points_dist(
            width_box_corner_pts_[0], width_box_corner_pts_[1])
        dist2 = calc_two_points_dist(
            width_box_corner_pts_[1], width_box_corner_pts_[2])
        if dist1 >= dist2:
            line_1 = [
                width_box_corner_pts_[0], width_box_corner_pts_[1]]
            line_2 = [
                width_box_corner_pts_[2], width_box_corner_pts_[3]]
        else:
            line_1 = [
                width_box_corner_pts_[1], width_box_corner_pts_[2]]
            line_2 = [
                width_box_corner_pts_[0], width_box_corner_pts_[3]]
        boundary_y = cos_t * line_1[0][1] + sin_t * line_1[0][2] + const_temp_y
        if boundary_y > 0:
            max_min_lines = [line_2, line_1]
        else:
            max_min_lines = [line_1, line_2]
        slice_max_min_y = calc_slice_endpoint_dist_to_edge(points, slice_max_min_pt_index,
                                                           max_min_lines, box_size_vec[0],
                                                           slice_max_min_y)
        width_trimed = update_with_multi_slice_width(
            slice_max_min_y, box_width, box_length, filter_radius)
    else:
        width_trimed = box_size_vec[1]
    return width_trimed


def radius_filter(points, radius, min_pts):
    # x y z
    points = np.asarray(points)

    # 创建一个掩码以标记保留的点
    mask = np.zeros(points.shape[0], dtype=bool)

    # 遍历每个点并计算其邻近点数
    for i in range(len(points)):
        # 计算当前点到所有点的距离
        distances = np.linalg.norm(points - points[i], axis=1)
        # 找到在半径内的点
        neighbors = np.sum(distances < radius)

        # 如果邻近点数满足条件，则标记为保留
        if neighbors >= min_pts:
            mask[i] = True

    # 使用掩码来过滤点云
    filtered_points = points[mask]
    removed_indices = np.where(~mask)[0]

    return filtered_points, removed_indices


def proc(points, boxes):
    for box in boxes:
        if box.track_id != 566:
            continue
        box_size_vec = get_updated_box_size_(points, box)
        return box_size_vec


def single_lidar(points, box, params=None):
    point_cloud_single_box = []
    for i in box.point_index:
        point_cloud_single_box.append(
            PointCloud2.create_from_protobuf(points.point[i]))
    len_cmp_pts = []
    if calc_length_compensate_pts(point_cloud_single_box, box, len_cmp_pts):
        print("failed to compensate length pt")

    # trim car mirror
    non_mirror_pc = []
    mirror_pc = []
    if clac_width_trim_pc(point_cloud_single_box, len_cmp_pts, non_mirror_pc, mirror_pc, box):
        return -1

    # update lenght and width by invoke compute_box_with_l_shape_
    save_points_to_pcd(non_mirror_pc, 'non_mirror_pc.pcd')
    # out = compute_box_with_l_shape_(non_mirror_pc)
    out = compute_box_with_l_shape_2(non_mirror_pc)

    return [out['length'], out['width']]


def azimuth_to_slope(azimuth):
    # 将方位角转换为与x轴的夹角
    if 0 < azimuth < 90:
        angle_with_x_axis = azimuth
    elif 90 < azimuth < 270:
        angle_with_x_axis = 270 - azimuth
    elif 270 < azimuth < 360:
        angle_with_x_axis = 450 - azimuth

    # 计算斜率
    slope = math.tan(math.radians(angle_with_x_axis))
    return slope


def slice_width(points, spindle, bounds, slipe_w=2):
    # y = k * x + b
    # spindle = 90 verticle x
    # spindle = 180 verticle y
    out = {}
    angle = spindle / 100.0
    slope_ = None

    if angle == 90 or angle == 270:
        i = bounds['min_y'][1]
        while i <= bounds['max_y'][1]:
            end = i + slipe_w
            out[f'{round(i, 2) * 100}-{round(end, 2) * 100}'] = {
                'range': [i, end],
                'points': [],
                'min_z': 1e9,
                'max_z': -1 * 1e9
            }
            i = end
    elif angle == 0 or angle == 180:
        i = bounds['min_z'][2]
        while i <= bounds['max_z'][2]:
            end = i + slipe_w
            out[f'{round(i, 2) * 100}-{round(end, 2) * 100}'] = {
                'range': [i, end],
                'points': [],
                'min_z': 1e9,
                'max_z': -1 * 1e9
            }
            i = end
    else:
        slope_ = azimuth_to_slope(angle)
        # k = -1 / k
        b1 = bounds['min_y'][1] - slope_ * bounds['min_y'][0]
        b2 = bounds['max_y'][1] - slope_ * bounds['max_y'][0]
        b3 = bounds['min_z'][1] - slope_ * bounds['min_z'][0]
        b4 = bounds['max_z'][1] - slope_ * bounds['max_z'][0]
        b_min = min([b1, b2, b3, b4])
        b_max = max([b1, b2, b3, b4])
        b1 = b_min
        while b1 <= b_max:
            b2 = b1 + slipe_w
            out[f'{round(b1, 2) * 100}-{round(b2, 2) * 100}'] = {
                'range': [b1, b2],
                'points': []
            }
            b1 = b2
    for pt in points:
        if angle == 90 or angle == 270:
            tmp_y = -pt[1]
            start = int((tmp_y - bounds['min_y'][0]) /
                        slipe_w) * slipe_w + bounds['min_y'][0]
            end = start + slipe_w
            key = f'{round(start, 2) * 100}-{round(end, 2) * 100}'
            out[key]['min_z'] = min(out[key]['min_z'], pt[2])
            out[key]['max_z'] = max(out[key]['max_z'], pt[2])
        elif angle == 0 or angle == 180:
            tmp_y = pt[2]
            start = int((tmp_y - bounds['min_z'][1]) /
                        slipe_w) * slipe_w + bounds['min_z'][1]
            end = start + slipe_w
            key = f'{round(start, 2) * 100}-{round(end, 2) * 100}'
            out[key]['min_z'] = min(out[key]['min_z'], pt[1])
            out[key]['max_z'] = max(out[key]['max_z'], pt[1])
        else:
            tmp_b = pt[2] - slope_ * (-1 * pt[1])
            start = int((tmp_b - b_min) / slipe_w) * slipe_w + b_min
            end = start + slipe_w
            key = f'{round(start, 2) * 100}-{round(end, 2) * 100}'
        if key not in out:
            continue
        out[key]['points'].append([-1 * pt[1], pt[2]])

    width = []
    points_threshold = min(80, len(points) / len(out))
    for k, v in out.items():
        if angle == 90 or angle == 270:
            width.append(abs(v['max_z'] - v['min_z']))
        else:
            valid_points = np.array(v['points'], dtype=np.float32)
            if len(valid_points) < points_threshold:
                continue
            max_distance = 0
            rect = cv2.minAreaRect(valid_points)
            rec_slope = np.tan(rect[2] * np.pi / 180.0)
            if (rec_slope - slope_) > (rec_slope + 1 / slope_):
                max_distance = rect[1][1]
            else:
                max_distance = rect[1][0]

            # width_tmp = [i for i in rect[1] if i > slipe_w]
            # if width_tmp:
            #     max_distance = min(width_tmp)
            # for p1, p2 in combinations(valid_points, 2):
            #     diff_vector = p2 - p1
            #     projection_length = abs(
            #         np.dot(diff_vector, normalized_direction))
            #     if projection_length > max_distance:
            #         max_distance = projection_length
                # write_pcd(v['points'], k)
                width.append(round(max_distance, 4))
    if not width:
        width.append(0)
    return width


def write_pcd(points, k):
    new_p = []
    for i in points:
        new_p.append([0, -1 * i[0], i[1]])
    new_p = np.array(new_p)
    # 写pcd文件
    pp = o3d.geometry.PointCloud()
    pp.points = o3d.utility.Vector3dVector(new_p)
    o3d.io.write_point_cloud(f"{k}-output.pcd", pp)


def write_pcd_2(points, k):
    new_p = np.array(points)
    # 写pcd文件
    pp = o3d.geometry.PointCloud()
    pp.points = o3d.utility.Vector3dVector(new_p)
    o3d.io.write_point_cloud(f"{k}-output.pcd", pp)


def cal_index(x, y, grip, min_x, min_y):
    return (x - min_x) // grip[0] + (y - min_y) // grip[1]


def filter_by_height(points, height_threshold, grip, min_x, min_y, min_pts=10):
    grips = {}
    for pt in points:
        index = cal_index(-pt[1], pt[2], grip, min_x, min_y)
        pt.append(index)
        if index not in grips:
            grips[index] = [pt[0]]
        else:
            grips[index].append(pt[0])
    # 1. filter by height
    points = np.array(points)
    for k, v in grips.items():
        if len(v) < min_pts:
            continue
        min_x = np.min(v)
        max_x = np.max(v)
        if max_x - min_x < height_threshold:
            points = points[points[:, 3] != k]
    return points


def dual_lidar(points, box, params={}):
    ret = [box.length, box.width]
    is_trim_useless_pts = getattr(params, 'is_trim_useless_pts', True)
    radius = getattr(params, 'radius', 0.3)
    min_pts = getattr(params, 'min_pts', 10)
    update_length = getattr(params, 'update_length', True)
    update_width = getattr(params, 'update_width', True)
    update_height = getattr(params, 'update_height', True)

    max_z = -1 * sys.float_info.max
    min_z = sys.float_info.max

    # 1. get point cloud for dual box
    for i in box.point_index:
        pt = points.point[i]
        if max_z < pt.z:
            max_z = pt.z
        if min_z > pt.z:
            min_z = pt.z

    # label far away point to trim
    for i in box.point_index:
        pt = points.point[i]
        if box.length > MIN_LENGTH_TO_TRIM and \
                (pt.z - min_z) / (max_z - min_z) > RATIO_FOR_FAR_AWAY:
            points.point[i].flags = 100
        else:
            points.point[i].flags = -100

    # original dual lidar logic, no change
    pcl_box_pc = {}
    pcl_box_pc['is_dense'] = True
    pcl_box_pc['width'] = len(box.point_index)
    pcl_box_pc['height'] = 1
    pcl_box_pc['points'] = []
    pcl_box_withzx_pc = copy.deepcopy(pcl_box_pc)

    bounds = {
        'min_y': [sys.float_info.max, 0],
        'max_y': [-1 * sys.float_info.max, 0],
        'min_z': [0, sys.float_info.max],
        'max_z': [0, -1 * sys.float_info.max],
    }
    center_x = 0
    center_y = 0

    useless_pt_cnt = 0
    for i in box.point_index:
        pt = points.point[i]
        if pt.flags == 100 and is_trim_useless_pts:
            useless_pt_cnt += 1
            continue
        point_tmp = [0.0, pt.y, pt.z]
        point_tmp_xy = [-pt.y, pt.z]
        center_x = center_x + point_tmp_xy[0]
        center_y = center_y + point_tmp_xy[1]
        if point_tmp_xy[0] > bounds['max_y'][0]:
            bounds['max_y'] = point_tmp_xy
        if point_tmp_xy[0] < bounds['min_y'][0]:
            bounds['min_y'] = point_tmp_xy
        if point_tmp_xy[1] > bounds['max_z'][1]:
            bounds['max_z'] = point_tmp_xy
        if point_tmp_xy[1] < bounds['min_z'][1]:
            bounds['min_z'] = point_tmp_xy
        pcl_box_pc['points'].append(point_tmp)
        pcl_box_withzx_pc['points'].append([pt.x, pt.y, pt.z])
    pcl_box_pc['width'] = pcl_box_pc['width'] - useless_pt_cnt
    pcl_box_withzx_pc['width'] = pcl_box_withzx_pc['width'] - useless_pt_cnt
    center_x = center_x / pcl_box_pc['width']
    center_y = center_y / pcl_box_pc['width']

    # downsample
    point_cloud_sampled_ = copy.deepcopy(pcl_box_pc)
    if len(pcl_box_pc['points']) > 4000:
        leaf_size = 0.1
        pcl_box_pc_tmp = np.array(pcl_box_pc['points'])
        cloud_filtered = o3d.geometry.PointCloud()
        cloud_filtered.points = o3d.utility.Vector3dVector(pcl_box_pc_tmp)
        point_cloud_sampled_tmp = cloud_filtered.voxel_down_sample(leaf_size)
        point_cloud_sampled_['points'] = np.asarray(
            point_cloud_sampled_tmp.points)
        point_cloud_sampled_['width'] = len(point_cloud_sampled_['points'])

    if len(point_cloud_sampled_['points']) > 1000:
        if box.length < 9.0:
            radius_tmp = 0.15
            min_pts_tmp = 5
            filtered_tmp, removed_indices = radius_filter(
                point_cloud_sampled_['points'].copy(), radius_tmp, min_pts_tmp)

            if len(filtered_tmp) == 0:
                point_cloud_filtered = point_cloud_sampled_.copy()
            else:
                box_size_vec_tmp, box_corner_pts_tmp, box_center_pt = compute_boxes(
                    filtered_tmp)
                removed_valid_cloud = get_inside_cloud(
                    point_cloud_sampled_['points'], removed_indices, box_corner_pts_tmp)
                point_cloud_filtered = np.vstack(
                    (filtered_tmp, removed_valid_cloud))
        else:
            point_cloud_filtered, removed_indices = radius_filter(
                point_cloud_sampled_['points'], radius, min_pts)
    else:
        point_cloud_filtered = point_cloud_sampled_['points']

    write_pcd_2(point_cloud_filtered, 'filter')
    # for width trimed
    origin_length = box.length
    origin_width = box.width
    if (update_length and len(point_cloud_filtered) > 100):
        box_size_vec_length, length_box_corner_pts_, box_center_pt = compute_boxes(
            point_cloud_filtered)
        origin_length = box_size_vec_length[0]
        origin_width = box_size_vec_length[1]
        ret[0] = box_size_vec_length[0]
        if ((box.position_z < -20.0 or box.position_z > 70.0) and
                abs(ret[0] - box.length) > 3.0):
            ret[0] = box.length

    if (update_width):
        start_time = time.time()
        ret[1] = get_rearview_removed_width(
            box, point_cloud_filtered, origin_length, origin_width, box_center_pt, radius, box_size_vec_length[1])
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"get_rearview_removed_width time: {elapsed_time:.6f} seconds {ret[1]}")
        start_time = time.time()
        tmp = slice_width(point_cloud_filtered, box.spindle,
                          bounds, 1)
        ret[1] = np.median(np.array(tmp))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"slice_width time: {elapsed_time:.6f} seconds {ret[1]}")
        if ((box.position_z < -20.0 or box.position_z > 70.0) and
                abs(ret[1] - box.width) > 1.0):
            ret[1] = box.width
    if update_height:
        pcl_box_withzx_pc, removed_indices = radius_filter(
            pcl_box_withzx_pc['points'], radius, min_pts)
        # ret[2] = box_height_refiner->box_refine_height_with_pts(pcl_box_withzx_pc,box);
    return ret


def calc_closeness_criterion(C_1, C_2):
    min_c_1 = min(C_1)
    max_c_1 = max(C_1)
    min_c_2 = min(C_2)
    max_c_2 = max(C_2)

    D_1 = [min(max_c_1 - c_1_element, c_1_element - min_c_1)
           for c_1_element in C_1]
    D_2 = [min(max_c_2 - c_2_element, c_2_element - min_c_2)
           for c_2_element in C_2]

    d_min = 0.05
    d_max = 0.5
    beta = 0
    for i in range(len(D_1)):
        d = min(max(min(D_1[i], D_2[i]), d_min), d_max)
        beta += d

    return 1 / beta


def compute_box_(points):
    pts = []
    for point in points:
        pts.append([point.y, point.z])


def compute_box_with_l_shape_(points, init_heading=0, heading_search_range=90.0):
    max_angle = 90.0
    if (init_heading + heading_search_range) <= 90:
        max_angle = (init_heading + heading_search_range) / 180.0 * math.pi
    min_angle = 0.0
    if (init_heading - heading_search_range) >= 0:
        min_angle = (init_heading - heading_search_range) / 180.0 * math.pi

    angle_reso = math.pi / 180.0 / 4
    pt_num = len(points)
    pt_step = 1
    if (pt_num > 2000):
        pt_step = int(pt_num / 1000)

    max_score = 0
    opt_theta = 0

    theta = min_angle
    while theta < max_angle:
        e_1 = np.array([math.cos(theta), math.sin(theta)])
        e_2 = np.array([-math.sin(theta), math.cos(theta)])
        c_1 = []
        c_2 = []
        for i in range(0, pt_num, pt_step):
            point = points[i]
            c_1.append(point.y * e_1[0] + point.z * e_1[1])
            c_2.append(point.y * e_2[0] + point.z * e_2[1])
        score = calc_closeness_criterion(c_1, c_2)
        if score > max_score:
            max_score = score
            opt_theta = theta
        theta += angle_reso
    e_1_star = np.array([math.cos(opt_theta), math.sin(opt_theta)])
    e_2_star = np.array([-math.sin(opt_theta), math.cos(opt_theta)])
    c_1_star = []
    c_2_star = []
    for point in points:
        # c_1_star.append(point.y * e_1_star[0] + point.z * e_1_star[1])
        # c_2_star.append(point.y * e_2_star[0] + point.z * e_2_star[1])
        pral = point.y * e_1_star[0] + point.z * e_1_star[1]
        verti = point.y * e_2_star[0] + point.z * e_2_star[1]
        c_1_star.append([pral, point.x, point.y, point.z])
        c_2_star.append([verti, point.x, point.y, point.z])
    c_1_star = pd.DataFrame(c_1_star, columns=['p', 'x', 'y', 'z'])
    c_2_star = pd.DataFrame(c_2_star, columns=['p', 'x', 'y', 'z'])
    # min_c_1_starr = min(c_1_star)
    # max_c_1_starr = max(c_1_star)
    # min_c_2_starr = min(c_2_star)
    # max_c_2_starr = max(c_2_star)
    min_c_1_starr = np.min(c_1_star['p'])
    max_c_1_starr = np.max(c_1_star['p'])
    min_c_2_starr = np.min(c_2_star['p'])
    max_c_2_starr = np.max(c_2_star['p'])
    a_1 = e_1_star[0]
    b_1 = e_1_star[1]
    c_1 = min_c_1_starr
    c_2 = min_c_2_starr
    c_3 = max_c_1_starr
    c_4 = max_c_2_starr
    intersection_y_1 = a_1 * c_1 - b_1 * c_2
    intersection_z_1 = a_1 * c_2 + b_1 * c_1
    intersection_y_2 = a_1 * c_3 - b_1 * c_4
    intersection_z_2 = a_1 * c_4 + b_1 * c_3
    diagonal_vec = [intersection_y_1 - intersection_y_2,
                    intersection_z_1 - intersection_z_2]
    length = abs(e_1_star.dot(diagonal_vec))
    width = abs(e_2_star.dot(diagonal_vec))
    size_swap = False
    if (length < width):
        temp = length
        length = width
        width = temp
        size_swap = True
    update_box_z = (intersection_z_1 + intersection_z_2) / 2.0
    update_box_y = (intersection_y_1 + intersection_y_2) / 2.0
    opt_theta = opt_theta - math.pi / 2.0
    if (size_swap):
        opt_theta = opt_theta + math.pi / 2.0
    return {
        'update_box_y': update_box_y,
        'update_box_z': update_box_z,
        'width': width,
        'length': length
    }


def compute_box_with_l_shape_2(points):
    points_2d = []
    cy = 0
    cz = 0

    for point in points:
        pt = [-point.y * 10000, point.z * 10000]
        cy = cy - point.y
        cz = cz + point.z
        points_2d.append(pt)
    cy = cy / len(points_2d)
    cz = cz / len(points_2d)
    points_2d = np.array(points_2d, dtype=np.int32)
    rect = cv2.minAreaRect(points_2d)
    box = cv2.boxPoints(rect)
    box = np.array(box)
    dx = box[1][0] - box[0][0]
    dy = box[1][1] - box[0][1]
    slope = 1.0 * dy / dx
    length = cal_length_2(slope, points_2d, box[1][0], box[1][1], 7, 4, stat=1)
    slope = -1.0 * dx / dy
    width = cal_length_2(slope, points_2d, box[1][0], box[1][1], 16, 2, stat=2)

    size_swap = False
    if (length < width):
        temp = length
        length = width
        width = temp
        size_swap = True
    opt_theta = opt_theta - math.pi / 2.0
    if (size_swap):
        opt_theta = opt_theta + math.pi / 2.0
    return {
        'width': width,
        'length': length
    }


def cal_length_2(slope, points, cy, cz, step=15, internal=2, stat=1):
    cz = cz / 10000.0
    cy = cy / 10000.0
    # 计算平行线的单位向量
    direction_vector = np.array([1, slope])  # 沿着平行线的方向
    normalized_direction = direction_vector / \
        np.linalg.norm(direction_vector)  # 单位向量
    b1 = cz - slope * cy
    max_distance_l = []
    farthest_points_l = []
    while b1 < cz - slope * cy + step:
        b2 = b1 + internal
        valid_points = []
        for point in points:
            x, y = point[0] / 10000.0, point[1] / 10000.0
            if (slope * x + b1 < y <= slope * x + b2):
                valid_points.append([x, y])

        valid_points = np.array(valid_points)
        if len(valid_points) < 100:
            b1 = b2
            continue
        max_distance = 0
        farthest_points = None
        for p1, p2 in combinations(valid_points, 2):
            diff_vector = p2 - p1
            projection_length = np.dot(diff_vector, normalized_direction)
            if projection_length > max_distance:
                max_distance = projection_length
                farthest_points = (p1, p2)
        max_distance_l.append(round(max_distance, 4))
        farthest_points_l.append(farthest_points)
        save_valid_points(
            valid_points, f"{int(b1)}-{int(max_distance * 10000)}.pcd")
        b1 = b2
    if stat == 1:
        return max(max_distance_l)
    return np.median(np.array(max_distance_l))


def cal_length(angle_radis, points, cy, cz):
    slope = math.tan(angle_radis)

    # 计算平行线的单位向量
    direction_vector = np.array([1, slope])  # 沿着平行线的方向
    normalized_direction = direction_vector / \
        np.linalg.norm(direction_vector)  # 单位向量
    b1 = cz + slope * cy - 15
    max_distance_l = []
    farthest_points_l = []
    while b1 < cz + slope * cy + 15:
        b2 = b1 + 1
        valid_points = []
        for point in points:
            x = -1 * point.y
            y = point.z
            if (slope * x + b1 < y <= slope * x + b2):
                valid_points.append([x, y])

        valid_points = np.array(valid_points)
        if len(valid_points) < 100:
            b1 = b2
            continue
        max_distance = 0
        farthest_points = None
        for p1, p2 in combinations(valid_points, 2):
            diff_vector = p2 - p1
            projection_length = np.dot(diff_vector, normalized_direction)
            if projection_length > max_distance:
                max_distance = projection_length
                farthest_points = (p1, p2)
        max_distance_l.append(round(max_distance, 4))
        farthest_points_l.append(farthest_points)
        save_valid_points(
            valid_points, f"{int(b1)}-{int(max_distance * 10000)}.pcd")
        b1 = b2
    return max(max_distance_l)


def save_valid_points(points, k):
    valid_points = []
    for point in points:
        pt = PointXYZI(x=0, y=-1*point[0], z=point[1], intensity=0)
        valid_points.append(pt)
    save_points_to_pcd(valid_points, k)
    return valid_points


def type_map(dtype):
    if dtype == np.float32:
        return ['F', 4]
    if dtype == np.float64:
        return ['F', 8]
    if dtype == np.uint16:
        return ['U', 2]
    if dtype == np.uint32:
        return ['U', 4]
    if dtype == np.int32:
        return ['I', 4]
    return ['F', 4]


def save_points_to_pcd(points, out_path):
    fields = 'FIELDS x y z intensity\n'
    size = 'SIZE 4 4 4 4\n'
    type_ = 'TYPE F F F F\n'
    count = 'COUNT 1 1 1 1\n'
    with open(out_path, 'w') as file:
        file.write("# .PCD v.7 - Point Cloud Data file format\n")
        file.write(fields)
        file.write(size)
        file.write(type_)
        file.write(count)
        file.write("WIDTH " + str(len(points)) + "\n")
        file.write("HEIGHT 1\n")
        file.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        file.write("POINTS " + str(len(points)) + "\n")
        file.write("DATA ascii\n")
        for pt in points:
            file.write(f"{pt.x} {pt.y} {pt.z} {pt.intensity}\n")


def clac_width_trim_pc(points, len_com_pts, non_mirror_pc, mirror_pc, box):
    mirror_pos_x = box.height * 0.66
    mirror_pos_x_thresh = box.height * 0.2
    unfilter_dist_forward_ratio = 0.0
    unfilter_dist_backward_ratio = 0.4
    filter_dist = 0.3
    is_drive_in = True   # z decrease
    if box.speed_z > 0:
        is_drive_in = False

    if len(points) < 100:
        return -1

    UNMARK = 10
    left_right_cor_pc = []
    mean_x = 0
    mean_z = 0
    min_x = sys.float_info.max
    for i in range(len(points)):
        pt = points[i]
        mean_x = mean_x + pt.x
        mean_z = mean_z + pt.z
        min_x = min(pt.x, min_x)
        left_right_cor_pc.append(
            PointXYZI(x=pt.x, y=pt.y, z=pt.z, intensity=UNMARK))
    mean_x = mean_x / len(points)
    mean_z = mean_z / len(points)
    mirror_pos_x = mirror_pos_x + min_x

    cv_box = []
    cvpts = []
    for i in range(len(points)):
        pt = points[i]
        if pt.x < mean_x:
            continue
        cvpts.append([pt.y, pt.z])
    cvpts_np = np.array(cvpts, dtype=np.float32)
    cvrect = cv2.minAreaRect(cvpts_np)
    vertices = cv2.boxPoints(cvrect)

    for i in range(4):
        pt = vertices[i]
        cv_box.append(PointXYZI(x=0, y=pt[0], z=pt[1], intensity=i))

    # Step1.Calculate box corners by border & spindle
    box_corners = []
    box_corners.append(cv_box[0])
    box_corners.append(cv_box[1])
    box_corners.append(cv_box[2])
    box_corners.append(cv_box[3])

    p1 = box_corners[0]
    p2 = box_corners[1]
    p3 = box_corners[2]
    p4 = box_corners[3]

    cv_length = abs(p2.z - p1.z)
    cv_width = abs(p1.z - p4.z)
    if (cv_length < cv_width):
        tmp = p1
        p1 = p3
        p3 = tmp
    if p1.z >= p2.z:
        tmp = p2
        p2 = p1
        p1 = tmp
        tmp = p4
        p4 = p3
        p3 = tmp

    left = Line([p1.y, p1.z], [p2.y, p2.z])
    front = Line([p1.y, p1.z], [p4.y, p4.z])
    back = Line([p2.y, p2.z], [p3.y, p3.z])
    right = Line([p4.y, p4.z], [p3.y, p3.z])

    CLOSE_TO_TOP_DOWN = 15
    DRIVE_IN_COLOSE_TO_FOWWARD = 40
    DIRVE_IN_COLOSE_TO_BACKEND = 45
    DRIVE_AWAY_COLOSE_TO_FOWWARD = 30
    DRIVE_AWAW_COLOSE_TO_BACKEND = 35
    CURV_BASE_VAL = 25
    LEN_COM_PT = 50
    MAX_SCAN_LINE = 200
    # Calculate distance to 6 borders(top buttom left right front backend)
    for i in range(len(left_right_cor_pc)):
        pt = left_right_cor_pc[i]
        if abs(pt.x - mirror_pos_x) > mirror_pos_x_thresh:
            pt.intensity = CLOSE_TO_TOP_DOWN
            continue
        dist_2_f = front.calc_dist_with_pt(pt.y, pt.z)
        dist_2_b = back.calc_dist_with_pt(pt.y, pt.z)
        if not is_drive_in:
            tmp = dist_2_b
            dist_2_b = dist_2_f
            dist_2_f = tmp
            if dist_2_b < dist_2_f and dist_2_b < box.length * unfilter_dist_backward_ratio:
                pt.intensity = DRIVE_AWAW_COLOSE_TO_BACKEND
                continue
            if dist_2_f < dist_2_b and dist_2_f < box.length * unfilter_dist_forward_ratio:
                pt.intensity = DRIVE_AWAY_COLOSE_TO_FOWWARD
                continue
        else:
            if dist_2_f < dist_2_b and dist_2_f < box.length * unfilter_dist_forward_ratio:
                pt.intensity = DRIVE_IN_COLOSE_TO_FOWWARD
                continue
            if dist_2_f > dist_2_b and dist_2_b < box.length * unfilter_dist_backward_ratio:
                pt.intensity = DIRVE_IN_COLOSE_TO_BACKEND
                continue

        dist_2_l = left.calc_dist_with_pt(pt.y, pt.z)
        dist_2_r = right.calc_dist_with_pt(pt.y, pt.z)
        pt.intensity = min(dist_2_r, dist_2_l)

    # Filter ROI region by distance to border
    roi_pc_idx = []
    non_roi_pc = []
    for i in range(len(left_right_cor_pc)):
        pt = left_right_cor_pc[i]
        if (pt.intensity < filter_dist):
            roi_pc_idx.append(i)
            mirror_pc.append(
                PointXYZI(x=pt.x, y=pt.y, z=pt.z, intensity=pt.intensity))
        else:
            non_roi_pc.append(
                PointXYZI(x=pt.x, y=pt.y, z=pt.z, intensity=pt.intensity))

    # add length compensate point
    for i in range(len(len_com_pts)):
        len_com_pt = len_com_pts[i]
        cmp_2_left_dsit = left.calc_dist_with_pt(len_com_pt.y, len_com_pt.z)
        cmp_2_right_dsit = right.calc_dist_with_pt(len_com_pt.y, len_com_pt.z)
        p1_com_pt_vec = np.array([len_com_pt.y - p1.y, len_com_pt.z - p1.z])
        p4_com_pt_vec = np.array([len_com_pt.y - p4.y, len_com_pt.z - p4.z])
        p1_com_pt_vec = p1_com_pt_vec / np.linalg.norm(p1_com_pt_vec)
        p4_com_pt_vec = p4_com_pt_vec / np.linalg.norm(p4_com_pt_vec)
        cross_left = p1_com_pt_vec[0] * \
            left.vec[1] - p1_com_pt_vec[1] * left.vec[0]
        cross_right = p4_com_pt_vec[0] * \
            right.vec[1] - p4_com_pt_vec[1] * right.vec[0]
        if (cross_left * cross_right < 0):
            non_roi_pc.append(
                PointXYZI(x=len_com_pt.x, y=len_com_pt.y, z=len_com_pt.z, intensity=LEN_COM_PT + min(cmp_2_left_dsit, cmp_2_right_dsit)))
            break

    # Reorgnize ROI region pointCloud by scan id
    scan_line_hash = {}
    for idx in roi_pc_idx:
        pt = points[idx]
        scanid = pt.scan_id
        if scanid not in scan_line_hash:
            scan_line_hash[scanid] = [idx]
        else:
            scan_line_hash[scanid].append(idx)

    # init scan line pc
    min_scan_idx = sys.float_info.max
    max_scan_idx = 0
    scan_line_pc = []
    for k, v in scan_line_hash.items():
        tmp = []
        for idx in v:
            pt = points[idx]
            tmp.append(PointXYZI(x=pt.x, y=pt.y,
                       z=pt.z, intensity=pt.scan_idx))
            if (pt.scan_idx > max_scan_idx):
                max_scan_idx = pt.scan_idx
            if (pt.scan_idx < min_scan_idx):
                min_scan_idx = pt.scan_idx
        sorted_tmp = sorted(tmp, key=lambda pts: sort_by_intensity(pts))
        scan_line_pc.append(sorted_tmp)

    scan_line_pc = sorted(
        scan_line_pc, key=lambda pts: sort_by_avg_distance(pts))

    # Calculate curve features
    curve_statistics_vec = []
    curve_idx_map = {}
    min_pt_per_line = 5
    curve_calc_nerghbour = 2
    for i in range(0, len(scan_line_pc)):
        if len(scan_line_pc[i]) < min_pt_per_line:
            continue
        pc_per_line = []
        for j in range(0, len(scan_line_pc[i])):
            pt = scan_line_pc[i][j]
            pc_per_line.append(pt)

        pt_size_per_line = len(pc_per_line)
        range_vec = []
        for pt in pc_per_line:
            range_vec.append(
                math.sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z))

        curve_vec = {}
        for k in range(curve_calc_nerghbour, pt_size_per_line - curve_calc_nerghbour):
            curve = 0
            for v in range(-curve_calc_nerghbour, curve_calc_nerghbour + 1):
                if v == 0:
                    continue
                curve = curve + range_vec[k + v]
                curve_vec[k] = (curve * curve)
        max = 0
        for k in range(curve_calc_nerghbour, pt_size_per_line - curve_calc_nerghbour):
            curve_statistics_vec.append(curve_vec[k])
            curve_idx_map[curve_vec[k]] = (i, k)
            if (curve_vec[k] > max):
                max = curve_vec[k]

    # Filter curve feautres
    mean_cruve = 0
    var_curve = 0
    for curve in curve_statistics_vec:
        mean_cruve = mean_cruve + curve
    mean_cruve = mean_cruve / len(curve_statistics_vec)
    for curve in curve_statistics_vec:
        var_curve = var_curve + \
            math.sqrt((curve - mean_cruve) * (curve - mean_cruve))
    var_curve = var_curve / len(curve_statistics_vec)

    # select non mirror pt in roi pc
    for k, iter in curve_idx_map.items():
        if k > (mean_cruve):
            continue
        yidx = iter[0]
        xidx = iter[1]
        pt = scan_line_pc[yidx][xidx]
        if left.calc_dist_with_pt(pt.y, pt.z) < filter_dist / 2 or \
                right.calc_dist_with_pt(pt.y, pt.z) < filter_dist / 2:
            continue
        non_roi_pc.append(
            PointXYZI(x=pt.x, y=pt.y, z=pt.z, intensity=CURV_BASE_VAL + min(left.calc_dist_with_pt(pt.y, pt.z),
                                                                            right.calc_dist_with_pt(pt.y, pt.z))))

    if non_roi_pc:
        for i in non_roi_pc:
            non_mirror_pc.append(i)
    return 0


def calc_length_compensate_pts(points, box, cmp_pts):
    min_pt_per_scan_line = 15
    min_roi_scan_line = 3
    pt_ratio_useful_per_scan_line = 0.25
    min_exp_pt_num = 3
    scale = 0.5
    buffer_exp_pt_size = 5

    box_cy = box.position_y
    box_cz = box.position_z

    cloud_size = len(points)
    if cloud_size < 100:
        return -1

    scan_line_hash = {}
    for i in range(0, cloud_size):
        pt = PointCloud2.create_from_protobuf(points[i])
        index = pt.scan_id
        if index in scan_line_hash:
            scan_line_hash[index].append([index, i])
        else:
            scan_line_hash[index] = [[index, i]]

    # scan line filter
    scan_line_pc = []
    for k, v in scan_line_hash.items():
        if len(v) < min_pt_per_scan_line:
            continue
        tmp = []
        for j in range(0, int(len(v) * pt_ratio_useful_per_scan_line)):
            idx = v[j][1]
            tmp.append(PointXYZI(x=points[idx].x, y=points[idx].y,
                       z=points[idx].z, intensity=points[idx].scan_id))
        scan_line_pc.append(tmp)

    # skip if rest scan line num < 3
    if len(scan_line_pc) < min_roi_scan_line:
        return -1

    # resort by  dist
    sorted_scan_line_pc = sorted(
        scan_line_pc, key=lambda pts: sort_by_avg_distance(pts))

    exp_pts = []
    for pts_per_scan_line in sorted_scan_line_pc:
        pt1 = pts_per_scan_line[0]
        pt2 = pts_per_scan_line[1]
        pt3 = pts_per_scan_line[2]
        if (pt1.z < pt2.z or pt2.z < pt3.z):
            continue
        len_pt1_pt2_m = math.sqrt((pt1.y - pt2.y) * (pt1.y - pt2.y) +
                                  (pt1.z - pt2.z) * (pt1.z - pt2.z))
        len_pt2_pt3_m = math.sqrt((pt2.y - pt3.y) * (pt2.y - pt3.y) +
                                  (pt2.z - pt3.z) * (pt2.z - pt3.z))
        if (len_pt1_pt2_m > box.length * 0.5 or len_pt2_pt3_m > box.length * 0.5):
            continue
        if (min(len_pt1_pt2_m, len_pt2_pt3_m) / max(len_pt1_pt2_m, len_pt2_pt3_m) < 0.5):
            continue

        offset_x = pt1.x - pt2.x
        offset_y = pt1.y - pt2.y
        offset_z = pt1.z - pt2.z

        if offset_y < 0:
            continue

        pt = PointXYZI(x=pt1.x + offset_x * scale, y=pt1.y +
                       offset_y * scale, z=pt1.z + offset_z * scale, intensity=pt1.intensity)
        exp_pts.append(pt)

    # skip if extrapolated pts num < 3
    exp_pt_size = len(exp_pts)
    if exp_pt_size < min_exp_pt_num:
        return -1

    dist_idx_map = []
    last_dist = 0
    for i in range(0, exp_pt_size):
        pt = exp_pts[i]
        dist = math.sqrt(pt.y * pt.y + pt.z * pt.z)
        # skip if not growing startforward
        if (dist < last_dist):
            continue
        dist_2_centroid = math.sqrt((pt.y - box_cy) * (pt.y - box_cy) +
                                    (pt.z - box_cz) * (pt.z - box_cz))
        if (dist_2_centroid > box.length * 0.9):
            continue
        last_dist = dist
        dist_idx_map.append((dist, i))

    cur_used_buffer_exp_pt = 0
    for iter in dist_idx_map:
        if cur_used_buffer_exp_pt >= buffer_exp_pt_size:
            break
        idx = iter[1]
        pt = exp_pts[idx]
        pt.intensity = VALID_LENGTH_CMP_PT
        cmp_pts.append(pt)
        cur_used_buffer_exp_pt = cur_used_buffer_exp_pt + 1

    if not cmp_pts:
        return -1

    return 0


def sort_by_avg_distance(pts):
    dist = sum(pt.y ** 2 + pt.z ** 2 for pt in pts) / len(pts)
    return dist


def sort_by_intensity(pt):
    return pt.intensity


def get_updated_box_size_(points, box, params={'update_length': True, 'update_width': True, 'radius_filter_min_pts': 10}):
    ret = [box.length, box.width]
    if not params['update_length'] and not params['update_width']:
        return ret
    if params['radius_filter_min_pts'] < 1:
        return ret
    if len(box.point_index) < 10:
        return ret
    if box.length < 4.0 or box.width < 1.0 or box.height < 1.0:
        return ret

    pc_num = len(points.point)

    if not is_dual_lidar(points):
        return single_lidar(points, box)
    else:
        return dual_lidar(points, box)


def get_obb_from_points(points, calcconvexhull=True):
    """ given a set of points, calculate the oriented bounding box.
    Parameters:
    points: numpy array of point coordinates with shape (n,2)
            where n is the number of points
    calcconvexhull: boolean, calculate the convex hull of the
            points before calculating the bounding box. You typically
            want to do that unless you know you are passing in a convex
            point set
    Output:
        tuple of corners, centre
    """
    if calcconvexhull:
        _ch = ConvexHull(points)
        points = _ch.points[_ch.vertices]
    cov_points = np.cov(points, y=None, rowvar=0, bias=1)
    v, vect = np.linalg.eig(cov_points)
    tvect = np.transpose(vect)
    # use the inverse of the eigenvectors as a rotation matrix and
    # rotate the points so they align with the x and y axes
    points_rotated = np.dot(points, np.linalg.inv(tvect))
    # get the minimum and maximum x and y
    mina = np.min(points_rotated, axis=0)
    maxa = np.max(points_rotated, axis=0)
    diff = (maxa - mina) * 0.5
    # the center is just half way between the min and max xy
    center = mina + diff
    # get the corners by subtracting and adding half the bounding boxes height and width to the center
    corners = np.array([center + [-diff[0], -diff[1]], center + [diff[0], -diff[1]], center + [diff[0], diff[1]],
                        center + [-diff[0], diff[1]], center + [-diff[0], -diff[1]]])
    # use the the eigenvectors as a rotation matrix and
    # rotate the corners and the center back
    corners = np.dot(corners, tvect)
    center = np.dot(center, tvect)
    return corners, center


def vis_points(xyz, outpath, other={}):

    yz = xyz[:, 1:3]
    a = np.array(yz)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.scatter(a[:, 0], a[:, 1])

    corners, center = get_obb_from_points(a)
    dealt_x = corners[0][0] - corners[1][0]
    dealt_y = corners[0][1] - corners[1][1]
    d1 = math.sqrt((dealt_x**2) + (dealt_y**2))

    dealt_x_2 = corners[1][0] - corners[2][0]
    dealt_y_2 = corners[1][1] - corners[2][1]
    d2 = math.sqrt((dealt_x_2**2)+(dealt_y_2**2))

    length = 0
    width = 0
    if d1 > d2:
        length = round(d1, 3)
        width = round(d2, 3)
    elif (d1 <= d2):
        length = round(d2, 3)
        width = round(d1, 3)

    ax.scatter([center[0]], [center[1]])
    ax.plot(corners[:, 0], corners[:, 1], 'k-')

    hull = ConvexHull(a)
    for simplex in hull.simplices:
        plt.plot(a[simplex, 0], a[simplex, 1], 'y--')

    title = f"vehicle length: {length}m, vehicle width: {width}m\n"

    for k, v in other.items():
        if v is None:
            continue
        title = title + \
            f"({k}-dis{round(v['position_z'], 3)}m) - vehicle length: {round(v['length'], 3)}m, vehicle width: {round(v['width'], 3)}m\n"

    plt.title(title)
    plt.axis('equal')
    fig.savefig(outpath)
    return length, width


if __name__ == '__main__':
    file = '/home/demo/Documents/datasets/size_filter_0923/11.00000'
    try:
        record = Record(file)
    except Exception as e:
        print(file)
    idxes = [141506]
    indata = {}
    out = []
    for tm_idx in idxes:
        indata[tm_idx] = {
            'boxes': None,
            'points': None,
        }
    for topic, message, t in record.read_messages():
        idx = None
        try:
            idx = message.idx
        except AttributeError:
            continue
        if idx not in indata:
            continue
        # if idx != 135803:
        #     continue
        if topic == 'omnisense/track_fusion/boxes':
            indata[idx]['boxes'] = message.box
        if topic == 'omnisense/track_fusion/foreground/dynamic_pointcloud':
            indata[idx]['points'] = message
    for k, v in indata.items():
        if not v['boxes'] or not v['points']:
            continue
        ret = proc(v['points'], v['boxes'])
        out.append([k, ret[0], ret[1]])
        print([k, ret[0], ret[1]])

    out = pd.DataFrame(out, columns=['idx', 'length', 'width'])
    out.to_csv('out.csv', index=False)
