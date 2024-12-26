"""
MUST_HAVE_       
    python=3.8
    open3d,
    yaml,
    cv2,
    numpy

RUN COMMAND
    python generate_parallel_matrix.py --dir ./test/ [optional --outdir ./] [optional --tqdm 1]
        --dir The directory where the PCD file is located. It is recommended to have 10 or more files.
        --outdir The directory where the output YAML file will be saved. If not provided, it will be saved in 'output' folder under the executable file directory.
        --tqdm 1 The execution time is relatively long; using this parameter allows you to see the progress bar.

CALL FUNCTION
    run_parallel_matrix(pcd_files:list): -> matrix:numpy array
"""


import os
import numpy as np
import open3d as o3d
import cv2
import logging
import time
import yaml
import sys


def write_yaml(path, data):
    with open(path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
        return True
    return False


def read_yaml(path):
    loaded_data = {}
    with open(path, 'r') as file:
        loaded_data = yaml.load(file, Loader=yaml.FullLoader)
    return loaded_data


def get_files_in_current_directory(dir, extension='.pcd'):
    pcd_files = []
    for item in os.listdir(dir):
        full_path = os.path.join(dir, item)
        if os.path.isfile(full_path) and item.endswith(extension):
            pcd_files.append(full_path)
    return pcd_files


def rotation_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    rot_matrix = np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                           [2 * (b * c + a * d), a * a + c * c -
                            b * b - d * d, 2 * (c * d - a * b)],
                           [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])
    return rot_matrix


def get_flatten_matrix(points, width=100, length=50.0, w_step=1.0, l_step=1.0):
    points = np.asarray(points)
    grids_width = int(length / l_step)
    grids_length = int(width / w_step)
    grids = [[10000, 0, 0]] * (grids_width * grids_length)
    for point in points:
        w = int((point[1] + width / 2) / w_step)
        l = int(point[2] / l_step)
        if w >= 0 and w < grids_width and l >= 0 and l < grids_length:
            if point[0] < grids[l*grids_width + w][0]:
                grids[l*grids_width + w] = [point[0], point[1], point[2]]
    surface_points = [grid for grid in grids if grid[0] < 10.0]
    surface_cloud = o3d.geometry.PointCloud()
    surface_cloud.points = o3d.utility.Vector3dVector(surface_points)
    plane_model, inliers = surface_cloud.segment_plane(
        distance_threshold=0.15, ransac_n=3, num_iterations=200)
    axis = np.cross(plane_model[:3], [1, 0, 0])
    angle = np.arccos(np.dot(plane_model[:3],  [1, 0, 0]))
    rot = rotation_matrix(axis, -angle)
    transf = np.eye(4)
    transf[:3, :3] = rot
    transf[0, 3] = plane_model[3]
    return transf


def get_output_directory(out_dir, file_type='output'):
    if not out_dir:
        out_dir = os.path.join(os.getcwd(), file_type)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def print_progress_bar(iteration, total, length=40, pre_message=''):
    percent = (iteration / total)
    arrow = '>' * int(round(percent * length) - 1)
    spaces = ' ' * (length - len(arrow))
    sys.stdout.write(
        f'\rProgress: {pre_message} | {arrow}{spaces}| {percent:.2%}')
    sys.stdout.flush()


class PointsAtrr:
    # up x front z right y
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z
        self._dis_2 = x * x + y * y + z * z
        self._dis = np.sqrt(self.dis_2)
        self._h = np.arctan2(y, z) * 180.0 / np.pi
        self._v = np.arcsin(x / self.dis) * 180.0 / np.pi

    @property
    def dis_2(self):
        return self._dis_2

    @property
    def dis(self):
        return self._dis

    @property
    def h(self):
        return self._h

    @property
    def v(self):
        return self._v

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z


class GroundFinder:
    def __init__(self, **kwargs):
        self._tqdm = kwargs.get('enable_tqdm', False)
        self._init_logger(**kwargs)
        self._init_params(**kwargs)
        self._init_experience_value()

    def parallel(self, pcds):
        self._init_valuse_for_parallel()
        if self._tqdm:
            print_progress_bar(0, 100, pre_message='start read_pcds')
        if not self._read_pcds(pcds):
            return False
        if self._tqdm:
            print_progress_bar(11, 100, pre_message='start rm_motion_obj')
        no_obj_cloud = self._rm_motion_obj()
        if no_obj_cloud.shape[0] == 0:
            self._log_message(
                f'no valid point cloud after rm_motion_obj .', logging.ERROR)
            return False
        if self._tqdm:
            print_progress_bar(21, 100, pre_message='start downsample')
        self._downsample_points = self._downsample(no_obj_cloud)
        if len(self._downsample_points.points) == 0:
            self._log_message(
                f'points after downsample is empty .', logging.ERROR)
            return False
        if self._tqdm:
            print_progress_bar(30, 100, pre_message='start estimate_normals')
        self._downsample_points = self._estimate_normals()
        if self._downsample_points is not None and np.asarray(self._downsample_points.normals).shape[0] == 0:
            self._log_message(
                f'normals after estimate_normals is empty .', logging.ERROR)
            return False
        if self._tqdm:
            print_progress_bar(35, 100, pre_message='start find road surface')
        regions = self._region_growth()
        if len(regions) == 0:
            self._log_message(f'can not find road surface .', logging.ERROR)
            return False
        if self._tqdm:
            print_progress_bar(55, 100, pre_message='start find_max_region')
        self._road_points = self._find_max_region(regions)
        # self._out_put_road_points()
        if len(self._road_points) == 0:
            self._log_message(f'can not find road surface .', logging.ERROR)
            return False
        if self._tqdm:
            print_progress_bar(
                60, 100, pre_message='start generate_ground_image')
        self._ground_lut, self._ground_lut_h = self._generate_ground_image()
        if self._tqdm:
            print_progress_bar(
                70, 100, pre_message='start find_single_ground')
        self._planes = self._find_single_ground()
        if self._tqdm:
            print_progress_bar(
                90, 100, pre_message='start filter_planes')
        if not self._filter_planes():
            return False
        if self._tqdm:
            print_progress_bar(
                98, 100, pre_message='start generate_parallel_transform')
        if not self._generate_parallel_transform():
            self._log_message(
                f'generate parallel transform failed .', logging.ERROR)
            return False
        if self._tqdm:
            print_progress_bar(
                100, 100, pre_message='parallel successfully')
        return True

    def _out_put_road_points(self, path='road_points.pcd'):
        points = []
        for i in self._road_points:
            points.append(self._downsample_points.points[i])
        points = np.array(points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(path, pcd)

    def get_parallel_transform(self):
        if self._valid_plane:
            return self._valid_plane[-1]
        return np.eye(4, np.float32)

    def _init_valuse_for_parallel(self):
        self._org_points = []
        self._pcds_attr = []
        self._downsample_points = None
        self._road_points = []
        self._ground_lut = np.array([])
        self._ground_lut_h = np.array([])
        self._planes = []
        self._valid_plane = []

    def _generate_parallel_transform(self):
        error_frame_count = 0
        for plane in self._planes:
            if plane[0] > 0.93:
                transform = self._get_parallel_transform(plane)
                self._valid_plane.append(transform)
            else:
                error_frame_count += 1
        if 1.0 * error_frame_count / len(self._planes) > 0.5:
            return False
        return True

    def _get_parallel_transform(self, plane_coeff):
        transform = np.eye(4, dtype=np.float32)
        if abs(plane_coeff[1]) < 1E-6 and abs(plane_coeff[2]) < 1E-6:
            transform[0, 3] = -plane_coeff[3]
        else:
            a = plane_coeff[0]
            b = plane_coeff[1]
            c = plane_coeff[2]
            d = plane_coeff[3]
            norm = np.sqrt(a * a + b * b + c * c)
            if norm > 0:
                a = a / norm
                b = b / norm
                c = c / norm
                d = d / norm
            if a < 0:
                a, b, c, d = -a, -b, -c, -d
            sina = np.sqrt(1 - a * a)
            b2 = b * b
            c2 = c * c
            base1 = 1.0 / (b2 + c2)
            base2 = np.sqrt(base1)
            aa = 1 - a
            support1 = base1 * aa
            support2 = base2 * sina
            transform[0, 0] = a
            transform[0, 1] = b * support2
            transform[0, 2] = c * support2
            transform[1, 0] = -transform[0, 1]
            transform[1, 1] = 1 - b2 * support1
            transform[1, 2] = -b * c * support1
            transform[2, 0] = -transform[0, 2]
            transform[2, 1] = transform[1, 2]
            transform[2, 2] = 1 - c2 * support1
            transform[0, 3] = d * support2
        return transform

    def _cal_dis_points_to_plane(self, points, plane):
        distance_list = []
        sum_distance_2 = 0
        sum_distance = 0
        n = len(points)
        if n > 0:
            for x, y, z in points:
                distance = x * plane[0] + y * \
                    plane[1] + z * plane[2] + plane[3]
                distance_list.append(distance)
                sum_distance_2 += distance * distance
                sum_distance += distance

            sum_distance_2 /= n
            sum_distance /= n
            stdev = np.sqrt(abs(sum_distance_2 - sum_distance * sum_distance))

        return distance_list, sum_distance, stdev

    def _filter_planes(self):
        error_frame_count = 0
        for i, plane in enumerate(self._planes):
            if np.all(plane == 0):
                continue

            distance_list, sum_distance, stdev = self._cal_dis_points_to_plane(
                self._org_points[i], plane)
            if not distance_list:
                error_frame_count += 1
                continue

            thresh_l = sum_distance - stdev * self._filter_ground_n_sigma
            thresh_h = sum_distance + stdev * self._filter_ground_n_sigma
            denoised_count = 0
            for j in range(len(distance_list)):
                if distance_list[j] > thresh_l and distance_list[j] < thresh_h:
                    denoised_count += 1
            if denoised_count < 1.0 * len(self._org_points[i]) / 2:
                error_frame_count += 1
        if 1.0 * error_frame_count / len(self._planes) > 0.5:
            self._log_message(
                f'error frame rate error_frame_count / files_num > 0.5 .', logging.ERROR)
            return False
        return True

    def _find_single_ground(self):
        default_v = 1e5
        out = []
        for pcd in self._org_points:
            bottom_mat = np.full(
                (self._z_size, self._y_size), default_v, dtype=np.float32)
            index_mat = np.zeros((self._z_size, self._y_size), dtype=np.int32)
            for j, (x, y, z) in enumerate(pcd):
                zid = int(z * self._inv_step)
                yid = int((y + self._y_range) * self._inv_step)
                if 0 <= zid < self._z_size and 0 <= yid < self._y_size and self._ground_lut[zid, yid] == 255 and x < self._ground_lut_h[zid, yid]:
                    bottom_mat[zid, yid] = min(x, bottom_mat[zid, yid])
                    index_mat[zid, yid] = j
            out_pcd_one = []
            for r in range(self._z_size):
                for c in range(self._y_size):
                    if bottom_mat[r, c] < default_v:
                        out_pcd_one.append(pcd[index_mat[r, c]])
            plane_coeff = np.zeros(4, dtype=np.float32)
            if len(out_pcd_one) >= 100:
                plane_coeff = self._fit_plane(out_pcd_one)
            out.append(plane_coeff)
        return out

    def _fit_plane(self, pc):
        plane_coeff = np.zeros(4, dtype=np.float32)
        if len(pc) < 3:
            return plane_coeff
        a = np.zeros((3, 3), dtype=np.float32)
        b = np.zeros(3, dtype=np.float32)
        for x, y, z in pc:
            a[0, 0] += x * x
            a[0, 1] += x * y
            a[0, 2] += x * z
            a[1, 1] += y * y
            a[1, 2] += y * z
            a[2, 2] += z * z
            b[0] += x
            b[1] += y
            b[2] += z
        a[1, 0] = a[0, 1]
        a[2, 0] = a[0, 2]
        a[2, 1] = a[1, 2]
        if np.abs(np.linalg.det(a)) < 1E-3:
            return plane_coeff
        s = -np.linalg.inv(a).dot(b)
        s /= np.linalg.norm(s)
        d = -np.sum(s * b) / len(pc)
        r_list = []
        sumr = sumr2 = 0.0
        for x, y, z in pc:
            r = np.abs(x * s[0] + y * s[1] + z * s[2] + d)
            r_list.append(r)
            sumr += r
            sumr2 += r * r

        inv_size = 1.0 / len(pc)
        sumr *= inv_size
        sumr2 *= inv_size
        thresh = sumr + np.sqrt(abs(sumr2 - sumr * sumr))

        count = 0
        for i, (x, y, z) in enumerate(pc):
            if r_list[i] > thresh:
                a[0, 0] -= x * x
                a[0, 1] -= x * y
                a[0, 2] -= x * z
                a[1, 1] -= y * y
                a[1, 2] -= y * z
                a[2, 2] -= z * z
                b[0] -= x
                b[1] -= y
                b[2] -= z
                count += 1
        a[1, 0] = a[0, 1]
        a[2, 0] = a[0, 2]
        a[2, 1] = a[1, 2]
        if len(pc) - count < 3:
            return plane_coeff
        s = -np.linalg.inv(a).dot(b)
        s /= np.linalg.norm(s)

        plane_coeff[:3] = s
        plane_coeff[3] = -np.sum(s * b) / (len(pc) - count)

        return plane_coeff

    def _generate_ground_image(self):
        ground_lut = np.zeros((self._z_size, self._y_size), dtype=np.uint8)
        ground_lut_h = np.zeros((self._z_size, self._y_size)) - np.inf
        for i in self._road_points:
            [x, y, z] = self._downsample_points.points[i]
            zid = int(z * self._inv_step)
            yid = int((y + self._y_range) * self._inv_step)
            if 0 <= zid < self._z_size and 0 <= yid < self._y_size:
                ground_lut[zid, yid] = 255
                ground_lut_h[zid, yid] = max(ground_lut_h[zid, yid], x)
        # ground_lut, ground_lut_h = self._dilate(ground_lut, ground_lut_h)
        return ground_lut, ground_lut_h

    def _erode(self, image, kernel=3):
        eroded_image = np.zeros_like(image)
        for i in range(image.shape[0] - kernel):
            for j in range(image.shape[1] - kernel):
                window = image[i:i + kernel, j:j + kernel]
                if np.any(window == 255):
                    a = 1
                if np.all(window == 255):
                    eroded_image[i, j] = 255
        return eroded_image

    def _dilate(self, image, image_attr, kernel=3):
        dilated_image = np.zeros_like(image)
        dilated_image_h = np.copy(image_attr)
        for i in range(image.shape[0] - kernel):
            for j in range(image.shape[1] - kernel):
                window = image[i:i + kernel, j:j + kernel]
                if np.any(window == 255):
                    dilated_image[i, j] = 255
                    if dilated_image_h[i, j] == -np.inf:
                        for ii in range(i, i + kernel + 1):
                            for jj in range(j, j + kernel + 1):
                                if image[ii, jj] == 255:
                                    dilated_image_h[i, j] = max(
                                        image_attr[ii, jj], dilated_image_h[i, j])
        return dilated_image, image_attr

    def _find_max_region(self, regions):
        out_region = []
        out_area = self._max_region_min_area
        for region in regions:
            points = []
            for i in region:
                points.append(
                    [self._downsample_points.points[i][2], -self._downsample_points.points[i][1]])
            points = np.array(points, dtype=np.float32)
            hull_indices = cv2.convexHull(points, returnPoints=False)
            if hull_indices is None or len(hull_indices) == 0:
                continue
            hull = points[hull_indices].reshape(-1, 2)
            hull_area = cv2.contourArea(hull)
            if hull_area > out_area:
                out_region = region
                out_area = hull_area
        return out_region

    def _region_growth(self):
        pcd_tree = o3d.geometry.KDTreeFlann(self._downsample_points)
        smoothness_threshold = self._region_growth_angle_diff_th / 180.0 * np.pi
        visited = np.zeros(len(self._downsample_points.points), dtype=bool)
        clusters = []
        for i in range(len(self._downsample_points.points)):
            if not visited[i]:
                cluster = []
                stack = [i]
                visited[i] = True
                while stack:
                    index = stack.pop()
                    cluster.append(index)
                    [k, idx, _] = pcd_tree.search_radius_vector_3d(
                        self._downsample_points.points[index], self._region_growth_radius_m)
                    for j in range(k):
                        neighbor_index = idx[j]
                        if not visited[neighbor_index]:
                            normal_diff = round(np.dot(
                                self._downsample_points.normals[index], self._downsample_points.normals[neighbor_index]), 5)
                            angle_diff = np.arccos(normal_diff)
                            if angle_diff < smoothness_threshold:
                                stack.append(neighbor_index)
                                visited[neighbor_index] = True
                if len(cluster) > self._max_region_min_point_num:
                    clusters.append(cluster)
        return clusters

    def _estimate_normals(self):
        self._downsample_points.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=5, max_nn=self._normal_estimate_neibo_num))

        normals = np.asarray(self._downsample_points.normals)
        error_normal_count = np.sum(np.isnan(normals))
        if error_normal_count / normals.shape[0] > 0.5:
            self._log_message(f'normals is invalid .', logging.ERROR)
            return None
        return self._downsample_points

    def _downsample(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        downsampled_pcd = pcd.voxel_down_sample(
            voxel_size=self._down_sample_leaf)
        return downsampled_pcd

    def _rm_motion_obj(self):
        hstep = 0.0625
        vstep = 0.0625
        hrange = 60
        vrange = 40
        hsize = 1920
        vsize = 1280
        min_dis = 0.01
        range_image = np.zeros((vsize, hsize))
        non_obj_cloud = []
        for i, pcd in enumerate(self._org_points):
            if self._tqdm:
                print_progress_bar(
                    10 + i + 1, 100, pre_message=f'start rm_motion_obj {i + 1}')
            pcd_attr = []
            for p in pcd:
                p_attr = PointsAtrr(p[0], p[1], p[2])
                pcd_attr.append(p_attr)
                if p_attr.dis_2 < min_dis:
                    continue
                hid = int((p_attr.h + hrange) / hstep)
                vid = int((p_attr.v + vrange) / vstep)
                if hid >= 0 and hid < hsize and vid >= 0 and vid < vsize:
                    range_image[vid][hid] = max(
                        p_attr.dis, range_image[vid][hid])
            self._pcds_attr.append(pcd_attr)
        for i in range(len(self._pcds_attr)):
            for p_attr in self._pcds_attr[i]:
                if p_attr.dis_2 < min_dis:
                    continue
                hid = int((p_attr.h + hrange) / hstep)
                vid = int((p_attr.v + vrange) / vstep)
                if hid >= 0 and hid < hsize and vid >= 0 and vid < vsize:
                    if range_image[vid][hid] - p_attr.dis < self._max_range_diff_th:
                        non_obj_cloud.append([p_attr.x, p_attr.y, p_attr.z])
        return np.array(non_obj_cloud)

    def _init_logger(self, **kwargs):
        self._log_is_enabled = kwargs.get('enable_logger', False)
        if not self._log_is_enabled:
            return
        self._log_is_enabled = True
        logger_name = kwargs.get('logger', 'GroundFinder')
        self._logger = logging.getLogger(logger_name)
        if self._logger.handlers:
            return
        level = kwargs.get('logger_level', logging.INFO)
        current_directory = kwargs.get('logger_dir', '')
        if not current_directory:
            current_directory = os.getcwd()
            current_directory = os.path.join(current_directory, 'log')
        os.makedirs(current_directory, exist_ok=True)
        file_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        path = os.path.join(current_directory,
                            f"ground_finder-{file_name}.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')
        self._logger.setLevel(level)
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(path)
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        self._logger.addHandler(file_handler)

    def _log_message(self, message, level):
        if self._log_is_enabled:
            self._logger.log(level, message)

    def _read_pcds(self, pcds):
        if len(pcds) < self._min_frame_num:
            self._log_message(
                f"Not enough files for ground finding, actual ({len(pcds)}), need ({self._min_frame_num}) .", logging.INFO)
            return False
        pcds.sort()
        self._org_points = []
        for pcd in pcds:
            if self._tqdm:
                count = len(self._org_points) + 1
                print_progress_bar(
                    count, 100, pre_message=f'start read_pcds {count}')
            if len(self._org_points) >= self._min_frame_num:
                break
            point_xyz = o3d.io.read_point_cloud(pcd)
            point_xyz = np.asarray(point_xyz.points)
            if point_xyz.shape[0] == 0:
                self._log_message(f"No points in file ({pcd}).", logging.INFO)
            elif point_xyz.shape[0] < self._points_num_frame:
                self._log_message(
                    f"Not enough points for ground finding ({self._points_num_frame}) in file ({pcd}).", logging.INFO)
            else:
                self._org_points.append(point_xyz)
        if len(self._org_points) < self._min_frame_num:
            self._log_message(
                f"Not enough valid files for ground finding, actual ({len(self._org_points)}), need ({self._min_frame_num}).", logging.ERROR)
            return False
        return True

    def _init_params(self, **kwargs):
        self._min_frame_num = kwargs.get('min_frame_num', 10)
        self._max_range_diff_th = kwargs.get('max_range_diff_th', 0.3)
        self._down_sample_leaf = kwargs.get('down_sample_leaf', 0.3)
        self._normal_estimate_neibo_num = kwargs.get(
            'normal_estimate_neibo_num', 20)
        self._region_growth_neibo_num = kwargs.get(
            'region_growth_neibo_num', 30)
        self._region_growth_curvature_th = kwargs.get(
            'region_growth_curvature_th', 1)
        self._region_growth_angle_diff_th = kwargs.get(
            'region_growth_angle_diff_th', 2)
        self._region_growth_radius_m = kwargs.get('region_growth_radius_m', 1)
        self._max_region_min_point_num = kwargs.get(
            'max_region_min_point_num', 100)
        self._max_region_min_area = kwargs.get('max_region_min_area', 20)
        self._filter_ground_n_sigma = kwargs.get('filter_ground_n_sigma', 1)
        self._points_num_frame = kwargs.get('points_num_frame', 1000)

    def update_params(self, **kwargs):
        self._min_frame_num = kwargs.get('min_frame_num', self._min_frame_num)
        self._max_range_diff_th = kwargs.get(
            'max_range_diff_th', self._max_range_diff_th)
        self._down_sample_leaf = kwargs.get(
            'down_sample_leaf', self._down_sample_leaf)
        self._normal_estimate_neibo_num = kwargs.get(
            'normal_estimate_neibo_num', self._normal_estimate_neibo_num)
        self._region_growth_neibo_num = kwargs.get(
            'region_growth_neibo_num', self._region_growth_neibo_num)
        self._region_growth_curvature_th = kwargs.get(
            'region_growth_curvature_th', self._region_growth_curvature_th)
        self._region_growth_angle_diff_th = kwargs.get(
            'region_growth_angle_diff_th', self._region_growth_angle_diff_th)
        self._max_region_min_point_num = kwargs.get(
            'max_region_min_point_num', self._max_region_min_point_num)
        self._max_region_min_area = kwargs.get(
            'max_region_min_area', self._max_region_min_area)
        self._filter_ground_n_sigma = kwargs.get(
            'filter_ground_n_sigma', self._filter_ground_n_sigma)
        self._points_num_frame = kwargs.get(
            'points_num_frame', self._points_num_frame)
        self._region_growth_radius_m = kwargs.get(
            'region_growth_radius_m', self._region_growth_radius_m)

    def _init_experience_value(self):
        self._z_size = 640
        self._y_size = 1024
        self._inv_step = 8
        self._y_range = 64


def matrix_format(matrix):
    formatted_string = np.array2string(np.array(matrix), formatter={
                                       'float_kind': lambda x: f'{x:.6f}'}, separator=' ')
    formatted_string = formatted_string.replace(
        '[', '').replace(']', '').replace('\n', '').strip()
    return formatted_string


def args_parser():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate parallel matrix Tool.')
    parser.add_argument('--dir', type=str, required=True,
                        help='The pcd files directory.')
    parser.add_argument('--outdir', type=str, default='',
                        help='The matrix out directory.')
    parser.add_argument('--log', type=int, default=0,
                        help='Log is enabled.')
    parser.add_argument('--logdir', type=str, default='',
                        help='Log file directory.')
    parser.add_argument('--tqdm', type=int, default=0,
                        help='Show progress bar or not?')
    return parser.parse_args()


def run_parallel_matrix(pcd_files, **kwargs):
    gf = GroundFinder(**kwargs)
    if gf.parallel(pcd_files):
        matrix = gf.get_parallel_transform()
    elif pcd_files:
        file = pcd_files[0]
        point_xyz = o3d.io.read_point_cloud(file)
        matrix = get_flatten_matrix(point_xyz.points)
    else:
        print("No point cloud files found.")
        return np.array([])
    return matrix


def main():
    args = args_parser()
    pcd_files = get_files_in_current_directory(args.dir)
    if not pcd_files:
        print("No point cloud files found.")
        return
    out_dir = get_output_directory(args.outdir)
    log_enable = True if args.log == 1 else False
    if log_enable:
        log_dir = get_output_directory(args.logdir, 'log')
    else:
        log_dir = ''
    tqdm_enable = True if args.tqdm == 1 else False
    gf = GroundFinder(enable_logger=log_enable,
                      logger_dir=log_dir, enable_tqdm=tqdm_enable)
    version = 'GroundFinder'
    project = 'GroundFinder'
    if gf.parallel(pcd_files):
        matrix = gf.get_parallel_transform()
    else:
        version = 'open3dSegmentPlane'
        file = pcd_files[0]
        project = os.path.basename(file).split('.')[0]
        point_xyz = o3d.io.read_point_cloud(file)
        matrix = get_flatten_matrix(point_xyz.points)
    matrix_str = matrix_format(matrix)
    date_str = time.strftime("%Y-%m-%d", time.localtime())
    yaml_data = {
        'version': version,
        'date': date_str,
        'project': project,
        'flat': {
            'transform': matrix_str
        }
    }
    write_yaml(os.path.join(out_dir, f'matrix-{date_str}.yaml'), yaml_data)


if __name__ == "__main__":
    main()
