import open3d as o3d
import numpy as np
import os
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import math


class Box():
    def __init__(self, **kwargs):
        self.cx = kwargs['cx']
        self.cy = kwargs['cy']
        self.cz = kwargs['cz']
        self.heading = kwargs['heading'] / 100.0
        self.l = kwargs['l']
        self.w = kwargs['w']
        self.h = kwargs['h']
        self.point_index = kwargs['point_index']
        self.points = []
        self.bounds = {}
        self.matrix = None
        self.rotate_points = []
        self.rotate_bounds = {}

    def matrix_2_global(self):
        theta = np.deg2rad(self.heading % 360)
        r = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
        t = np.array([self.cz, self.cy, self.cx])
        trans_matrix = np.eye(4)
        trans_matrix[0:3, 0:3] = r
        trans_matrix[0:3, 3] = t
        self.matrix = trans_matrix

    def set_ponits(self, points):
        self.bounds = {
            'min_y': [1e9, -1e9],
            'max_y': [-1e9, -1e9],
            'min_z': [1e9, 1e9],
            'max_z': [-1e9, -1e9]
        }
        self.points = []
        for i in self.point_index:
            pt = points.point[i]
            self.points.append([pt.x, pt.y, pt.z])
            if pt.y < self.bounds['min_y'][0]:
                self.bounds['min_y'] = [pt.y, pt.z]
            if pt.y > self.bounds['max_y'][0]:
                self.bounds['max_y'] = [pt.y, pt.z]
            if pt.z < self.bounds['min_z'][1]:
                self.bounds['min_y'] = [pt.y, pt.z]
            if pt.z > self.bounds['max_z'][1]:
                self.bounds['max_y'] = [pt.y, pt.z]

    def cal_rotate_points(self):
        self.rotate_bounds = {
            'min_y': 1e9,
            'max_y': -1e9,
            'min_z': 1e9,
            'max_z': -1e9
        }
        self.rotate_points = []
        angle = np.pi / 180 * (360 - self.heading)
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)
        const_temp_y = -cos_t * self.cy - sin_t * self.cz
        const_temp_z = sin_t * self.cy - cos_t * self.cz
        for pt in self.points:
            r_x = pt[0]
            r_y = cos_t * pt[1] + sin_t * pt[2] + const_temp_y
            r_z = -sin_t * pt[1] + cos_t * pt[2] + const_temp_z
            self.rotate_bounds['min_y'] = min(self.rotate_bounds['min_y'], r_y)
            self.rotate_bounds['min_z'] = min(self.rotate_bounds['min_z'], r_z)
            self.rotate_bounds['max_y'] = max(self.rotate_bounds['max_y'], r_y)
            self.rotate_bounds['max_z'] = max(self.rotate_bounds['max_z'], r_z)
            self.rotate_points.append([r_x, r_y, r_z])


def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return pcd.points


def raster(points, grip, min_x, min_y, max_x, max_y):
    rows = int((max_x - min_x) // grip[0]) + 1
    cols = int((max_y - min_y) // grip[0]) + 1
    img = np.zeros((rows, cols))
    for pt in points:
        x = int((pt[0] - min_x) // grip[0])
        y = int((pt[1] - min_y) // grip[1])
        img[x, y] = img[x, y] + 1
    return img


def find_bounds(points):
    min_x = 1e9
    min_y = 1e9
    max_x = -1e9
    max_y = -1e9
    for i in points:
        min_x = min(min_x, i[0])
        min_y = min(min_y, i[1])
        max_x = max(max_x, i[0])
        max_y = max(max_y, i[1])
    return min_x, min_y, max_x, max_y


def rasterize_points(box, grid_size=0.1):

    points = np.array(box.rotate_points)

    min_z = box.rotate_bounds['min_z']
    max_z = box.rotate_bounds['max_z']
    min_y = box.rotate_bounds['min_y']
    max_y = box.rotate_bounds['max_y']

    rows = int((max_z - min_z) / grid_size) + 1
    cols = int((max_y - min_y) / grid_size) + 1

    # 创建栅格 (二维数组)
    raster_grid = [[] for i in range(rows * cols)]
    raster_grid_l = np.zeros([rows, cols])

    # 将点放入相应的栅格中
    for i in range(len(points)):
        point = points[i]
        grid_x = int((point[2] - min_z) / grid_size)
        grid_y = int((point[1] - min_y) / grid_size)
        index = grid_x * cols + grid_y
        if grid_x >= 0 and grid_y >= 0 and grid_x < rows and grid_y < cols:
            raster_grid[index].append(i)
            raster_grid_l[grid_x, grid_y] += 1

    return raster_grid, raster_grid_l, rows, cols


def read_box_pcd(filepath, idx, id, box_channle='omnisense/track_fusion/boxes', points_channel='omnisense/track_fusion/foreground/dynamic_pointcloud'):
    from cyber_record.record import Record
    box = None
    pcd = None
    try:
        record = Record(filepath)
    except Exception as e:
        print(file)
        return box, pcd
    for topic, message, t in record.read_messages():
        try:
            tidx = message.idx
        except AttributeError:
            continue
        if idx != tidx:
            continue
        if topic == box_channle:
            for t_box in message.box:
                if t_box.track_id == id:
                    tmp = {
                        'cx': t_box.position_x,
                        'cy': t_box.position_y,
                        'cz': t_box.position_z,
                        'heading': t_box.spindle,
                        'l': t_box.length,
                        'w': t_box.width,
                        'h': t_box.height,
                        'point_index': t_box.point_index
                    }
                    box = Box(**tmp)
        elif topic == points_channel:
            pcd = message
        if box and pcd:
            return box, pcd
    return box, pcd


def write_pcd(points, k):
    new_p = []
    for i in points:
        new_p.append([0, i[1], i[0]])
    new_p = np.array(new_p)
    # 写pcd文件
    pp = o3d.geometry.PointCloud()
    pp.points = o3d.utility.Vector3dVector(new_p)
    o3d.io.write_point_cloud(f"{k}-output.pcd", pp)


def write_pcd_3d(points, k):
    new_p = np.array(points)
    # 写pcd文件
    pp = o3d.geometry.PointCloud()
    pp.points = o3d.utility.Vector3dVector(new_p)
    o3d.io.write_point_cloud(k, pp)


def filter_points(box, points):
    pcd = []
    bounds = {
        'min_y': [1e9, -1e9],
        'max_y': [-1e9, -1e9],
        'min_z': [1e9, 1e9],
        'max_z': [-1e9, -1e9]
    }
    for i in box.point_index:
        pt = points.point[i]
        pcd.append([pt.x, pt.y, pt.z])
        if pt.y < bounds['min_y'][0]:
            bounds['min_y'] = [pt.y, pt.z]
        if pt.y > bounds['max_y'][0]:
            bounds['max_y'] = [pt.y, pt.z]
        if pt.z < bounds['min_z'][1]:
            bounds['min_y'] = [pt.y, pt.z]
        if pt.z > bounds['max_z'][1]:
            bounds['max_y'] = [pt.y, pt.z]
    return np.array(pcd), bounds


def filter_points_org(box, points):
    bounds = {
        'min_y': [1e9, -1e9],
        'max_y': [-1e9, -1e9],
        'min_z': [1e9, 1e9],
        'max_z': [-1e9, -1e9]
    }
    pcd = []
    for i in box.point_index:
        pt = points.point[i]
        pcd.append([pt.x, pt.y, pt.z])
        if pt.y < bounds['min_y'][0]:
            bounds['min_y'] = [pt.y, pt.z]
        if pt.y > bounds['max_y'][0]:
            bounds['max_y'] = [pt.y, pt.z]
        if pt.z < bounds['min_z'][1]:
            bounds['min_y'] = [pt.y, pt.z]
        if pt.z > bounds['max_z'][1]:
            bounds['max_y'] = [pt.y, pt.z]
    return np.array(pcd), bounds


def rotate_points(points, box):
    rotation_matrix = np.linalg.inv(box.matrix)

    # 将点集进行旋转
    num_points = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((num_points, 1))))
    rotated_points = homogeneous_points @ rotation_matrix.T
    return rotated_points[:, 0:3]  # z y x


def eight_neighborhood_labeling(image):
    # 获取图像尺寸
    rows, cols = image.shape
    # 创建一个和原图相同大小的新图像，初始值为0
    labeled_image = np.zeros((rows, cols), dtype=np.uint8)
    # 标签计数器
    label = 1

    for row in range(1, rows-1):
        for col in range(1, cols-1):
            # 如果当前像素是边界点
            if image[row, col] == 0:
                # 检查8邻域内是否有非边界点
                if image[row-1, col] > 0 or image[row+1, col] > 0 or \
                   image[row, col-1] > 0 or image[row, col+1] > 0:
                    # 为该边界点分配一个标签
                    labeled_image[row, col] = label
                    label += 1

    return labeled_image


def label_connected_components(image, labeled_image, r, c, l):
    rows, cols = image.shape
    for dr, dc in [(1, 0), (1, 1)]:
        nc, nr = c + dc, r + dr
        if 0 <= nr < rows and 0 <= nc < cols and image[nr, nc] > 0:
            labeled_image[nr, nc] += 1
            l.append([nr, nc])
            label_connected_components(image, labeled_image, nr, nc, l)
    return


def neighbour(rasterize_):
    rows, cols = rasterize_.shape
    labeled_image = np.zeros([rows, cols], dtype=np.uint8)
    del_points = []
    for i in range(1, cols - 1):
        if np.sum(rasterize_[:, i]) > 0 and np.sum(rasterize_[:, i + 1]) == 0:
            continue
        for j in range(1, rows - 1):
            if rasterize_[j, i] > 0 and labeled_image[j, i] == 0:
                p_set = [[j, i]]
                label_connected_components(
                    rasterize_, labeled_image, j, i, p_set)
                # r = j
                # c = i
                # is_track = True
                # while is_track:
                #     is_track = False
                #     for dr, dc in [(1, 0), (1, 1)]:
                #         nc, nr = c + dc, r + dr
                #         if 0 <= nr < rows and 0 <= nc < cols and rasterize_[nr, nc] > 0:
                #             labeled_image[nr, nc] += 1
                #             p_set.append([nr, nc])
                #             r, c = nr, nc
                #             is_track = True
                #             break

                if len(p_set) < 10:
                    for p in p_set:
                        rasterize_[p[0], p[1]] = 0
                    del_points.extend(p_set)
    return del_points


def main_new_nosie(file, idx, boxid, out_dir, r_num, c_num):
    box, points = read_box_pcd(
        file, idx, boxid, box_channle='omnisense/distortion_correction/01/boxes', points_channel='omnisense/distortion_correction/01/PointCloud')
    if box is None:
        return
    print(box.l)
    print(box.w)
    print(box.h)
    box.set_ponits(points)
    if not box.points:
        print(f"box points is none")
        return
    box.cal_rotate_points()

    grip_size = 0.2
    rasterize_points_, rasterize_, rows, cols = rasterize_points(
        box, grip_size)
    rasterize_ = np.array(rasterize_)
    column_sums = rasterize_.sum(axis=0)
    max_sum_index = np.argmax(column_sums)
    non_zero_counts = np.count_nonzero(rasterize_, axis=0)
    max_non_zero_index = np.argmax(non_zero_counts)
    start_col = 0
    end_col = cols - 1
    if cols - 1 - max_sum_index < max_sum_index:
        end_col = max_sum_index
    else:
        start_col = max_sum_index
    if end_col - max_non_zero_index < 0:
        end_col = max_non_zero_index
    if start_col > max_non_zero_index:
        start_col = max_non_zero_index
    if not (cols - 1 - end_col > 0 and cols - 1 - end_col < c_num):
        end_col = cols
    if not (start_col < c_num):
        start_col = 0

    row_sums = rasterize_.sum(axis=1)
    max_sum_index = np.argmax(row_sums)
    non_zero_counts = np.count_nonzero(rasterize_, axis=1)
    max_non_zero_index = np.argmax(non_zero_counts)
    start_row = 0
    end_row = rows - 1
    if rows - 1 - max_sum_index < max_sum_index:
        end_row = max_sum_index
    else:
        start_row = max_sum_index
    if end_row - max_non_zero_index < 0:
        end_row = max_non_zero_index
    if start_row > max_non_zero_index:
        start_row = max_non_zero_index
    if not (rows - 1 - end_row > 0 and rows - 1 - end_row < r_num):
        end_row = rows
    if not (start_row < r_num):
        start_row = 0

    print(
        f"rows = {rows}, cols = {cols}, start_col = {start_col}, end_col = {end_col}")
    filter_points_index = []
    for col in range(end_col + 1, cols):
        for row in range(0, rows):
            index = row * cols + col
            filter_points_index.extend(rasterize_points_[index])
    filter_pts = []
    for i in range(len(box.points)):
        if i in filter_points_index:
            continue
        filter_pts.append(box.points[i])
    org_path = os.path.join(out_dir, f'org_idx_{idx}_box_id_{boxid}.pcd')
    write_pcd_3d(box.points, org_path)
    filter_path = os.path.join(
        out_dir, f'filter_idx_{idx}_box_id_{boxid}_row_{start_row}_{end_row}_col_{start_col}_{end_col}.pcd')
    write_pcd_3d(filter_pts, filter_path)
    png_path = os.path.join(out_dir, f'png_{idx}_box_id_{boxid}.png')
    vis_points(org_path, filter_path, png_path)


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


def vis_points(org_path, filter_path, out_path):
    org_xyz_all = np.array([0, 0, 0])
    if org_path.find(".pcd") != 0:
        org_pcd = o3d.io.read_point_cloud(org_path)
        xyz = np.asarray(org_pcd.points)
        org_xyz_all = np.vstack((org_xyz_all, xyz))
    org_yz = org_xyz_all[:, 1:3]
    org_yz = org_yz[1:, :]
    org_yz = np.array(org_yz)

    fig = plt.figure(figsize=(24, 24))
    ax = fig.add_subplot(111)
    ax.scatter(org_yz[:, 0], org_yz[:, 1])
    corners, center = get_obb_from_points(org_yz)
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
    title = f"pre vehicle length: {length}m, vehicle width: {width}m\n"

    org_xyz_all = np.array([0, 0, 0])
    if filter_path.find(".pcd") != 0:
        filter_pcd = o3d.io.read_point_cloud(filter_path)
        xyz = np.asarray(filter_pcd.points)
        org_xyz_all = np.vstack((org_xyz_all, xyz))
    org_yz = org_xyz_all[:, 1:3]
    org_yz = org_yz[1:, :]
    org_yz = np.array(org_yz)

    ax.scatter(org_yz[:, 0], org_yz[:, 1])
    corners, center = get_obb_from_points(org_yz)
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
    ax.plot(corners[:, 0], corners[:, 1], 'y--')
    title = title + \
        f"\n filter vehicle length: {length}m, vehicle width: {width}m\n"

    plt.title(title)
    plt.axis('equal')
    plt.legend()
    fig.savefig(out_path)


if __name__ == '__main__':
    scence = '11'  # 152812
    boxid = 3
    file = f'/home/demo/Documents/datasets/1010s/{scence}.{boxid}.00000'
    out_dir = f'/home/demo/Documents/datasets/1010s/test/{scence}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in range(141457, 141777):
        main_new_nosie(file, i, boxid, out_dir, r_num=10, c_num=10)
