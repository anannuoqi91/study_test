import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import math
import copy


class PointXYZI:
    x: float
    y: float
    z: float
    intensity: float


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
        self.matrix = None
        self.matrix_2_global()

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


def projection_range(points, k):
    max_p = [-1e9, 0]
    min_p = [1e9, 0]
    min_y_p = [1e9, 1e9, 1e9]
    min_z_p = [1e9, 1e9, 1e9]
    max_y_p = [-1e9, -1e9, -1e9]
    max_z_p = [-1e9, -1e9, -1e9]
    for x, y, z in points:
        b = z + 1 / k * y
        y_ = b / (k + 1 / k)
        z_ = k * y_
        if y_ < min_p[0]:
            min_p = [y_, z_]
        if y_ > max_p[0]:
            max_p = [y_, z_]
        if y < min_y_p[1]:
            min_y_p = [x, y, z]
        if y > max_y_p[1]:
            max_y_p = [x, y, z]
        if z < min_z_p[2]:
            min_z_p = [x, y, z]
        if z > max_z_p[2]:
            max_z_p = [x, y, z]

    projected_values = math.sqrt(
        (min_p[0] - max_p[0])**2 + (min_p[1] - max_p[1])**2)
    bounds = {
        'projected_values': projected_values,
        'min_y_p': min_y_p,
        'max_y_p': max_y_p,
        'min_z_p': min_z_p,
        'max_z_p': max_z_p
    }
    return bounds


def find_best_slope(points):
    best_k = None
    min_range = -1e9

    # 这里简单取一些斜率的样本值
    slopes = np.linspace(0, 90, 90)  # 可根据需求调整范围
    fin_pro_re = None

    for k in slopes[1:-1]:
        slope = math.tan((90 - k) * math.pi / 180)
        pro_re = projection_range(points, slope)
        if pro_re['projected_values'] > min_range:
            min_range = pro_re['projected_values']
            best_k = slope
            fin_pro_re = copy.deepcopy(pro_re)

    return best_k, fin_pro_re


def create_rec(points):
    k, bounds = find_best_slope(points)
    p1 = bounds['min_y_p']
    p2 = bounds['max_y_p']
    p3 = bounds['min_z_p']
    p4 = bounds['max_z_p']
    ps = [p1, p2, p3, p4]
    dis_l = 0
    a = p1
    b = p2
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            dis = math.sqrt((ps[i][1] - ps[j][1]) ** 2 +
                            (ps[i][2] - ps[j][2]) ** 2)
            if dis > dis_l:
                a = ps[i]
                b = ps[j]
    # 计算4条边
    # y = k * x + b
    l1_b = a[2] - k * a[1]
    l2_b = b[2] - (-1 / k) * b[1]
    l3_b = a[2] - (-1 / k) * a[1]
    l4_b = b[2] - k * b[1]
    n_b_2_a_y = (l2_b - l1_b) / (k + 1 / k)
    n_b_2_a_z = k * (l2_b - l1_b) / (k + 1 / k) + l1_b
    n_a_2_b_y = (l3_b - l4_b) / (k + 1 / k)
    n_a_2_b_z = k * (l3_b - l4_b) / (k + 1 / k) + l4_b

    coners = [[a[1], a[2]], [n_a_2_b_y, n_a_2_b_z], [
        b[1], b[2]], [n_b_2_a_y, n_b_2_a_z], [a[1], a[2]]]

    return coners


def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return pcd.points


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


def vis_points(points, out_path):
    org_xyz_all = np.array([0, 0, 0])
    org_xyz_all = np.vstack((org_xyz_all, points))
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

    rec = create_rec(points)
    rec = np.array(rec)
    dealt_x = rec[0][0] - rec[1][0]
    dealt_y = rec[0][1] - rec[1][1]
    d1 = math.sqrt((dealt_x**2) + (dealt_y**2))
    dealt_x_2 = rec[1][0] - rec[2][0]
    dealt_y_2 = rec[1][1] - rec[2][1]
    d2 = math.sqrt((dealt_x_2**2)+(dealt_y_2**2))
    length = 0
    width = 0
    if d1 > d2:
        length = round(d1, 3)
        width = round(d2, 3)
    elif (d1 <= d2):
        length = round(d2, 3)
        width = round(d1, 3)

    ax.plot(rec[:, 0], rec[:, 1], 'y--')
    title = title + \
        f"\n filter vehicle length: {length}m, vehicle width: {width}m\n"

    plt.title(title)
    plt.axis('equal')
    plt.legend()
    fig.savefig(out_path)


def rotate_points(points, box):
    rotation_matrix = np.linalg.inv(box.matrix)

    # 将点集进行旋转
    num_points = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((num_points, 1))))
    rotated_points = homogeneous_points @ rotation_matrix.T
    return rotated_points[:, 0:3]  # z y x


def rasterize_points(points, box, grid_size=0.1):

    min_z = -1 * box.l
    max_z = box.l
    min_y = -1 * box.w
    max_y = box.w

    rows = int((max_z - min_z) / grid_size) + 1
    cols = int((max_y - min_y) / grid_size) + 1

    # 创建栅格 (二维数组)
    raster_grid = [[] for i in range(rows * cols)]

    # 将点放入相应的栅格中
    for i in range(len(points)):
        point = points[i]
        grid_x = int((point[0] - min_z) / grid_size)
        grid_y = int((point[1] - min_y) / grid_size)
        if grid_x > 0 and grid_y > 0 and grid_x < rows and grid_y < cols:
            p = PointXYZI(x=0.0, y=point[0], z=point[1])
            raster_grid[grid_x][grid_y].append(p)

    return raster_grid, [min_z, max_z, min_y, max_y]


filename = '/home/demo/Documents/datasets/pcd/scence_04_track_id_7_idx_166801.pcd'
# 示例点集
points = read_pcd(filename)
points = np.asarray(points)
vis_points(points, 'tt.png')
# best_k, min_range = find_best_slope(points)
# print(f"最佳斜率: {best_k}, 最小范围: {min_range}")
