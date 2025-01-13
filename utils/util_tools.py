import numpy as np
from .util_angle import norm_angle_radius
import sys


def projection_bev(arrow, yaw_rad):
    """
    Project a 2D arrow onto the BEV (bird's eye view) coordinate system.

    Parameters
    ----------
    arrow : array_like(front, right)
        The 2D arrow to project, with the x-axis pointing to the front of the object.
    yaw_rad : float
        The yaw angle of the object (in radians) in the world coordinate system.

    Returns
    -------
    projected_vector : array_like
        The projected vector in the BEV coordinate system.
    """
    dist = np.linalg.norm(arrow)
    diff = norm_angle_radius(np.arctan2(arrow[0], arrow[1])) - yaw_rad
    projected_vector = [dist * np.cos(diff), dist * np.sin(diff)]

    return projected_vector


def coord_transform(arrow, yaw_rad, dx, dy):
    """
    Transform a 2D arrow in the world coordinate system to a 2D arrow in the BEV coordinate system.

    Parameters
    ----------
    arrow : array_like(front, right)
        The 2D arrow to transform, with the x-axis pointing to the front of the object.
    yaw_rad : float
        The yaw angle of the object (in radians) in the world coordinate system.
    dx : float
        The x-coordinate of the object's center in the world coordinate system.
    dy : float
        The y-coordinate of the object's center in the world coordinate system.

    Returns
    -------
    transformed_arrow : array_like(front, right)
        The transformed 2D arrow in the BEV coordinate system.


    x' = x * cos(θ) - y * sin(θ) + dx
    y' = x-cx) * sin(θ) + (y-cy) * cos(θ) + cy
    """
    pass


def print_progress_bar(iteration, total, length=40, pre_message=''):
    percent = (iteration / total)
    arrow = '>' * int(round(percent * length) - 1)
    spaces = ' ' * (length - len(arrow))
    sys.stdout.write(
        f'\rProgress: {pre_message} | {arrow}{spaces}| {percent:.2%}')
    sys.stdout.flush()


def get_rotation_traslation(trans_matrix):
    """
    从 4x4 转换矩阵中提取旋转矩阵和平移向量

    参数:
    transform_matrix: np.ndarray, 形状为 (4, 4) 的转换矩阵

    返回:
    rotation_matrix: np.ndarray, 形状为 (3, 3) 的旋转矩阵
    translation_vector: np.ndarray, 形状为 (3,) 的平移向量
    """

    if trans_matrix.shape != (4, 4):
        raise ValueError("transform_matrix must have shape (4, 4)")

    rotation_matrix = trans_matrix[:3, :3]
    translation_vector = trans_matrix[:3, 3]

    return rotation_matrix, translation_vector


def points_transformation(points, matrix):
    points = np.array(points)
    if points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    rotation_matrix, translation_vector = get_rotation_traslation(matrix)
    transformed_points = points @ rotation_matrix.T + translation_vector
    return transformed_points


def inverse_transform(points, transform_matrix):
    """
    从转换后的点云集合 C 和转换矩阵 B 恢复出原始点云集合 A。

    参数:
    points: np.ndarray, 形状为 (N, 3) 的点云集合 C（齐次坐标）
    transform_matrix_B: np.ndarray, 形状为 (4, 4) 的转换矩阵 B

    返回:
    point_cloud: np.ndarray, 形状为 (N, 3) 的点云集合 A（齐次坐标）
    """

    n = points.shape[0]
    homogeneous = np.hstack((points, np.ones((n, 1))))

    # 计算逆转换矩阵 B^{-1}
    transform_matrix_inv = np.linalg.inv(transform_matrix)

    # 使用逆矩阵将点云 C 转换回 A
    homogeneous_out = homogeneous @ transform_matrix_inv.T  # 矩阵乘法

    # 去掉齐次坐标的最后一维，得到 A
    return homogeneous_out[:, :3]


def cartesian_to_polar(x, y, z):
    # 计算 r
    r = np.sqrt(x**2 + y**2 + z**2)

    # 计算 θ
    theta = np.arctan2(y, x)  # 使用 arctan2 考虑象限

    # 计算 φ
    phi = np.arccos(z / r) if r != 0 else 0  # 避免除以零的情况

    return r, theta, phi
