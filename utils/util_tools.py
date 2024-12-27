import numpy as np
from .util_angle import norm_angle_radius
import math


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


def calculate_bev_corners_by_center_yaw(arrow, yaw_rad):
    """
    Calculate the corners of a 2D arrow in the BEV (bird's eye view) coordinate system.

    Parameters
    ----------
    arrow : array_like(front, right)
        The 2D arrow to project, with the x-axis pointing to the front of the object.
    yaw_rad : float
        The yaw angle of the object (in radians) in the world coordinate system.

    Returns
    -------
    corners : array_like
        The corners of the arrow in the BEV coordinate system.
    """

    cosval = math.cos(yaw_rad)
    sinval = math.sin(yaw_rad)
    dz_fl = 0.5 * self._length * cosval - 0.5 * self._width * sinval
    dy_fl = 0.5 * self._length * sinval + 0.5 * self._width * cosval

    dz_bl = -0.5 * self._length * cosval - 0.5 * self._width * sinval
    dy_bl = -0.5 * self._length * sinval + 0.5 * self._width * cosval

    out['upper_front_left'] = PointXYZ(
        0.5 * self._height, self._center.y + dy_fl, self._center.z + dz_fl)
    out['upper_back_left'] = PointXYZ(
        0.5 * self._height, self._center.y + dy_bl, self._center.z + dz_bl)
    out['upper_back_right'] = PointXYZ(
        0.5 * self._height, self._center.y - dy_fl, self._center.z - dz_fl)
    out['upper_front_right'] = PointXYZ(
        0.5 * self._height, self._center.y - dy_bl, self._center.z - dz_bl)
    out['lower_front_left'] = PointXYZ(
        -0.5 * self._height, self._center.y + dy_fl, self._center.z + dz_fl)
    out['lower_back_left'] = PointXYZ(
        -0.5 * self._height, self._center.y + dy_bl, self._center.z + dz_bl)
    out['lower_back_right'] = PointXYZ(
        -0.5 * self._height, self._center.y - dy_fl, self._center.z - dz_fl)
    out['lower_front_right'] = PointXYZ(
        -0.5 * self._height, self._center.y - dy_bl, self._center.z - dz_bl)
