import numpy as np
from .util_angle import norm_angle_radius


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
