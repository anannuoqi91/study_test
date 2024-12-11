import math


def norm_angle_radius(angle: float) -> float:
    """
    Normalize an angle to [0, 2π).

    Parameters
    ----------
    angle : float
        The angle to be normalized.

    Returns
    -------
    float
        The normalized angle.
    """

    two_pi = 2.0 * math.pi
    angle_norm = angle % two_pi
    return angle_norm if angle_norm >= 0 else angle_norm + two_pi


def diff_angle_radius(angle1: float, angle2: float) -> float:
    diff = norm_angle_radius(angle1 - angle2)
    return min(diff, 2 * math.pi - diff)
