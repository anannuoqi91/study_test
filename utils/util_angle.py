import math


def norm_angle(angle: float) -> float:
    # 返回的角度在 [0, 2π) 范围内
    two_pi = 2.0 * math.pi
    angle_norm = angle % two_pi
    return angle_norm if angle_norm >= 0 else angle_norm + two_pi
