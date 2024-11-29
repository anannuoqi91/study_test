import numpy as np


def heading_diff(heading1: float, heading2: float):
    h1 = [np.cos(heading1), np.sin(heading1), 0, 1]
    h2 = [np.cos(heading2), np.sin(heading2), 0, 1]
    dot_product = h1[0] * h2[0] + h1[1] * h2[1]
    h1_l = np.sqrt(h1[0] * h1[0] + h1[1] * h1[1])
    h2_l = np.sqrt(h2[0] * h2[0] + h2[1] * h2[1])
    return abs(np.acos(dot_product / (h1_l * h2_l + 1e-6)))
