import yaml
import numpy as np
from utils.util_angle import norm_angle_radius
import matplotlib.pyplot as plt


def read_matrix_from_yaml(file):
    out = {}
    with open(file, 'r') as f:
        data = yaml.safe_load(f)
        for key, value in data.items():
            if 'lidar_' in key and 'transform' in value:
                tmp = [float(i) for i in value['transform'].strip().split(' ')]
                out[key] = np.array(tmp).reshape(4, 4)
    return out


def cal_car(center, size, heading):
    x, y, z = center
    l, w, h = size
    heading_azimuth_rad = norm_angle_radius(heading)
    out = {}
    # front-left, back-left, back-right, front-right
    cosval = np.cos(heading_azimuth_rad)
    sinval = np.sin(heading_azimuth_rad)
    dz_fl = 0.5 * l * cosval - 0.5 * w * sinval
    dy_fl = 0.5 * l * sinval + 0.5 * w * cosval

    dz_bl = -0.5 * l * cosval - 0.5 * w * sinval
    dy_bl = -0.5 * l * sinval + 0.5 * w * cosval

    out['upper_front_left'] = [x + 0.5 * h, y + dy_fl, z + dz_fl]
    out['upper_back_left'] = [x + 0.5 * h, y + dy_bl, z + dz_bl]
    out['upper_back_right'] = [x + 0.5 * h, y - dy_fl, z - dz_fl]
    out['upper_front_right'] = [x + 0.5 * h, y - dy_bl, z - dz_bl]
    out['lower_front_left'] = [x - 0.5 * h, y + dy_fl, z + dz_fl]
    out['lower_back_left'] = [x - 0.5 * h, y + dy_bl, z + dz_bl]
    out['lower_back_right'] = [x - 0.5 * h, y - dy_fl, z - dz_fl]
    out['lower_front_right'] = [x - 0.5 * h, y - dy_bl, z - dz_bl]
    return out


def plan_img(img, outpath, title=''):
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap='viridis', interpolation='nearest')

    plt.colorbar(label='Value')
    plt.axis('off')  # 不显示坐标轴
    plt.title(title)
    plt.savefig(outpath, format='png', dpi=300)


def convert_speed(kmh):
    """
    将速度从 km/h 转换为 m/s 和 m/ms

    参数:
    kmh: float, 以 km/h 为单位的速度

    返回:
    tuple: (m/s, m/ms)
    """
    mps = kmh / 3.6                # 转换为 m/s
    m_per_ms = mps / 1000         # 转换为 m/ms
    return m_per_ms
