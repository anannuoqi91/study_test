"""
车沿着z轴行使
车长 l
车宽 w
车高 h
模拟点云
假设车辆行使方向与z轴完全一致
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from utils.util_tools import inverse_transform


def read_matrix_from_yaml(file):
    out = {}
    with open(file, 'r') as f:
        data = yaml.safe_load(f)
        for key, value in data.items():
            if 'lidar_' in key and 'transform' in value:
                tmp = [float(i) for i in value['transform'].strip().split(' ')]
                out[key] = np.array(tmp).reshape(4, 4)
    return out


def car_points(l, w, h, center):
    x, y, z = center
    x_h = [x - h / 2, x + h / 2]
    y_w = [y - w / 2, y + w / 2]
    z_l = [z - l / 2, z + l / 2]
    points = np.array([
        [x_h[0], y_w[0], z_l[0]],
        [x_h[0], y_w[0], z_l[1]],
        [x_h[0], y_w[1], z_l[0]],
        [x_h[0], y_w[1], z_l[1]],
        [x_h[1], y_w[0], z_l[0]],
        [x_h[1], y_w[0], z_l[1]],
        [x_h[1], y_w[1], z_l[0]],
        [x_h[1], y_w[1], z_l[1]],

    ])
    points = pd.DataFrame(points, columns=['x', 'y', 'z'])
    points['bz'] = ['d-l-s', 'd-l-e', 'd-r-s',
                    'd-r-e', 'u-l-s', 'u-l-e', 'u-r-s', 'u-r-e']
    return points


def cartesian_to_polar(x, y, z):
    # 计算 r
    r = np.sqrt(x**2 + y**2 + z**2)

    # 计算 θ
    theta = np.arctan2(y, x)  # 使用 arctan2 考虑象限

    # 计算 φ
    phi = np.arccos(z / r) if r != 0 else 0  # 避免除以零的情况

    return r, theta, phi


def read_polar(path):
    model_name = os.path.basename(path).replace('.csv', '').split('_')
    theta_min = float(model_name[-1]) / 10000
    phi_min = float(model_name[-2]) / 10000
    res = float(model_name[-4]) / 1000
    time_model = pd.read_csv(path)
    time_model = np.array(time_model)
    out = {
        'model': time_model,
        'theta_min': theta_min,
        'phi_min': phi_min,
        'res': res
    }
    return out


def cal_deata_ms(phi, theta, time_model, time_model_min, time_model_res):
    rs, cs = time_model.shape
    phi_min, theta_min = time_model_min
    r = int((phi - phi_min) / time_model_res)
    c = int((theta - theta_min) / time_model_res)
    if r >= 0 and r < rs and c >= 0 and c < cs:
        return time_model[r][c]
    return 0


def cal_points_ms(time_model, points):
    points['deata_ms_1'] = 0
    for index, row in points.iterrows():
        x, y, z = row['x'], row['y'], row['z']
        r, theta, phi = cartesian_to_polar(z, y, x)
        deata_ms = cal_deata_ms(phi, theta, time_model['model'], [
                                time_model['phi_min'], time_model['theta_min']], time_model['res'])
        points.at[index, 'deata_ms_1'] = deata_ms
    return points


def cal_points_ms_2(time_model, points):
    points['deata_ms_2'] = 0
    for index, row in points.iterrows():
        x, y, z = row['x_2'], row['y_2'], row['z_2']
        r, theta, phi = cartesian_to_polar(z, y, x)
        deata_ms = cal_deata_ms(phi, theta, time_model['model'], [
                                time_model['phi_min'], time_model['theta_min']], time_model['res'])
        points.at[index, 'deata_ms_2'] = deata_ms
    return points


def main_z(car, center_z, model_path, outpath='./out_csv/'):
    out = None
    model_dic = read_polar(model_path)
    for i_z in center_z:
        print(i_z)
        points = car_points(car['l'], car['w'], car['h'], [
                            car['h'] / 2, car['w'] / 2, i_z])
        points = cal_points_ms(model_dic, points)
        points['z'] = i_z
        if out is None:
            out = points
        else:
            out = pd.concat([out, points])
    if len(out) == 0:
        print(center_z)
        return
    out.to_csv(
        f'{outpath}_lwh-{car["l"]}_{car["w"]}_{car["h"]}_z-{center_z[0]}_{center_z[-1]}.csv', index=False)
    plan_line(
        out, 'bz', f'{outpath}_lwh-{car["l"]}_{car["w"]}_{car["h"]}_z-{center_z[0]}_{center_z[-1]}.png')
    return out


def convert_coordinate(points_df, matrix):
    data_02 = points_df[['x', 'y', 'z']]
    data_02_org = inverse_transform(data_02, matrix)
    points_df[['x_2', 'y_2', 'z_2']] = data_02_org
    return points_df


def main_z_single(car, center_y, center_z, model_path, matrixpath, outpath='./out_csv/'):
    out = None
    model_dic_1 = read_polar(model_path[0])
    model_dic_2 = read_polar(model_path[1])
    fusion_matrix = read_matrix_from_yaml(matrixpath)['lidar_02']
    for i_z in center_z:
        points = car_points(car['l'], car['w'], car['h'], [
                            car['h'] / 2, center_y, i_z])
        points = cal_points_ms(model_dic_1, points)
        points = convert_coordinate(points, fusion_matrix)
        points = cal_points_ms_2(model_dic_2, points)
        points['z'] = i_z
        if out is None:
            out = points
        else:
            out = pd.concat([out, points])
    if len(out) == 0:
        print(center_z)
        return

    out['deata_ms'] = out.apply(lambda r: abs(
        r['deata_ms_1'] - r['deata_ms_2']), axis=1)
    out.to_csv(
        f'{outpath}m_lwh-{car["l"]}_{car["w"]}_{car["h"]}_z-{center_z[0]}_{center_z[-1]}.csv', index=False)
    plan_line(
        out, 'bz', f'{outpath}m_lwh-{car["l"]}_{car["w"]}_{car["h"]}_z-{center_z[0]}_{center_z[-1]}.png')
    return out


def main_y_single(car, center_y, center_z, model_path, matrixpath, outpath='./out_csv/'):
    out = None
    model_dic_1 = read_polar(model_path[0])
    model_dic_2 = read_polar(model_path[1])
    fusion_matrix = read_matrix_from_yaml(matrixpath)['lidar_02']
    for i_z in center_y:
        points = car_points(car['l'], car['w'], car['h'], [
                            car['h'] / 2, i_z, center_z])
        points = cal_points_ms(model_dic_1, points)
        points = convert_coordinate(points, fusion_matrix)
        points = cal_points_ms_2(model_dic_2, points)
        points['y'] = i_z
        if out is None:
            out = points
        else:
            out = pd.concat([out, points])
    if len(out) == 0:
        return

    out['deata_ms'] = out.apply(lambda r: abs(
        r['deata_ms_1'] - r['deata_ms_2']), axis=1)
    out.to_csv(
        f'{outpath}m_lwh-{car["l"]}_{car["w"]}_{car["h"]}_y-{center_y[0]}_{center_y[-1]}.csv', index=False)
    plan_line(
        out, 'bz', f'{outpath}m_lwh-{car["l"]}_{car["w"]}_{car["h"]}_y-{center_y[0]}_{center_y[-1]}.png', x_col='y')
    return out


def plan_line(df, col, outpath, x_col='z'):
    plt.figure(figsize=(15, 5))
    col_l = np.unique(df[col])
    for i in col_l:
        df_l = df[df[col] == i]
        df_l = df_l.sort_values(by=x_col)
        x = df_l[x_col].tolist()
        y = df_l['deata_ms'].tolist()
        plt.plot(x, y, label=i)
    plt.xlabel(f"{x_col} m")
    plt.ylabel("deat_time_ms")
    plt.legend()
    plt.savefig(f'{outpath}', format='png', dpi=300)


if __name__ == '__main__':
    dual_model_path = './out_polar/dual_25_10.0_avg_11459_-10062.csv'
    model_path_1 = './out_polar/24_96__10.0_avg_13528_-10463.csv'
    model_path_2 = './out_polar/24_97__10.0_avg_13588_-10461.csv'
    car_1 = {
        'l': 4.640,
        'w': 1.780,
        'h': 1.435
    }
    car_2 = {
        'l': 8.640,
        'w': 2.380,
        'h': 2.435
    }
    car_3 = {
        'l': 15.640,
        'w': 2.380,
        'h': 3.435
    }
    car_4 = {
        'l': 17.640,
        'w': 2.780,
        'h': 4.135
    }

    fusion_file = '/home/demo/Documents/datasets/s228/two0905/matrix/fusion_falcon.yaml'

    for i in [car_1, car_2, car_3, car_4]:

        center_z = np.arange(10, 50.5, 0.5).tolist()
        center_y = -10.0
        main_z_single(i, center_y, center_z, [model_path_1, model_path_2],
                      fusion_file, outpath=f'./out_csv/y_{center_y}')
        center_y = np.arange(-20.5, 0, 1.0).tolist()
        center_z = 30.0
        main_y_single(i, center_y, center_z, [model_path_1, model_path_2],
                      fusion_file, outpath=f'./out_csv/z_{center_z}')
    # main_z(car_b, center_z, dual_model_path)
