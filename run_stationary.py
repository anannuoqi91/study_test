import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
import yaml
from utils.util_tools import inverse_transform
from pyntcloud import PyntCloud
import os


def read_matrix_from_yaml(file):
    out = {}
    with open(file, 'r') as f:
        data = yaml.safe_load(f)
        for key, value in data.items():
            if 'lidar_' in key and 'transform' in value:
                tmp = [float(i) for i in value['transform'].strip().split(' ')]
                out[key] = np.array(tmp).reshape(4, 4)
    return out


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
    return mps, m_per_ms


def dis_to_o(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


def dis(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x1 - x2) * (x1 - x2) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1))


def extract_points(files, dir):
    cloud = PyntCloud.from_file(os.path.join(dir, files['01']['name']))
    points_df = cloud.points
    points_df_01 = points_df[(points_df['x'] > files['01']['x'][0]) & (
        points_df['x'] < files['01']['x'][1]) & (points_df['y'] > files['01']['y'][0]) & (
        points_df['y'] < files['01']['y'][1]) & (points_df['z'] > files['01']['z'][0]) & (
        points_df['z'] < files['01']['z'][1])]
    cloud = PyntCloud.from_file(os.path.join(dir, files['02']['name']))
    points_df = cloud.points
    points_df_02 = points_df[(points_df['x'] > files['02']['x'][0]) & (
        points_df['x'] < files['02']['x'][1]) & (points_df['y'] > files['02']['y'][0]) & (
        points_df['y'] < files['02']['y'][1]) & (points_df['z'] > files['02']['z'][0]) & (
        points_df['z'] < files['02']['z'][1])]

    points_df_01['dis_02'] = np.nan
    points_df_01['time_02'] = np.nan
    for index, row in points_df_01.iterrows():
        x = row['x']
        y = row['y']
        z = row['z']
        points_df_02['tmp_dis'] = points_df_02.apply(
            lambda r: dis(x, y, z, r['x'], r['y'], r['z']), axis=1)
        min_row_index = points_df_02['tmp_dis'].idxmin()
        if points_df_02.at[min_row_index, 'tmp_dis'] < 1.0:
            points_df_01.at[index,
                            'dis_02'] = points_df_02.at[min_row_index, 'tmp_dis']
            points_df_01.at[index,
                            'time_02'] = points_df_02.at[min_row_index, 'time_ms']
            points_df_02.drop(min_row_index, inplace=True)
    return points_df_01
    # points_df_01.to_csv('./points_df_01.csv', index=False)


def grid_data(df, res):
    key_f = "{row}-{col}"
    out = {}
    y_min = df['y'].min()
    z_min = df['z'].min()
    for index, row in df.iterrows():
        r = int((row['z'] - z_min) / res)
        c = int((row['y'] - y_min) / res)
        k = key_f.format(row=r, col=c)
        if k in out:
            out[k]['x'].append(row['x'])
            out[k]['deat_t'].append(row['time_ms'] - row['time_02'])
        else:
            out[k] = {
                'x': [row['x']],
                'deat_t': [row['time_ms'] - row['time_02']]
            }
    return out


def grid_data_h(df, res=0.3):
    out = {}
    y_min = df['y'].min()
    z_min = df['z'].min()
    y_max = df['y'].max()
    z_max = df['z'].max()
    rs = int((z_max - z_min) / res) + 1
    cs = int((y_max - y_min) / res) + 1
    out = [[None for j in range(cs)] for i in range(rs)]
    for index, row in df.iterrows():
        r = int((row['z'] - z_min) / res)
        c = int((row['y'] - y_min) / res)
        if out[r][c] is None:
            out[r][c] = [row['time_ms'] - row['time_02']]
        else:
            out[r][c].append(row['time_ms'] - row['time_02'])
    return out, rs, cs


def scaler_to_10(value):
    while value > 10:
        value = value / 10.0
    return int(value)


def plan(data, out_dir):
    plt.figure(figsize=(3, 10))  # 设置图形尺寸
    # 使用 viridis 配色方案
    plt.imshow(data, cmap='viridis', interpolation='nearest')

    # 添加颜色条
    plt.colorbar(label='Value')  # 显示对应颜色的值

    # 设置标题和标签
    plt.title('Grid Plot (deata time ms)')
    plt.xlabel('y m')
    plt.ylabel('z m')

    # 显示图形
    plt.savefig(out_dir, format='png')


def plan_line(data_dic, out_dir):
    for k, v in data_dic.items():
        if len(v['x']) > 5:
            data = pd.DataFrame(columns=['h', 'v'])
            data['h'] = v['x']
            data['v'] = v['deat_t']
            data = data.sort_values(by='h')
            x = data['h'].to_list()
            y = data['v'].to_list()
            plt.figure(figsize=(15, 5))
            plt.plot(x, y, marker='o')  # 'o' 表示在数据点上加上圆圈标记
            plt.title("deat_time_ms")
            plt.xlabel("x m")
            plt.xticks(fontsize=8)
            plt.ylabel("deat_time_ms")

            # 显示网格
            plt.grid()
            plt.savefig(os.path.join(
                out_dir, f'{k}.png'), format='png', dpi=300)


if __name__ == "__main__":
    # fusion_file = '/home/demo/Documents/datasets/s228/two0905/matrix/fusion_falcon.yaml'
    # fusion_matrix = read_matrix_from_yaml(fusion_file)
    # data = pd.read_csv('./s228_stationary.csv')
    # data.columns = [i.lower() for i in data.columns]
    # data_02 = data[['x_2', 'y_2', 'z_2']]
    # data_02_org = inverse_transform(data_02, fusion_matrix['lidar_02'])
    # data[['x_2', 'y_2', 'z_2']] = data_02_org
    # data['dis_01'] = data.apply(
    #     lambda r: dis_to_o(0, r['y_1'], r['z_1']), axis=1)
    # data['dis_02'] = data.apply(
    #     lambda r: dis_to_o(0, r['y_2'], r['z_2']), axis=1)
    files = {
        '01': {
            'name': '01/trans_T20240905_083426_LiDAR_96_D1000/T20240905_083426_LiDAR_96_D1000-1.pcd',
            'x': [-0.144745, 3.93939],
            'y': [-5.81986, -2.57225],
            'z': [19.5904, 36.477]
        },
        '02': {
            'name': '01/trans_T20240905_083426_LiDAR_97_D1000/T20240905_083426_LiDAR_97_D1000-0.pcd',
            'x': [-0.259938, 3.80575],
            'y': [-5.81742, -2.78213],
            'z': [19.8024, 36.9669]
        }
    }
    dir = '/home/demo/Documents/datasets/pcd/'
    stationary_points = extract_points(files, dir)
    stationary_points.to_csv('./points_df_01.csv', index=False)
    # stationary_points = pd.read_csv('./points_df_01.csv')
    stationary_points.columns = [i.lower() for i in stationary_points.columns]
    df_no_nan_rows = stationary_points.dropna()
    res = 10
    h_1, r, c = grid_data_h(df_no_nan_rows, res / 10.0)
    data_avg = [[0 for j in range(c)] for i in range(r)]
    data_min = [[0 for j in range(c)] for i in range(r)]
    data_max = [[0 for j in range(c)] for i in range(r)]
    data_median = [[0 for j in range(c)] for i in range(r)]
    for i in range(r):
        for j in range(c):
            if h_1[i][j] is None:
                continue
            data_avg[i][j] = np.average(np.array(h_1[i][j]))
            data_min[i][j] = np.min(np.array(h_1[i][j]))
            data_max[i][j] = np.max(np.array(h_1[i][j]))
            data_median[i][j] = np.median(np.array(h_1[i][j]))
    plan(data_avg, f'./data_avg_{res}.png')
    plan(data_min, f'./data_min_{res}.png')
    plan(data_max, f'./data_max_{res}.png')
    plan(data_median, f'./data_median_{res}.png')
    grid_df = grid_data(df_no_nan_rows, res / 10.0)
    plan_line(grid_df, f'./time_line/')
