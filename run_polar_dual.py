from pyntcloud import PyntCloud
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from utils.util_tools import inverse_transform
import yaml


def convert_polar(points_df):
    points_df['all'] = points_df.apply(
        lambda r: cartesian_to_polar(r['z'], r['y'], r['x']), axis=1
    )
    points_df['rho'] = points_df['all'].apply(lambda x: x[0])
    points_df['theta'] = points_df['all'].apply(lambda x: x[1])
    points_df['phi'] = points_df['all'].apply(lambda x: x[2])
    points_df.drop('all', axis=1, inplace=True)
    return points_df


def cartesian_to_polar(x, y, z):
    # 计算 r
    r = np.sqrt(x**2 + y**2 + z**2)

    # 计算 θ
    theta = np.arctan2(y, x)  # 使用 arctan2 考虑象限

    # 计算 φ
    phi = np.arccos(z / r) if r != 0 else 0  # 避免除以零的情况

    return r, theta, phi


def cal_img(df, res, bz='median'):
    points_df = convert_polar(df)
    phi_min = points_df['phi'].min()
    phi_max = points_df['phi'].max()
    theta_min = points_df['theta'].min()
    theta_max = points_df['theta'].max()
    r = int((phi_max - phi_min) / res) + 1
    c = int((theta_max - theta_min) / res) + 1
    imgs = {}
    func = {
        'median': np.median,
        'avg': np.average,
        'min': np.min,
        'max': np.max,
    }
    if bz == 'all':
        imgs = {
            'median': np.zeros((r, c)),
            'avg': np.zeros((r, c)),
            'min': np.zeros((r, c)),
            'max': np.zeros((r, c)),
        }
    else:
        imgs[bz] = np.zeros((r, c))
    for i in range(r):
        start_phi = i * res + phi_min
        end_phi = (i + 1) * res + phi_min
        for j in range(c):
            start_theta = j * res + theta_min
            end_theta = (j + 1) * res + theta_min
            tmp = points_df[(points_df['phi'] >= start_phi) &
                            (points_df['phi'] < end_phi) & (points_df['theta'] >= start_theta) &
                            (points_df['theta'] < end_theta)]
            if len(tmp) > 0:
                if bz == 'all':
                    for k in func:
                        imgs[k][i][j] = func[k](tmp['deata_ms'])
                else:
                    imgs[bz][i][j] = func[bz](tmp['deata_ms'])

    return imgs, phi_min, theta_min


def cal_img_yz(df, res, bz='median'):
    points_df = df
    x_min = points_df['z'].min()
    x_max = points_df['z'].max()
    y_min = points_df['y'].min()
    y_max = points_df['y'].max()
    r = int((x_max - x_min) / res) + 1
    c = int((y_max - y_min) / res) + 1
    imgs = {}
    func = {
        'median': np.median,
        'avg': np.average,
        'min': np.min,
        'max': np.max,
    }
    if bz == 'all':
        imgs = {
            'median': np.zeros((r, c)),
            'avg': np.zeros((r, c)),
            'min': np.zeros((r, c)),
            'max': np.zeros((r, c)),
        }
    else:
        imgs[bz] = np.zeros((r, c))
    for i in range(r):
        start_phi = i * res + x_min
        end_phi = (i + 1) * res + x_min
        for j in range(c):
            start_theta = j * res + y_min
            end_theta = (j + 1) * res + y_min
            tmp = points_df[(points_df['z'] >= start_phi) &
                            (points_df['z'] < end_phi) & (points_df['y'] >= start_theta) &
                            (points_df['y'] < end_theta)]
            if len(tmp) > 0:
                if bz == 'all':
                    for k in func:
                        imgs[k][i][j] = func[k](tmp['deata_ms'])
                else:
                    imgs[bz][i][j] = func[bz](tmp['deata_ms'])

    return imgs, x_min, y_min


def cal_actual_ms(df):
    df['actual_ms'] = df.apply(
        lambda r: r['time_s'] * 1000 + r['time_ms'], axis=1)
    return df


def cal_deata_ms(df):
    min_ms = df['actual_ms'].min()
    df['deata_ms'] = df['actual_ms'].apply(
        lambda x: x - min_ms)
    return df


def get_pts(dir, matrix):
    files = os.listdir(dir)
    out = None
    i = 0
    for file in files:
        if i > 20:
            break
        i = i + 1
        file_path = os.path.join(dir, file)
        if not os.path.exists(file_path) and not file.endwith('.pcd'):
            continue
        lidar_02_path = file_path.replace('_96_', '_97_')
        lidar_01 = deal_pts(file_path)
        lidar_02 = deal_pts(lidar_02_path)
        tmp = pd.concat([lidar_01, lidar_02], axis=0).reset_index(drop=True)
        tmp = convert_coordinate(tmp, matrix)
        tmp = cal_deata_ms(tmp)
        if out is None:
            out = tmp
        else:
            out = pd.concat([out, tmp], axis=0).reset_index(drop=True)

    return out, len(files)


def get_pts_yz(dir):
    files = os.listdir(dir)
    out = None
    i = 0
    for file in files:
        if i > 10:
            break
        i = i + 1
        file_path = os.path.join(dir, file)
        if not os.path.exists(file_path) and not file.endwith('.pcd'):
            continue
        lidar_02_path = file_path.replace('_96_', '_97_')
        lidar_01 = deal_pts(file_path)
        lidar_02 = deal_pts(lidar_02_path)
        tmp = pd.concat([lidar_01, lidar_02], axis=0).reset_index(drop=True)
        tmp = cal_deata_ms(tmp)
        if out is None:
            out = tmp
        else:
            out = pd.concat([out, tmp], axis=0).reset_index(drop=True)

    return out, len(files)


def convert_coordinate(points_df, matrix):
    data_02 = points_df[['x', 'y', 'z']]
    data_02_org = inverse_transform(data_02, matrix)
    points_df[['x', 'y', 'z']] = data_02_org
    return points_df


def deal_pts(file):
    cloud = PyntCloud.from_file(file)
    points_df = cloud.points
    points_df = cal_actual_ms(points_df)
    return points_df


def save_result(imgs, res, outpath, phi_min, theta_min):
    for k, img in imgs.items():
        r, c = img.shape
        columns_r = [f'c_{i}' for i in range(c)]
        df = pd.DataFrame(img, columns=columns_r)
        df.to_csv(
            f'{outpath}_{res*1000}_{k}_{round(phi_min * 10000)}_{round(theta_min * 10000)}.csv', index=False)
        img_path = f'{outpath}_{res*1000}_{k}_{round(phi_min * 10000)}_{round(theta_min * 10000)}.png'
        save_fig(img, img_path)


def read_matrix_from_yaml(file):
    out = {}
    with open(file, 'r') as f:
        data = yaml.safe_load(f)
        for key, value in data.items():
            if 'lidar_' in key and 'transform' in value:
                tmp = [float(i) for i in value['transform'].strip().split(' ')]
                out[key] = np.array(tmp).reshape(4, 4)
    return out


def save_fig(img, outpath):
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap='viridis', interpolation='nearest')

    plt.colorbar(label='Value')
    plt.axis('off')  # 不显示坐标轴
    plt.savefig(outpath, format='png', dpi=300)


if __name__ == '__main__':
    # dir = '/home/demo/Documents/datasets/pcd/08_2/T20240905_082424_LiDAR_96_D1000/'
    # out_dir = './out_polor/'
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    # res = 0.01
    # parral_file = '/home/demo/Documents/datasets/s228/two0905/matrix/01_parallel_falcon.yaml'
    # parral_matrix = read_matrix_from_yaml(parral_file)
    # df, file_num = get_pts(dir, parral_matrix['lidar_01'])
    # imgs, phi_min, theta_min = cal_img(df, res, 'all')
    # save_result(imgs, res, f'{out_dir}dual_{file_num}', phi_min, theta_min)

    dir = '/home/demo/Documents/datasets/pcd/08_2/trans_T20240905_082424_LiDAR_96_D1000/'
    out_dir = './out_yz/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    res = 0.3
    df, file_num = get_pts_yz(dir)
    imgs, x_min, y_min = cal_img_yz(df, res, 'all')
    save_result(imgs, res, f'{out_dir}dual_{file_num}', x_min, y_min)
