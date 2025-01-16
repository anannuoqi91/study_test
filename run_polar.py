from pyntcloud import PyntCloud
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from utils.util_tools import points_transformation, inverse_transform
from pre_fusion.tools import read_matrix_from_yaml


def convert_polar(points_df, x_name='x', y_name='y', z_name='z'):
    points_df['all'] = points_df.apply(
        lambda r: cartesian_to_polar(r[z_name], r[y_name], r[x_name]), axis=1
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


def cal_img(df, res, bz='median', x_name='x', y_name='y', z_name='z'):
    points_df = convert_polar(df, x_name, y_name, z_name)
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


def cal_deata_ms(df):
    df['actual_ms'] = df.apply(
        lambda r: r['time_s'] * 1000 + r['time_ms'], axis=1)
    min_ms = df['actual_ms'].min()
    df['deata_ms'] = df['actual_ms'].apply(lambda x: x - min_ms)
    return df


def get_pts(dir, bz='_96_'):
    files = os.listdir(dir)
    cal_files = []
    for file in files:
        one_dir = os.path.join(dir, file)
        if not os.path.isdir(one_dir):
            continue
        next_files = os.listdir(one_dir)
        for next_file in next_files:
            next_dir = os.path.join(one_dir, next_file)
            if not os.path.isdir(next_dir):
                continue
            if 'trans_' in next_file or 'parallel' in next_file:
                continue
            if bz not in next_file:
                continue

            final_files = os.listdir(next_dir)
            random_files = random.sample(final_files, 1)
            cal_files.extend([os.path.join(next_dir, i_f)
                              for i_f in random_files])
    return deal_pts(cal_files), len(cal_files)


def get_pts_yz(dir, bz='_96_'):
    files = os.listdir(dir)
    cal_files = []
    for file in files:
        one_dir = os.path.join(dir, file)
        if not os.path.isdir(one_dir):
            continue
        next_files = os.listdir(one_dir)
        for next_file in next_files:
            next_dir = os.path.join(one_dir, next_file)
            if not os.path.isdir(next_dir):
                continue
            if 'trans_' in next_file or 'parallel' in next_file:
                if bz not in next_file:
                    continue
                final_files = os.listdir(next_dir)
                random_files = random.sample(final_files, 2)
                cal_files.extend([os.path.join(next_dir, i_f)
                                  for i_f in random_files])
    return deal_pts(cal_files), len(cal_files)


def deal_pts(files):
    out = None
    for file in files:
        cloud = PyntCloud.from_file(file)
        points_df = cloud.points
        points_df = cal_deata_ms(points_df)
        if points_df['deata_ms'].max() > 100:
            points_df = points_df[points_df['deata_ms'] < 100]
        if out is None:
            out = points_df
        else:
            out = pd.concat([out, points_df], axis=0).reset_index(drop=True)
    return out


def save_result(imgs, res, outpath, phi_min, theta_min):
    for k, img in imgs.items():
        r, c = img.shape
        columns_r = [f'c_{i}' for i in range(c)]
        df = pd.DataFrame(img, columns=columns_r)
        df.to_csv(
            f'{outpath}_{res*1000}_{k}_{round(phi_min * 10000)}_{round(theta_min * 10000)}.csv', index=False)
        img_path = f'{outpath}_{res*1000}_{k}_{round(phi_min * 10000)}_{round(theta_min * 10000)}.png'
        save_fig(img, img_path)


def save_fig(img, outpath):

    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap='viridis', interpolation='nearest')

    plt.colorbar(label='Value')
    plt.axis('off')  # 不显示坐标轴
    plt.savefig(outpath, format='png', dpi=300)


def out_h():
    parral_file = '/home/demo/Documents/datasets/s228/two0905/matrix/01_parallel_falcon.yaml'
    parral_matrix = read_matrix_from_yaml(parral_file)
    dir = '/home/demo/Documents/datasets/pcd/'
    out_dir = './out_polor_h/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    res = 0.01
    bz = '_96_'
    df, file_num = get_pts(dir, bz=bz)
    tmp_points = df[['x', 'y', 'z']]
    df[['parallel_x', 'parallel_y', 'parallel_z']
       ] = points_transformation(tmp_points, parral_matrix['lidar_01'])
    h = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    for i in range(len(h) + 1):
        if i == 0:
            tmp_df = df[df['parallel_x'] < h[i]]
        elif i == len(h):
            tmp_df = df[df['parallel_x'] > h[i - 1]]
        else:
            tmp_df = df[(df['parallel_x'] >= h[i - 1])
                        & (df['parallel_x'] < h[i])]
        imgs, phi_min, theta_min = cal_img(tmp_df, res, 'avg')
        save_result(
            imgs, res, f'{out_dir}{file_num}{bz}{i}', phi_min, theta_min)


def out_2_to_1():
    parral_file = '/home/demo/Documents/datasets/s228/two0905/matrix/01_parallel_falcon.yaml'
    parral_matrix = read_matrix_from_yaml(parral_file)
    fusion_file = '/home/demo/Documents/datasets/s228/two0905/matrix/fusion_falcon.yaml'
    fusion_matrix = read_matrix_from_yaml(fusion_file)
    dir = '/home/demo/Documents/datasets/pcd/'
    out_dir = './out_polor_2_1/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    res = 0.01
    bz = '_97_'
    df, file_num = get_pts(dir, bz=bz)
    tmp_points = df[['x', 'y', 'z']]
    df[['parallel_x', 'parallel_y', 'parallel_z']] = points_transformation(
        tmp_points, parral_matrix['lidar_02'])
    tmp_points = df[['parallel_x', 'parallel_y', 'parallel_z']]
    df[['fusion_x', 'fusion_y', 'fusion_z']] = points_transformation(
        tmp_points, fusion_matrix['lidar_02']
    )
    tmp_points = df[['fusion_x', 'fusion_y', 'fusion_z']]
    df[['i_fusion_x', 'i_fusion_y', 'i_fusion_z']] = inverse_transform(
        tmp_points, fusion_matrix['lidar_01']
    )

    imgs, phi_min, theta_min = cal_img(
        df, res, bz='avg', x_name='i_fusion_x', y_name='i_fusion_y', z_name='i_fusion_z')
    save_result(
        imgs, res, f'{out_dir}{file_num}{bz}', phi_min, theta_min)


if __name__ == '__main__':
    out_2_to_1()
    # parral_file = '/home/demo/Documents/datasets/s228/two0905/matrix/01_parallel_falcon.yaml'
    # parral_matrix = read_matrix_from_yaml(parral_file)
    # dir = '/home/demo/Documents/datasets/pcd/'
    # out_dir = './out_polor_h/'
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    # res = 0.01
    # bz = '_96_'
    # df, file_num = get_pts(dir, bz=bz)

    # imgs, phi_min, theta_min = cal_img(df, res, 'avg')
    # save_result(
    #     imgs, res, f'{out_dir}{file_num}{bz}', phi_min, theta_min)

    # out_dir = './out_yz/'
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    # res = 0.3
    # bz = '_96_'
    # df, file_num = get_pts_yz(dir, bz=bz)
    # imgs, x_min, y_min = cal_img_yz(df, res, 'all')
    # save_result(imgs, res, f'{out_dir}{file_num}{bz}', x_min, y_min)

# file = '/home/demo/Documents/datasets/pcd/08_2/T20240905_082424_LiDAR_97_D1000/T20240905_082424_LiDAR_97_D1000-45.pcd'
# cloud= PyntCloud.from_file(file)
# points_df= cloud.points
# points_df= convert_polar(points_df)
# phi_min= points_df['phi'].min()
# phi_max= points_df['phi'].max()
# theta_min= points_df['theta'].min()
# theta_max= points_df['theta'].max()
# min_ms= points_df['time_ms'].min()
# res= 0.01
# r= int((phi_max - phi_min) / res) + 1
# c= int((theta_max - theta_min) / res) + 1
# img= np.zeros((r, c))
# for i in range(r):
#     start_phi= i * res + phi_min
#     end_phi= (i + 1) * res + phi_min
#     for j in range(c):
#         start_theta= j * res + theta_min
#         end_theta= (j + 1) * res + theta_min
#         tmp= points_df[(points_df['phi'] >= start_phi) &
#                         (points_df['phi'] < end_phi) & (points_df['theta'] >= start_theta) &
#                         (points_df['theta'] < end_theta)]
#         if len(tmp) > 0:
#             # tmp_ms = np.median(tmp['time_ms'])
#             tmp_ms= np.average(tmp['time_ms'])
#             img[i][j]= int(tmp_ms - min_ms)

# columns_r= [f'c_{i}' for i in range(c)]
# df= pd.DataFrame(img, columns=columns_r)
# df.to_csv(f'./out_csv/polar_97_{res*1000}.csv', index=False)

# print(phi_min)
# print(theta_min)
# # 创建一个新的图形
# plt.figure(figsize=(8, 6))

# # 显示图像
# plt.imshow(img)
# plt.axis('off')  # 不显示坐标轴
# plt.title('polar')
# plt.savefig('./out_img/polar_96_avg.png', format='png', dpi=300)
# # 显示图形
# plt.show()
