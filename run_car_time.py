from utils.car import *
from utils.util_tools import cartesian_to_polar
from pyntcloud import PyntCloud
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def convert_polar(points_df):
    points_df['all'] = points_df.apply(
        lambda r: cartesian_to_polar(r['z'], r['y'], r['x']), axis=1
    )
    points_df['rho'] = points_df['all'].apply(lambda x: x[0])
    points_df['phi'] = points_df['all'].apply(lambda x: x[1])
    points_df['theta'] = points_df['all'].apply(lambda x: x[2])
    points_df.drop('all', axis=1, inplace=True)
    return points_df


def extract_points(file_dic, dir):
    cloud = PyntCloud.from_file(os.path.join(dir, file_dic['name']))
    points_df = cloud.points
    points_df_01 = points_df[(points_df['x'] > file_dic['x'][0]) & (
        points_df['x'] < file_dic['x'][1]) & (points_df['y'] > file_dic['y'][0]) & (
        points_df['y'] < file_dic['y'][1]) & (points_df['z'] > file_dic['z'][0]) & (
        points_df['z'] < file_dic['z'][1])]
    return points_df_01


def extract_points_2(filepath, filter=None):
    cloud = PyntCloud.from_file(filepath)
    points_df = cloud.points
    if filter is not None:
        points_df = points_df[(points_df['x'] > filter['x'][0]) & (points_df['x'] < filter['x'][1]) & (points_df['y'] > filter['y'][0]) & (
            points_df['y'] < filter['y'][1]) & (points_df['z'] > filter['z'][0]) & (points_df['z'] < filter['z'][1])]
    return points_df


def cal_actual_ms(df):
    df['actual_ms'] = df.apply(
        lambda r: r['time_s'] * 1000 + r['time_ms'], axis=1
    )
    return df, df['actual_ms'].min()


def cal_deata_ms(phi, theta, time_model, time_model_min, time_model_res):
    rs, cs = time_model.shape
    phi_min, theta_min = time_model_min
    r = int((phi - phi_min) / time_model_res)
    c = int((theta - theta_min) / time_model_res)
    if r >= 0 and r < rs and c >= 0 and c < cs:
        return time_model[r][c]
    return 0


def plan_polar(imgs, titles, out_dir):
    for i in range(len(imgs)):
        i_img = imgs[i]
        title = titles[i]
        plt.figure(figsize=(8, 6))
        plt.imshow(i_img, cmap='viridis', interpolation='nearest')

        plt.colorbar(label='Value')
        plt.axis('off')  # 不显示坐标轴
        plt.title(title)
        plt.savefig(f'{out_dir}_{title}.png', format='png', dpi=300)


def convert_img(df):
    phi_min = df['phi'].min()
    phi_max = df['phi'].max()
    theta_min = df['theta'].min()
    theta_max = df['theta'].max()
    res = 0.01
    r = int((phi_max - phi_min) / res) + 1
    c = int((theta_max - theta_min) / res) + 1
    img_median = np.zeros((r, c))
    img_avg = np.zeros((r, c))
    img_min = np.zeros((r, c))
    img_max = np.zeros((r, c))

    for i in range(r):
        start_phi = i * res + phi_min
        end_phi = (i + 1) * res + phi_min
        for j in range(c):
            start_theta = j * res + theta_min
            end_theta = (j + 1) * res + theta_min
            tmp = df[(df['phi'] >= start_phi) &
                     (df['phi'] < end_phi) & (df['theta'] >= start_theta) &
                     (df['theta'] < end_theta)]
            if len(tmp) > 0:
                img_median[i][j] = np.median(tmp['deata_ms'])
                img_avg[i][j] = np.average(tmp['deata_ms'])
                img_min[i][j] = np.min(tmp['deata_ms'])
                img_max[i][j] = np.max(tmp['deata_ms'])
    return [img_median, img_avg, img_min, img_max], ['median', 'avg', 'min', 'max']


def convert_img_xy(df):
    x_min = df['z'].min()
    x_max = df['z'].max()
    y_min = df['y'].min()
    y_max = df['y'].max()
    res = 0.1
    r = int((x_max - x_min) / res) + 1
    c = int((y_max - y_min) / res) + 1
    img_median = np.zeros((r, c))
    img_avg = np.zeros((r, c))
    img_min = np.zeros((r, c))
    img_max = np.zeros((r, c))

    for i in range(r):
        start_x = i * res + x_min
        end_x = (i + 1) * res + x_min
        for j in range(c):
            start_y = j * res + y_min
            end_y = (j + 1) * res + y_min
            tmp = df[(df['z'] >= start_x) &
                     (df['z'] < end_x) & (df['y'] >= start_y) &
                     (df['y'] < end_y)]
            if len(tmp) > 0:
                img_median[i][j] = np.median(tmp['deata_ms'])
                img_avg[i][j] = np.average(tmp['deata_ms'])
                img_min[i][j] = np.min(tmp['deata_ms'])
                img_max[i][j] = np.max(tmp['deata_ms'])
    return [img_median, img_avg, img_min, img_max], ['median', 'avg', 'min', 'max']


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


def single_lidar(filepath, time_model, filter=None, out_dir='./out_img/01_96_car'):
    points_df = extract_points_2(filepath, filter)
    points_df, actualtime_min = cal_actual_ms(points_df)
    points_df = convert_polar(points_df)
    points_df['deata_ms'] = points_df.apply(
        lambda r: cal_deata_ms(r['phi'], r['theta'], time_model['model'], [time_model['phi_min'], time_model['theta_min']], time_model['res']), axis=1)
    print('car time_ms: ', points_df['deata_ms'].max(
    ) - points_df['deata_ms'].min())
    print('car time_ms: ', points_df['actual_ms'].max(
    ) - points_df['actual_ms'].min())

    img, t = convert_img(points_df)
    plan_polar(img, t, f'{out_dir}_polar')
    img, t = convert_img_xy(points_df)
    plan_polar(img, t, f'{out_dir}_zy')
    points_df['deata_ms'] = points_df['actual_ms'].apply(
        lambda x: x - actualtime_min)
    img, t = convert_img(points_df)
    plan_polar(img, t, f'{out_dir}_actual_polar')
    img, t = convert_img_xy(points_df)
    plan_polar(img, t, f'{out_dir}_actual_zy')


def dual_lidar(filepath, time_model, filter=None, out_dir='./out_img/dual_car'):
    lidar_01 = extract_points_2(filepath[0], filter[0])
    lidar_02 = extract_points_2(filepath[1], filter[1])

    # time
    lidar_01, actualtime_min_01 = cal_actual_ms(lidar_01)
    lidar_02, actualtime_min_02 = cal_actual_ms(lidar_02)
    actualtime_min = min(actualtime_min_01, actualtime_min_02)
    points_df = pd.concat([lidar_01, lidar_02]).reset_index(drop=True)
    points_df = convert_polar(points_df)
    points_df['deata_ms'] = points_df.apply(
        lambda r: cal_deata_ms(r['phi'], r['theta'], time_model['model'], [time_model['phi_min'], time_model['theta_min']], time_model['res']), axis=1)
    print('car time_ms: ', points_df['deata_ms'].max(
    ) - points_df['deata_ms'].min())
    print('car time_ms: ', points_df['actual_ms'].max(
    ) - points_df['actual_ms'].min())

    img, t = convert_img(points_df)
    plan_polar(img, t, f'{out_dir}_polar')
    img, t = convert_img_xy(points_df)
    plan_polar(img, t, f'{out_dir}_zy')
    points_df['deata_ms'] = points_df['actual_ms'].apply(
        lambda x: x - actualtime_min)
    img, t = convert_img(points_df)
    plan_polar(img, t, f'{out_dir}_actual_polar')
    img, t = convert_img_xy(points_df)
    plan_polar(img, t, f'{out_dir}_actual_zy')


if __name__ == '__main__':
    dir = '/home/demo/Documents/datasets/pcd/'
    # model_path = './out_polar/24_96__10.0_avg_13528_-10463.csv'
    # model_dic_96 = read_polar(model_path)
    # files_path = os.path.join(dir, files_01['01']['name'])
    # single_lidar(files_path, model_dic_96,
    #              filter=files_01['01'], out_dir='./out_img/01_96_car')

    files_path = [os.path.join(dir, files_01['01']['name']), os.path.join(
        dir, files_01['02']['name'])]
    filters = [files_01['01'], files_01['02']]
    dual_model_path = './out_polar/dual_25_10.0_avg_11459_-10062.csv'
    model_dic = read_polar(dual_model_path)
    dual_lidar(files_path, model_dic, filter=filters)
