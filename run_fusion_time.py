from pre_fusion.tools import read_matrix_from_yaml
from scipy.spatial import cKDTree
import numpy as np
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
import os
import pandas as pd
from utils.util_tools import (
    points_transformation, inverse_transform, cartesian_to_polar, polar_to_cartesian, distance_along_heading)
from utils.util_file import mkdir_directory
from pre_fusion.tools import cal_car, convert_speed


def read_points(file_path):
    cloud = PyntCloud.from_file(file_path)
    df = cloud.points
    df['actual_ms'] = df.apply(
        lambda r: r['time_s'] * 1000 + r['time_ms'], axis=1)
    min_ms = df['actual_ms'].min()
    df['deata_ms'] = df['actual_ms'].apply(lambda x: x - min_ms)
    return df


def join_region(lidar1_points, lidar2_transformed):
    # 创建 KDTree
    tree1 = cKDTree(lidar1_points)
    tree2 = cKDTree(lidar2_transformed)

    # 找到在一定距离内的相交点
    distance_threshold = 0.3  # 定义相交的距离阈值
    indices1 = tree1.query_ball_tree(tree2, distance_threshold)

    # 收集相交的点
    intersection_points = []
    for i, neighbors in enumerate(indices1):
        if len(neighbors) > 0:
            intersection_points.append(lidar1_points[i])

    intersection_points = np.array(intersection_points)
    out = {
        'x': [np.min(intersection_points[:, 0]), np.max(intersection_points[:, 0])],
        'y': [np.min(intersection_points[:, 1]), np.max(intersection_points[:, 1])],
        'z': [np.min(intersection_points[:, 2]), np.max(intersection_points[:, 2])]
    }

    return out


def filter(df, region):
    df = df[(df['x'] > region['x'][0]) & (df['x'] < region['x'][1]) &
            (df['y'] > region['y'][0]) & (df['y'] < region['y'][1]) &
            (df['z'] > region['z'][0]) & (df['z'] < region['z'][1])]
    return df


def parallel(points, matrix):
    tmp_points = points[['x', 'y', 'z']]
    points[['x', 'y', 'z']] = points_transformation(tmp_points, matrix)
    return points


def convert_polar(points_df, x_name='x', y_name='y', z_name='z'):
    points_df['all'] = points_df.apply(
        lambda r: cartesian_to_polar(r[z_name], r[y_name], r[x_name]), axis=1
    )
    points_df['rho'] = points_df['all'].apply(lambda x: x[0])
    points_df['phi'] = points_df['all'].apply(lambda x: x[1])
    points_df['theta'] = points_df['all'].apply(lambda x: x[2])
    points_df.drop('all', axis=1, inplace=True)
    return points_df


def cal_img(df, r, c, bz='median'):
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
        for j in range(c):
            tmp_df = df[(df['r'] == i) & (df['c'] == j)]
            if len(tmp_df) > 0:
                if bz == 'all':
                    for k in func:
                        imgs[k][i][j] = func[k](tmp_df['deata_ms'])
                else:
                    imgs[bz][i][j] = func[bz](tmp_df['deata_ms'])

    return imgs


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
    plt.axis('on')  # 不显示坐标轴
    plt.savefig(outpath, format='png', dpi=300)


def cal_csv_img():
    file_path_1 = '/home/demo/Documents/datasets/pcd/08_2/T20240905_082424_LiDAR_96_D1000/T20240905_082424_LiDAR_96_D1000-45.pcd'
    file_path_2 = '/home/demo/Documents/datasets/pcd/08_2/T20240905_082424_LiDAR_97_D1000/T20240905_082424_LiDAR_97_D1000-45.pcd'
    parral_file = '/home/demo/Documents/datasets/s228/two0905/matrix/01_parallel_falcon.yaml'
    fusion_file = '/home/demo/Documents/datasets/s228/two0905/matrix/fusion_falcon.yaml'
    m_p = read_matrix_from_yaml(parral_file)
    m_f = read_matrix_from_yaml(fusion_file)
    p1 = read_points(file_path_1)
    p2 = read_points(file_path_2)
    p_1 = parallel(p1, m_p['lidar_01'])
    p_2 = parallel(p2, m_p['lidar_02'])
    t_2 = parallel(p_2, m_f['lidar_02'])
    region = join_region(np.array(p_1[['x', 'y', 'z']]),
                         np.array(t_2[['x', 'y', 'z']]))
    filter_p_1 = filter(p_1, region)
    filter_t_2 = filter(t_2, region)
    filter_p_o_1 = inverse_transform(
        filter_p_1[['x', 'y', 'z']], m_p['lidar_01'])
    filter_p_1[['x', 'y', 'z']] = filter_p_o_1
    filter_p_o_2 = inverse_transform(
        filter_t_2[['x', 'y', 'z']], m_p['lidar_01'])
    filter_t_2[['x', 'y', 'z']] = filter_p_o_2
    filter_p_o_1 = convert_polar(filter_p_1)
    filter_p_o_2 = convert_polar(filter_t_2)
    phi_min = min(filter_p_o_1['phi'].min(), filter_p_o_2['phi'].min())
    phi_max = max(filter_p_o_1['phi'].max(), filter_p_o_2['phi'].max())
    theta_min = min(filter_p_o_1['theta'].min(), filter_p_o_2['theta'].min())
    theta_max = max(filter_p_o_1['theta'].max(), filter_p_o_2['theta'].max())
    res = 0.01
    c = int((phi_max - phi_min) / res) + 1   # 方位角
    r = int((theta_max - theta_min) / res) + 1   # 俯仰角
    filter_p_o_1['r'] = filter_p_o_1['theta'].apply(
        lambda x: int((x - theta_min) / res)
    )
    filter_p_o_1['c'] = filter_p_o_1['phi'].apply(
        lambda x: int((x - phi_min) / res)
    )
    filter_p_o_2['r'] = filter_p_o_2['theta'].apply(
        lambda x: int((x - theta_min) / res)
    )
    filter_p_o_2['c'] = filter_p_o_2['phi'].apply(
        lambda x: int((x - phi_min) / res)
    )
    imgs_1 = cal_img(filter_p_o_1, r, c, bz='avg')
    imgs_2 = cal_img(filter_p_o_2, r, c, bz='avg')
    save_result(
        imgs_1, res, f'./lidar_01', phi_min, theta_min)
    save_result(
        imgs_2, res, f'./lidar_02', phi_min, theta_min)
    inner_merge = pd.merge(filter_p_o_1, filter_p_o_2,
                           on=['r', 'c'], how='inner')
    inner_merge['deata_ms'] = inner_merge.apply(
        lambda r: abs(r['actual_ms_y'] - r['actual_ms_x']), axis=1)
    imgs_f = cal_img(inner_merge, r, c, bz='avg')
    save_result(
        imgs_f, res, f'./lidar_fusion', phi_min, theta_min)


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


def plan_pts(x, y, v, out_path):
    plt.clf()
    plt.scatter(x, y, c=v, cmap='viridis')
    plt.colorbar(label='Value')
    plt.xlabel('z m')
    plt.ylabel('y m')

    # 设置标题
    plt.title('Scatter Plot with Color Representation')

    plt.savefig(out_path, format='png', dpi=300)


def points_img():
    lidar_fusion = './lidar_fusion_10.0_avg_-14638_12950.csv'
    model_time = read_polar(lidar_fusion)
    r, c = model_time['model'].shape
    z_range = [10, 60]
    y_range = [-30, 30]
    h = 0
    time_np = model_time['model']
    while h <= 5.0:
        dis_2d = 10
        out_dir = f'./out_{h}/'
        mkdir_directory(out_dir)
        while dis_2d < 100:
            rho = np.sqrt(dis_2d**2 + h**2)
            out = []
            out_path = os.path.join(out_dir, f'{dis_2d}.png')
            dis_2d = dis_2d + 1.0
            for i in range(r):
                for j in range(c):
                    phi = j * model_time['res'] + model_time['phi_min']
                    theta = i * model_time['res'] + model_time['theta_min']
                    z, y, x = polar_to_cartesian(rho, theta, phi)
                    if z < z_range[0] or y < y_range[0] or abs(x - h) > 0.5 or z > z_range[1] or y > y_range[1]:
                        continue
                    out.append([x, y, z, time_np[i][j]])
            out = pd.DataFrame(out, columns=['x', 'y', 'z', 'deata_ms'])
            # out.sort_values(by='deata_ms', inplace=True)
            if len(out) == 0:
                continue
            print(
                f'h = {h} m dis_2d = {dis_2d} m: deata_ms min = {out["deata_ms"].min()}, max = {out["deata_ms"].max()}')
            plan_pts(out['z'], out['y'], out['deata_ms'], out_path)
        h = h + 1.0


def ms_xy_h():
    lidar_fusion = './lidar_fusion_10.0_avg_-14638_12950.csv'
    model_time = read_polar(lidar_fusion)
    r, c = model_time['model'].shape
    z_range = [10, 60]
    y_range = [-30, 30]
    time_np = model_time['model']
    deata_ms = 5
    while deata_ms <= 20:
        h = 0
        print(f'deata_ms = {deata_ms}')
        while h <= 5.0:
            out_path = f'./out_{h}_deata_ms_{deata_ms}.png'
            out = []
            for i in range(r):
                for j in range(c):
                    phi = j * model_time['res'] + model_time['phi_min']
                    theta = i * model_time['res'] + model_time['theta_min']
                    if time_np[i][j] > deata_ms:
                        continue
                    start_cal = True
                    rho = 10
                    while start_cal and rho < 70:
                        z, y, x = polar_to_cartesian(rho, theta, phi)
                        if abs(x - h) < 0.5 and abs(rho**2 - z**2 - y**2) < 0.5 and z > z_range[0] and y > y_range[0] and z < z_range[1] and y < y_range[1]:
                            start_cal = False
                            out.append([x, y, z, time_np[i][j]])
                        rho = rho + 0.5

            out = pd.DataFrame(out, columns=['x', 'y', 'z', 'deata_ms'])
            # out.sort_values(by='deata_ms', inplace=True)
            if len(out) > 0:
                plan_pts(out['z'], out['y'], out['deata_ms'], out_path)
            h = h + 0.5
        deata_ms = deata_ms + 5


def find_ms(xyz, model_time):
    rs, cs = model_time['model'].shape
    x, y, z = xyz
    r, phi, theta = cartesian_to_polar(z, y, x)
    c = int((phi - model_time['phi_min']) / model_time['res'])
    r = int((theta - model_time['theta_min']) / model_time['res'])
    if r >= 0 and r < rs and c >= 0 and c < cs:
        return model_time['model'][r][c]
    return 0.0


def car_csv(carsize):
    lidar_1_file = './lidar_01_10.0_avg_-14638_12950.csv'
    lidar_2_file = './lidar_02_10.0_avg_-14638_12950.csv'
    model_time_1 = read_polar(lidar_1_file)
    model_time_2 = read_polar(lidar_2_file)

    heading = -45

    car_l, car_w, car_h = carsize

    while heading <= 45:
        bz = 0
        out = []
        z = 10
        while z < 60:
            y = -30
            while y < 30:
                x = car_h / 2 - 0.5
                bz = bz + 1
                while x <= car_h / 2 + 0.5:
                    ms_1 = find_ms([x, y, z], model_time_1)
                    ms_2 = find_ms([x, y, z], model_time_2)
                    out.append([bz, x, y, z, ms_1, ms_2, 'center'])
                    car_pts = cal_car(
                        [x, y, z], [car_l, car_w, car_h], heading)
                    for k, v in car_pts.items():
                        ms_1 = find_ms(v, model_time_1)
                        ms_2 = find_ms(v, model_time_2)
                        out.append([bz, v[0], v[1], v[2], ms_1, ms_2, k])
                    x = x + 0.5
                y = y + 0.5
            z = z + 0.5

        out = pd.DataFrame(
            out, columns=['bz', 'x', 'y', 'z', 'ms_1', 'ms_2', 'type'])
        out.to_csv(
            f'./lwh_{car_l}-{car_w}-{car_h}_heading_{heading}.csv', index=False)
        heading = heading + 10


def cal_dis(filepath):
    df = pd.read_csv(filepath)
    df['deata_ms'] = df.apply(lambda r: abs(r['ms_1'] - r['ms_2']), axis=1)
    df['front_or_follow'] = df['type'].apply(
        lambda x: x.split('_')[1] if x != 'center' else x)
    df_bz = df.groupby(['bz', 'front_or_follow'])[
        'deata_ms'].agg(['max', 'min']).reset_index()
    df_center = df[df['type'] == 'center']
    df_center = df_center.groupby(
        ['bz'])[['x', 'y', 'z']].agg(['mean']).reset_index()
    df_center.columns = ['bz', 'x', 'y', 'z']
    final_df = pd.merge(df_bz, df_center, on='bz', how='left')
    bz = np.unique(final_df['bz'])
    out = []
    for i in bz:
        tmp = final_df[final_df['bz'] == i]
        if len(tmp) == 0:
            continue
        one_r = tmp[tmp['front_or_follow'] == 'center']
        center_row = one_r.iloc[0] if not one_r.empty else None
        one_r = tmp[tmp['front_or_follow'] == 'front']
        front_row = one_r.iloc[0] if not one_r.empty else None
        one_r = tmp[tmp['front_or_follow'] == 'back']
        back_row = one_r.iloc[0] if not one_r.empty else None
        if center_row is None or front_row is None or back_row is None:
            continue
        out.append([center_row['x'], center_row['y'], center_row['z'],
                    center_row['max'], center_row['min'], front_row['max'], front_row['min'], back_row['max'], back_row['min']])
    out = pd.DataFrame(out, columns=['x', 'y', 'z', 'center_max', 'center_min',
                                     'front_max', 'front_min', 'back_max', 'back_min'])
    for c_name in ['front_max', 'front_min', 'back_max', 'back_min']:
        for v in [30, 50, 70, 100, 120, 150, 200]:
            new_name = f'{c_name}_{v}_dis_m'
            out[new_name] = out[c_name].apply(
                lambda x: round(convert_speed(v) * x, 3))
    out.to_csv(filepath.replace('.csv', '_dis.csv'), index=False)


if __name__ == '__main__':
    # cal_csv_img()
    # points_img()
    # ms_xy_h()
    car_csv([4.640, 1.780, 1.435])
    # car_csv([8.640, 2.380, 2.435])
    # car_csv([15.640, 2.380, 3.435])
    # car_csv([17.640, 2.780, 4.135])
    # for head in [-45, -35, -25, -15, -5, 5, 15, 25, 35, 45]:
    #     filepath = f'./lwh_17.64-2.78-4.135_heading_{head}.csv'
    #     cal_dis(filepath)
    # dir = './car/'
    # files = os.listdir(dir)
    # out = []
    # for file in files:
    #     filepath = os.path.join(dir, file)
    #     if not os.path.isfile(filepath) or '_dis.csv' not in file:
    #         continue

    #     data = pd.read_csv(filepath)
    #     df = data[(data['y'] < 1) & (data['y'] > -20)
    #               & (data['z'] > 13) & data['z'] < 60]
    #     carsize = file.split('_')[1].split('-')
    #     head = float(file.split('_')[-2])
    #     max_row = df.loc[df['front_max'].idxmax()]
    #     out.append([carsize, [max_row['x'], max_row['y'],
    #                max_row['z']], head, max_row['front_max'], max_row['back_max'], 'front_max', [max_row['front_max_30_dis_m'], max_row['back_max_30_dis_m']], [max_row['front_max_120_dis_m'], max_row['back_max_120_dis_m']], [max_row['front_max_200_dis_m'], max_row['back_max_200_dis_m']]])
    #     max_row = df.loc[df['back_max'].idxmax()]
    #     out.append([carsize, [max_row['x'], max_row['y'],
    #                max_row['z']], head, max_row['front_max'], max_row['back_max'], 'back_max', [max_row['front_max_30_dis_m'], max_row['back_max_30_dis_m']], [max_row['front_max_120_dis_m'], max_row['back_max_120_dis_m']], [max_row['front_max_200_dis_m'], max_row['back_max_200_dis_m']]])
    #     max_row = df.loc[df['front_max'].idxmin()]
    #     out.append([carsize, [max_row['x'], max_row['y'],
    #                max_row['z']], head, max_row['front_max'], max_row['back_max'], 'front_min', [max_row['front_max_30_dis_m'], max_row['back_max_30_dis_m']], [max_row['front_max_120_dis_m'], max_row['back_max_120_dis_m']], [max_row['front_max_200_dis_m'], max_row['back_max_200_dis_m']]])
    #     max_row = df.loc[df['back_max'].idxmin()]
    #     out.append([carsize, [max_row['x'], max_row['y'],
    #                max_row['z']], head, max_row['front_max'], max_row['back_max'], 'back_min', [max_row['front_max_30_dis_m'], max_row['back_max_30_dis_m']], [max_row['front_max_120_dis_m'], max_row['back_max_120_dis_m']], [max_row['front_max_200_dis_m'], max_row['back_max_200_dis_m']]])

    # out = pd.DataFrame(out, columns=[
    #                    'carsize', 'center_pos', 'head', 'time_front', 'time_back', 'bz', '30', '70', '120'])
    # out['l'] = out['carsize'].apply(lambda x: float(x[0]))
    # out = out.sort_values(by=['l', 'head', 'bz'])
    # out.drop(columns='l', inplace=True)
    # out.to_csv('./res.csv', index=False)
