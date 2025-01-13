import os
from pyntcloud import PyntCloud
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from utils.util_tools import cartesian_to_polar
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


fit_models_dis = {
    '01': {
        '1': [-6.96227824e-02,  3.70974429e+00, -6.40654133e+01,  3.61201682e+02, 0],
        '2': [1.03321530e-03,  4.70165666e-02, -2.31978132e+00,  2.57841839e+01, 0],
        '3': [1.30329758e-04, -2.23810882e-02,  1.49037197e+00, -1.45076194e+00, 0]},
    '02': {
        '1': [-5.22383883e-03,  4.45566642e-01, -1.19902263e+01,  1.03739288e+02],
        '2': [7.20288684e-02, -7.61804311e+00,  2.68734135e+02, -3.14836534e+03],
        '3': [-2.95343693e-04,  5.02700131e-02, -2.45167752e+00,  5.88518245e+01]}
}

fit_models_dis_2 = {
    '01': {
        '1': [5.66713545e-01, -5.16790931e+01,  2.82089144e+03,  1.70113401e+04, -1.28091011e+04],
        '2': [3.12531028e-01, -3.86798405e+01,  2.86924778e+03,  2.35386538e+04, -1.51928206e+04],
        '3': [7.62114382e-04, -1.71324405e-01,  2.20908126e+01,  3.08184604e+02, -1.47963779e+02]},
    '02': {
        '1': [9.44421594e-02, -1.22854108e+01,  9.62570367e+02,  8.38004869e+03, -5.24815665e+03],
        '2': [2.99805277e-01, -4.89060589e+01,  4.74256271e+03,  5.05087552e+04, -2.86387508e+04],
        '3': [-3.80692130e-03,  1.09246056e+00, -1.84819190e+02, -3.42250972e+03, 1.47553141e+03]}
}


files_01 = {
    '01': {
        'name': '01/trans_T20240905_083426_LiDAR_96_D1000/T20240905_083426_LiDAR_96_D1000-1.pcd',
        'x': [-0.144745, 3.93939],
        'y': [-5.81986, -2.57225],
        'z': [19.5904, 36.477],
        'start_ms': 399.13653564
    },
    '02': {
        'name': '01/trans_T20240905_083426_LiDAR_97_D1000/T20240905_083426_LiDAR_97_D1000-0.pcd',
        'x': [-0.259938, 3.80575],
        'y': [-5.81742, -2.78213],
        'z': [19.8024, 36.9669],
        'start_ms': 399.57522583
    }
}


def read_pcd(filepath, filer_range=None):
    cloud = PyntCloud.from_file(filepath)
    points_df = cloud.points
    if filer_range is not None:
        if 'x' in filer_range:
            points_df = points_df[(points_df['x'] > filer_range['x'][0]) & (
                points_df['x'] < filer_range['x'][1])]
        if 'y' in filer_range:
            points_df = points_df[(points_df['y'] > filer_range['y'][0]) & (
                points_df['y'] < filer_range['y'][1])]
        if 'z' in filer_range:
            points_df = points_df[(points_df['z'] > filer_range['z'][0]) & (
                points_df['z'] < filer_range['z'][1])]
    return points_df


def model(xyz, a, b, c, d, e, f, g, h, i, j):
    x, y, z = xyz
    return a * x * x + b * x + c * y * y + d * y + e * z * z + f * z + g * x * y + h * x * z + i * y * z + j


def fit_model(xyz, popt):
    x, y, z = xyz
    a, b, c, d, e, f, g, h, i, j = popt
    return a * x * x + b * x + c * y * y + d * y + e * z * z + f * z + g * x * y + h * x * z + i * y * z + j


def extract_points(file, dir):
    cloud = PyntCloud.from_file(os.path.join(dir, file['name']))
    points_df = cloud.points
    points_df_01 = points_df[(points_df['x'] > file['x'][0]) & (
        points_df['x'] < file['x'][1]) & (points_df['y'] > file['y'][0]) & (
        points_df['y'] < file['y'][1]) & (points_df['z'] > file['z'][0]) & (
        points_df['z'] < file['z'][1])]

    return points_df_01


def dis_2d(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


def plan_line(x, y_dic, outpath='./plan_line.png'):
    plt.figure(figsize=(15, 5))
    # l = []
    # for k, v in y_dic.items():
    #     plt.plot(x, v)
    #     l.append(k)
    # plt.legend(l)

    # plt.savefig(outpath, format='png', dpi=300)
    plt.figure(figsize=(15, 5))
    l = []
    for k, v in y_dic.items():
        plt.plot(v, x)
        l.append(k)
    plt.legend(l)

    plt.savefig(outpath, format='png', dpi=300)


def scanline_dis_median(files, filter_range=None, out_dir='./'):
    out = None
    i = 0
    for file in files:
        if file.endswith('.pcd'):
            points_df = read_pcd(file, filter_range)
            points_df[f'dis_2d_{i}'] = points_df.apply(
                lambda r: dis_2d(r['y'], r['z'], 0, 0), axis=1)
            if out is None:
                out = points_df.groupby('scanline')[
                    f'dis_2d_{i}'].median().reset_index()
                out = out.rename(
                    columns={f'median': f'dis_2d_{i}_median'})

            else:
                tmp_sc = points_df.groupby('scanline')[
                    f'dis_2d_{i}'].median().reset_index()
                tmp_sc = tmp_sc.rename(
                    columns={f'median': f'dis_2d_{i}_median'})
                out = pd.merge(out, tmp_sc, how='outer', on='scanline')
            i += 1
    out.sort_values('scanline', inplace=True)
    x = out['scanline'].to_list()
    y_dic = {}
    for i in out.columns:
        if 'dis_2d_' in i:
            y_dic[i] = out[i].to_list()
    plan_line(x, y_dic, f'{out_dir}trans_scanline_dis_median.png')
    print(files)


def convert_polar(points_df):
    points_df['all'] = points_df.apply(
        lambda r: cartesian_to_polar(r['z'], r['y'], r['x']), axis=1
    )
    points_df['rho'] = points_df['all'].apply(lambda x: x[0])
    points_df['theta'] = points_df['all'].apply(lambda x: x[1])
    points_df['phi'] = points_df['all'].apply(lambda x: x[2])
    points_df.drop('all', axis=1, inplace=True)
    return points_df


def scanline_polar_median(files, filter_range=None, out_dir='./'):
    out_rho = None
    out_theta = None
    out_phi = None
    i = 0
    for file in files:
        if file.endswith('.pcd'):
            points_df = read_pcd(file, filter_range)
            points_df = convert_polar(points_df)
            points_df = points_df.rename(
                columns={'rho': f'rho_{i}', 'theta': f'theta_{i}', 'phi': f'phi_{i}'})
            if out_rho is None:
                out_rho = points_df.groupby('scanline')[
                    f'rho_{i}'].median().reset_index()
                out_rho = out_rho.rename(
                    columns={f'median': f'rho_{i}_median'})
                out_theta = points_df.groupby('scanline')[
                    f'theta_{i}'].median().reset_index()
                out_theta = out_theta.rename(
                    columns={f'median': f'theta_{i}_median'})
                out_phi = points_df.groupby('scanline')[
                    f'phi_{i}'].median().reset_index()
                out_phi = out_phi.rename(
                    columns={f'median': f'phi_{i}_median'})

            else:
                tmp_sc_rho = points_df.groupby('scanline')[
                    f'rho_{i}'].median().reset_index()
                tmp_sc_rho = tmp_sc_rho.rename(
                    columns={f'median': f'rho_{i}_median'})
                out_rho = pd.merge(out_rho, tmp_sc_rho,
                                   how='outer', on='scanline')
                tmp_sc_theta = points_df.groupby('scanline')[
                    f'theta_{i}'].median().reset_index()
                tmp_sc_theta = tmp_sc_theta.rename(
                    columns={f'median': f'theta_{i}_median'})
                out_theta = pd.merge(out_theta, tmp_sc_theta,
                                     how='outer', on='scanline')
                tmp_sc_phi = points_df.groupby('scanline')[
                    f'phi_{i}'].median().reset_index()
                tmp_sc_phi = tmp_sc_phi.rename(
                    columns={f'median': f'phi_{i}_median'})
                out_phi = pd.merge(out_phi, tmp_sc_phi,
                                   how='outer', on='scanline')
            i += 1
    out_phi.sort_values('scanline', inplace=True)
    x = out_phi['scanline'].to_list()
    y_dic = {}
    for i in out_phi.columns:
        if 'phi_' in i:
            y_dic[i] = out_phi[i].to_list()
    plan_line(x, y_dic, f'{out_dir}all_scanline_phi_median.png')
    out_theta.sort_values('scanline', inplace=True)
    x = out_theta['scanline'].to_list()
    y_dic = {}
    for i in out_theta.columns:
        if 'theta_' in i:
            y_dic[i] = out_theta[i].to_list()
    plan_line(x, y_dic, f'{out_dir}all_scanline_theta_median.png')
    out_rho.sort_values('scanline', inplace=True)
    x = out_rho['scanline'].to_list()
    y_dic = {}
    for i in out_rho.columns:
        if 'rho_' in i:
            y_dic[i] = out_rho[i].to_list()
    plan_line(x, y_dic, f'{out_dir}all_scanline_rho_median.png')
    print(len(files))


def main_plan_dis(dir):
    files = os.listdir(dir)
    cal_files = []
    for file in files:
        one_dir = os.path.join(dir, file)
        if not os.path.isdir(one_dir):
            continue
        next_files = os.listdir(one_dir)
        for next_file in next_files:
            next_dir = os.path.join(one_dir, next_file)
            if os.path.isdir(next_dir) and ('trans_' in next_file and '_96_' in next_file):
                # if not os.path.isdir(next_dir) or 'parallel_' not in next_file or '_97_' not in next_file:
                # if os.path.isdir(next_dir) and (('trans_' in next_file and '_96_' in next_file) or ('parallel_' in next_file and '_97_' in next_file)):
                final_files = os.listdir(next_dir)
                random_files = random.sample(final_files, 10)
                cal_files.extend([os.path.join(next_dir, i_f)
                                  for i_f in random_files])
    scanline_dis_median(cal_files, out_dir='./96_')


def main_plan_polar(dir):
    files = os.listdir(dir)
    cal_files = []
    for file in files:
        one_dir = os.path.join(dir, file)
        if not os.path.isdir(one_dir):
            continue
        next_files = os.listdir(one_dir)
        for next_file in next_files:
            next_dir = os.path.join(one_dir, next_file)
            if not os.path.isdir(next_dir) or 'trans_' in next_file or '_97_' not in next_file:
                continue
            final_files = os.listdir(next_dir)
            random_files = random.sample(final_files, 10)
            cal_files.extend([os.path.join(next_dir, i_f)
                             for i_f in random_files])
    scanline_polar_median(cal_files, out_dir='./97_')


def create_df(files, filter_range=None, outpath=None):
    df = pd.DataFrame(columns=['scanline', 'dis_2d'])
    for file in files:
        if file.endswith('.pcd'):
            points_df = read_pcd(file, filter_range)
            points_df[f'dis_2d'] = points_df.apply(
                lambda r: dis_2d(r['y'], r['z'], 0, 0), axis=1)
            tmp = points_df.groupby('scanline')[
                'dis_2d'].median().reset_index()
            df = pd.concat([df, tmp], axis=0).reset_index(drop=True)
    if outpath is not None:
        df.to_csv(outpath, index=False)
    return df


def model_dis(dis, a, b, c, d, e):
    return e * np.sqrt(dis) + a * dis**3 + b * dis**2 + c * dis + d


def create_model(df, phase_dic):
    out_models = {}
    for k, v in phase_dic.items():
        tmp_df = df[(df['dis_2d'] >= v[0]) &
                    (df['dis_2d'] <= v[1])]
        if len(tmp_df) == 0:
            print(v)
            out_models[k] = [0, 0, 0, 0]
            continue
        v = np.array(tmp_df['scanline'].to_list())
        popt, _ = curve_fit(model_dis, tmp_df['dis_2d'], v)
        out_models[k] = popt
    return out_models


def fit_dis(files, phase_dic, outpath='./dis_mdedian.csv'):
    print(len(files))
    df = create_df(files, outpath=outpath)
    model = create_model(df, phase_dic)
    print(model)
    return model


def main_fit_dis(dir):
    files = os.listdir(dir)
    lidar_01_phase_dic = {
        '1': [0, 20],
        '2': [20, 30],
        '3': [30, 80],

    }
    lidar_02_phase_dic = {
        '1': [0, 30],
        '2': [30, 40],
        '3': [40, 80],

    }
    cal_files_01 = []
    cal_files_02 = []
    for file in files:
        one_dir = os.path.join(dir, file)
        if not os.path.isdir(one_dir):
            continue
        next_files = os.listdir(one_dir)
        for next_file in next_files:
            next_dir = os.path.join(one_dir, next_file)
            if not os.path.isdir(next_dir):
                continue
            if 'trans_' in next_file and '_96_' in next_file:
                final_files = os.listdir(next_dir)
                random_files = random.sample(final_files, 5)
                cal_files_01.extend([os.path.join(next_dir, i_f)
                                     for i_f in random_files])
            elif 'parallel_' in next_file and '_97_' in next_file:
                final_files = os.listdir(next_dir)
                random_files = random.sample(final_files, 5)
                cal_files_02.extend([os.path.join(next_dir, i_f)
                                     for i_f in random_files])

    models_01 = fit_dis(cal_files_01, lidar_01_phase_dic,
                        outpath='./96_dis_mdedian.csv')
    plane_model_dis(models_01, lidar_01_phase_dic,
                    outpath='./96_dis_plane.png')
    models_02 = fit_dis(cal_files_02, lidar_02_phase_dic,
                        outpath='./97_dis_mdedian.csv')
    plane_model_dis(models_02, lidar_02_phase_dic,
                    outpath='./97_dis_plane.png')


def fit_model_dis(model, x):
    a, b, c, d, e = model
    out = e * np.sqrt(x) + a * x**3 + b * x**2 + c * x + d
    if out > 37:
        out = 0
    return out


def plane_model_dis(model, phase_dic, outpath='./plane_dis.png'):
    plt.figure(figsize=(15, 5))
    l = []
    for k, v in model.items():
        x = np.linspace(phase_dic[k][0], phase_dic[k][1], 100)
        y = [fit_model_dis(v, i) for i in x]
        out_x = []
        out_y = []
        for i in range(len(y)):
            if y[i] > 37:
                continue
            out_x.append(x[i])
            out_y.append(y[i])
        plt.plot(out_x, out_y)
        l.append(f'{phase_dic[k][0]}-{phase_dic[k][1]}')
    plt.legend(l)
    plt.savefig(outpath, format='png', dpi=300)


def fit_model_dis_by_pahse(model, phase, x):
    for k, v in phase.items():
        if x >= v[0] and x <= v[1]:
            return fit_model_dis(model[k], x)
    print(x)
    return None


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


if __name__ == '__main__':
    dir = '/home/demo/Documents/datasets/pcd/'
    # main_plan_dis(dir)
    # main_fit_dis(dir)
    lidar_01_phase_dic = {
        '1': [0, 20],
        '2': [20, 30],
        '3': [30, 80],
    }
    lidar_02_phase_dic = {
        '1': [0, 30],
        '2': [30, 40],
        '3': [40, 80],
    }
    # plane_model_dis(fit_models_dis_2['01'], lidar_01_phase_dic,
    #                 outpath='./96_plane_dis_2.png')
    # plane_model_dis(fit_models_dis_2['02'], lidar_02_phase_dic,
    #                 outpath='./97_plane_dis_2.png')

    # scline = pd.read_csv('./out_csv/scanline.csv')

    # lidar_01_xyz_pre = [0.499230, -14.012615, 38.510430]
    # lidar_01_xyz_gt = [0.527008, -14.145591, 37.748840]

    # lidar_01_xyz_pre = [1.996998, -13.929734, 38.250145]
    # lidar_01_xyz_gt = [1.978159, -13.951413, 37.696663]

    fusion_file = '/home/demo/Documents/datasets/s228/two0905/matrix/fusion_falcon.yaml'
    fusion_matrix = read_matrix_from_yaml(fusion_file)

    from utils.car import *
    bz_i = 1
    scene_bz = '05'
    car_scanline = car_70_t[scene_bz]['96'][bz_i]
    car_e = car_70[scene_bz]['96'][bz_i]
    car_start_t_1 = car_70_start_t[scene_bz]['96'][bz_i]
    car_start_t_2 = car_70_start_t[scene_bz]['97'][bz_i]

    car_s = [car_e[0], car_e[1], car_e[2] + 5.0]
    lidar_02_xyz = inverse_transform(
        np.array([car_s]), fusion_matrix['lidar_02'])[0]
    print(lidar_02_xyz)
    in_x_2 = dis_2d(lidar_02_xyz[1], lidar_02_xyz[2], 0, 0)
    scanline_2 = fit_model_dis_by_pahse(
        fit_models_dis_2['02'], lidar_02_phase_dic, in_x_2)
    scanline_1 = fit_model_dis_by_pahse(
        fit_models_dis_2['01'], lidar_01_phase_dic, dis_2d(car_s[1], car_s[2], 0, 0))
    deata_1 = (scanline_1 - car_scanline) * 2.5
    deata_2 = round(scanline_2) * 2.5
    dic = {
        '96': {
            'scanline': round(scanline_1, 2),
            'deata_time_ms': round(deata_1, 3),
            'time_ms': round(scanline_1 * 2.5 + car_start_t_1, 3),
            'dis_m': round(deata_1 * convert_speed(120), 3)
        },
        '97': {
            'scanline': round(scanline_2, 2),
            'deata_time_ms': round(deata_2, 3),
            'time_ms': round(scanline_2 * 2.5 + car_start_t_2, 3),
            'dis_m': round(deata_2 * convert_speed(120), 3)
        },
        'deata_time_ms': round(scanline_1 * 2.5 + car_start_t_1 - (scanline_2 * 2.5 + car_start_t_2), 3),
        'lidar_02_s': [round(i, 3) for i in lidar_02_xyz.tolist()]

    }
    print(dic)
    print(f'lidar_01 scanline: {scanline_1}')
    print(f'lidar_02 scanline: {scanline_2}')
    print(f'lidar_01 deata_time_ms: {deata_1}')
    print(f'lidar_02 deata_time_ms: {deata_2}')
    print(f'lidar_01 dis_m: {deata_1 * convert_speed(120)}')
    print(f'lidar_02 dis_m: {deata_2}')
