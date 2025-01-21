from pre_fusion.tools import read_matrix_from_yaml, cal_car, convert_speed
from utils.util_tools import (
    points_transformation, inverse_transform, cartesian_to_polar, polar_to_cartesian, distance_along_heading)
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.util_file import mkdir_directory
from tqdm import tqdm
from multiprocessing import Pool
import copy


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


def find_ms(xyz, model_time):
    rs, cs = model_time['model'].shape
    x, y, z = xyz
    r, phi, theta = cartesian_to_polar(z, y, x)
    c = int((phi - model_time['phi_min']) / model_time['res'])
    r = int((theta - model_time['theta_min']) / model_time['res'])
    # c = int((phi - model_time['theta_min']) / model_time['res'])
    # r = int((theta - model_time['phi_min']) / model_time['res'])
    if r >= 0 and r < rs and c >= 0 and c < cs:
        return round(model_time['model'][r][c], 2)
    return None


def change_cord(xyz):
    return [xyz[2], xyz[1], xyz[0]]


def out_car(carsize, car_pt, heading, time_model_1, time_model_2, pt_1, pt_2, fusion_matrix, parral_matrix):
    pt_2 = [round(value, 2) for value in pt_2]
    pt_1 = [round(value, 2) for value in pt_1]
    cord_xyz_1 = change_cord(pt_1)
    cord_xyz_2 = change_cord(pt_2)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*cord_xyz_1, color='r', s=100,
               label=f'Point lidar_01 {pt_1}')
    ax.scatter(*cord_xyz_2, color='b', s=100,
               label=f'Point lidar_02 {pt_2}')

    car_l, car_w, car_h = carsize
    x, y, z = car_pt
    car_pts = cal_car(
        [x, y, z], [car_l, car_w, car_h], heading)
    lower = ['lower_front_left', 'lower_back_left', 'lower_back_right',
             'lower_front_right', 'lower_front_left']
    up = ['upper_front_left', 'upper_front_right',
          'upper_back_right', 'upper_back_left', 'upper_front_left']
    four_l = [['lower_front_left', 'upper_front_left'],
              ['lower_back_right', 'upper_back_right'],
              ['lower_front_right', 'upper_front_right'],
              ['lower_back_left', 'upper_back_left']]
    for line in [lower, up]:
        x_pts, y_pts, z_pts = [], [], []
        for k in line:
            tmp_xyz = change_cord(car_pts[k])
            x_pts.append(tmp_xyz[0])
            y_pts.append(tmp_xyz[1])
            z_pts.append(tmp_xyz[2])
        ax.plot(x_pts, y_pts, z_pts, color='g', linewidth=2)
    for line in four_l:
        x_pts, y_pts, z_pts = [], [], []
        for k in line:
            tmp_xyz = change_cord(car_pts[k])
            x_pts.append(tmp_xyz[0])
            y_pts.append(tmp_xyz[1])
            z_pts.append(tmp_xyz[2])
            ax.plot(x_pts, y_pts, z_pts, color='g', linewidth=2)

    for k, v in car_pts.items():
        v_1 = inverse_transform(np.array([v]), parral_matrix['lidar_01'])
        ms_1 = find_ms(v_1[0], time_model_1)
        v_2 = inverse_transform(np.array([v]), fusion_matrix['lidar_02'])
        v_2 = inverse_transform(v_2, parral_matrix['lidar_02'])
        ms_2 = find_ms(v_2[0], time_model_2)
        if ms_1 is None or ms_2 is None:
            print(f'lidar_{1 if ms_1 is None else 2}out range')
            continue
        cord_v = change_cord(v)
        ax.scatter(cord_v[0], cord_v[1], cord_v[2], color='y', s=100,
                   label=f'{k} ms_1 {ms_1} ms_2 {ms_2}, deata {round(ms_2 - ms_1, 2)}')

    ax.set_xlabel('z')
    ax.set_ylabel('y')
    ax.set_zlabel('x')

    # 设置坐标轴范围
    ax.set_xlim(0, 60)
    ax.set_ylim(-20, 10)
    ax.set_zlim(0, 30)

    ax.legend()
    plt.show()


def cal_car_ms(carsize, car_pt, heading, time_model_1, time_model_2, fusion_matrix, parral_matrix):
    speed = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    speed_ms = [convert_speed(i) for i in speed]
    out = []
    car_l, car_w, car_h = carsize
    x, y, z = car_pt
    car_pts = cal_car([x, y, z], [car_l, car_w, car_h], heading)
    for k, v in car_pts.items():
        v_1 = inverse_transform(np.array([v]), parral_matrix['lidar_01'])
        ms_1 = find_ms(v_1[0], time_model_1)
        v_2 = inverse_transform(np.array([v]), fusion_matrix['lidar_02'])
        v_2 = inverse_transform(v_2, parral_matrix['lidar_02'])
        ms_2 = find_ms(v_2[0], time_model_2)
        if ms_1 is None or ms_2 is None:
            # print(f'lidar_{1 if ms_1 is None else 2} out range')
            continue
        deata_ms = round(abs(ms_2 - ms_1), 2)
        tmp = [carsize, car_pt, heading, k, ms_1, ms_2, deata_ms]
        for i in speed_ms:
            tmp.append(round(i * deata_ms, 2))
        out.append(tmp)
    return pd.DataFrame(out, columns=['carsize', 'car_pt', 'heading', 'car_id', 'ms_1', 'ms_2', 'deata_ms'] + [f'v_{i}' for i in speed])


def out_car_csv(parral_file, fusion_file, time_model_1, time_model_2, out_dir, head, z, y):
    mkdir_directory(out_dir)
    m_p = read_matrix_from_yaml(parral_file)
    m_f = read_matrix_from_yaml(fusion_file)
    time_model_1 = read_polar(time_model_1)
    time_model_2 = read_polar(time_model_2)
    params = {'fusion_matrix': m_f,
              'parallel_matrix': m_p,
              'head': head,
              'z': z,
              'y': y,
              'time_model_1': time_model_1,
              'time_model_2': time_model_2}

    carsize = [[4.640, 1.780, 1.435], [8.640, 2.380, 2.435],
               [15.640, 2.380, 3.435], [17.640, 2.780, 4.135]]
    cut_arg = []
    for size in carsize:
        tmp = copy.deepcopy(params)
        tmp['carsize'] = size
        tmp['outpath'] = os.path.join(
            out_dir, f'{size[0]}_{size[1]}_{size[2]}.csv')
        cut_arg.append(tmp)
    with Pool(len(carsize)) as pool:
        pool.map(single_car_for_pool, cut_arg)

    # for size in carsize:
    #     outpath = os.path.join(
    #         out_dir, f'{size[0]}_{size[1]}_{size[2]}.csv')
    #     out = None
    #     for h in head:
    #         for x_ in z:
    #             for y_ in y:
    #                 car_pt = [size[2] / 2, y_, x_]
    #                 df = cal_car_ms(size, car_pt, h, time_model_1,
    #                                 time_model_2, m_f, m_p)
    #                 if out is None:
    #                     out = df
    #                 else:
    #                     out = pd.concat([out, df], axis=0)
    #     out.to_csv(outpath, index=False)


def single_car_for_pool(args):
    carsize = args['carsize']
    outpath = args['outpath']
    fusion_matrix = args['fusion_matrix']
    parallel_matrix = args['parallel_matrix']
    head = args['head']
    z = args['z']
    y = args['y']
    time_model_1 = args['time_model_1']
    time_model_2 = args['time_model_2']
    out = None
    for h in tqdm(head, desc='cal head...'):
        for x_ in z:
            for y_ in y:
                car_pt = [carsize[2] / 2, y_, x_]
                df = cal_car_ms(carsize, car_pt, h, time_model_1,
                                time_model_2, fusion_matrix, parallel_matrix)
                if out is None:
                    out = df
                else:
                    out = pd.concat([out, df], axis=0)
    print(f'{carsize} data size {len(out)}')
    out.to_csv(outpath, index=False)


def plan_test(parral_file, fusion_file, time_model_1, time_model_2):

    m_p = read_matrix_from_yaml(parral_file)
    m_f = read_matrix_from_yaml(fusion_file)

    p_2_o = [0, 0, 0]
    p_2_o_to_1 = [[0, 0, 0]]
    p_2_o_to_1 = points_transformation(
        np.array(p_2_o_to_1), m_f['lidar_02'])[0]
    extend = {
        'y': [min(p_2_o_to_1[1], 0), max(p_2_o_to_1[1], 0)],
        'z': [min(p_2_o_to_1[2], 0), max(p_2_o_to_1[2], 0)]
    }

    time_model_1 = read_polar(time_model_1)
    time_model_2 = read_polar(time_model_2)

    carsize = [4.640, 1.780, 1.435]
    car_pt = [carsize[2] / 2, 0, 10]
    heading = 10
    out_car(carsize, car_pt, heading, time_model_1,
            time_model_2, p_2_o, p_2_o_to_1, m_f, m_p)


def out_range(parral_file, fusion_file):
    m_p = read_matrix_from_yaml(parral_file)
    m_f = read_matrix_from_yaml(fusion_file)

    p_2_o_to_1 = [[0, 0, 0]]
    p_2_o_to_1 = points_transformation(
        np.array(p_2_o_to_1), m_f['lidar_02'])[0]
    extend = {
        'y': [min(p_2_o_to_1[1], 0), max(p_2_o_to_1[1], 0)],
        'z': [min(p_2_o_to_1[2], 0), max(p_2_o_to_1[2], 0)]
    }

    print(f"lidar_02 point: {[round(i, 2) for i in p_2_o_to_1]}")
    print(extend)


def plan_single_car(filepath):
    out_dir = os.path.dirname(filepath)
    name = os.path.basename(filepath).split(".csv")[0]
    out_dir = os.path.join(out_dir, name)
    mkdir_directory(out_dir)
    df = pd.read_csv(filepath)
    head = np.unique(df['heading'])
    for h in head:
        tmp_df = df[df['heading'] == h]
        car_pt = np.unique(tmp_df['car_pt'])
        x, y, v = [], [], []
        for i in car_pt:
            tmp = df[df['car_pt'] == i]
            if tmp.empty:
                continue
            max_t = tmp['deata_ms'].max()
            center_pt = i.replace('[', '').replace(
                ']', '').replace(' ', '').split(',')
            cy, cz = float(center_pt[1]), float(center_pt[2])
            x.append(cy)
            y.append(cz)
            v.append(int(max_t))

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(x, y, c=v, cmap='viridis', s=100)
        cbar = plt.colorbar(scatter)
        cbar.set_label('deata_ms')
        plt.xlabel('y m')
        plt.ylabel('z m')
        plt.title(f'lwh={name}, head={h}')
        # plt.show()
        plt.savefig(os.path.join(out_dir, f'{h}.png'), format='png', dpi=300)
        plt.close()


def out_v_time_dis():
    out = []
    for v in range(10, 200, 5):
        for time in range(10, 90, 10):
            out.append([v, time, round(convert_speed(v) * time, 2)])
    out = pd.DataFrame(out, columns=['v', 'time', 'dis'])
    out.to_csv('./out_car/out_v_time_dis.csv', index=False)


if __name__ == "__main__":
    """
    s228
    """
    # parral_file = '/home/demo/Documents/datasets/s228/two0905/matrix/01_parallel_falcon.yaml'
    # fusion_file = '/home/demo/Documents/datasets/s228/two0905/matrix/fusion_falcon.yaml'
    # time_model_1 = './out_polar/24_96__10.0_avg_13528_-10463.csv'
    # time_model_2 = './out_polar/24_97__10.0_avg_13588_-10461.csv'
    # head = [i for i in range(-45, 50, 5)]
    # z = [i for i in range(0, 60, 1)]
    # y = [i for i in range(-20, 10, 2)]
    """
    qing  2024-05-23_16_25_05_30Min
    """
    parral_file = '/home/demo/Documents/datasets/qing/2024-05-23/matrix/01_parallel.yaml'
    fusion_file = '/home/demo/Documents/datasets/qing/2024-05-23/matrix/fusion.yaml'
    time_model_1 = './qing/polar/10_220__10.0_avg_-10464_13532.csv'
    time_model_2 = './qing/polar/10_221__10.0_avg_-10465_13831.csv'
    head = [i for i in range(0, 360, 10)]
    z = [i for i in range(0, 100, 1)]
    y = [i for i in range(-50, 50, 1)]

    out_range(parral_file, fusion_file)
    # plan_test(parral_file, fusion_file, time_model_1, time_model_2)
    out_car_csv(parral_file, fusion_file, time_model_1,
                time_model_2, './qing/out_car', head, z, y)
    # plan_single_car('./out_car/17.64_2.78_4.135.csv')

    # out_v_time_dis()
