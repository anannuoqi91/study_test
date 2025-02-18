from utils.util_file import (
    list_files_in_current_directory, mkdir_directory, delete_file)
from utils.util_pcd import write_pcd
import os
from pyntcloud import PyntCloud
import yaml
import numpy as np
from utils.util_tools import points_transformation
from sdk.sdk_tools import get_org_pcd, rewrite_pcd


def read_matrix_from_yaml(file):
    out = {}
    with open(file, 'r') as f:
        data = yaml.safe_load(f)
        for key, value in data.items():
            if 'lidar_' in key and 'transform' in value:
                tmp = [float(i) for i in value['transform'].strip().split(' ')]
                out[key] = np.array(tmp).reshape(4, 4)
    return out


def transformation_points(files, matrix, out_dir, bz='lidar_01', fusion_matrix=None):
    for file in files:
        cloud = PyntCloud.from_file(file)
        points_df = cloud.points
        points = points_df[['x', 'y', 'z']]
        points_df[['x', 'y', 'z']] = points_transformation(points, matrix)
        if bz == 'lidar_02' and fusion_matrix is not None:
            points_df[['x', 'y', 'z']] = points_transformation(
                points_df[['x', 'y', 'z']], fusion_matrix)
        column_order = ['x', 'y', 'z', 'reflectance', 'channel', 'roi', 'facet', 'is_2nd_return', 'multi_return',
                        'confid_level', 'flag', 'elongation', 'time_s', 'time_ms', 'scanline', 'scan_idx', 'frame_id', 'ring_id']
        points = points_df[column_order].values.tolist()
        num_points = len(points)
        if num_points == 0 or len(points[0]) != 18:
            raise ValueError("points value not attached to headers .")
        headers = ['x:F', 'y:F', 'z:F', 'reflectance:U', 'channel:U', 'roi:U', 'facet:U', 'is_2nd_return:U', 'multi_return:U',
                   'confid_level:U', 'flag:U', 'elongation:U', 'time_s:I32', 'time_ms:F', 'scanline:U', 'scan_idx:U', 'frame_id:U', 'ring_id:U']
        write_pcd(os.path.join(out_dir, os.path.basename(file)), points, headers)


def rewrite_and_parallel(dir_01, dir_02, parral_file, fusion_file):
    parral_matrix = read_matrix_from_yaml(parral_file)
    fusion_matrix = read_matrix_from_yaml(fusion_file)
    if dir_01.endswith('/'):
        dir_01 = dir_01[:-1]
    out_trans_dir = f"{dir_01}_trans"
    files = list_files_in_current_directory(dir_01, '.pcd')
    rewrite_pcd(files)
    os.makedirs(out_trans_dir)
    transformation_points(
        files, parral_matrix['lidar_01'], out_trans_dir)
    if dir_02.endswith('/'):
        dir_02 = dir_02[:-1]
    out_trans_dir = f"{dir_02}_trans"
    files = list_files_in_current_directory(dir_02, '.pcd')
    rewrite_pcd(files)
    os.makedirs(out_trans_dir)
    transformation_points(
        files, parral_matrix['lidar_02'], out_trans_dir, 'lidar_02', fusion_matrix['lidar_02'])


def format_filename(files):
    out = []
    for file in files:
        name = os.path.basename(file)
        time_l = name.split('_')[-1]
        time_l = time_l.split('.')
        time_s = float(time_l[0])
        time_ms = float(time_l[1])
        out.append([file, time_s * 1000000 + time_ms])
    return out


def match_timestamp(files_01, files_02):
    files_01 = sorted(files_01, key=lambda x: x[1])
    files_02 = sorted(files_02, key=lambda x: x[1])
    out_01 = [x[0] for x in files_01]
    out_02 = [x[0] for x in files_02]
    list_01 = [x[1] for x in files_01]
    list_02 = [x[1] for x in files_02]
    if list_01 and list_02:
        start = 0
        if list_02[0] > list_01[0]:
            for i in range(len(list_01) - 1):
                if list_01[i] < list_02[0] <= list_01[i + 1]:
                    if (list_02[0] - list_01[i]) > (list_01[i + 1] - list_02[0]):
                        start = i + 1
                    else:
                        start = i
                    break
            out_01 = out_01[start:]
        else:
            for i in range(len(list_02) - 1):
                if list_02[i] < list_01[0] <= list_02[i + 1]:
                    if (list_01[0] - list_02[i]) > (list_02[i + 1] - list_01[0]):
                        start = i + 1
                    else:
                        start = i
                    break
            out_02 = out_02[start:]
    num = min(len(out_01), len(out_02))
    out_01 = out_01[:num]
    out_02 = out_02[:num]
    return out_01, out_02


def out_org_pcd():
    inno_pc_1 = '/home/demo/Documents/datasets/qing/2024-05-23/2024-05-23_16_25_05_30Min/LIDAR_220_18000MB_1716452708.inno_pc'
    inno_pc_2 = '/home/demo/Documents/datasets/qing/2024-05-23/2024-05-23_16_25_05_30Min/LIDAR_221_18000MB_1716452708.inno_pc'
    out_dir = '/home/demo/Documents/datasets/pcd'
    file_number = 3600
    files_1 = get_org_pcd(inno_pc_1, out_dir=out_dir,
                          file_number=file_number, with_timestamp=True)
    files_2 = get_org_pcd(inno_pc_2, out_dir=out_dir,
                          file_number=file_number, with_timestamp=True)
    files_1 = format_filename(files_1)
    files_2 = format_filename(files_2)
    files_1_n, files_2_n = match_timestamp(files_1, files_2)
    files_1 = [r[0] for r in files_1]
    files_2 = [r[0] for r in files_2]
    result = list(set(files_1) - set(files_1_n))
    for i in result:
        delete_file(i)
    result = list(set(files_2) - set(files_2_n))
    for i in result:
        delete_file(i)


def write_time_to_pcd():
    out_dir = '/home/demo/Documents/datasets/pcd'
    file_dir = ['LIDAR_220_18000MB_1716452708', 'LIDAR_221_18000MB_1716452708']
    for i in file_dir:
        files = list_files_in_current_directory(
            os.path.join(out_dir, i), '.pcd')
        rewrite_pcd(files)


def parral_pcds():
    out_dir = '/home/demo/Documents/datasets/pcd'
    file_dir = ['LIDAR_220_18000MB_1716452708', 'LIDAR_221_18000MB_1716452708']
    parral_file = '/home/demo/Documents/datasets/qing/2024-05-23/matrix/01_parallel.yaml'
    parral_matrix = read_matrix_from_yaml(parral_file)
    bz = 1
    for i in file_dir:
        out_trans_dir = os.path.join(out_dir, f'parallel_{i}')
        mkdir_directory(out_trans_dir)

        files = list_files_in_current_directory(
            os.path.join(out_dir, file_dir[0]), '.pcd')
        transformation_points(
            files, parral_matrix[f'lidar_0{bz}'], out_trans_dir)
        bz = bz + 1


if __name__ == "__main__":
    # out_org_pcd()
    # write_time_to_pcd()
    parral_pcds()

    # out_dir = '/home/demo/Documents/datasets/pcd/02/'
    # file_name = 'T20240905_083450_LiDAR_97_D1000'
    # parral_file = '/home/demo/Documents/datasets/s228/two0905/matrix/01_parallel_falcon.yaml'
    # fusion_file = '/home/demo/Documents/datasets/s228/two0905/matrix/fusion_falcon.yaml'
    # out_trans_dir = os.path.join(out_dir, f'trans_{file_name}')
    # inno_pc = f'/home/demo/Documents/datasets/s228/two0905/14/{file_name}.inno_pc'
    # file_number = 3600
    # files = get_org_pcd(
    #     inno_pc, out_dir=out_dir, file_number=file_number)

    # files = list_files_in_current_directory(os.path.join(
    #     out_dir, file_name), '.pcd')
    # rewrite_pcd(files)

    # os.makedirs(out_trans_dir)
    # parral_matrix = read_matrix_from_yaml(parral_file)
    # fusion_matrix = read_matrix_from_yaml(fusion_file)
    # if '96' in file_name:
    #     transformation_points(
    #         files, parral_matrix['lidar_01'], out_trans_dir)
    # else:
    #     transformation_points(
    #         files, parral_matrix['lidar_02'], out_trans_dir, 'lidar_02', fusion_matrix['lidar_02'])

    # dir = '/home/demo/Documents/datasets/pcd/'
    # files = os.listdir(dir)
    # cal_files = []

    # for file in files:
    #     one_dir = os.path.join(dir, file)
    #     if not os.path.isdir(one_dir):
    #         continue
    #     next_files = os.listdir(one_dir)
    #     for next_file in next_files:
    #         next_dir = os.path.join(one_dir, next_file)
    #         if not os.path.isdir(next_dir) or 'trans_' in next_file or '_96_' in next_file:
    #             continue
    #         out_trans_dir = os.path.join(one_dir, f'parallel_{next_file}')
    #         os.makedirs(out_trans_dir)
    #         final_files = list_files_in_current_directory(next_dir, '.pcd')
    #         transformation_points(
    #             final_files, parral_matrix['lidar_02'], out_trans_dir)
