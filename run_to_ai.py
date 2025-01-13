from utils.util_file import delete_folder, mkdir_directory
from utils.util_pcd import write_pcd
import os
from pyntcloud import PyntCloud
import yaml
import numpy as np
from utils.util_tools import points_transformation
from sdk.sdk_tools import get_org_pcd, rewrite_pcd
import pandas as pd


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


def read_matrix_from_yaml(file):
    out = {}
    with open(file, 'r') as f:
        data = yaml.safe_load(f)
        for key, value in data.items():
            if 'lidar_' in key and 'transform' in value:
                tmp = [float(i) for i in value['transform'].strip().split(' ')]
                out[key] = np.array(tmp).reshape(4, 4)
    return out


def read_pcd(file):
    cloud = PyntCloud.from_file(file)
    points_df = cloud.points
    points_df = points_df.rename(
        columns={'reflectance': 'intensity'})
    points_df['timestamp'] = points_df['timestamp'] * 1000000
    return points_df


def parallel(points, matrix):
    tmp_points = points[['x', 'y', 'z']]
    points[['x', 'y', 'z']] = points_transformation(tmp_points, matrix)
    return points


def fusion(points, matrix):
    tmp_points = points[['x', 'y', 'z']]
    points[['x', 'y', 'z']] = points_transformation(tmp_points, matrix)
    return points


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


def get_org_pcd_and_match(file_path_1, file_path_2, out_dir='./'):
    mkdir_directory(out_dir)
    files_01 = get_org_pcd(file_path_1, out_dir, with_timestamp=True)
    files_01 = format_filename(files_01)
    files_02 = get_org_pcd(file_path_2, out_dir, with_timestamp=True)
    files_02 = format_filename(files_02)
    files_01, files_02 = match_timestamp(files_01, files_02)
    return files_01, files_02


def write_pcd_i(points, path):
    column_order = ['x', 'y', 'z', 'intensity', 'channel', 'roi', 'facet', 'is_2nd_return', 'multi_return',
                    'confid_level', 'flag', 'elongation', 'timestamp', 'scanline', 'scan_idx', 'frame_id', 'ring_id']
    headers = ['x:F', 'y:F', 'z:F', 'intensity:U', 'channel:U', 'roi:U', 'facet:U', 'is_2nd_return:U', 'multi_return:U',
               'confid_level:U', 'flag:U', 'elongation:U', 'timestamp:I64', 'scanline:U', 'scan_idx:U', 'frame_id:U', 'ring_id:U']
    points = points[column_order].values.tolist()
    num_points = len(points)
    if num_points == 0 or len(points[0]) != 17:
        raise ValueError("points value not attached to headers .")
    write_pcd(path, points, headers)


def pipline(file_path_1, file_path_2, parral_file, fusion_file, out_dir='./', filter=None):
    scene = os.path.basename(file_path_1).split('.')[0]
    out_dir_fusion = os.path.join(out_dir, f"fusion_{scene}")
    mkdir_directory(out_dir_fusion)
    files_01, files_02 = get_org_pcd_and_match(file_path_1, file_path_2)
    num = len(files_01)
    parral_matrix = read_matrix_from_yaml(parral_file)
    fusion_matrix = read_matrix_from_yaml(fusion_file)
    if filter is None or scene not in filter:
        start_index = np.random.randint(0, num - 10)
        print(f'{scene} not in filter')
    else:
        start_index = filter[scene]
    end_index = min(start_index + 10, num)
    for i in range(start_index, end_index):
        file_01 = files_01[i]
        file_02 = files_02[i]
        name = os.path.basename(file_01)
        pcd_01 = read_pcd(file_01)
        pcd_02 = read_pcd(file_02)
        pcd_01 = parallel(pcd_01, parral_matrix['lidar_01'])
        pcd_02 = parallel(pcd_02, parral_matrix['lidar_02'])
        pcd_02 = fusion(pcd_02, fusion_matrix['lidar_02'])
        all_pcd = pd.concat([pcd_01, pcd_02])
        write_pcd_i(all_pcd, os.path.join(out_dir_fusion, name))
    delete_folder(os.path.join(out_dir, f"{scene}"))
    delete_folder(os.path.join(
        out_dir, f"{os.path.basename(file_path_2).split('.')[0]}"))


def pipline_fusion(files_1, files_2, fusion_file, out_dir='./'):
    files_02 = format_filename(files_02)
    files_01, files_02 = match_timestamp(files_01, files_02)
    num = len(files_01)
    fusion_matrix = read_matrix_from_yaml(fusion_file)
    start_index = np.random.randint(0, num - 10)
    end_index = min(start_index + 10, num)
    for i in range(start_index, end_index):
        file_01 = files_01[i]
        file_02 = files_02[i]
        name = os.path.basename(file_01)
        pcd_01 = read_pcd(file_01)
        pcd_02 = read_pcd(file_02)
        pcd_02 = fusion(pcd_02, fusion_matrix['lidar_02'])
        all_pcd = pd.concat([pcd_01, pcd_02])
        write_pcd_i(all_pcd, os.path.join(out_dir, name))


def read_filter(dir='/home/demo/Documents/datasets/pcd'):
    out = {}
    files = os.listdir(dir)
    for file in files:
        next_dir = os.path.join(dir, file)
        if not os.path.isdir(next_dir):
            continue
        files_next = os.listdir(next_dir)
        for dir_ in files_next:
            final_dir = os.path.join(next_dir, dir_)
            if 'trans_' in dir_ or 'parallel_' in dir_ or '_97_' in dir_ or not os.path.isdir(final_dir):
                continue
            files_final = os.listdir(final_dir)
            files_final.sort()
            num = files_final[0].split('.')[0].split('-')[-1]
            out[dir_] = int(num)
    return out


if __name__ == "__main__":
    file_path_1 = '/home/demo/Documents/datasets/s228/two0905/06/T20240905_081447_LiDAR_96_D1000.inno_pc'
    file_path_2 = '/home/demo/Documents/datasets/s228/two0905/06/T20240905_081447_LiDAR_97_D1000.inno_pc'
    parral_file = '/home/demo/Documents/datasets/s228/two0905/matrix/01_parallel_falcon.yaml'
    fusion_file = '/home/demo/Documents/datasets/s228/two0905/matrix/fusion_falcon.yaml'
    filter = read_filter()

    pipline(file_path_1, file_path_2, parral_file, fusion_file, filter=filter)
