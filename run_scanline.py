import os
from pyntcloud import PyntCloud
import pandas as pd
import matplotlib.pyplot as plt
from utils.util_tools import cartesian_to_polar


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


def dis_2d(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


def scanline_time(points_df):
    start_ms = points_df['time_ms'].min()
    points_df['scanline'] = points_df['scanline'].astype(int)
    points_df['deata_ms'] = points_df['time_ms'] - start_ms
    out = points_df.groupby('scanline')['deata_ms'].agg(
        ['max', 'min']).reset_index()
    out = out.rename(
        columns={'max': 'max_time_ms', 'min': 'min_time_ms'})
    return out


def scaline_dis(points_df):
    points_df['scanline'] = points_df['scanline'].astype(int)
    points_df['dis_2d'] = points_df.apply(
        lambda r: dis_2d(r['y'], r['z'], 0, 0), axis=1)
    out = points_df.groupby('scanline').agg(
        {'dis_2d': ['max', 'min', 'median'], 'x': ['max', 'min', 'median']}).reset_index()
    out.columns = out.columns.map(lambda x: '_'.join(x))
    out = out.rename(
        columns={'scanline_': 'scanline'})
    return out


def plan_line(x, y_dic, outpath='./plan_line.png'):
    plt.figure(figsize=(15, 5))
    l = []
    for k, v in y_dic.items():
        plt.plot(x, v)
        l.append(k)
    plt.legend(l)

    plt.savefig(outpath, format='png', dpi=300)


def out_scaline_t(params, filer_range=None, out_path='./scanline.csv'):
    dir, lidar = params
    files = os.listdir(os.path.join(dir, lidar))
    sc_t = None
    i = 0
    files.sort()
    for file in files:
        if i > 1:
            break
        if file.endswith('.pcd'):
            points_df = read_pcd(os.path.join(dir, lidar, file), filer_range)
            if sc_t is None:
                sc_t = scanline_time(points_df)
                sc_t['file'] = i
            else:
                tmp_sc = scanline_time(points_df)
                tmp_sc['file'] = i
                sc_t = pd.merge(sc_t, tmp_sc, how='outer', on='scanline')
            i += 1
    sc_t.to_csv(out_path, index=False)


def out_scanline_dis(params, filer_range=None, out_dir='./'):
    dir, lidar = params
    files = os.listdir(os.path.join(dir, lidar))
    sc_t = None
    i = 0
    files.sort()
    for file in files:
        if i > 1:
            break
        if file.endswith('.pcd'):
            points_df = read_pcd(os.path.join(dir, lidar, file), filer_range)
            if sc_t is None:
                sc_t = scaline_dis(points_df)
                sc_t['file'] = i
            else:
                tmp_sc = scaline_dis(points_df)
                tmp_sc['file'] = i
                sc_t = pd.merge(sc_t, tmp_sc, how='outer', on='scanline')
            i += 1
    sc_t.sort_values('scanline', inplace=True)
    x = sc_t['scanline'].to_list()
    y_dic = {
        '0_min_x': sc_t['x_min_x'].to_list(),
        '0_max_x': sc_t['x_max_x'].to_list(),
        '0_median_x': sc_t['x_median_x'].to_list(),
        '1_median_x': sc_t['x_median_y'].to_list(),
        '1_min_x': sc_t['x_min_y'].to_list(),
        '1_max_x': sc_t['x_max_y'].to_list(),
    }
    plan_line(x, y_dic, f'{out_dir}trans_scanline_h.png')
    y_dic = {
        '0_dis_x_min': sc_t['dis_2d_min_x'].to_list(),
        '0_dis_x_max': sc_t['dis_2d_max_x'].to_list(),
        '0_dis_x_median': sc_t['dis_2d_median_x'].to_list(),
        '1_dis_x_median': sc_t['dis_2d_median_y'].to_list(),
        '1_dis_x_min': sc_t['dis_2d_min_y'].to_list(),
        '1_dis_x_max': sc_t['dis_2d_max_y'].to_list(),
    }
    plan_line(x, y_dic, f'{out_dir}trans_scanline_dis.png')


def out_scanline_dis_median(params, filer_range=None, out_dir='./'):
    dir, lidar = params
    files = os.listdir(os.path.join(dir, lidar))
    sc_t = None
    i = 0
    files.sort()
    for file in files:
        if i > 10:
            break
        if file.endswith('.pcd'):
            points_df = read_pcd(os.path.join(dir, lidar, file), filer_range)
            points_df[f'dis_2d_{i}'] = points_df.apply(
                lambda r: dis_2d(r['y'], r['z'], 0, 0), axis=1)
            if sc_t is None:
                sc_t = points_df.groupby('scanline')[
                    f'dis_2d_{i}'].median().reset_index()
                sc_t = sc_t.rename(
                    columns={f'median': f'dis_2d_{i}_median'})

            else:
                tmp_sc = points_df.groupby('scanline')[
                    f'dis_2d_{i}'].median().reset_index()
                tmp_sc = tmp_sc.rename(
                    columns={f'median': f'dis_2d_{i}_median'})
                sc_t = pd.merge(sc_t, tmp_sc, how='outer', on='scanline')
            i += 1
    sc_t.sort_values('scanline', inplace=True)
    x = sc_t['scanline'].to_list()
    y_dic = {}
    for i in sc_t.columns:
        if 'dis_2d_' in i:
            y_dic[i] = sc_t[i].to_list()
    plan_line(x, y_dic, f'{out_dir}trans_scanline_dis_median.png')


def convert_polar(points_df):
    points_df['all'] = points_df.apply(
        lambda r: cartesian_to_polar(r['x'], r['y'], r['z']), axis=1
    )
    points_df['rho'] = points_df['all'].apply(lambda x: x[0])
    points_df['theta'] = points_df['all'].apply(lambda x: x[1])
    points_df['phi'] = points_df['all'].apply(lambda x: x[2])
    points_df.drop('all', axis=1, inplace=True)
    return points_df


def scanline_r_theta_phi(points_df):
    points_df['scanline'] = points_df['scanline'].astype(int)
    points_df['dis_2d'] = points_df.apply(
        lambda r: dis_2d(r['y'], r['z'], 0, 0), axis=1)
    out = points_df.groupby('scanline').agg(
        {'rho': ['max', 'min'], 'theta': ['max', 'min', ], 'phi': ['max', 'min']}).reset_index()
    out.columns = out.columns.map(lambda x: '_'.join(x))
    out = out.rename(
        columns={'scanline_': 'scanline'})
    return out


def out_scanline_r_theta_phi(params, filer_range=None, out_dir='./'):
    files = os.listdir(os.path.join(dir, lidar))
    sc_t = None
    i = 0
    files.sort()
    for file in files:
        if i > 1:
            break
        if file.endswith('.pcd'):
            points_df = read_pcd(os.path.join(dir, lidar, file))


if __name__ == '__main__':
    dir = '/home/demo/Documents/datasets/pcd/10'
    lidar = 'trans_T20240905_071539_LiDAR_96_D1000'
    filer_range = {'x': [-1, 6], 'y': [-10, 10], 'z': [0, 100]}
    # out_scaline_t((dir, lidar))
    out_scanline_dis_median((dir, lidar), out_dir='./10_')
    # from utils.util_tools import cartesian_to_polar
    # files = os.listdir(os.path.join(dir, lidar))
    # sc_t = None
    # i = 0
    # files.sort()
    # for file in files:
    #     if i > 1:
    #         break
    #     if file.endswith('.pcd'):
    #         points_df = read_pcd(os.path.join(dir, lidar, file))
    #         if sc_t is None:
    #             sc_t = convert_polar(points_df)
    #             sc_t = scanline_r_theta_phi(sc_t)
    #             sc_t['file'] = i
    #         else:
    #             tmp_sc = convert_polar
    #             tmp_sc['file'] = i
    #             sc_t = pd.merge(sc_t, tmp_sc, how='outer', on='scanline')
    #         i += 1
    # sc_t.sort_values('scanline', inplace=True)
    # x = sc_t['scanline'].to_list()
    # y_dic = {
    #     '0_min_x': sc_t['x_min_x'].to_list(),
    #     '0_max_x': sc_t['x_max_x'].to_list(),
    #     '1_min_x': sc_t['x_min_y'].to_list(),
    #     '1_max_x': sc_t['x_max_y'].to_list(),
    # }
    # plan_line(x, y_dic, './trans_scanline_h.png')
    # y_dic = {
    #     '0_dis_x_min': sc_t['dis_2d_min_x'].to_list(),
    #     '0_dis_x_max': sc_t['dis_2d_max_x'].to_list(),
    #     '1_dis_x_min': sc_t['dis_2d_min_y'].to_list(),
    #     '1_dis_x_max': sc_t['dis_2d_max_y'].to_list(),
    # }
    # plan_line(x, y_dic, './trans_scanline_dis.png')
