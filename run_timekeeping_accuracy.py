from utils.util_file import delete_folder
import os
import pandas as pd
from sdk.sdk_tools import get_org_pcd


def get_timestamp_list(inno_pc_path):
    tmp_dir = os.path.abspath('./')
    file_name = os.path.basename(inno_pc_path).split('.')[0]
    files_01 = get_org_pcd(
        inno_pc_path, out_dir=tmp_dir, with_timestamp=True)
    files_01 = [float(os.path.basename(i).split(
        '.pcd')[0].split('_')[-1]) for i in files_01]
    delete_folder(os.path.join(tmp_dir, file_name))
    return files_01


def match_timestamp(list_01, list_02):
    list_01.sort()
    list_02.sort()
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
            list_01 = list_01[start:]
        else:
            for i in range(len(list_02) - 1):
                if list_02[i] < list_01[0] <= list_02[i + 1]:
                    if (list_01[0] - list_02[i]) > (list_02[i + 1] - list_01[0]):
                        start = i + 1
                    else:
                        start = i
                    break
            list_02 = list_02[start:]
    return list_01, list_02


def time_list():
    time_df = pd.DataFrame(
        columns=['scene', 'id', 'num', 'lidar_01', 'lidar_02', 'deat_time_ms'])
    base_dir = '/home/demo/Documents/datasets'
    scence = ['s228']  # , 'qing'
    bz = {
        's228': ['96', '97'],
        # 'qing': ['220', '221']
    }
    for s in scence:
        tmp = {
            'scene': s
        }
        cur_dir = os.path.join(base_dir, s)
        for item in os.listdir(cur_dir):
            item_path = os.path.join(cur_dir, item)
            if not os.path.isdir(item_path):
                continue
            tmp['id'] = item
            for item_num in os.listdir(item_path):
                item_path_num = os.path.join(item_path, item_num)
                if not os.path.isdir(item_path_num):
                    continue
                tmp['num'] = item_num
                files_01 = []
                files_02 = []
                for item_file in os.listdir(item_path_num):
                    full_path = os.path.join(item_path_num, item_file)
                    if os.path.isfile(full_path) and full_path.endswith('.inno_pc'):
                        if bz[s][0] in item_file:
                            files_01 = get_timestamp_list(full_path)
                        elif bz[s][1] in item_file:
                            files_02 = get_timestamp_list(full_path)

                if files_01 and files_02:
                    files_01, files_02 = match_timestamp(files_01, files_02)
                    tmp['lidar_01'] = files_01[0]
                    tmp['lidar_02'] = files_02[0]
                    tmp['deat_time_ms'] = abs(files_02[0] - files_01[0]) * 1000
                    time_df.loc[len(time_df)] = tmp
    return time_df


def plan_line(df, out_path="deat_time_ms.png"):
    import matplotlib.pyplot as plt
    x = df.apply(
        lambda r: f"{r['scene']}\n{r['id']}\n{r['num']}", axis=1).to_list()
    y = df['deat_time_ms'].to_list()
    plt.figure(figsize=(15, 5))
    plt.plot(x, y, marker='o')  # 'o' 表示在数据点上加上圆圈标记
    plt.title("deat_time_ms")
    plt.xlabel("Scene")
    plt.xticks(fontsize=8)
    plt.ylabel("deat_time_ms")

    # 显示网格
    plt.grid()
    plt.savefig(out_path, format='png', dpi=300)

    # 显示图形
    # plt.show()


if __name__ == "__main__":
    # time_df = time_list()
    # time_df.to_csv('./timekeeping.csv', index=False)
    time_df = pd.read_csv('./timekeeping.csv')
    plan_line(time_df)
