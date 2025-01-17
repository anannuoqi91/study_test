import shapely.geometry as geo
from src.apollo_info.box import get_traces_from_record
import pandas as pd
import math
import argparse
import os


def decode_stable_region(stable_region_file):
    out = []
    # read txt
    with open(stable_region_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                continue
            line = line.split('(')[-1].split(')')[0]
            line = line.split(';')
            if len(line) != 3:
                raise ValueError('stable region format error')
            out.append([float(line[1]), float(line[-1])])
    if out:
        return geo.Polygon(out)
    return out


def diff_angle_radius(angle1: float, angle2: float) -> float:
    radius1 = math.radians(angle1)
    radius2 = math.radians(angle2)
    two_pi = 2.0 * math.pi
    angle_norm = (radius1 - radius2) % two_pi
    diff = angle_norm if angle_norm >= 0 else angle_norm + two_pi
    return min(diff, 2 * math.pi - diff)


def poly_intersect(region1, region2):
    return region1.intersects(region2)


def detect_change(i_range, trace, size_thres={'length': 0.1, 'width': 0.1, 'height': 0.1}, head_thres=45.0):
    l = len(trace)
    out = pd.DataFrame(columns=['box_id', 'idx', 'obj_type', 'is_type_change',
                                'is_size_change', 'is_head_change', 'type_change_to', 'size', 'size_change_to', 'head', 'head_change_to', 'position'])
    radius_thres = math.radians(head_thres)

    for i in range(i_range[0], i_range[1] + 1):
        if i == l - 1:
            break
        i_b = trace[i]
        nex_b = trace[i + 1]
        lwh_i = [round(i_b.length, 2), round(
            i_b.width, 2), round(i_b.height, 2)]
        lwh_n = [round(nex_b.length, 2), round(
            nex_b.width, 2), round(nex_b.height, 2)]
        tmp = {
            'box_id': i_b.track_id,
            'idx': i_b.idx,
            'obj_type': i_b.object_type,
            'is_type_change': False,
            'is_size_change': False,
            'is_head_change': False,
            'type_change_to': None,
            'size': lwh_i,
            'size_change_to': None,
            'head': trace[i].heading,
            'head_change_to': None,
            'position': [round(i_b.position_x, 3), round(i_b.position_y, 3), round(i_b.position_z, 3)]
        }
        if nex_b.object_type != i_b.object_type:
            tmp['is_type_change'] = True
            tmp['type_change_to'] = nex_b.object_type
        if abs(lwh_i[0] - lwh_n[0]) > size_thres['length'] or abs(lwh_i[1] - lwh_n[1]) > size_thres['width'] or abs(lwh_i[2] - lwh_n[2]) > size_thres['height']:
            tmp['is_size_change'] = True
            tmp['size_change_to'] = lwh_n
        if diff_angle_radius(i_b.heading, nex_b.heading) > radius_thres:
            tmp['is_head_change'] = True
            tmp['head_change_to'] = nex_b.heading
        out.loc[len(out)] = tmp
    out = out[out['is_type_change'] |
              out['is_size_change'] | out['is_head_change']]
    return out


def cut_trace(trace, stable_region):
    num = len(trace)
    out = []
    start = None
    for i in range(0, num):
        t_b = trace[i]
        d_2 = t_b.cal_car()
        t_b_poly = geo.Polygon(d_2)
        if poly_intersect(t_b_poly, stable_region):
            if start is None:
                start = i
        elif start is not None:
            out.append([start, i - 1])
            start = None
    if start is not None:
        out.append([start, num - 1])
    return out


def appear_unexpected(appear_t, data_start_t, trace_start_t):
    # 突然出现的逻辑
    # 第一帧出现在稳定区域内，且不是录制的起始帧
    if appear_t == trace_start_t and appear_t != data_start_t:
        return True
    return False


def disappear_unexpected(disappear_t, data_end_t, trace_end_t):
    # 突然消失的逻辑
    # 最后一帧在稳定区域内，且不是录制的结束帧
    if disappear_t == trace_end_t and disappear_t != data_end_t:
        return True
    return False


def main(args):
    stable_region = decode_stable_region(args.stable_region_file)
    if not stable_region:
        raise ValueError('stable region is null')
    file_path = []
    for root, dirs, files in os.walk(args.record_dir):
        for file_name in files:
            if '.' in file_name:
                file_path.append(file_name)
    if len(file_path) == 0:
        print('record is null')
        exit()
    file_path.sort()
    file_path = [os.path.join(args.record_dir, i) for i in file_path]
    traces, start_end_idx, idx, bounds = get_traces_from_record(
        file_path, topic_name=args.channel_name)
    if len(traces) == 0:
        raise ValueError('traces is null')
    out = None
    for key in traces:
        trace = traces[key]
        k_trace_time = cut_trace(trace, stable_region)
        if len(k_trace_time) == 0:
            continue
        k_s_i = k_trace_time[0][0]
        k_e_i = k_trace_time[-1][1]
        is_appear = appear_unexpected(
            trace[k_s_i].idx, start_end_idx[0], trace[0].idx)
        is_disappear = disappear_unexpected(
            trace[k_e_i].idx, start_end_idx[1], trace[-1].idx)
        key_df = None
        for i in k_trace_time:
            tmp_key = detect_change(i, trace)
            if len(tmp_key) == 0:
                continue
            if key_df is None:
                key_df = tmp_key
            else:
                key_df = pd.concat([key_df, tmp_key], axis=0)
        if key_df is None:
            continue
        key_df['is_appear'] = is_appear
        key_df['appear_idx'] = trace[k_s_i].idx
        key_df['is_disappear'] = is_disappear
        key_df['disappear_idx'] = trace[k_e_i].idx
        if out is None:
            out = key_df
        else:
            out = pd.concat([out, key_df], axis=0)
    out.to_csv(args.output_file, index=False)


if __name__ == '__main__':
    stable_region_file = r'/home/demo/Documents/datasets/A01-03/matrix/stable_region.txt'
    parser = argparse.ArgumentParser(description='track assess')
    parser.add_argument(
        '-ir',
        '--stable_region_file',
        help='区域文件路径',
        default=stable_region_file
    )
    parser.add_argument(
        '-ic',
        '--channel_name',
        help='track的channel',
        default='omnisense/event/05/boxes'
    )
    parser.add_argument(
        '-if',
        '--record_dir',
        help='record文件路径',
        default='/home/seyond_user/od/SW/A01-02'
    )
    parser.add_argument(
        '-io',
        '--output_file',
        help='record文件路径',
        default='./test_02.csv'
    )
    args = parser.parse_args()
    main(args)
