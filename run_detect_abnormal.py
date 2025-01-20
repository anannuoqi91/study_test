import shapely.geometry as geo
from cyber_record.record import Record
import pandas as pd
import math
import argparse
import os
from multiprocessing import Pool
from tqdm import tqdm


def norm_angle_radius(angle: float) -> float:
    """
        Normalize an angle to [0, 2π).

        Parameters
        ----------
        angle : float
            The angle to be normalized.

        Returns
        -------
        float
            The normalized angle.
        """

    two_pi = 2.0 * math.pi
    angle_norm = angle % two_pi
    return angle_norm if angle_norm >= 0 else angle_norm + two_pi


class Box:
    def __init__(self, box=None):
        if box is not None:
            self.set_value_from_box(box)
        else:
            self._init_values()

    def _init_values(self):
        self.timestamp = None
        self.position_x = None
        self.position_y = None
        self.position_z = None
        self.pose_w = None
        self.pose_x = None
        self.pose_y = None
        self.pose_z = None
        self.length = None
        self.height = None
        self.width = None
        self.track_id = None
        self.object_type = None
        self.heading = None
        self.idx = None
        self.poly = None
        self.bounds = None
        self.region = None

    def set_value_from_box(self, box):
        self.timestamp = box.timestamp
        self.position_x = box.position_x
        self.position_y = box.position_y
        self.position_z = box.position_z
        self.pose_w = box.pose_w
        self.pose_x = box.pose_x
        self.pose_y = box.pose_y
        self.pose_z = box.pose_z
        self.length = box.length
        self.height = box.height
        self.width = box.width
        self.track_id = box.track_id
        self.object_type = box.object_type
        self.heading = box.spindle / 100.0
        self.speed = box.speed

    def set_idx(self, idx):
        self.idx = idx

    def cal_car(self):
        y, z = self.position_y, self.position_z
        l, w = self.length, self.width
        heading_azimuth_rad = norm_angle_radius(self.heading)
        out = {}
        # front-left, back-left, back-right, front-right
        cosval = math.cos(heading_azimuth_rad)
        sinval = math.sin(heading_azimuth_rad)
        dz_fl = 0.5 * l * cosval - 0.5 * w * sinval
        dy_fl = 0.5 * l * sinval + 0.5 * w * cosval

        dz_bl = -0.5 * l * cosval - 0.5 * w * sinval
        dy_bl = -0.5 * l * sinval + 0.5 * w * cosval

        front_left = [y + dy_fl, z + dz_fl]
        back_left = [y + dy_bl, z + dz_bl]
        back_right = [y - dy_fl, z - dz_fl]
        front_right = [y - dy_bl, z - dz_bl]
        return [front_left, back_left, back_right, front_right]


def read_file(args):
    filepath = args['filepath']
    filename = os.path.basename(filepath)
    topic_name = args['topic_name']
    box_type = args.get('box_type', [])
    traces = {}
    start_idx = 1e10
    end_idx = -1
    record = Record(filepath)
    for topic, message, t in record.read_messages(topic_name):
        end_idx = max(end_idx, message.idx)
        start_idx = min(start_idx, message.idx)
        for box in message.box:
            if box_type and box.object_type not in box_type:
                continue
            tmp = Box(box)
            tmp.set_idx(message.idx)
            if tmp.track_id in traces:
                trace = traces[tmp.track_id]
                trace.append(tmp)
                traces[tmp.track_id] = trace
            else:
                traces[tmp.track_id] = [tmp]

    return filename, traces, start_idx, end_idx


def get_traces_from_record(filepath,
                           topic_name='omnisense/event/05/boxes',
                           filter_num=0,
                           box_type=[], pool_num=4):
    traces = {}
    start_idx = 1e10
    end_idx = -1
    file_num = len(filepath)
    cut_arg = []
    sort_filename = []
    # for i in range(file_num):
    for i in tqdm(range(file_num), desc='get traces from record'):
        sort_filename.append(os.path.basename(filepath[i]))
        cut_arg.append(
            {'filepath': filepath[i], 'topic_name': topic_name, 'box_type': box_type})
        if (len(cut_arg) == pool_num) or \
                ((i == file_num - 1) and len(cut_arg) > 0):
            with Pool(pool_num) as pool:
                results = pool.starmap(read_file, [(arg,) for arg in cut_arg])
                for tmp in results:
                    traces[tmp[0]] = tmp[1]
                    start_idx = min(start_idx, tmp[2])
                    end_idx = max(end_idx, tmp[3])
            cut_arg = []
    # sort
    new_traces = {}
    sort_filename.sort()
    for i in sort_filename:
        tmp = traces[i]
        for key in tmp:
            if key in new_traces:
                new_traces[key] += tmp[key]
            else:
                new_traces[key] = tmp[key]
    if filter_num > 0:
        for key in new_traces:
            if len(new_traces[key]) < filter_num:
                del new_traces[key]
    return new_traces, start_idx, end_idx


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


def detect_trace(args):
    trace = args['trace']
    stable_region = args['stable_region']
    start_idx = args['start_idx']
    end_idx = args['end_idx']
    trace_df = None
    k_trace_time = cut_trace(trace, stable_region)
    if len(k_trace_time) == 0:
        return trace_df
    k_s_i = k_trace_time[0][0]
    k_e_i = k_trace_time[-1][1]
    is_appear = appear_unexpected(
        trace[k_s_i].idx, start_idx, trace[0].idx)
    is_disappear = disappear_unexpected(
        trace[k_e_i].idx, end_idx, trace[-1].idx)

    for i in k_trace_time:
        tmp_key = detect_change(i, trace)
        if len(tmp_key) == 0:
            continue
        if trace_df is None:
            trace_df = tmp_key
        else:
            trace_df = pd.concat([trace_df, tmp_key], axis=0)
    if trace_df is None:
        return trace_df
    trace_df['is_appear'] = is_appear
    trace_df['appear_idx'] = trace[k_s_i].idx
    trace_df['is_disappear'] = is_disappear
    trace_df['disappear_idx'] = trace[k_e_i].idx
    return trace_df


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
        raise ValueError('file path is null')
    file_path.sort()
    file_path = [os.path.join(args.record_dir, i) for i in file_path]
    traces, start_idx, end_idx = get_traces_from_record(
        file_path, topic_name=args.channel_name)
    if len(traces) == 0:
        raise ValueError('traces is null')
    out = None
    traces_num = len(traces)
    start_i = 0
    cut_arg = []
    # for key in traces:
    for key in tqdm(traces.keys(), desc='Detecting traces'):
        trace = traces[key]
        cut_arg.append({'trace': trace, 'stable_region': stable_region,
                       'start_idx': start_idx, 'end_idx': end_idx})
        start_i += 1
        if (len(cut_arg) == args.pool_num) or \
                ((start_i == traces_num) and len(cut_arg) > 0):
            with Pool(args.pool_num) as pool:
                results = pool.starmap(
                    detect_trace, [(arg,) for arg in cut_arg])
                for tmp in results:
                    if tmp is None:
                        continue
                    if out is None:
                        out = tmp
                    else:
                        out = pd.concat([out, tmp], axis=0)
            cut_arg = []
    if out is None:
        raise ValueError('out is null')
    out.to_csv(args.output_file, index=False)


def parse_args():
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
        default='/home/seyond_user/od/SW/A01-03'
    )
    parser.add_argument(
        '-io',
        '--output_file',
        help='record文件路径',
        default='./test_03.csv'
    )
    parser.add_argument(
        '-p',
        '--pool_num',
        help='并行线程数',
        default=4
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
