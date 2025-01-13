import os
import subprocess
from pyntcloud import PyntCloud
from utils.util_file import mkdir_directory, list_files_in_current_directory
from utils.util_pcd import write_pcd


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
SDK_PATH = os.path.join(CUR_DIR, 'get_pcd_3.102.6_x86')


def prepare_getpcd_cmd(inno_pc, out_dir, frame_start=0, frame_number=1, file_number=60*60, sdk=SDK_PATH, with_timestamp=False):
    if not inno_pc.endswith('.inno_pc'):
        raise Exception('inno_pc must be .inno_pc')
    file_name = os.path.basename(inno_pc)
    file_name = file_name.split('.')[0]
    oud_dir = os.path.join(out_dir, file_name)
    oud_dir = os.path.abspath(oud_dir)
    mkdir_directory(oud_dir)
    if not oud_dir.endswith('/'):
        oud_dir += '/'
    cmd = [
        os.path.abspath(sdk),
        "--inno-pc-filename", inno_pc,
        "--frame-start", str(frame_start),
        "--frame-number", str(frame_number),
        "--file-number", str(file_number)]
    if with_timestamp:
        cmd.extend(["--output-filename-with-timestamp", f"{oud_dir}.pcd"])
    else:
        cmd.extend(["--output-filename", f"{oud_dir}{file_name}.pcd"])
    return cmd, oud_dir


def rewrite_pcd(files):
    for file in files:
        cloud = PyntCloud.from_file(file)
        points_df = cloud.points
        points_df['time_s'] = cloud.points['timestamp'].apply(lambda x: int(x))
        points_df['time_ms'] = cloud.points.apply(lambda row: (
            row['timestamp'] - row['time_s']) * 1000, axis=1)
        points_df.drop('timestamp', axis=1, inplace=True)
        column_order = ['x', 'y', 'z', 'reflectance', 'channel', 'roi', 'facet', 'is_2nd_return', 'multi_return',
                        'confid_level', 'flag', 'elongation', 'time_s', 'time_ms', 'scanline', 'scan_idx', 'frame_id', 'ring_id']
        points = points_df[column_order].values.tolist()
        num_points = len(points)
        if num_points == 0 or len(points[0]) != 18:
            raise ValueError("points value not attached to headers .")
        headers = ['x:F', 'y:F', 'z:F', 'reflectance:U', 'channel:U', 'roi:U', 'facet:U', 'is_2nd_return:U', 'multi_return:U',
                   'confid_level:U', 'flag:U', 'elongation:U', 'time_s:I32', 'time_ms:F', 'scanline:U', 'scan_idx:U', 'frame_id:U', 'ring_id:U']
        write_pcd(file, points, headers)


def get_org_pcd(inno_pc, out_dir, frame_start=0, frame_number=1, file_number=60*60, sdk=SDK_PATH, with_timestamp=False):
    cmd, oud_dir = prepare_getpcd_cmd(
        inno_pc, out_dir, frame_start=frame_start, frame_number=frame_number, file_number=file_number, sdk=sdk, with_timestamp=with_timestamp)
    subprocess.run(cmd, capture_output=True, text=True)
    return list_files_in_current_directory(oud_dir, '.pcd')
