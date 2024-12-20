import subprocess
import time
import re
import threading
import atexit
import csv
import importlib.util

package_name = 'pandas'
if importlib.util.find_spec(package_name) is not None:
    import pandas as pd
    PANDAS_USEFUL = True
else:
    PANDAS_USEFUL = False


class FrameMonitor:
    def __init__(self, channels_name=[], interval_s=1, logger=None):
        self._channels_name = channels_name
        self._logger = logger
        self._command = ["cyber_monitor", "-c", ""]
        self._info_df = []
        self._columns = ["timestamp", "name", "frame_rate"]
        self._timestamp = time.time()
        self._is_running = False
        self._thread = None
        self._sleep = interval_s
        self._end = None
        self._setup_bash()

    def _setup_bash(self):
        try:
            result = subprocess.run(
                ["source cyber/setup.bash"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            if self._logger is not None:
                self._logger.error(
                    f"Error occurred: FrameMonitor _setup_bash {e}")
            else:
                print(f"Error occurred: {e}")

    def _get_channel_frame_rate(self, c_name):
        self._command[2] = c_name
        try:
            process = subprocess.Popen(
                self._command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            i = 0
            while i < 3:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    match = re.search(r'(\d+\.\d+)$', output)
                    if match:
                        frame_rate = float(match.group(1).strip())
                        self._info_df.append(
                            [self._timestamp, c_name, frame_rate])
                        break
                i += 1
            process.kill()
        except Exception as e:
            if self._logger is not None:
                self._logger.error(
                    f"Error occurred: FrameMonitor _get_channel_frame_rate {e}")
            else:
                print(f"Error occurred: {e}")
        finally:
            process.kill()

    def _start(self):
        while self._is_running and ((self._end is not None and time.time() < self._end) or self._end is None):
            self._timestamp = time.time()
            for c_name in self._channels_name:
                self._get_channel_frame_rate(c_name)
            time.sleep(self._sleep)

    def start(self):
        self._is_running = True
        self._thread = threading.Thread(target=self._start)
        self._thread.start()
        atexit.register(self.end)

    def end(self):
        self._is_running = False
        if self._thread is not None:
            self._thread.join()

    def info_out(self, out_path):
        if PANDAS_USEFUL:
            df = pd.DataFrame(self._info_df, columns=self._columns)
            df.to_csv(out_path, index=False)
        else:
            with open(out_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(self._columns)
                writer.writerows(self._info_df)

    def set_duration(self, duration_s):
        self._end = time.time() + duration_s


if __name__ == "__main__":
    channel_name_01 = ["omnisense/lidar/01/PointCloud",
                       "omnisense/preprocess/01/parallel_up_dynamic_points",
                       "omnisense/segmentation/01/boxes",
                       "omnisense/track/01/boxes",
                       "omnisense/cluster/01/boxes",
                       "omnisense/bkg/01/outlier_boxes",
                       "omnisense/event/01/web",
                       "omnisense/roi/01/boxes",
                       "omnisense/roi/01/points"]
    channel_name_02 = ["omnisense/lidar/02/PointCloud",
                       "omnisense/preprocess/02/parallel_up_dynamic_points",
                       "omnisense/segmentation/02/boxes",
                       "omnisense/track/02/boxes",
                       "omnisense/cluster/02/boxes",
                       "omnisense/bkg/02/outlier_boxes",
                       "omnisense/event/02/web",
                       "omnisense/roi/02/boxes",
                       "omnisense/roi/02/points"]
    channel_name_fusion = ["omnisense/track_fusion/boxes",
                           "omnisense/roi/03/boxes", "omnisense/roi/03/points"]
    frame_m = FrameMonitor(channel_name_01)
    frame_m.start()
    time.sleep(10)  # Wait for 10 seconds
    frame_m.end()
    frame_m.info_out("frame_rate.csv")
