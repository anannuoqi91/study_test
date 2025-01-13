
from . import _append_path
from utils.util_file import prepare_directory
import subprocess
import pandas as pd
import sys
import re
import os
import atexit


class BashScriptManager:
    def __init__(self, script_path, params: list):
        self._script_path = script_path
        self._process = None
        self._run_command = ['bash', self._script_path]
        self._run_command.extend(params)

    def start_script(self):
        """启动 Bash 脚本并保存进程信息"""
        try:
            # 启动 Bash 脚本
            self._process = subprocess.Popen(
                self._run_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"Bash script started with PID: {self._process.pid}")

            # 注册退出时的清理函数
            atexit.register(self.cleanup)

        except Exception as e:
            print(f"Failed to start the script: {e}")

    def stop_script(self):
        """终止 Bash 脚本的进程"""
        if self._process and self._process.poll() is None:  # 检查进程是否仍在运行
            print(f"Stopping Bash script with PID: {self._process.pid}")
            self._process.terminate()  # 尝试正常终止进程
            try:
                self._process.wait(timeout=5)  # 等待进程结束
            except subprocess.TimeoutExpired:
                print("Process did not terminate in time, killing it.")
                self._process.kill()  # 强制杀死进程

    def cleanup(self):
        """清理操作，包括停止脚本进程"""
        self.stop_script()


class SystemMonitorLog:
    def __init__(self, log_path, key_word='', interval_s=5, time_consuming_min=1) -> None:
        self._log_path = log_path
        self._key_word = key_word
        self._time_interval_s = interval_s
        self._time_consume = time_consuming_min * 60
        self._parse_header = [self._parse_header_top, self._parse_header_task,
                              self._parse_header_CPU, self._parse_header_MEM, self._parse_header_SWAP]
        self._bash_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'system_monitor.bash')
        self._init_df()
        self._bash_manager = BashScriptManager(
            self._bash_path, [self._log_path, str(self._time_interval_s), self._key_word])

    def start_monitor_script(self):
        self._bash_manager.start_script()

    def _init_df(self):
        self._top_df = pd.DataFrame(columns=[
                                    'timestamp', 'Time', 'Uptime', 'Users', 'Load_Average_1', 'Load_Average_5', 'Load_Average_15'])
        self._task_df = pd.DataFrame(columns=[
                                     'timestamp', 'Total_Tasks', 'Running_Tasks', 'Sleeping_Tasks', 'Stopped_Tasks', 'Zombie_Tasks'])
        self._cpu_df = pd.DataFrame(columns=['timestamp', 'Cpu_User', 'Cpu_System', 'Cpu_Nice',
                                    'Cpu_Idle', 'Cpu_Wait', 'Cpu_Hardware_Interrupt', 'Cpu_Software_Interrupt', 'Cpu_Stolen'])
        self._mem_df = pd.DataFrame(
            columns=['timestamp', 'Mem_Total', 'Mem_Free', 'Mem_Used', 'Mem_Buff_Cache'])
        self._swap_df = pd.DataFrame(
            columns=['timestamp', 'Swap_Total', 'Swap_Free', 'Swap_Used', 'Swap_Available'])
        self._process_df = pd.DataFrame(columns=[
                                        'timestamp', 'PID', 'USER', 'PR', 'NI', 'VIRT', 'RES', 'SHR', 'STATE', 'CPU', 'MEM', 'TIME', 'COMMAND'])

    def _parse_header_top(self, txt, timestamp):
        out = {'timestamp': timestamp}
        time_load_pattern = r"^top\s+-\s+(\d{2}:\d{2}:\d{2})\s+up\s+(\d{1,2}:\d{2}),\s+(\d+)\s+user,\s+load\s+average:\s+([\d.]+),\s+([\d.]+),\s+([\d.]+)$"
        time_load_match = re.search(time_load_pattern, txt)
        if time_load_match:
            out['Time'] = time_load_match.group(1)
            out['Uptime'] = time_load_match.group(2)
            out['Users'] = int(time_load_match.group(3))
            out['Load_Average_1'] = float(time_load_match.group(4))
            out['Load_Average_5'] = float(time_load_match.group(5))
            out['Load_Average_15'] = float(time_load_match.group(6))
        self._top_df.loc[len(self._top_df)] = out

    def _parse_header_task(self, txt, timestamp):
        out = {'timestamp': timestamp}
        tasks_pattern = r"^任务:\s+(\d+)\s+total,\s+(\d+)\s+running,\s+(\d+)\s+sleeping,\s+(\d+)\s+stopped,\s+(\d+)\s+zombie$"
        tasks_match = re.search(tasks_pattern, txt)
        if tasks_match:
            out['Total_Tasks'] = int(tasks_match.group(1))
            out['Running_Tasks'] = int(tasks_match.group(2))
            out['Sleeping_Tasks'] = int(tasks_match.group(3))
            out['Stopped_Tasks'] = int(tasks_match.group(4))
            out['Zombie_Tasks'] = int(tasks_match.group(5))
        self._task_df.loc[len(self._task_df)] = out

    def _parse_header_CPU(self, txt, timestamp):
        out = {'timestamp': timestamp}
        cpu_pattern = r"^%Cpu\(s\):\s+([\d.]+)\s+us,\s+([\d.]+)\s+sy,\s+([\d.]+)\s+ni,\s+([\d.]+)\s+id,\s+([\d.]+)\s+wa,\s+([\d.]+)\s+hi,\s+([\d.]+)\s+si,\s+([\d.]+)\s+st$"
        cpu_match = re.search(cpu_pattern, txt)
        if cpu_match:
            out['Cpu_User'] = float(cpu_match.group(1))
            out['Cpu_System'] = float(cpu_match.group(2))
            out['Cpu_Nice'] = float(cpu_match.group(3))
            out['Cpu_Idle'] = float(cpu_match.group(4))
            out['Cpu_Wait'] = float(cpu_match.group(5))
            out['Cpu_Hardware_Interrupt'] = float(cpu_match.group(6))
            out['Cpu_Software_Interrupt'] = float(cpu_match.group(7))
            out['Cpu_Stolen'] = float(cpu_match.group(8))
        self._cpu_df.loc[len(self._cpu_df)] = out

    def _parse_header_MEM(self, txt, timestamp):
        out = {'timestamp': timestamp}
        mem_pattern = r"^MiB Mem\s*:\s+([\d.]+)\s+total,\s+([\d.]+)\s+free,\s+([\d.]+)\s+used,\s+([\d.]+)\s+buff/cache$"
        mem_match = re.search(mem_pattern, txt)
        if mem_match:
            out['Mem_Total'] = float(mem_match.group(1))
            out['Mem_Free'] = float(mem_match.group(2))
            out['Mem_Used'] = float(mem_match.group(3))
            out['Mem_Buff_Cache'] = float(mem_match.group(4))
        self._mem_df.loc[len(self._mem_df)] = out

    def _parse_header_SWAP(self, txt, timestamp):
        out = {'timestamp': timestamp}
        swap_pattern = \
            r"^MiB Swap\s*:\s+([\d.]+)\s+total,\s+([\d.]+)\s+free,\s+([\d.]+)\s+used.\s+([\d.]+)\s+avail\s+Mem\s+$"
        swap_match = re.search(swap_pattern, txt)
        if swap_match:
            out['Swap_Total'] = float(swap_match.group(1))
            out['Swap_Free'] = float(swap_match.group(2))
            out['Swap_Used'] = float(swap_match.group(3))
            out['Swap_Available'] = float(swap_match.group(4))
        self._swap_df.loc[len(self._swap_df)] = out

    def _parse_process(self, txt, timestamp):
        out = {'timestamp': timestamp}
        match = re.match(
            r"^(\d+)\s+(\w+)\s+([-\w]+)\s+(-?\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d:]+\.\d{2})\s+(\S+)$", txt.strip().rstrip('\n'))
        if match:
            out.update({
                'PID': int(match.group(1)),
                'USER': match.group(2),
                'PR': match.group(3),
                'NI': int(match.group(4)),
                'VIRT': match.group(5),
                'RES': int(match.group(6)),
                'SHR': int(match.group(7)),
                'STATE': match.group(8),
                'CPU': float(match.group(9)),
                'MEM': float(match.group(10)),
                'TIME': match.group(11),
                'COMMAND': match.group(12)
            })
            self._process_df.loc[len(self._process_df)] = out

    def parse_log_to_dataframe(self):
        count = 0
        timestamp = -1
        with open(self._log_path, 'r') as file:
            for line in file:
                if "Timestamp:" in line:
                    timestamp = line.strip().split("Timestamp: ")[1]
                    count = 0
                elif count < 5:
                    self._parse_header[count](line, timestamp)
                    count += 1
                else:
                    self._parse_process(line, timestamp)

    @property
    def process_df(self):
        return self._process_df.copy()

    @property
    def header_cpu_df(self):
        return self._cpu_df.copy()

    @property
    def header_top_df(self):
        return self._top_df.copy()

    @property
    def header_task_df(self):
        return self._task_df.copy()

    @property
    def header_mem_df(self):
        return self._mem_df.copy()

    @property
    def header_swap_df(self):
        return self._swap_df.copy()

    def write_log_2_excel(self, out_path):
        prepare_directory(out_path)
        with pd.ExcelWriter(out_path) as writer:
            if not self._process_df.empty:
                self._process_df.to_excel(writer, sheet_name='process',
                                          index=False)
            if not self._cpu_df.empty:
                self._cpu_df.to_excel(writer, sheet_name='header_cpu',
                                      index=False)
            if not self._top_df.empty:
                self._top_df.to_excel(writer, sheet_name='header_top',
                                      index=False)
            if not self._task_df.empty:
                self._task_df.to_excel(writer, sheet_name='header_task',
                                       index=False)
            if not self._mem_df.empty:
                self._mem_df.to_excel(writer, sheet_name='header_mem',
                                      index=False)
            if not self._swap_df.empty:
                self._swap_df.to_excel(writer, sheet_name='header_swap',
                                       index=False)


if __name__ == "__main__":
    c_sys_m = SystemMonitorLog(
        '/home/demo/Documents/code/study_test/resource_usage/top_log.txt')
    # c_sys_m.start_monitor_script()
    try:
        # time.sleep(20)
        c_sys_m.parse_log_to_dataframe()
        c_sys_m.write_log_2_excel(
            '/home/demo/Documents/code/study_test/resource_usage/sys_excel.xlsx')
    except KeyboardInterrupt:
        print("Program interrupted, exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
