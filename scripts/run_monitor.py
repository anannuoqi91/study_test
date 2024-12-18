import requests
import glob
import importlib
import json
import logging
import subprocess
import sys
import copy
from threading import Event


class DiskManager:
    def __init__(self, mount_path):
        self._mount_path = mount_path
        self._mount_point = None
        self._is_mounted = False
        self._usage = None
        self._available = None
        self._available_rate = None
        self.smart_mount_info = {}
        self.check()

    def check(self):
        self._mount_disk()
        self.check_mounted()
        self.smart_mount()

    def check_mounted(self):
        if not self._mount_point:
            return
        try:
            self._is_mounted = False
            result = subprocess.run(
                ['df', '-h'], capture_output=True, text=True, check=True)
            if self._mount_point in result.stdout:
                lines = result.stdout.splitlines()
                if len(lines) > 1:
                    for i in range(1, len(lines)):
                        info = lines[i].split()
                        if self._mount_point == info[0]:
                            self._usage = self._usage_number(info[1])
                            self._available = self._usage_number(info[3])
                            self._is_mounted = True
        except subprocess.CalledProcessError:
            self._is_mounted = False

    def _usage_number(self, number_str):
        if 'G' in number_str:
            return float(number_str.replace('G', ''))
        if 'M' in number_str:
            return float(number_str.replace('M', '')) / 1024
        if 'KB' in number_str:
            return float(number_str.replace('KB', '')) / 1024 / 1024
        if 'B' in number_str:
            return float(number_str.replace('B', '')) / 1024 / 1024
        return float(number_str)

    def _mount_disk(self):
        try:
            # 使用 df 命令获取挂载信息
            result = subprocess.run(
                ['df', '-h', self._mount_path], capture_output=True, text=True, check=True)
            lines = result.stdout.splitlines()
            if len(lines) > 1:
                # 提取文件系统和挂载点
                info = lines[1].split()
                self._mount_point = info[0]
                self._usage = self._usage_number(info[1])
                self._available = self._usage_number(info[3])
                self._available_rate = round(self._available / self._usage, 4)
                self._is_mounted = True
        except Exception as e:
            print(f"Error retrieving disk information: {e}")
            return None, None, None

    def smart_mount(self):
        self.smart_mount_info = {
            'warning': 'No',
            'temperature': -1,
        }
        try:
            result = subprocess.run(['sudo', 'smartctl', '-a', self._mount_point],
                                    capture_output=True, text=True, check=True)
            if 'SMART/Health Information (NVMe Log 0x02)' in result.stdout:
                tmp_re = result.stdout.split(
                    'SMART/Health Information (NVMe Log 0x02)')[1].split('\n')
                for i in tmp_re:
                    if 'Critical Warning' in i:
                        i_split = i.split('Critical Warning:')
                        if len(i_split) > 1:
                            warn = i_split[1].replace(' ', '')
                            if warn != '0x00':
                                self.smart_mount_info['warning'] = warn
                    elif 'Temperature' in i:
                        i_split = i.split('Temperature:')
                        if len(i_split) > 1:
                            warn = i_split[1].strip().split(' ')
                            if len(warn) > 1:
                                self.smart_mount_info['temperature'] = float(
                                    warn[0])
                    elif 'Available Spare' in i:
                        i_split = i.split('Available Spare:')
                        if len(i_split) > 1:
                            warn = i_split[1].strip()
                            self.smart_mount_info['available_spare'] = warn
                    elif 'Available Spare Threshold' in i:
                        i_split = i.split('Available Spare Threshold:')
                        if len(i_split) > 1:
                            warn = i_split[1].strip()
                            self.smart_mount_info['available_spare_threshold'] = warn
                    elif 'Percentage Used' in i:
                        i_split = i.split('Percentage Used:')
                        if len(i_split) > 1:
                            warn = i_split[1].strip()
                            self.smart_mount_info['percentage_used'] = warn
                    elif 'Media and Data Integrity Errors' in i:
                        i_split = i.split('Media and Data Integrity Errors:')
                        if len(i_split) > 1:
                            warn = i_split[1].strip()
                            self.smart_mount_info['media_data_integrity_errors'] = int(
                                warn)
                    elif 'Error Information Log Entries' in i:
                        i_split = i.split('Error Information Log Entries:')
                        if len(i_split) > 1:
                            warn = i_split[1].strip()
                            self.smart_mount_info['error_information_log_entries'] = int(
                                warn)
                    elif 'Warning Comp. Temperature Time' in i:
                        i_split = i.split('Warning Comp. Temperature Time:')
                        if len(i_split) > 1:
                            warn = i_split[1].strip()
                            self.smart_mount_info['warning_comp_temperature_time'] = int(
                                warn)
                    elif 'Critical Comp. Temperature Time' in i:
                        i_split = i.split('Critical Comp. Temperature Time:')
                        if len(i_split) > 1:
                            warn = i_split[1].strip()
                            self.smart_mount_info['critical_comp_temperature_time'] = int(
                                warn)
        except subprocess.CalledProcessError as e:
            logging.error(
                f"[SMART/Health Information] Error mounting disk: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    @property
    def is_mounted(self):
        return self._is_mounted

    @property
    def mount_point(self):
        return self._mount_point

    @property
    def usage(self):
        return self._usage

    @property
    def available(self):
        return self._available

    @property
    def available_rate(self):
        return self._available_rate


class Container:
    def __init__(self, name) -> None:
        self.name = name
        self.exists = False
        self.running = False
        self._init_mounts_source = {}
        self._mounts_source = {}
        self._mounts_status = False
        self._not_mounted = []
        self._mounts_diff = None
        self._other_info = {}

        self.check()

    def check(self):
        result = run_command(f"docker inspect {self.name}")
        if result:
            container_info = json.loads(result)[0]
            state = container_info.get('State', None)
            is_running = state['Running'] if state else False
            self.exists = True
            self.running = is_running
            if not self._mounts_source:
                self._decode_mounts(container_info)
        self.check_mounts_status()
        self.check_mounts_diff()

    def check_mounts_diff(self):
        if self._init_mounts_source.keys() == self._mounts_source.keys():
            return
        for key, v in self._init_mounts_source.items():
            if key in self._mounts_source:
                continue
            v['mount'].check()
            if not v['mount'].is_mounted:
                self._mounts_diff = f"{key} is change bind container [{self.name}] and not mounted."
        self._mounts_diff = f"container [{self.name}] mounts change but mounted."

    def _decode_mounts(self, inspect_info):
        self._mounts_source = {}
        mounts = inspect_info.get('Mounts', [])
        for tmp_info in mounts:
            if tmp_info['Type'] == 'bind':
                tmp = DiskManager(tmp_info['Source'])
                if tmp.mount_point not in self._mounts_source:
                    self._mounts_source[tmp.mount_point] = {
                        'mount': tmp,
                        'destination': [tmp_info['Destination']]
                    }
                else:
                    self._mounts_source[tmp.mount_point]['destination'].append(
                        tmp_info['Destination'])
        if not self._init_mounts_source:
            self._init_mounts_source = copy.deepcopy(self._mounts_source)

    def start(self):
        re = run_command(f"docker start {self.name}")
        if re and re == self.name:
            self.exists = True
            self.running = True
            return True
        else:
            return False

    def restart(self):
        re = run_command(f"docker restart {self.name}")
        if re and re == self.name:
            self.exists = True
            self.running = True
            return True
        else:
            return False

    def stop(self):
        re = run_command(f"docker stop {self.name}")
        if re and re == self.name:
            self.exists = True
            self.running = False
            return True
        else:
            return False

    def check_mounts_status(self):
        self._not_mounted = []
        self._mounts_status = True
        for mount_p, v in self._mounts_source.items():
            v['mount'].check()
            if not v['mount'].is_mounted:
                self._mounts_status = False
                self._not_mounted.append(mount_p)

    def mounts_status(self):
        return self._mounts_status, self._not_mounted

    @setattr
    def other_info(self, **args):
        self._other_info = copy.deepcopy(args)

    @property
    def other_info(self):
        return copy.deepcopy(self._other_info)

    @property
    def mounts_info(self):
        if self._mounts_status:
            return "All mounted."
        else:
            return f"Not mounted: {', '.join(self._not_mounted)}"

    def ssd_usage_warn(self, threshold=0.3):
        out = {}
        for mount_p, v in self._mounts_source.items():
            v['mount'].check()
            if v['mount'].is_mounted:
                if v['mount'].available_rate < threshold:
                    out[mount_p] = {
                        'message': f'Insufficient mem: {mount_p} bind {self.name}',
                        'usage': v['mount'].usage,
                        'available': v['mount'].available,
                        'available_rate': v['mount'].available_rate
                    }
        return out

    @property
    def mounts_source(self):
        return self._mounts_source

    @property
    def mounts_diff(self):
        return self._mounts_diff


def od_is_crash(container='OmniVidi_VL'):
    global MONITORED_DOCKER_CONTAINERS
    data = {}
    try:
        result = run_command(f"docker logs {container} --tail 1000")
        if 'exit, respawn' in result and 'INFO Start process' in result:
            tmp = result.split(' exit, respawn!')
            if len(tmp) > 1:
                tmp = tmp[1].split('INFO Start process [')
                if len(tmp) > 1:
                    tmp = tmp[1].split(']')[0]
                    data[tmp] = 'crash'
                    MONITORED_DOCKER_CONTAINERS[container].other_info = {
                        'od_crash': {
                            'module': tmp,
                            'restart': False
                        }
                    }
                    return data
    except subprocess.CalledProcessError as e:
        logging.error(
            f"[OD Check] Error checking OD status of container {container}: {e}")
    return data


def install_and_import(package):
    try:
        importlib.import_module(package)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package])
    # finally:
    #     globals()[package] = importlib.import_module(package)


def install_smartctl():
    try:
        subprocess.run(['sudo', 'apt-get', 'update'], check=True)
        # 安装 smartmontools
        subprocess.run(['sudo', 'apt-get', 'install',
                       '-y', 'smartmontools'], check=True)
        print("smartmontools install compeleted.")
    except subprocess.CalledProcessError as e:
        print(f"Faile: {e}")


install_and_import('requests')


def run_command(command):
    """执行命令并返回输出"""
    logging.info(f"Command executed: {command}")
    try:
        result = subprocess.run(
            command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode('utf-8').strip()
    except subprocess.CalledProcessError as e:
        logging.error(
            f"Command failed: {command}\nError: {e.stderr.decode('utf-8')}")
        return None


def get_thermal_temperature():
    """
    For Jetson devices, the thermal zone contains:
        CPU-therm, GPU-therm, Tdiode_tegra,
        CV0-therm, CV1-therm, CV2-therm,
        SOC0-therm, SOC1-therm, SOC2-therm,
        tj-therm, Tboard_tegra, e.t.c
    For X86 devices, the thermal zone contains:
        x86_pkg_temp, e.t.c
    :return: a dictionary containing the temperature of each thermal zone
    """
    try:
        thermal_zones_temp = glob.glob('/sys/class/thermal/thermal_zone*/temp')
        thermal_zones_type = glob.glob('/sys/class/thermal/thermal_zone*/type')
        temperatures = {}
        for i in range(len(thermal_zones_temp)):
            with open(thermal_zones_temp[i], 'r') as f:
                temperature = f.read().strip()
                temperature = int(temperature) / 1000
                with open(thermal_zones_type[i], 'r') as f:
                    sensor_type = f.read().strip()
                    if temperatures.get(sensor_type, None):
                        temperatures[sensor_type] += [temperature]
                    else:
                        temperatures[sensor_type] = [temperature]
        return temperatures
    except Exception as e:
        print(f"An error occurred: {e}")
    return None


def is_jetson_device():
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            return True
    except FileNotFoundError:
        pass
    return False


def get_nvidia_gpu_temp_via_nvidia_smi():
    try:
        result = run_command(
            "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader")
        if result is not None:
            return float(result)
    except Exception as e:
        print(f"An error occurred: {e}")
    return None


def get_cpu_gpu_temperature():
    if is_jetson_device():
        sys_temp = get_thermal_temperature()
        cpu_temp = sys_temp.get('CPU-therm', None)
        avg_cpu_temp = sum(cpu_temp) / len(cpu_temp) if cpu_temp else None
        gpu_temp = sys_temp.get('GPU-therm', None)
        avg_gpu_temp = sum(gpu_temp) / len(gpu_temp) if gpu_temp else None
        return avg_cpu_temp, avg_gpu_temp
    else:
        sys_temp = get_thermal_temperature()
        cpu_temp = sys_temp.get('x86_pkg_temp', None)
        avg_cpu_temp = sum(cpu_temp) / len(cpu_temp) if cpu_temp else None
        avg_gpu_temp = get_nvidia_gpu_temp_via_nvidia_smi()
        return avg_cpu_temp, avg_gpu_temp


def get_cpu_usage():
    try:
        result = run_command(
            "top -bn1 | awk '/^%Cpu/{printf(\"%.1f\", 100-$8)}'")
        if result is not None:
            return float(result.rstrip('%'))
    except Exception as e:
        print(f"An error occurred: {e}")
    return None


def get_memory_usage():
    try:
        result = run_command(
            "free -m | awk 'NR==2{printf \"%.2f\", $3*100/$2 }'")
        if result is not None:
            return float(result)
    except Exception as e:
        print(f"An error occurred: {e}")
    return None


# 设置日志记录
logging.basicConfig(
    filename='system_info_script.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_system_info():
    """获取系统信息"""
    cpu_t, gpu_t = get_cpu_gpu_temperature()
    info = {
        'cpu_usage': get_cpu_usage(),
        'memory_usage': get_memory_usage(),
        'cpu_temp': cpu_t,
        'gpu_temp': gpu_t
    }
    return info


def check_mountpoint(temperature_thres=10):
    global MONITORED_MOUNT_POINTS, MONITORED_DOCKER_CONTAINERS
    data = {}
    for mountpoint in MONITORED_MOUNT_POINTS:
        mountpoint.check()
        if not mountpoint.is_mounted:
            data[
                mountpoint.mount_point] = f"Mount point does not exist. Temperature = {mountpoint.smart_mount_info.get('temperature', -1)}"
        elif mountpoint.smart_mount_info.get('temperature', -1) > temperature_thres:
            data[
                mountpoint.mount_point] = f"Temperature is too high. Temperature = {mountpoint.smart_mount_info.get('temperature', -1)}"
        else:
            data[mountpoint.mount_point] = "OK"
    for container, v in MONITORED_DOCKER_CONTAINERS.items():
        if v.mounts_status()[0]:
            continue
        for i in v.mounts_status()[1]:
            if i in data:
                continue
            data[i] = f"Mount(binds {container}) point does not exist. Temperature = {v._mounts_source[i].get('temperature', -1)}"
    return data


def send_to_server(data):
    """将数据发送到远程服务器"""
    url = "http://yourserver.com/api/data"
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            logging.info("Data sent successfully")
        else:
            logging.error(
                f"Failed to send data. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")


def check(od_container='OmniVidi_VL'):
    global MONITORED_DOCKER_CONTAINERS
    all_info = {
        'system_info': get_system_info(),
        'docker_info': check_containers_state(),
        'mount_point': check_mountpoint(),
        'od_crash': od_is_crash(od_container)
    }
    msgs = []
    # check cpu & gpu temperature
    if all_info['system_info']['cpu_temp'] > 90:
        msgs.append("CPU temperature is too high!")
    if all_info['system_info']['gpu_temp'] > 90:
        msgs.append("GPU temperature is too high!")
    # check docker containers
    for container, info in all_info['docker_info'].items():
        if not info['exists']:
            msgs.append(f"Container '{container}' does not exist!")
        elif not info['running']:
            msgs.append(f"Container '{container}' is not running!")
    # check mount points
    for mountpoint, status in all_info['mount_point'].items():
        if status != 'OK':
            msgs.append(f"Mount point '{mountpoint}' does not exist!")
    for module, status in all_info['od_crash'].items():
        msgs.append(f"{module} exit, respawn !")
    if msgs:
        all_info['msg'] = " ".join(msgs)
    logging.info(f"Collected info: {all_info}")

    if msgs:
        msg = " ".join(msgs)
        status = "down"
    else:
        msg = "OK"
        status = "up"

    return status, msg


def send_status(url, status, msg):
    # 发送请求
    payload = {'status': status, 'msg': msg}
    response = requests.get(url, params=payload)
    # 打印响应状态
    print(
        f"Sent status: {status}, message: {msg}, response: {response.status_code}")


def main_loop(stop_event, push_url):
    stop_event.wait(1)
    while not stop_event.is_set():
        status, msg = check()
        send_status(push_url, status, msg)
        control_containers()
        stop_event.wait(INTERVAL_SEC)


def check_containers_state():
    global MONITORED_DOCKER_CONTAINERS
    out_info = {}
    for container, info in MONITORED_DOCKER_CONTAINERS.items():
        info.check()
        out_info[container] = {
            'exists': True,
            'running': True,
            'mounts_info': info.mounts_info,
            'ssd_usage': info.ssd_usage_warn(),
            'mounts_diff': info.mounts_diff,
        }
        if not info.exists:
            logging.error(f"Container '{container}' does not exist!")
            out_info[container]['exists'] = False
            out_info[container]['running'] = False
        elif not info.running:
            logging.error(f"Container '{container}' is not running!")
            out_info[container]['running'] = False
        if out_info[container]['ssd_usage']:
            logging.error(f"Container '{container}' ssd usage warn!")
        if not info.mounts_diff:
            out_info[container]['running'] = False
            logging.error(f"Container '{container}' {info.mounts_diff}!")
    return out_info


def control_containers():
    for container, info in MONITORED_DOCKER_CONTAINERS.items():
        if info.running:
            if info.other_info.get('od_crash', None):
                if not info.other_info['od_crash'].get(['restart'], False):
                    logging.error(
                        f"Container '{container}' od crash restart!")
                    info.restart()
                    info.other_info['od_crash']['restart'] = True
                else:
                    logging.info(
                        f"Container '{container}' od crash not restart!")
        else:
            if info.exists:
                if info.mounts_diff or info.mounts_status()[0]:
                    info.start()
                else:
                    logging.error(
                        f"Container '{container}' can't start because of mount point or status!")
            else:
                logging.error(f"Container '{container}' does not exist!")


def init_monitos(**args):
    global MONITORED_DOCKER_CONTAINERS, MONITORED_MOUNT_POINTS
    containers = args['containers']
    MONITORED_MOUNT_POINTS = [DiskManager(i) for i in args['mounts']]
    # 初始检查
    logging.info(f"Initial check: {containers}")
    for container in containers:
        if container not in MONITORED_DOCKER_CONTAINERS:
            MONITORED_DOCKER_CONTAINERS[container] = Container(container)


if __name__ == '__main__':
    INTERVAL_SEC = 10
    MONITORED_DOCKER_CONTAINERS = {}
    MONITORED_MOUNT_POINTS = []

    containers = [
        "OmniVidi_VL",
        "city-admin",
        "mysql-sc",
        "nginx-web-sc",
        "redis-sc"
    ]
    mounts = ['/']

    push_url = 'http://localhost:3001/api/push/6MAFL4VS78'
    stop_event = Event()
    init_monitos(containers=containers, mounts=mounts)
    try:
        main_loop(stop_event, push_url)
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt. Stopping...")
        stop_event.set()

    # od_is_crash('OmniVidi_VL')
