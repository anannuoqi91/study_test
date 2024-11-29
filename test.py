import requests
import glob
import importlib
import json
import logging
import subprocess
import sys
from threading import Event


def install_and_import(package):
    try:
        importlib.import_module(package)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package])
    # finally:
    #     globals()[package] = importlib.import_module(package)


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


def check_mountpoint_exists(mountpoint):
    """检查挂载点是否存在"""
    try:
        subprocess.run(['mountpoint', '-q', mountpoint], check=True)
        return True
    except subprocess.CalledProcessError:
        return False


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


def check_containers(expected_containers):
    data = {}
    for container in expected_containers:
        result = run_command(f"docker inspect {container}")
        if result:
            container_info = json.loads(result)[0]
            state = container_info.get('State', None)
            data[container] = {
                'exists': True,
                'running': state['Running'] if state else False,
                'state': state
            }
        else:
            data[container] = {
                'exists': False,
                'running': False,
                'state': None
            }
    return data


def check_mountpoint(mountpoints):
    data = {}
    for mountpoint in mountpoints:
        if not check_mountpoint_exists(mountpoint):
            data[mountpoint] = "Mount point does not exist"
        else:
            data[mountpoint] = "OK"
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


def check():
    all_info = {
        'system_info': get_system_info(),
        'docker_info': check_containers(MONITORED_DOCKER_CONTAINERS),
        'mount_point': check_mountpoint(MONITORED_MOUNT_POINTS)
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


def main_loop(interval_sec, stop_event):
    stop_event.wait(1)
    while not stop_event.is_set():
        status, msg = check()
        send_status(PUSH_URL, status, msg)
        stop_event.wait(interval_sec)


if __name__ == '__main__':
    INTERVAL_SEC = 10
    MONITORED_DOCKER_CONTAINERS = [
        "OmniVidi_VL",
        "city-admin",
        "mysql-sc",
        "nginx-web-sc",
        "redis-sc"
    ]
    MONITORED_MOUNT_POINTS = [

    ]
    PUSH_URL = 'http://localhost:3001/api/push/oXYrjPqHof'
    stop_event = Event()
    check_containers(MONITORED_DOCKER_CONTAINERS)
    try:
        main_loop(INTERVAL_SEC, stop_event)
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt. Stopping...")
        stop_event.set()
